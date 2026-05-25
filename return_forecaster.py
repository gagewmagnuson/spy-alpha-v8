"""
SPY Alpha v7 — Hierarchical Return Forecaster
=================================================

Core architectural innovation in v7: two-layer forecasting.

Layer 1 — Regime → ETF/Sector Returns:
    Identical to v6's proven ensemble (60% historical regime estimator + 40% Ridge).
    Predicts expected returns for all ETFs in the trading universe
    (Tier 1 macro + Tier 2 sectors + Tier 3 thematic = 22 ETFs).

Layer 2 — Sector → Stock Alpha:
    Price-only relative momentum model. Computes alpha score for each
    individual stock relative to its parent sector ETF.
    No earnings or fundamental data in v7.0.

Output:
    Single DataFrame of expected returns for ALL ~33 assets, plus
    confidence scores and method labels. The asset selector consumes
    this to choose the portfolio.

Design Principles (from v6):
    - No binary decisions: forecasts are continuous and probability-weighted.
    - Graceful degradation: low conviction → momentum fallback → equal weight.
    - Uncertainty is a first-class output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data_pipeline import (
    TIER1_MACRO,
    TIER2_SECTORS,
    TIER3_THEMATIC,
    TIER4_STOCKS,
    STOCK_SECTOR_MAP,
)

logger = logging.getLogger("spy_alpha_v8.return_forecaster")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# All ETFs that Layer 1 forecasts (Tier 1 macro + Tier 2 sectors + Tier 3 thematic)
# V8: UPRO and SHY now included — they participate in normal forecasting
# and selection. Leverage and defense handled by meta-allocator and risk engine.
LAYER1_ETFS: List[str] = TIER1_MACRO + TIER2_SECTORS + TIER3_THEMATIC

# Sector groups by economic sensitivity (used for momentum fallback)
SECTOR_GROUPS: Dict[str, List[str]] = {
    "cyclical":  ["XLK", "XLY", "XLF", "XLI", "XLB", "SMH"],
    "defensive": ["XLV", "XLP", "XLU", "XLRE", "XBI"],
    "mixed":     ["XLE", "XLC", "XME"],
}

# Which sector groups to favor in each regime (for fallback allocation)
REGIME_SECTOR_PREFERENCE: Dict[str, List[str]] = {
    "Bull":              ["cyclical", "mixed"],
    "Slowdown":          ["defensive", "mixed"],
    "Crisis-Deflation":  ["defensive"],
    "Crisis-Inflation":  ["mixed"],  # commodities, pricing power
    "Crisis":            ["defensive"],  # v6 compat
    "Inflation":         ["cyclical", "mixed"],
}

DEFAULT_HORIZON = 21  # trading days (~1 month)
MIN_REGIME_SAMPLES = 63  # ~3 months

# Conviction thresholds for fallback cascade
HIGH_CONVICTION_THRESHOLD = 0.65
LOW_CONVICTION_THRESHOLD = 0.40

# Layer 2 configuration
ALPHA_SCALING_FACTOR = 0.30  # How much stock alpha adjusts sector forecast
LAYER2_LOOKBACK = 126  # 6 months of relative performance data


# ---------------------------------------------------------------------------
# Forecast Output
# ---------------------------------------------------------------------------

@dataclass
class AssetForecast:
    """
    Container for multi-asset return forecasts with uncertainty.

    v7 change from v6's SectorForecast: now covers all ~33 assets,
    not just 11 sector ETFs.
    """
    # Point forecasts: expected return per asset (annualized)
    expected_returns: pd.Series              # index = asset tickers

    # Confidence per asset (0-1 scale)
    confidence: pd.Series                    # index = asset tickers

    # Overall forecast confidence (scalar)
    overall_confidence: float

    # Forecast method used per asset
    method: pd.Series                        # index = asset tickers

    # Regime context
    regime_probs: pd.Series                  # current regime probabilities
    dominant_regime: str
    regime_conviction: float

    # Stock alpha scores (Layer 2 output, stocks only)
    stock_alpha_scores: Dict[str, float] = field(default_factory=dict)

    # Metadata
    forecast_date: str = ""
    horizon_days: int = 21
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layer 1: Regime-Conditioned Historical Estimator
# ---------------------------------------------------------------------------

class HistoricalRegimeEstimator:
    """
    Estimate ETF returns conditioned on regime using historical averages.

    For each regime, computes the average forward return of each ETF
    during historical periods when that regime was dominant.
    """

    def __init__(self, horizon: int = DEFAULT_HORIZON, min_samples: int = MIN_REGIME_SAMPLES):
        self.horizon = horizon
        self.min_samples = min_samples
        self._regime_returns: Optional[pd.DataFrame] = None
        self._regime_vols: Optional[pd.DataFrame] = None
        self._regime_counts: Optional[pd.Series] = None

    def fit(
        self,
        adj_close: pd.DataFrame,
        regime_probs: pd.DataFrame,
        assets: Optional[List[str]] = None,
    ) -> "HistoricalRegimeEstimator":
        """Compute historical regime-conditioned return statistics."""
        assets = assets or [s for s in LAYER1_ETFS if s in adj_close.columns]

        fwd_returns = adj_close[assets].pct_change(self.horizon).shift(-self.horizon)
        fwd_returns_ann = fwd_returns * (252 / self.horizon)

        common_idx = regime_probs.index.intersection(fwd_returns_ann.dropna().index)
        probs = regime_probs.loc[common_idx]
        returns = fwd_returns_ann.loc[common_idx]

        dominant = probs.idxmax(axis=1)

        regime_means = {}
        regime_vols = {}
        regime_counts = {}

        for regime in probs.columns:
            mask = dominant == regime
            count = mask.sum()
            regime_counts[regime] = count

            if count >= self.min_samples:
                regime_means[regime] = returns.loc[mask].mean()
                regime_vols[regime] = returns.loc[mask].std()
            else:
                regime_means[regime] = pd.Series(np.nan, index=assets)
                regime_vols[regime] = pd.Series(np.nan, index=assets)
                logger.warning(
                    f"  Regime '{regime}': only {count} samples "
                    f"(need {self.min_samples}), will use fallback"
                )

        self._regime_returns = pd.DataFrame(regime_means, columns=probs.columns)
        self._regime_vols = pd.DataFrame(regime_vols, columns=probs.columns)
        self._regime_counts = pd.Series(regime_counts)

        logger.info(f"Historical estimator fit: {len(common_idx)} days, {len(assets)} assets")
        for regime, count in regime_counts.items():
            logger.info(f"  {regime}: {count} samples")

        return self

    def predict(self, current_probs: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Generate probability-weighted return forecast."""
        if self._regime_returns is None:
            raise RuntimeError("Estimator not fitted.")

        weighted_return = pd.Series(0.0, index=self._regime_returns.index)
        weighted_var = pd.Series(0.0, index=self._regime_returns.index)

        for regime in current_probs.index:
            if regime in self._regime_returns.columns:
                p = current_probs[regime]
                r = self._regime_returns[regime]
                v = self._regime_vols[regime]

                valid = r.notna()
                weighted_return.loc[valid] += p * r.loc[valid]
                weighted_var.loc[valid] += p * (v.loc[valid] ** 2)

        uncertainty = np.sqrt(weighted_var)
        return weighted_return, uncertainty

    @property
    def regime_counts(self) -> Optional[pd.Series]:
        return self._regime_counts


# ---------------------------------------------------------------------------
# Layer 1: Ridge Regression Forecaster
# ---------------------------------------------------------------------------

class RidgeRegimeForecaster:
    """
    Regularized Ridge regression for regime-conditioned ETF forecasting.

    Fits per-ETF Ridge models using sector features + regime probabilities.
    """

    def __init__(
        self,
        horizon: int = DEFAULT_HORIZON,
        alpha: float = 10.0,
        min_train_samples: int = 252,
    ):
        self.horizon = horizon
        self.alpha = alpha
        self.min_train_samples = min_train_samples
        self._models: Dict[str, Ridge] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._is_fitted = False
        self._train_scores: Dict[str, float] = {}

    def fit(
        self,
        sector_features: pd.DataFrame,
        adj_close: pd.DataFrame,
        regime_probs: pd.DataFrame,
        assets: Optional[List[str]] = None,
    ) -> "RidgeRegimeForecaster":
        """Fit Ridge models for each ETF that has sector features."""
        assets = assets or [s for s in LAYER1_ETFS if s in adj_close.columns]
        available_tickers = sector_features.index.get_level_values("ticker").unique()

        for ticker in assets:
            if ticker not in available_tickers:
                # Tier 1 macro assets (SPY, QQQ, etc.) don't have sector features
                # They get forecast from historical estimator only
                continue

            ticker_feats = sector_features.xs(ticker, level="ticker")

            fwd_ret = adj_close[ticker].pct_change(self.horizon).shift(-self.horizon)
            fwd_ret_ann = fwd_ret * (252 / self.horizon)

            common_idx = (
                ticker_feats.index
                .intersection(regime_probs.index)
                .intersection(fwd_ret_ann.dropna().index)
            )

            if len(common_idx) < self.min_train_samples:
                continue

            X_sector = ticker_feats.loc[common_idx]
            X_regime = regime_probs.loc[common_idx]
            X = pd.concat([X_sector, X_regime], axis=1)
            y = fwd_ret_ann.loc[common_idx]

            valid_mask = X.notna().all(axis=1) & y.notna()
            X = X.loc[valid_mask]
            y = y.loc[valid_mask]

            if len(X) < self.min_train_samples:
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = Ridge(alpha=self.alpha, random_state=42)
            model.fit(X_scaled, y)

            self._models[ticker] = model
            self._scalers[ticker] = scaler
            self._train_scores[ticker] = float(model.score(X_scaled, y))

            logger.info(
                f"  {ticker}: Ridge fitted on {len(X)} samples, R²={self._train_scores[ticker]:.4f}"
            )

        self._is_fitted = len(self._models) > 0
        return self

    def predict(
        self,
        sector_features: pd.DataFrame,
        regime_probs: pd.Series,
        date: pd.Timestamp,
        assets: Optional[List[str]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate Ridge-based return forecasts."""
        assets = assets or list(self._models.keys())
        predictions = {}
        confidences = {}

        for ticker in assets:
            if ticker not in self._models:
                predictions[ticker] = np.nan
                confidences[ticker] = 0.0
                continue

            try:
                ticker_feats = sector_features.xs(ticker, level="ticker")
                if date in ticker_feats.index:
                    x_sector = ticker_feats.loc[[date]]
                else:
                    nearest = ticker_feats.index[ticker_feats.index <= date]
                    if len(nearest) == 0:
                        predictions[ticker] = np.nan
                        confidences[ticker] = 0.0
                        continue
                    x_sector = ticker_feats.loc[[nearest[-1]]]

                x_regime = regime_probs.to_frame().T
                x_regime.index = x_sector.index

                X = pd.concat([x_sector, x_regime], axis=1)

                if X.isnull().any().any():
                    X = X.fillna(0)

                X_scaled = self._scalers[ticker].transform(X)
                pred = self._models[ticker].predict(X_scaled)[0]

                predictions[ticker] = float(pred)
                r2 = self._train_scores.get(ticker, 0)
                confidences[ticker] = float(np.clip(r2 * 2, 0, 1))

            except Exception as e:
                logger.warning(f"  Ridge prediction failed for {ticker}: {e}")
                predictions[ticker] = np.nan
                confidences[ticker] = 0.0

        return pd.Series(predictions), pd.Series(confidences)


# ---------------------------------------------------------------------------
# Layer 1: Momentum Fallback
# ---------------------------------------------------------------------------

def momentum_fallback(
    raw_close: pd.DataFrame,
    regime_probs: pd.Series,
    assets: List[str],
    lookback: int = 63,
) -> Tuple[pd.Series, pd.Series]:
    """
    Pure momentum-based return estimate for low-conviction periods.

    Uses regime-aware sector group preferences to tilt momentum.
    """
    available = [s for s in assets if s in raw_close.columns]
    if not available:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Compute momentum
    returns = raw_close[available].pct_change(lookback).iloc[-1]
    returns_ann = returns * (252 / lookback)

    # Regime-aware tilt
    dominant = regime_probs.idxmax()
    preferred_groups = REGIME_SECTOR_PREFERENCE.get(dominant, ["cyclical", "defensive"])

    preferred_tickers = set()
    for group in preferred_groups:
        preferred_tickers.update(SECTOR_GROUPS.get(group, []))

    confidence = pd.Series(0.2, index=returns_ann.index)
    for ticker in returns_ann.index:
        if ticker in preferred_tickers:
            confidence[ticker] = 0.3

    return returns_ann, confidence


# ---------------------------------------------------------------------------
# Layer 2: Stock Alpha Model
# ---------------------------------------------------------------------------

class StockAlphaModel:
    """
    Price-only relative momentum model for individual stock alpha.

    Computes alpha score for each stock relative to its parent sector ETF.
    No earnings or fundamental data in v7.0.

    Alpha Score = 0.6 * RelMom63 + 0.4 * RelMom126 - 0.2 * (VolRatio - 1.0)

    Eligibility: Stock must have RelMom63 > 0 (outperforming sector).
    If ineligible, alpha_score = -999 (sector ETF preferred).
    """

    def __init__(
        self,
        lookback: int = LAYER2_LOOKBACK,
        alpha_scaling: float = ALPHA_SCALING_FACTOR,
    ):
        self.lookback = lookback
        self.alpha_scaling = alpha_scaling

    def compute_alpha_scores(
        self,
        raw_close: pd.DataFrame,
        date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, float]:
        """
        Compute alpha scores for all eligible stocks.

        Returns dict of ticker → alpha_score.
        Ineligible stocks get -999.
        """
        if date is None:
            date = raw_close.index[-1]

        # Get data up to date
        prices = raw_close.loc[:date]
        if len(prices) < self.lookback:
            logger.warning(f"Insufficient data for stock alpha ({len(prices)} < {self.lookback})")
            return {}

        alpha_scores = {}

        for stock, sector_etf in STOCK_SECTOR_MAP.items():
            if stock not in prices.columns or sector_etf not in prices.columns:
                continue

            stock_prices = prices[stock].dropna()
            sector_prices = prices[sector_etf].dropna()

            if len(stock_prices) < self.lookback or len(sector_prices) < self.lookback:
                continue

            # Relative momentum (63-day)
            stock_ret_63 = stock_prices.iloc[-1] / stock_prices.iloc[-63] - 1 if len(stock_prices) >= 63 else 0
            sector_ret_63 = sector_prices.iloc[-1] / sector_prices.iloc[-63] - 1 if len(sector_prices) >= 63 else 0
            rel_mom_63 = stock_ret_63 - sector_ret_63

            # Relative momentum (126-day)
            stock_ret_126 = stock_prices.iloc[-1] / stock_prices.iloc[-126] - 1 if len(stock_prices) >= 126 else 0
            sector_ret_126 = sector_prices.iloc[-1] / sector_prices.iloc[-126] - 1 if len(sector_prices) >= 126 else 0
            rel_mom_126 = stock_ret_126 - sector_ret_126

            # Volatility ratio (penalize unstable stocks)
            stock_vol = stock_prices.pct_change().tail(20).std()
            sector_vol = sector_prices.pct_change().tail(20).std()
            vol_ratio = stock_vol / max(sector_vol, 1e-8)

            # Eligibility filter
            if rel_mom_63 <= 0:
                alpha_scores[stock] = -999.0
                continue

            # Alpha score
            alpha = 0.6 * rel_mom_63 + 0.4 * rel_mom_126 - 0.2 * (vol_ratio - 1.0)
            alpha_scores[stock] = float(alpha)

        # Normalize across eligible stocks
        eligible = {k: v for k, v in alpha_scores.items() if v > -999}
        if eligible:
            values = np.array(list(eligible.values()))
            mean_a = values.mean()
            std_a = values.std()
            if std_a > 0:
                for k in eligible:
                    alpha_scores[k] = (alpha_scores[k] - mean_a) / std_a

        return alpha_scores

    def compute_stock_expected_returns(
        self,
        etf_expected_returns: pd.Series,
        alpha_scores: Dict[str, float],
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Combine Layer 1 ETF forecasts with Layer 2 stock alpha.

        stock_expected_return = sector_etf_expected_return + alpha_score * ALPHA_SCALING_FACTOR

        Returns expected_returns, confidence, method for eligible stocks.
        """
        stock_returns = {}
        stock_confidence = {}
        stock_methods = {}

        for stock, sector_etf in STOCK_SECTOR_MAP.items():
            if stock not in alpha_scores:
                continue

            alpha = alpha_scores[stock]

            if alpha <= -999:
                # Ineligible — don't include this stock
                continue

            # Get sector ETF expected return
            sector_ret = etf_expected_returns.get(sector_etf, 0.0)
            if pd.isna(sector_ret):
                sector_ret = 0.0

            stock_ret = sector_ret + alpha * self.alpha_scaling
            stock_returns[stock] = stock_ret

            # Confidence based on alpha strength
            stock_confidence[stock] = float(np.clip(abs(alpha) * 0.3, 0.1, 0.8))
            stock_methods[stock] = "sector_forecast+stock_alpha"

        return (
            pd.Series(stock_returns),
            pd.Series(stock_confidence),
            pd.Series(stock_methods),
        )


# ---------------------------------------------------------------------------
# Main Forecaster (combines Layer 1 + Layer 2)
# ---------------------------------------------------------------------------

class ReturnForecaster:
    """
    Hierarchical return forecaster: Layer 1 (ETFs) + Layer 2 (stock alpha).

    Output covers ALL ~33 tradeable assets with expected returns,
    confidence scores, and method labels.
    """

    def __init__(
        self,
        horizon: int = DEFAULT_HORIZON,
        ridge_alpha: float = 10.0,
        high_conviction: float = HIGH_CONVICTION_THRESHOLD,
        low_conviction: float = LOW_CONVICTION_THRESHOLD,
        historical_weight: float = 0.6,
        momentum_lookback: int = 63,
        alpha_scaling: float = ALPHA_SCALING_FACTOR,
    ):
        self.horizon = horizon
        self.ridge_alpha = ridge_alpha
        self.high_conviction = high_conviction
        self.low_conviction = low_conviction
        self.historical_weight = historical_weight
        self.momentum_lookback = momentum_lookback

        self._hist_estimator = HistoricalRegimeEstimator(horizon=horizon)
        self._ridge_forecaster = RidgeRegimeForecaster(
            horizon=horizon, alpha=ridge_alpha
        )
        self._stock_alpha = StockAlphaModel(alpha_scaling=alpha_scaling)
        self._is_fitted = False

    def fit(
        self,
        adj_close: pd.DataFrame,
        raw_close: pd.DataFrame,
        regime_probs: pd.DataFrame,
        sector_features: pd.DataFrame,
        assets: Optional[List[str]] = None,
    ) -> "ReturnForecaster":
        """Fit Layer 1 sub-models (historical + Ridge)."""
        assets = assets or [s for s in LAYER1_ETFS if s in adj_close.columns]
        logger.info(f"Fitting ReturnForecaster: {len(assets)} ETFs, horizon={self.horizon}d")

        self._hist_estimator.fit(adj_close, regime_probs, assets)
        self._ridge_forecaster.fit(sector_features, adj_close, regime_probs, assets)

        self._is_fitted = True
        return self

    def generate_forecast(
        self,
        regime_probs: pd.DataFrame,
        sector_features: pd.DataFrame,
        adj_close: pd.DataFrame,
        raw_close: pd.DataFrame,
        forecast_date: Optional[pd.Timestamp] = None,
        include_stocks: bool = True,
    ) -> AssetForecast:
        """
        Generate forecasts for all assets (ETFs + stocks).

        Pipeline:
            1. Layer 1: Forecast ETF returns using regime-conditioned ensemble
            2. Layer 2: Compute stock alpha scores (if include_stocks=True)
            3. Combine into single forecast output
        """
        etf_assets = [s for s in LAYER1_ETFS if s in adj_close.columns]

        if not self._is_fitted:
            self.fit(adj_close, raw_close, regime_probs, sector_features, etf_assets)

        if forecast_date is None:
            forecast_date = regime_probs.index[-1]

        # Current regime state
        if forecast_date in regime_probs.index:
            current_probs = regime_probs.loc[forecast_date]
        else:
            available = regime_probs.index[regime_probs.index <= forecast_date]
            if len(available) == 0:
                raise ValueError(f"No regime data available at or before {forecast_date}")
            current_probs = regime_probs.loc[available[-1]]

        dominant_regime = current_probs.idxmax()
        regime_conviction = current_probs.max()

        # ---- Layer 1: ETF Forecasts ----
        if regime_conviction >= self.high_conviction:
            etf_returns, etf_confidence, etf_methods = self._model_forecast(
                current_probs, sector_features, forecast_date, raw_close, etf_assets
            )
        elif regime_conviction >= self.low_conviction:
            model_ret, model_conf, model_methods = self._model_forecast(
                current_probs, sector_features, forecast_date, raw_close, etf_assets
            )
            mom_ret, mom_conf = momentum_fallback(
                raw_close, current_probs, etf_assets, self.momentum_lookback
            )
            common = model_ret.index.intersection(mom_ret.index)
            blend = 0.5
            etf_returns = blend * model_ret.loc[common] + (1 - blend) * mom_ret.loc[common]
            etf_confidence = blend * model_conf.loc[common] + (1 - blend) * mom_conf.loc[common]
            etf_methods = pd.Series("model+momentum_blend", index=common)
        else:
            etf_returns, etf_confidence = momentum_fallback(
                raw_close, current_probs, etf_assets, self.momentum_lookback
            )
            etf_methods = pd.Series("momentum_fallback", index=etf_returns.index)

        # ---- Layer 2: Stock Alpha ----
        stock_alpha_scores = {}
        stock_returns = pd.Series(dtype=float)
        stock_confidence = pd.Series(dtype=float)
        stock_methods = pd.Series(dtype=str)

        if include_stocks:
            stock_alpha_scores = self._stock_alpha.compute_alpha_scores(
                raw_close, date=forecast_date
            )
            if stock_alpha_scores:
                stock_returns, stock_confidence, stock_methods = (
                    self._stock_alpha.compute_stock_expected_returns(
                        etf_returns, stock_alpha_scores
                    )
                )

        # ---- Combine Layer 1 + Layer 2 ----
        all_returns = pd.concat([etf_returns, stock_returns])
        all_confidence = pd.concat([etf_confidence, stock_confidence])
        all_methods = pd.concat([etf_methods, stock_methods])

        # Remove duplicates (stocks override their sector if present)
        all_returns = all_returns[~all_returns.index.duplicated(keep='last')]
        all_confidence = all_confidence[~all_confidence.index.duplicated(keep='last')]
        all_methods = all_methods[~all_methods.index.duplicated(keep='last')]

        # Overall confidence
        avg_confidence = all_confidence.mean() if len(all_confidence) > 0 else 0.0
        overall_confidence = float(0.5 * regime_conviction + 0.5 * avg_confidence)

        return AssetForecast(
            expected_returns=all_returns,
            confidence=all_confidence,
            overall_confidence=overall_confidence,
            method=all_methods,
            regime_probs=current_probs,
            dominant_regime=dominant_regime,
            regime_conviction=regime_conviction,
            stock_alpha_scores=stock_alpha_scores,
            forecast_date=str(forecast_date.date()),
            horizon_days=self.horizon,
            metadata={
                "n_etfs_forecast": len(etf_returns),
                "n_stocks_eligible": sum(1 for v in stock_alpha_scores.values() if v > -999),
                "n_stocks_total": len(stock_alpha_scores),
                "high_conviction_threshold": self.high_conviction,
                "low_conviction_threshold": self.low_conviction,
                "historical_weight": self.historical_weight,
                "regime_counts": (
                    self._hist_estimator.regime_counts.to_dict()
                    if self._hist_estimator.regime_counts is not None
                    else {}
                ),
            },
        )

    def _model_forecast(
        self,
        current_probs: pd.Series,
        sector_features: pd.DataFrame,
        forecast_date: pd.Timestamp,
        raw_close: pd.DataFrame,
        assets: List[str],
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate ensemble model forecast (Historical + Ridge)."""
        hist_ret, hist_unc = self._hist_estimator.predict(current_probs)
        ridge_ret, ridge_conf = self._ridge_forecaster.predict(
            sector_features, current_probs, forecast_date, assets
        )

        w_hist = self.historical_weight
        w_ridge = 1 - w_hist

        expected_returns = pd.Series(dtype=float)
        confidence = pd.Series(dtype=float)
        methods = pd.Series(dtype=str)

        for ticker in assets:
            has_hist = ticker in hist_ret.index and pd.notna(hist_ret.get(ticker))
            has_ridge = ticker in ridge_ret.index and pd.notna(ridge_ret.get(ticker))

            if has_hist and has_ridge:
                ret = w_hist * hist_ret[ticker] + w_ridge * ridge_ret[ticker]
                conf = w_hist * max(0, 1 - hist_unc.get(ticker, 0.5)) + w_ridge * ridge_conf.get(ticker, 0)
                conf = np.clip(conf, 0, 1)
                method = "ensemble"
            elif has_hist:
                ret = hist_ret[ticker]
                conf = max(0, 1 - hist_unc.get(ticker, 0.5))
                conf = np.clip(conf, 0, 1)
                method = "historical_only"
            elif has_ridge:
                ret = ridge_ret[ticker]
                conf = ridge_conf.get(ticker, 0.3)
                method = "ridge_only"
            else:
                mom_ret, mom_conf = momentum_fallback(
                    raw_close, current_probs, [ticker], self.momentum_lookback
                )
                ret = mom_ret.get(ticker, 0.0)
                conf = mom_conf.get(ticker, 0.1)
                method = "momentum_fallback"

            expected_returns[ticker] = ret
            confidence[ticker] = conf
            methods[ticker] = method

        return expected_returns, confidence, methods

    def generate_forecast_series(
        self,
        regime_probs: pd.DataFrame,
        sector_features: pd.DataFrame,
        adj_close: pd.DataFrame,
        raw_close: pd.DataFrame,
        dates: Optional[pd.DatetimeIndex] = None,
        include_stocks: bool = True,
    ) -> List[AssetForecast]:
        """Generate forecasts for multiple dates (for backtesting)."""
        if not self._is_fitted:
            assets = [s for s in LAYER1_ETFS if s in adj_close.columns]
            self.fit(adj_close, raw_close, regime_probs, sector_features, assets)

        if dates is None:
            dates = regime_probs.index[::5]  # weekly rebalancing

        forecasts = []
        for date in dates:
            try:
                fc = self.generate_forecast(
                    regime_probs, sector_features, adj_close, raw_close,
                    forecast_date=date, include_stocks=include_stocks,
                )
                forecasts.append(fc)
            except Exception as e:
                logger.warning(f"  Forecast failed for {date}: {e}")

        logger.info(f"Generated {len(forecasts)} forecasts over {len(dates)} dates")
        return forecasts


# ---------------------------------------------------------------------------
# Forecast Diagnostics
# ---------------------------------------------------------------------------

def print_forecast(forecast: AssetForecast) -> None:
    """Pretty-print a single forecast."""
    print(f"\n{'='*60}")
    print(f"ASSET RETURN FORECAST — {forecast.forecast_date}")
    print(f"{'='*60}")
    print(f"  Regime:             {forecast.dominant_regime} ({forecast.regime_conviction:.3f})")
    print(f"  Overall confidence: {forecast.overall_confidence:.3f}")
    print(f"  Horizon:            {forecast.horizon_days} trading days")
    print(f"  ETFs forecast:      {forecast.metadata.get('n_etfs_forecast', '?')}")
    print(f"  Stocks eligible:    {forecast.metadata.get('n_stocks_eligible', 0)}")

    print(f"\n  {'Asset':<8s} {'Expected Return':>16s} {'Confidence':>12s} {'Method':<25s}")
    print(f"  {'-'*61}")

    sorted_assets = forecast.expected_returns.sort_values(ascending=False)
    for ticker in sorted_assets.index:
        ret = forecast.expected_returns[ticker]
        conf = forecast.confidence.get(ticker, 0)
        method = forecast.method.get(ticker, "?")
        marker = " ★" if ticker in TIER4_STOCKS else ""
        print(f"  {ticker:<8s} {ret:>15.2%} {conf:>11.3f}  {method:<25s}{marker}")


def summarize_forecast_series(forecasts: List[AssetForecast]) -> Dict[str, Any]:
    """Summarize a series of forecasts for diagnostics."""
    if not forecasts:
        return {"error": "No forecasts to summarize"}

    methods_count: Dict[str, int] = {}
    confidences = []
    regime_convictions = []
    dominant_regimes: Dict[str, int] = {}
    stocks_eligible = []

    for fc in forecasts:
        confidences.append(fc.overall_confidence)
        regime_convictions.append(fc.regime_conviction)
        stocks_eligible.append(fc.metadata.get("n_stocks_eligible", 0))

        regime = fc.dominant_regime
        dominant_regimes[regime] = dominant_regimes.get(regime, 0) + 1

        for method in fc.method.values:
            methods_count[method] = methods_count.get(method, 0) + 1

    return {
        "n_forecasts": len(forecasts),
        "date_range": f"{forecasts[0].forecast_date} → {forecasts[-1].forecast_date}",
        "mean_confidence": float(np.mean(confidences)),
        "min_confidence": float(np.min(confidences)),
        "mean_regime_conviction": float(np.mean(regime_convictions)),
        "dominant_regime_distribution": dominant_regimes,
        "method_distribution": methods_count,
        "mean_stocks_eligible": float(np.mean(stocks_eligible)),
        "pct_model_based": sum(
            v for k, v in methods_count.items()
            if "ensemble" in k or "historical" in k or "ridge" in k
        ) / max(sum(methods_count.values()), 1),
        "pct_fallback": sum(
            v for k, v in methods_count.items() if "fallback" in k
        ) / max(sum(methods_count.values()), 1),
    }