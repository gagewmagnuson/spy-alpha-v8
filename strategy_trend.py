"""
SPY Alpha v8 — Strategy 2: Trend / CTA
=========================================
 
NEW module in v8. Pure price-based trend following across multiple timeframes.
Completely independent of the HMM — no regime data, no macro data, no latent features.
 
Design Principles:
    - Only uses price data (moving averages and realized volatility)
    - Genuinely orthogonal to Strategy 1 (different information, different behavior)
    - Excellent drawdown properties (tested standalone: Max DD -15.9%)
    - Modest standalone alpha (Sharpe ~0.85)
    - Value is in diversification of failure modes with Strategy 1
    - When the HMM misclassifies (2012-2014, 2019), trend stays invested
 
Signal Generation:
    For each asset in TREND_UNIVERSE:
        Compute whether price is above/below 50d, 100d, 200d MAs
        trend_score = weighted average: 30% on 50d, 35% on 100d, 35% on 200d
 
    For assets with positive trend_score:
        Weight by inverse 63-day realized volatility (risk parity)
 
    Assets with negative trend_score: weight = 0 (sit in cash/SHY)
 
Rebalance: Every 5 trading days (weekly)
"""
 
from __future__ import annotations
 
import logging
from typing import Any, Dict, List, Optional
 
import numpy as np
import pandas as pd
 
from strategy_regime import StrategyOutput
 
logger = logging.getLogger("spy_alpha_v8.strategy_trend")
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
# Trend universe — liquid assets with meaningful trend behavior
# Deliberately smaller than the full trading universe to keep the
# strategy simple and interpretable
TREND_UNIVERSE: List[str] = [
    "SPY", "QQQ", "IWM", "VWO",    # Equity
    "TLT",                           # Rates
    "GLD",                           # Commodities
    "SHY",                           # Cash safety
    "XLK", "XLE", "XLF", "XLV",     # Sectors
]
 
# Moving average windows and weights (from build spec)
MA_WINDOWS: List[int] = [50, 100, 200]
MA_WEIGHTS: List[float] = [0.30, 0.35, 0.35]
 
# Volatility lookback for inverse-vol weighting
VOL_LOOKBACK: int = 63
 
# Minimum number of days required to compute signals
MIN_HISTORY: int = 252
 
# Rebalance frequency
REBALANCE_EVERY: int = 5
 
 
# ---------------------------------------------------------------------------
# Trend Signal Computation
# ---------------------------------------------------------------------------
 
def compute_trend_scores(
    raw_close: pd.DataFrame,
    assets: Optional[List[str]] = None,
    ma_windows: Optional[List[int]] = None,
    ma_weights: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Compute trend scores for each asset on each trading day.
 
    Trend score = weighted average of above/below MA signals.
    Score of 1.0 = price above all MAs (strong uptrend)
    Score of 0.0 = price below all MAs (strong downtrend)
 
    Returns DataFrame with one column per asset, one row per trading day.
    """
    if assets is None:
        assets = TREND_UNIVERSE
    if ma_windows is None:
        ma_windows = MA_WINDOWS
    if ma_weights is None:
        ma_weights = MA_WEIGHTS
 
    available = [a for a in assets if a in raw_close.columns]
    if not available:
        logger.warning("No trend universe assets found in price data")
        return pd.DataFrame()
 
    scores = {}
 
    for asset in available:
        price = raw_close[asset].dropna()
        if len(price) < max(ma_windows):
            logger.warning(f"  {asset}: insufficient history ({len(price)} days), skipping")
            continue
 
        asset_score = pd.Series(0.0, index=price.index)
 
        for window, weight in zip(ma_windows, ma_weights):
            ma = price.rolling(window, min_periods=window).mean()
            signal = (price > ma).astype(float)
            asset_score += signal * weight
 
        scores[asset] = asset_score
 
    df = pd.DataFrame(scores)
    logger.info(f"Trend scores computed for {len(scores)} assets")
    return df
 
 
def compute_inverse_vol_weights(
    raw_close: pd.DataFrame,
    assets: List[str],
    lookback: int = VOL_LOOKBACK,
) -> pd.DataFrame:
    """
    Compute inverse-volatility weights for risk parity sizing.
 
    Assets with lower realized volatility get higher weight.
    This ensures each asset contributes roughly equal risk.
 
    Returns DataFrame with one column per asset, weights sum to 1.0 per row.
    """
    available = [a for a in assets if a in raw_close.columns]
    if not available:
        return pd.DataFrame()
 
    returns = raw_close[available].pct_change()
 
    # Rolling annualized volatility
    rolling_vol = returns.rolling(lookback, min_periods=21).std() * np.sqrt(252)
 
    # Inverse vol (higher vol → lower weight)
    inv_vol = 1.0 / rolling_vol.replace(0, np.nan)
 
    # Normalize to sum to 1.0 per row
    row_sums = inv_vol.sum(axis=1)
    weights = inv_vol.div(row_sums, axis=0)
 
    return weights
 
 
# ---------------------------------------------------------------------------
# Strategy 2: Trend/CTA
# ---------------------------------------------------------------------------
 
class TrendCTAStrategy:
    """
    Strategy 2: Pure price-based trend following.
 
    No HMM dependency. No macro data. Only price and volatility.
    Provides genuine orthogonality to Strategy 1.
 
    The strategy:
        1. Computes trend scores (above/below MAs) for each asset
        2. Assets with positive trend get weight, others get zero
        3. Positive-trend assets sized by inverse realized volatility
        4. Cash (SHY) absorbs unallocated capital when few assets trend up
    """
 
    def __init__(
        self,
        trend_universe: Optional[List[str]] = None,
        ma_windows: Optional[List[int]] = None,
        ma_weights: Optional[List[float]] = None,
        vol_lookback: int = VOL_LOOKBACK,
        rebalance_every: int = REBALANCE_EVERY,
        min_trend_score: float = 0.30,
    ):
        self.trend_universe = trend_universe or TREND_UNIVERSE
        self.ma_windows = ma_windows or MA_WINDOWS
        self.ma_weights = ma_weights or MA_WEIGHTS
        self.vol_lookback = vol_lookback
        self.rebalance_every = rebalance_every
        self.min_trend_score = min_trend_score
 
        # Computed during build
        self.trend_scores: Optional[pd.DataFrame] = None
        self.inv_vol_weights: Optional[pd.DataFrame] = None
 
    def build(self, snapshot: Dict[str, Any]) -> None:
        """
        Pre-compute trend scores and volatility weights for the full period.
 
        This is lightweight compared to Strategy 1 — no model training,
        just rolling computations on price data.
        """
        from data_pipeline import get_raw_close
 
        logger.info("Strategy 2 (Trend/CTA): Building signals...")
 
        raw_close = get_raw_close(snapshot)
 
        # Compute trend scores
        self.trend_scores = compute_trend_scores(
            raw_close,
            assets=self.trend_universe,
            ma_windows=self.ma_windows,
            ma_weights=self.ma_weights,
        )
 
        # Compute inverse-vol weights
        self.inv_vol_weights = compute_inverse_vol_weights(
            raw_close,
            assets=self.trend_universe,
            lookback=self.vol_lookback,
        )
 
        if self.trend_scores.empty:
            raise RuntimeError("No trend scores computed — check price data")
 
        logger.info(
            f"Strategy 2 (Trend/CTA): Built successfully, "
            f"{self.trend_scores.shape[0]} days, {self.trend_scores.shape[1]} assets"
        )
 
    def generate_signals(
        self,
        snapshot: Dict[str, Any],
        rebalance_dates: Optional[pd.DatetimeIndex] = None,
    ) -> List[StrategyOutput]:
        """
        Generate strategy signals for each rebalance date.
 
        If rebalance_dates is provided, signals are generated only on those
        dates (for alignment with Strategy 1). Otherwise, signals are generated
        every rebalance_every trading days.
        """
        if self.trend_scores is None:
            raise RuntimeError("Call build() before generate_signals()")
 
        # Determine rebalance dates
        if rebalance_dates is not None:
            # Use provided dates, filtered to available data
            available_dates = self.trend_scores.index
            dates = rebalance_dates.intersection(available_dates)
        else:
            # Generate our own rebalance schedule
            available_dates = self.trend_scores.dropna(how="all").index
            # Start after enough history for the longest MA
            start_idx = max(MA_WINDOWS) + self.vol_lookback
            if start_idx >= len(available_dates):
                logger.warning("Insufficient data for trend signals")
                return []
            dates = available_dates[start_idx::self.rebalance_every]
 
        outputs = []
        for date in dates:
            output = self._generate_single_signal(date)
            if output is not None:
                outputs.append(output)
 
        logger.info(f"Strategy 2: Generated {len(outputs)} signals")
        return outputs
 
    def _generate_single_signal(self, date: pd.Timestamp) -> Optional[StrategyOutput]:
        """Generate a single strategy output for a given date."""
        if date not in self.trend_scores.index:
            return None
 
        scores = self.trend_scores.loc[date].dropna()
        if scores.empty:
            return None
 
        # ---- Identify assets in uptrend ----
        in_uptrend = scores[scores >= self.min_trend_score].index.tolist()
 
        # ---- Compute weights ----
        weights = {}
 
        if not in_uptrend:
            # No assets in uptrend — go to full cash
            weights["SHY"] = 1.0
            confidence = 0.1
            active_assets = ["SHY"]
        else:
            # Get inverse-vol weights for uptrending assets
            if date in self.inv_vol_weights.index:
                vol_weights = self.inv_vol_weights.loc[date]
 
                # Filter to uptrending assets with valid vol weights
                valid_uptrend = [a for a in in_uptrend
                                 if a in vol_weights.index and pd.notna(vol_weights[a])]
 
                if valid_uptrend:
                    raw_weights = vol_weights[valid_uptrend]
 
                    # Renormalize to sum to 1.0
                    total = raw_weights.sum()
                    if total > 0:
                        normalized = raw_weights / total
                        for asset, w in normalized.items():
                            if w > 1e-6:
                                weights[asset] = float(w)
 
            # If inverse-vol failed, fall back to equal weight
            if not weights:
                equal_w = 1.0 / len(in_uptrend)
                for asset in in_uptrend:
                    weights[asset] = equal_w
 
            active_assets = list(weights.keys())
 
            # Confidence based on breadth: more assets in uptrend = higher confidence
            n_trending = len(in_uptrend)
            n_total = len(scores)
            confidence = float(n_trending / n_total) if n_total > 0 else 0.5
 
        # ---- Build metadata ----
        n_uptrend = len(in_uptrend)
        n_total = len(scores)
 
        trend_direction = "bullish" if n_uptrend > n_total * 0.6 else \
                         "bearish" if n_uptrend < n_total * 0.3 else "mixed"
 
        metadata = {
            "date": date.strftime("%Y-%m-%d"),
            "trend_direction": trend_direction,
            "assets_in_uptrend": n_uptrend,
            "assets_total": n_total,
            "uptrend_assets": in_uptrend,
            "trend_scores": {asset: float(scores[asset])
                            for asset in scores.index if asset in self.trend_universe},
        }
 
        return StrategyOutput(
            strategy_name="trend_cta",
            proposed_weights=weights,
            confidence=confidence,
            active_assets=active_assets,
            strategy_metadata=metadata,
        )
 
    def get_trend_scores(self) -> Optional[pd.DataFrame]:
        """Return the full trend score matrix for diagnostics."""
        return self.trend_scores
 
 
# ---------------------------------------------------------------------------
# Standalone Backtest Support
# ---------------------------------------------------------------------------
 
def backtest_trend_standalone(
    snapshot: Dict[str, Any],
    rebalance_every: int = REBALANCE_EVERY,
) -> pd.DataFrame:
    """
    Run a standalone backtest of the trend strategy.
 
    Returns a DataFrame with columns:
        - portfolio_return: daily portfolio return
        - spy_return: daily SPY return (benchmark)
        - cumulative: cumulative portfolio return
        - spy_cumulative: cumulative SPY return
 
    This is a simplified backtest for validating the strategy independently
    before integration with the meta-allocator.
    """
    from data_pipeline import get_adj_close, get_raw_close
 
    raw_close = get_raw_close(snapshot)
    adj_close = get_adj_close(snapshot)
 
    # Build strategy
    strategy = TrendCTAStrategy(rebalance_every=rebalance_every)
    strategy.build(snapshot)
 
    # Generate signals on own schedule
    outputs = strategy.generate_signals(snapshot)
 
    if not outputs:
        raise RuntimeError("No signals generated")
 
    # Build weight history — map each signal to its effective dates
    # (signal holds until next rebalance)
    signal_dates = [pd.Timestamp(o.strategy_metadata["date"]) for o in outputs]
 
    # Daily returns for all assets
    available_assets = [a for a in TREND_UNIVERSE if a in adj_close.columns]
    daily_returns = adj_close[available_assets].pct_change()
 
    # SPY benchmark
    spy_returns = adj_close["SPY"].pct_change() if "SPY" in adj_close.columns else pd.Series(0, index=adj_close.index)
 
    # Walk through signals and compute portfolio returns
    portfolio_returns = []
    current_weights = {}
 
    all_dates = daily_returns.index
    signal_idx = 0
 
    for i, date in enumerate(all_dates):
        # Update weights if we've reached a signal date
        while signal_idx < len(signal_dates) and signal_dates[signal_idx] <= date:
            current_weights = outputs[signal_idx].proposed_weights
            signal_idx += 1
 
        # Skip if no signal yet
        if not current_weights:
            portfolio_returns.append({
                "date": date,
                "portfolio_return": 0.0,
                "spy_return": float(spy_returns.get(date, 0.0)),
            })
            continue
 
        # Compute weighted portfolio return
        port_ret = 0.0
        for asset, weight in current_weights.items():
            if asset in daily_returns.columns and pd.notna(daily_returns.loc[date, asset]):
                port_ret += weight * daily_returns.loc[date, asset]
 
        portfolio_returns.append({
            "date": date,
            "portfolio_return": port_ret,
            "spy_return": float(spy_returns.get(date, 0.0)),
        })
 
    result = pd.DataFrame(portfolio_returns).set_index("date")
 
    # Cumulative returns
    result["cumulative"] = (1 + result["portfolio_return"]).cumprod()
    result["spy_cumulative"] = (1 + result["spy_return"]).cumprod()
 
    return result
 
 
def print_trend_backtest_report(result: pd.DataFrame) -> None:
    """Print a performance report for the standalone trend backtest."""
    returns = result["portfolio_return"].dropna()
    spy_returns = result["spy_return"].dropna()
 
    # Align
    common = returns.index.intersection(spy_returns.index)
    returns = returns.loc[common]
    spy_returns = spy_returns.loc[common]
 
    # Skip initial zero-return period
    first_nonzero = returns[returns != 0].index[0] if (returns != 0).any() else returns.index[0]
    returns = returns.loc[first_nonzero:]
    spy_returns = spy_returns.loc[first_nonzero:]
 
    n_years = len(returns) / 252
 
    # CAGR
    cum = (1 + returns).cumprod()
    cagr = cum.iloc[-1] ** (1 / n_years) - 1
 
    spy_cum = (1 + spy_returns).cumprod()
    spy_cagr = spy_cum.iloc[-1] ** (1 / n_years) - 1
 
    # Sharpe
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
 
    # Sortino
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (returns.mean() * 252) / downside_vol
 
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_peak = cumulative.expanding().max()
    drawdown = (cumulative - rolling_peak) / rolling_peak
    max_dd = drawdown.min()
 
    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0
 
    # Annualized vol
    ann_vol = returns.std() * np.sqrt(252)
 
    # Win rate
    win_rate = (returns > 0).mean()
 
    # Rolling correlation with SPY
    corr_63d = returns.rolling(63).corr(spy_returns)
    mean_corr = corr_63d.mean()
 
    print(f"\n{'='*60}")
    print(f"STRATEGY 2 (TREND/CTA) — STANDALONE BACKTEST")
    print(f"{'='*60}")
    print(f"--- Return Summary ---")
    print(f"  Period:              {n_years:.1f} years ({len(returns)} trading days)")
    print(f"  CAGR:                {cagr:>8.1%}    (Benchmark: {spy_cagr:>8.1%})")
    print(f"  Total Return:        {(cum.iloc[-1]-1):>8.1%}    (Benchmark: {(spy_cum.iloc[-1]-1):>8.1%})")
    print(f"--- Risk-Adjusted Metrics ---")
    print(f"  Sharpe Ratio:        {sharpe:>8.2f}")
    print(f"  Sortino Ratio:       {sortino:>8.2f}")
    print(f"  Calmar Ratio:        {calmar:>8.2f}")
    print(f"--- Risk Metrics ---")
    print(f"  Annualized Vol:      {ann_vol:>8.1%}")
    print(f"  Max Drawdown:        {max_dd:>8.1%}")
    print(f"  Win Rate:            {win_rate:>8.1%}")
    print(f"--- Correlation with SPY ---")
    print(f"  Mean 63d corr:       {mean_corr:>8.3f}")
 
    # Year-by-year returns
    print(f"\n--- Annual Returns ---")
    yearly = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    spy_yearly = spy_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    print(f"  {'Year':<6s} {'Strategy':>10s} {'SPY':>10s} {'Excess':>10s}")
    print(f"  {'-'*36}")
    for date in yearly.index:
        yr = date.year
        strat_r = yearly.loc[date]
        spy_r = spy_yearly.loc[date] if date in spy_yearly.index else 0.0
        print(f"  {yr:<6d} {strat_r:>9.1%} {spy_r:>9.1%} {strat_r - spy_r:>9.1%}")
 
 
# ---------------------------------------------------------------------------
# Blend Testing Support
# ---------------------------------------------------------------------------
 
def blend_equal_weight(
    strategy1_outputs: List[StrategyOutput],
    strategy2_outputs: List[StrategyOutput],
    weight_s1: float = 0.50,
    weight_s2: float = 0.50,
) -> List[Dict[str, float]]:
    """
    Create equal-weight blended portfolios from two strategies.
 
    Aligns outputs by date and produces blended weights.
    Used for testing diversification benefit before the meta-allocator.
    """
    # Build date-keyed lookups
    s1_by_date = {o.strategy_metadata["date"]: o for o in strategy1_outputs}
    s2_by_date = {o.strategy_metadata["date"]: o for o in strategy2_outputs}
 
    # Find common dates
    common_dates = sorted(set(s1_by_date.keys()) & set(s2_by_date.keys()))
 
    blended = []
    for date in common_dates:
        s1_weights = s1_by_date[date].proposed_weights
        s2_weights = s2_by_date[date].proposed_weights
 
        # Blend: w_final = weight_s1 * s1 + weight_s2 * s2
        combined = {}
        all_assets = set(list(s1_weights.keys()) + list(s2_weights.keys()))
        for asset in all_assets:
            w1 = s1_weights.get(asset, 0.0) * weight_s1
            w2 = s2_weights.get(asset, 0.0) * weight_s2
            total = w1 + w2
            if total > 1e-6:
                combined[asset] = total
 
        blended.append({
            "date": date,
            "weights": combined,
        })
 
    return blended