"""
SPY Alpha v7 — Portfolio Optimizer
====================================

Assigns weights to selected assets with overlay instruments (UPRO, SHY, TLT, GLD).

Changes from v6:
    - Dynamic asset universe (8-11 selected assets, not fixed 11 sectors)
    - TLT crisis-type conditional overlay (deflation → floor, inflation → zero)
    - GLD inflation/crisis-inflation overlay
    - 5-regime probability support
    - Rank-based allocation across selected assets
    - All v6 proven parameters preserved (smoothing, circuit breakers, vol target)

Design Principles:
    - Continuous probability-weighted allocation — no binary thresholds
    - UPRO/SHY/TLT/GLD overlays are regime-driven, independent of selection
    - Circuit breakers and crash filters ported exactly from v6
    - Weight smoothing: 0.7 new + 0.3 previous (proven in v6)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from return_forecaster import AssetForecast
from asset_selector import AssetSelection

logger = logging.getLogger("spy_alpha_v8.portfolio_optimizer")


# ---------------------------------------------------------------------------
# Configuration (all v6 proven values)
# ---------------------------------------------------------------------------

DEFAULT_VOL_TARGET_LOW = 0.08
DEFAULT_VOL_TARGET_HIGH = 0.14
DEFAULT_VOL_TARGET_MID = 0.12

DEFAULT_SMOOTHING = 0.7
DEFAULT_MAX_WEIGHT = 0.30
DEFAULT_MAX_TURNOVER = 0.40
DEFAULT_MAX_UPRO_WEIGHT = 0.45
DEFAULT_MAX_TLT_WEIGHT = 0.25
DEFAULT_MAX_GLD_WEIGHT = 0.20

FULL_CONVICTION_THRESHOLD = 0.55
ZERO_CONVICTION_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Allocation Output
# ---------------------------------------------------------------------------

@dataclass
class PortfolioAllocation:
    """Container for a portfolio allocation decision."""
    weights: pd.Series                    # all assets + overlays, sum to 1
    selected_assets: List[str]            # assets chosen by selector
    upro_weight: float
    upro_leverage_factor: float
    shy_weight: float
    tlt_weight: float
    gld_weight: float

    target_volatility: float
    realized_volatility: float
    conviction_shrinkage: float
    turnover: float
    smoothing_applied: bool

    dominant_regime: str
    regime_conviction: float
    overall_confidence: float
    risk_on_prob: float

    allocation_date: str
    n_stocks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Covariance Estimation
# ---------------------------------------------------------------------------

def estimate_covariance(
    adj_close: pd.DataFrame,
    assets: List[str],
    lookback: int = 252,
    shrinkage: float = 0.3,
) -> pd.DataFrame:
    """Estimate annualized covariance with Ledoit-Wolf-style shrinkage."""
    available = [s for s in assets if s in adj_close.columns]
    returns = adj_close[available].pct_change().dropna().tail(lookback)

    if len(returns) < 63:
        shrinkage = max(shrinkage, 0.6)

    sample_cov = returns.cov() * 252
    diag_target = pd.DataFrame(
        np.diag(np.diag(sample_cov.values)),
        index=sample_cov.index, columns=sample_cov.columns,
    )
    return (1 - shrinkage) * sample_cov + shrinkage * diag_target


# ---------------------------------------------------------------------------
# Core Optimizer
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """
    Portfolio optimizer for v7's dynamic asset universe.

    Pipeline:
        1. Compute conviction shrinkage
        2. Compute overlay allocations (UPRO, SHY, TLT, GLD)
        3. Apply circuit breakers
        4. Distribute remaining budget to selected assets via rank-based allocation
        5. Apply vol targeting, smoothing, turnover limits
    """

    def __init__(
        self,
        vol_target: float = DEFAULT_VOL_TARGET_MID,
        vol_target_low: float = DEFAULT_VOL_TARGET_LOW,
        vol_target_high: float = DEFAULT_VOL_TARGET_HIGH,
        weight_smoothing: float = DEFAULT_SMOOTHING,
        max_weight: float = DEFAULT_MAX_WEIGHT,
        max_turnover: float = DEFAULT_MAX_TURNOVER,
        max_upro_weight: float = DEFAULT_MAX_UPRO_WEIGHT,
        max_tlt_weight: float = DEFAULT_MAX_TLT_WEIGHT,
        max_gld_weight: float = DEFAULT_MAX_GLD_WEIGHT,
        cov_lookback: int = 252,
        cov_shrinkage: float = 0.3,
    ):
        self.vol_target = vol_target
        self.vol_target_low = vol_target_low
        self.vol_target_high = vol_target_high
        self.weight_smoothing = weight_smoothing
        self.max_weight = max_weight
        self.max_turnover = max_turnover
        self.max_upro_weight = max_upro_weight
        self.max_tlt_weight = max_tlt_weight
        self.max_gld_weight = max_gld_weight
        self.cov_lookback = cov_lookback
        self.cov_shrinkage = cov_shrinkage

    def optimize(
        self,
        forecast: AssetForecast,
        selection: AssetSelection,
        adj_close: pd.DataFrame,
        previous_weights: Optional[pd.Series] = None,
    ) -> PortfolioAllocation:
        """
        Generate optimal allocation for selected assets.

        V8 MODIFICATION: Label-driven overlay removed.
        UPRO/SHY/TLT/GLD are treated as normal selected assets.
        Sizing is determined by rank-based allocation from expected returns,
        not by regime labels. The meta-allocator and risk constraint layer
        (built in later steps) will handle leverage and defense decisions.
        """
        selected = selection.selected_assets
        if not selected:
            raise ValueError("No assets selected for optimization")

        probs = forecast.regime_probs
        confidence = forecast.overall_confidence

        # ---- Regime probabilities (kept for diagnostics only) ----
        bull_prob = probs.get("Bull", 0.0)
        slowdown_prob = probs.get("Slowdown", 0.0)
        crisis_defl_prob = probs.get("Crisis-Deflation", probs.get("Crisis", 0.0))
        crisis_infl_prob = probs.get("Crisis-Inflation", 0.0)
        inflation_prob = probs.get("Inflation", 0.0)

        risk_on_prob = bull_prob + slowdown_prob
        total_crisis_prob = crisis_defl_prob + crisis_infl_prob

        # ---- Step 1: Conviction shrinkage ----
        shrinkage = self._compute_shrinkage(confidence)

        # ---- Step 2: Covariance estimation ----
        cov_matrix = estimate_covariance(adj_close, selected, self.cov_lookback, self.cov_shrinkage)

        # ---- Step 3: Rank-based allocation for ALL selected assets ----
        # No overlay — UPRO, SHY, TLT, GLD are sized by expected returns
        # just like every other asset. The meta-allocator handles leverage
        # and defense in later v8 steps.
        asset_weights = self._rank_allocate(
            forecast, selected, shrinkage, budget=1.0
        )

        # ---- Step 4: Apply position limits ----
        # Cap individual instrument weights
        if "UPRO" in asset_weights:
            asset_weights["UPRO"] = min(asset_weights["UPRO"], self.max_upro_weight)
        if "TLT" in asset_weights:
            asset_weights["TLT"] = min(asset_weights["TLT"], self.max_tlt_weight)
        if "GLD" in asset_weights:
            asset_weights["GLD"] = min(asset_weights["GLD"], self.max_gld_weight)

        # Cap any single asset
        asset_weights = asset_weights.clip(upper=self.max_weight)

        # Renormalize after capping
        if asset_weights.sum() > 0:
            asset_weights = asset_weights / asset_weights.sum()

        # ---- Step 5: Vol targeting ----
        available_for_vol = [t for t in asset_weights.index if t in cov_matrix.index]
        if len(available_for_vol) >= 2:
            port_vol = self._estimate_vol(asset_weights.reindex(available_for_vol, fill_value=0), cov_matrix)
            if port_vol > 0 and (port_vol < self.vol_target_low or port_vol > self.vol_target_high):
                scale = self.vol_target / port_vol
                asset_weights *= scale
        else:
            port_vol = self.vol_target

        # ---- Normalize final weights ----
        final_weights = asset_weights.copy()
        total = final_weights.sum()
        if total > 0:
            final_weights = final_weights / total
        final_weights = final_weights.clip(lower=0)

        # ---- Step 6: Smoothing ----
        smoothing_applied = False
        if previous_weights is not None:
            final_weights, smoothing_applied = self._smooth(final_weights, previous_weights)

        # ---- Step 7: Turnover limit ----
        turnover = 0.0
        if previous_weights is not None:
            turnover = self._turnover(final_weights, previous_weights)
            if turnover > self.max_turnover:
                final_weights = self._limit_turnover(final_weights, previous_weights, self.max_turnover)
                turnover = self._turnover(final_weights, previous_weights)

        # Extract final overlay weights for reporting
        final_tlt = float(final_weights.get("TLT", 0.0))
        final_gld = float(final_weights.get("GLD", 0.0))
        final_upro = float(final_weights.get("UPRO", 0.0))
        final_shy = float(final_weights.get("SHY", 0.0))

        return PortfolioAllocation(
            weights=final_weights,
            selected_assets=selected,
            upro_weight=final_upro,
            upro_leverage_factor=final_upro * 3.0,
            shy_weight=final_shy,
            tlt_weight=final_tlt,
            gld_weight=final_gld,
            target_volatility=self.vol_target,
            realized_volatility=port_vol,
            conviction_shrinkage=shrinkage,
            turnover=turnover,
            smoothing_applied=smoothing_applied,
            dominant_regime=forecast.dominant_regime,
            regime_conviction=forecast.regime_conviction,
            overall_confidence=confidence,
            risk_on_prob=risk_on_prob,
            allocation_date=forecast.forecast_date,
            n_stocks=selection.n_stocks,
            metadata={
                "total_crisis_prob": total_crisis_prob,
                "crisis_defl_prob": crisis_defl_prob,
                "crisis_infl_prob": crisis_infl_prob,
                "inflation_prob": inflation_prob,
                "overlay_removed": True,
            },
        )

    # ------------------------------------------------------------------
    # Conviction Shrinkage
    # ------------------------------------------------------------------

    def _compute_shrinkage(self, confidence: float) -> float:
        if confidence >= FULL_CONVICTION_THRESHOLD:
            return 1.0
        elif confidence <= ZERO_CONVICTION_THRESHOLD:
            return 0.0
        else:
            return (confidence - ZERO_CONVICTION_THRESHOLD) / (
                FULL_CONVICTION_THRESHOLD - ZERO_CONVICTION_THRESHOLD
            )

    # ------------------------------------------------------------------
    # Overlay Allocations
    # ------------------------------------------------------------------

    def _compute_upro(
        self, risk_on_prob: float, total_crisis_prob: float,
        portfolio_vol: float, confidence: float,
    ) -> float:
        """Continuous UPRO leverage scaling (ported from v6)."""
        vol_ratio = self.vol_target / max(portfolio_vol, 0.01)
        vol_adj = np.clip(vol_ratio, 0.3, 1.5)

        if total_crisis_prob > 0.50:
            upro = 0.0
        elif risk_on_prob > 0.3:
            upro_base = self.max_upro_weight * ((risk_on_prob - 0.3) / 0.7) ** 1.5
            upro = upro_base * vol_adj * confidence
        else:
            upro = 0.0

        return float(np.clip(upro, 0.0, self.max_upro_weight))

    def _compute_shy(self, risk_on_prob: float, total_crisis_prob: float) -> float:
        """Continuous SHY defensive allocation (ported from v6)."""
        if risk_on_prob > 0.7 and total_crisis_prob < 0.2:
            shy = 0.0
        else:
            shy = 0.50 * (total_crisis_prob ** 1.2)
        return float(np.clip(shy, 0.0, 0.50))

    def _compute_tlt_overlay(
        self, selected_weight: float, crisis_defl_prob: float, crisis_infl_prob: float,
    ) -> float:
        """
        TLT crisis-type conditional overlay.

        Crisis-Deflation: TLT gets a FLOOR (bonds rally in deflation).
        Crisis-Inflation: TLT forced to ZERO (bonds fall in inflation).
        """
        if crisis_defl_prob > 0.4:
            tlt_floor = 0.10 + (crisis_defl_prob - 0.4) * 0.40
            tlt = max(selected_weight, np.clip(tlt_floor, 0.0, self.max_tlt_weight))
        elif crisis_infl_prob > 0.4:
            tlt = 0.0  # Force zero
        else:
            tlt = selected_weight

        return float(np.clip(tlt, 0.0, self.max_tlt_weight))

    def _compute_gld_overlay(
        self, selected_weight: float, inflation_prob: float, crisis_infl_prob: float,
    ) -> float:
        """GLD inflation & crisis-inflation overlay."""
        if inflation_prob > 0.3:
            gld_floor = 0.08 + (inflation_prob - 0.3) * 0.30
            gld = max(selected_weight, np.clip(gld_floor, 0.0, self.max_gld_weight))
        elif crisis_infl_prob > 0.4:
            gld_floor = 0.10 + (crisis_infl_prob - 0.4) * 0.25
            gld = max(selected_weight, np.clip(gld_floor, 0.0, self.max_gld_weight))
        else:
            gld = selected_weight

        return float(np.clip(gld, 0.0, self.max_gld_weight))

    # ------------------------------------------------------------------
    # Circuit Breaker (ported exactly from v6)
    # ------------------------------------------------------------------

    def _circuit_breaker(
        self, upro: float, shy: float,
        adj_close: pd.DataFrame, forecast_dt: pd.Timestamp,
    ) -> Tuple[float, float]:
        """Dual-timeframe circuit breaker + crash momentum filter."""
        if "SPY" not in adj_close.columns:
            return upro, shy

        spy_up_to_date = adj_close["SPY"].loc[:forecast_dt]
        spy_recent = spy_up_to_date.tail(60)

        if len(spy_recent) < 10:
            return upro, shy

        current_price = spy_recent.iloc[-1]

        # Crash momentum filter: 5-day return < -4%
        if len(spy_recent) >= 6:
            spy_5d_ret = current_price / spy_recent.iloc[-6] - 1
            if spy_5d_ret < -0.04:
                upro *= 0.5
                logger.info(f"  CRASH MOMENTUM FILTER: 5d return={spy_5d_ret:.1%}, halving UPRO")

        # Fast breaker: 10-day
        rolling_peak_10 = spy_recent.rolling(10).max().iloc[-1]
        fast_dd = (current_price - rolling_peak_10) / rolling_peak_10

        # Slow breaker: 40-day
        slow_dd = 0.0
        if len(spy_recent) >= 40:
            rolling_peak_40 = spy_recent.rolling(40).max().iloc[-1]
            slow_dd = (current_price - rolling_peak_40) / rolling_peak_40

        worst_dd = min(fast_dd, slow_dd)

        if worst_dd < -0.15:
            logger.info(f"  CIRCUIT BREAKER SEVERE: dd={worst_dd:.1%}")
            upro = 0.0
            shy = 0.50
        elif worst_dd < -0.07:
            logger.info(f"  CIRCUIT BREAKER MODERATE: dd={worst_dd:.1%}")
            upro *= 0.3
            shy = max(shy, 0.30)

        return upro, shy

    # ------------------------------------------------------------------
    # Rank-Based Allocation
    # ------------------------------------------------------------------

    def _rank_allocate(
        self, forecast: AssetForecast,
        assets: List[str], shrinkage: float, budget: float,
    ) -> pd.Series:
        """
        Rank-based weight assignment for selected assets.

        High conviction: exponential rank weighting.
        Low conviction: shrink toward equal weight.
        """
        n = len(assets)
        if n == 0:
            return pd.Series(dtype=float)

        equal_weight = pd.Series(budget / n, index=assets)

        # Get scores for selected assets
        scores = forecast.expected_returns.reindex(assets, fill_value=0)
        confidence = forecast.confidence.reindex(assets, fill_value=0.5)

        adjusted = scores * confidence
        ranks = adjusted.rank(ascending=True)

        # Exponential rank weighting
        rank_scores = np.exp(ranks / n * 2.0)

        # Zero out bottom 3 when conviction is high
        if forecast.regime_conviction > 0.7 and n > 5:
            bottom = ranks.nsmallest(3).index
            rank_scores[bottom] *= 0.1

        weights = rank_scores / rank_scores.sum() * budget

        # Cap individual weights
        weights = weights.clip(upper=self.max_weight * budget / (budget if budget > 0 else 1))

        # Renormalize
        if weights.sum() > 0:
            weights = weights / weights.sum() * budget

        # Conviction shrinkage: blend with equal weight
        final = shrinkage * weights + (1 - shrinkage) * equal_weight

        return final

    # ------------------------------------------------------------------
    # Vol / Smoothing / Turnover
    # ------------------------------------------------------------------

    def _estimate_vol(self, weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
        common = [s for s in weights.index if s in cov_matrix.index]
        if len(common) < 2:
            return self.vol_target
        w = weights[common].values
        sigma = cov_matrix.loc[common, common].values
        port_var = w @ sigma @ w
        return float(np.sqrt(max(port_var, 0)))

    def _smooth(self, new: pd.Series, prev: pd.Series) -> Tuple[pd.Series, bool]:
        all_t = new.index.union(prev.index)
        n = new.reindex(all_t, fill_value=0)
        p = prev.reindex(all_t, fill_value=0)
        smoothed = self.weight_smoothing * n + (1 - self.weight_smoothing) * p
        smoothed = smoothed.clip(lower=0)
        if smoothed.sum() > 0:
            smoothed = smoothed / smoothed.sum()
        return smoothed, True

    def _turnover(self, new: pd.Series, prev: pd.Series) -> float:
        all_t = new.index.union(prev.index)
        return float((new.reindex(all_t, fill_value=0) - prev.reindex(all_t, fill_value=0)).abs().sum())

    def _limit_turnover(self, new: pd.Series, prev: pd.Series, max_to: float) -> pd.Series:
        all_t = new.index.union(prev.index)
        n = new.reindex(all_t, fill_value=0)
        p = prev.reindex(all_t, fill_value=0)
        full_to = (n - p).abs().sum()
        if full_to <= max_to:
            return new
        alpha = max_to / max(full_to, 1e-8)
        limited = alpha * n + (1 - alpha) * p
        limited = limited.clip(lower=0)
        if limited.sum() > 0:
            limited = limited / limited.sum()
        return limited

    # ------------------------------------------------------------------
    # Backtest Support
    # ------------------------------------------------------------------

    def optimize_series(
        self,
        forecasts: List[AssetForecast],
        selections: List[AssetSelection],
        adj_close: pd.DataFrame,
    ) -> List[PortfolioAllocation]:
        """Generate allocation series for backtesting."""
        allocations = []
        previous_weights = None

        for forecast, selection in zip(forecasts, selections):
            try:
                alloc = self.optimize(
                    forecast=forecast,
                    selection=selection,
                    adj_close=adj_close,
                    previous_weights=previous_weights,
                )
                allocations.append(alloc)
                previous_weights = alloc.weights
            except Exception as e:
                logger.warning(f"  Optimization failed for {forecast.forecast_date}: {e}")

        logger.info(f"Optimized {len(allocations)} allocations")
        return allocations


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_allocation(alloc: PortfolioAllocation) -> None:
    """Pretty-print a single allocation."""
    print(f"\n{'='*60}")
    print(f"PORTFOLIO ALLOCATION — {alloc.allocation_date}")
    print(f"{'='*60}")
    print(f"  Regime:            {alloc.dominant_regime} ({alloc.regime_conviction:.3f})")
    print(f"  Confidence:        {alloc.overall_confidence:.3f}")
    print(f"  Risk-on prob:      {alloc.risk_on_prob:.3f}")
    print(f"  Conviction shrink: {alloc.conviction_shrinkage:.3f}")
    print(f"  Target vol:        {alloc.target_volatility:.1%}")
    print(f"  Estimated vol:     {alloc.realized_volatility:.1%}")
    print(f"  Turnover:          {alloc.turnover:.1%}")
    print(f"  Stocks held:       {alloc.n_stocks}")

    print(f"\n  UPRO:  {alloc.upro_weight:>7.1%}  (leverage: {alloc.upro_leverage_factor:.2f}x)")
    print(f"  SHY:   {alloc.shy_weight:>7.1%}")
    print(f"  TLT:   {alloc.tlt_weight:>7.1%}")
    print(f"  GLD:   {alloc.gld_weight:>7.1%}")

    print(f"\n  {'Asset':<8s} {'Weight':>8s}")
    print(f"  {'-'*16}")
    for ticker, weight in alloc.weights.sort_values(ascending=False).items():
        if weight > 0.001:
            marker = ""
            if ticker == "UPRO":
                marker = " ◆ leverage"
            elif ticker == "SHY":
                marker = " ◆ defensive"
            elif ticker == "TLT":
                marker = " ◆ bonds"
            elif ticker == "GLD":
                marker = " ◆ gold"
            print(f"  {ticker:<8s} {weight:>7.1%}{marker}")

    print(f"\n  Total:  {alloc.weights.sum():>7.1%}")


def summarize_allocation_series(allocations: List[PortfolioAllocation]) -> Dict[str, Any]:
    """Summarize allocation series for diagnostics."""
    if not allocations:
        return {"error": "No allocations"}

    upro = [a.upro_weight for a in allocations]
    shy = [a.shy_weight for a in allocations]
    tlt = [a.tlt_weight for a in allocations]
    gld = [a.gld_weight for a in allocations]
    turnovers = [a.turnover for a in allocations]
    n_stocks = [a.n_stocks for a in allocations]

    regime_dist = {}
    for a in allocations:
        r = a.dominant_regime
        regime_dist[r] = regime_dist.get(r, 0) + 1

    # Asset frequency
    asset_freq: Dict[str, int] = {}
    for a in allocations:
        for t in a.selected_assets:
            asset_freq[t] = asset_freq.get(t, 0) + 1

    return {
        "n_allocations": len(allocations),
        "date_range": f"{allocations[0].allocation_date} → {allocations[-1].allocation_date}",
        "mean_upro_weight": float(np.mean(upro)),
        "max_upro_weight": float(np.max(upro)),
        "mean_shy_weight": float(np.mean(shy)),
        "mean_tlt_weight": float(np.mean(tlt)),
        "mean_gld_weight": float(np.mean(gld)),
        "mean_turnover": float(np.mean(turnovers)),
        "mean_stocks_held": float(np.mean(n_stocks)),
        "regime_distribution": regime_dist,
        "asset_selection_frequency": dict(sorted(asset_freq.items(), key=lambda x: -x[1])),
    }