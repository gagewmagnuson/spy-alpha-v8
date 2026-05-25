"""
SPY Alpha v8 — Strategy Health Tracking
==========================================
 
NEW module in v8. Tracks rolling performance metrics for each strategy,
providing the meta-allocator with information about recent strategy quality.
 
This is the "Capital Allocation Memory" from the build spec — it gives
the meta-allocator context about which strategies are performing well
and which are struggling.
 
Metrics tracked per strategy:
    - Rolling 63-day Sharpe ratio
    - Rolling 63-day max drawdown
    - Rolling 21-day hit rate (% of positive return days)
    - Rolling 21-day turnover
    - Strategy stability score (std of weight changes)
 
Design Principles:
    - Pure tracking — no allocation decisions
    - All metrics are backward-looking (no lookahead bias)
    - Metrics computed from realized returns, not forecasted returns
    - NaN-safe — handles missing data gracefully
"""
 
from __future__ import annotations
 
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
 
import numpy as np
import pandas as pd
 
logger = logging.getLogger("spy_alpha_v8.strategy_health")
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
SHARPE_WINDOW: int = 63       # ~3 months
DRAWDOWN_WINDOW: int = 63     # ~3 months
HIT_RATE_WINDOW: int = 21     # ~1 month
TURNOVER_WINDOW: int = 21     # ~1 month
STABILITY_WINDOW: int = 21    # ~1 month
 
 
# ---------------------------------------------------------------------------
# Strategy Health Metrics
# ---------------------------------------------------------------------------
 
@dataclass
class StrategyHealthSnapshot:
    """Health metrics for a single strategy at a single point in time."""
    strategy_name: str
    date: str
    rolling_sharpe: float
    rolling_max_dd: float
    rolling_hit_rate: float
    rolling_turnover: float
    stability_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
 
 
class StrategyHealthTracker:
    """
    Tracks rolling performance metrics for each strategy.
 
    Consumes daily strategy returns and weight histories to compute
    health metrics that feed into the meta-allocator as features.
    """
 
    def __init__(
        self,
        sharpe_window: int = SHARPE_WINDOW,
        drawdown_window: int = DRAWDOWN_WINDOW,
        hit_rate_window: int = HIT_RATE_WINDOW,
        turnover_window: int = TURNOVER_WINDOW,
        stability_window: int = STABILITY_WINDOW,
    ):
        self.sharpe_window = sharpe_window
        self.drawdown_window = drawdown_window
        self.hit_rate_window = hit_rate_window
        self.turnover_window = turnover_window
        self.stability_window = stability_window
 
    def compute_health_series(
        self,
        strategy_returns: pd.Series,
        weight_changes: Optional[pd.Series] = None,
        turnover_series: Optional[pd.Series] = None,
        strategy_name: str = "unknown",
    ) -> pd.DataFrame:
        """
        Compute rolling health metrics for a single strategy.
 
        Args:
            strategy_returns: Daily returns for this strategy
            weight_changes: Daily total absolute weight change (for stability)
            turnover_series: Turnover at each rebalance (for rolling turnover)
            strategy_name: Name for column prefixing
 
        Returns:
            DataFrame with one row per trading day, columns:
                {name}_sharpe_63d
                {name}_max_dd_63d
                {name}_hit_rate_21d
                {name}_turnover_21d
                {name}_stability_21d
        """
        prefix = strategy_name
        features = {}
 
        # ---- Rolling Sharpe (annualized) ----
        roll_mean = strategy_returns.rolling(
            self.sharpe_window, min_periods=21
        ).mean() * 252
 
        roll_std = strategy_returns.rolling(
            self.sharpe_window, min_periods=21
        ).std() * np.sqrt(252)
 
        features[f"{prefix}_sharpe_63d"] = roll_mean / roll_std.replace(0, np.nan)
 
        # ---- Rolling Max Drawdown ----
        cum = (1 + strategy_returns).cumprod()
 
        def rolling_max_dd(window):
            result = pd.Series(0.0, index=cum.index)
            for i in range(window, len(cum)):
                window_cum = cum.iloc[i - window:i + 1]
                peak = window_cum.expanding().max()
                dd = (window_cum - peak) / peak
                result.iloc[i] = dd.min()
            return result
 
        features[f"{prefix}_max_dd_63d"] = rolling_max_dd(self.drawdown_window)
 
        # ---- Rolling Hit Rate ----
        positive_days = (strategy_returns > 0).astype(float)
        features[f"{prefix}_hit_rate_21d"] = positive_days.rolling(
            self.hit_rate_window, min_periods=10
        ).mean()
 
        # ---- Rolling Turnover ----
        if turnover_series is not None and not turnover_series.empty:
            # Turnover is sparse (only at rebalances), forward-fill
            turnover_filled = turnover_series.reindex(strategy_returns.index).ffill()
            features[f"{prefix}_turnover_21d"] = turnover_filled.rolling(
                self.turnover_window, min_periods=1
            ).mean()
        else:
            features[f"{prefix}_turnover_21d"] = pd.Series(
                np.nan, index=strategy_returns.index
            )
 
        # ---- Stability Score ----
        # Lower = more stable (less erratic weight changes)
        if weight_changes is not None and not weight_changes.empty:
            weight_changes_filled = weight_changes.reindex(strategy_returns.index).ffill().fillna(0)
            features[f"{prefix}_stability_21d"] = weight_changes_filled.rolling(
                self.stability_window, min_periods=5
            ).std()
        else:
            features[f"{prefix}_stability_21d"] = pd.Series(
                np.nan, index=strategy_returns.index
            )
 
        df = pd.DataFrame(features)
        logger.info(
            f"Health metrics for {strategy_name}: {len(df)} days, "
            f"mean Sharpe={df[f'{prefix}_sharpe_63d'].mean():.2f}"
        )
        return df
 
    def compute_all_health(
        self,
        strategy_daily_returns: Dict[str, pd.Series],
        strategy_weight_changes: Optional[Dict[str, pd.Series]] = None,
        strategy_turnovers: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """
        Compute health metrics for all strategies and combine into one DataFrame.
 
        Args:
            strategy_daily_returns: {strategy_name: daily_return_series}
            strategy_weight_changes: {strategy_name: weight_change_series}
            strategy_turnovers: {strategy_name: turnover_series}
 
        Returns:
            Combined DataFrame with all strategy health features
        """
        all_health = []
 
        for name, returns in strategy_daily_returns.items():
            weight_changes = None
            if strategy_weight_changes and name in strategy_weight_changes:
                weight_changes = strategy_weight_changes[name]
 
            turnover = None
            if strategy_turnovers and name in strategy_turnovers:
                turnover = strategy_turnovers[name]
 
            health = self.compute_health_series(
                returns, weight_changes, turnover, strategy_name=name
            )
            all_health.append(health)
 
        if not all_health:
            return pd.DataFrame()
 
        combined = pd.concat(all_health, axis=1)
        logger.info(
            f"Combined health metrics: {combined.shape[0]} days × {combined.shape[1]} features"
        )
        return combined
 
    def get_latest_health(
        self,
        health_df: pd.DataFrame,
        strategy_names: List[str],
    ) -> Dict[str, StrategyHealthSnapshot]:
        """
        Get the most recent health snapshot for each strategy.
        """
        if health_df.empty:
            return {}
 
        latest = health_df.iloc[-1]
        date = health_df.index[-1]
 
        snapshots = {}
        for name in strategy_names:
            snapshots[name] = StrategyHealthSnapshot(
                strategy_name=name,
                date=date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                rolling_sharpe=float(latest.get(f"{name}_sharpe_63d", np.nan)),
                rolling_max_dd=float(latest.get(f"{name}_max_dd_63d", np.nan)),
                rolling_hit_rate=float(latest.get(f"{name}_hit_rate_21d", np.nan)),
                rolling_turnover=float(latest.get(f"{name}_turnover_21d", np.nan)),
                stability_score=float(latest.get(f"{name}_stability_21d", np.nan)),
            )
 
        return snapshots
 
 
# ---------------------------------------------------------------------------
# Utility: Compute Daily Strategy Returns from Signals
# ---------------------------------------------------------------------------
 
def compute_strategy_daily_returns(
    strategy_outputs: list,
    adj_close: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily portfolio returns for a strategy from its signal outputs.
 
    Maps each signal to its effective dates (holds until next signal)
    and computes the weighted daily return.
    """
    if not strategy_outputs:
        return pd.Series(dtype=float)
 
    daily_returns = adj_close.pct_change()
 
    signal_dates = [pd.Timestamp(o.strategy_metadata["date"]) for o in strategy_outputs]
    signal_weights = [o.proposed_weights for o in strategy_outputs]
 
    portfolio_returns = pd.Series(0.0, index=daily_returns.index)
    current_weights = {}
    signal_idx = 0
 
    for date in daily_returns.index:
        while signal_idx < len(signal_dates) and signal_dates[signal_idx] <= date:
            current_weights = signal_weights[signal_idx]
            signal_idx += 1
 
        if current_weights:
            ret = sum(
                current_weights.get(a, 0) * daily_returns.loc[date, a]
                for a in current_weights
                if a in daily_returns.columns and pd.notna(daily_returns.loc[date, a])
            )
            portfolio_returns.loc[date] = ret
 
    return portfolio_returns
 
 
def compute_weight_changes(
    strategy_outputs: list,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Compute the total absolute weight change at each rebalance.
 
    Returns a sparse series (non-zero only at rebalance dates).
    """
    changes = pd.Series(0.0, index=index)
 
    prev_weights = {}
    for o in strategy_outputs:
        date = pd.Timestamp(o.strategy_metadata["date"])
        if date not in index:
            continue
 
        curr_weights = o.proposed_weights
        all_assets = set(list(prev_weights.keys()) + list(curr_weights.keys()))
 
        total_change = sum(
            abs(curr_weights.get(a, 0) - prev_weights.get(a, 0))
            for a in all_assets
        )
 
        changes.loc[date] = total_change
        prev_weights = curr_weights
 
    return changes
 
 
# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
 
def print_health_summary(
    health_df: pd.DataFrame,
    strategy_names: List[str],
) -> None:
    """Print a summary of strategy health metrics."""
    print(f"\n{'='*60}")
    print(f"STRATEGY HEALTH SUMMARY")
    print(f"{'='*60}")
    print(f"  Period: {health_df.index[0].date()} → {health_df.index[-1].date()}")
    print()
 
    print(f"  {'Strategy':<20s} {'Sharpe':>8s} {'MaxDD':>8s} {'HitRate':>8s} {'Turnover':>8s} {'Stability':>10s}")
    print(f"  {'-'*54}")
 
    for name in strategy_names:
        sharpe = health_df[f"{name}_sharpe_63d"].mean()
        max_dd = health_df[f"{name}_max_dd_63d"].mean()
        hit_rate = health_df[f"{name}_hit_rate_21d"].mean()
        turnover = health_df[f"{name}_turnover_21d"].mean()
        stability = health_df[f"{name}_stability_21d"].mean()
 
        print(
            f"  {name:<20s} {sharpe:>7.2f} {max_dd:>7.1%} {hit_rate:>7.1%} "
            f"{turnover:>7.3f} {stability:>9.4f}"
        )