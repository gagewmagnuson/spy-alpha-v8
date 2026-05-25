"""
SPY Alpha v8 — Strategy 3: Defensive / Tail Protection
=========================================================
 
NEW module in v8. A strategy designed to generate positive returns during
market stress. Expected to have low or negative standalone Sharpe — its
value is in portfolio-level convexity improvement.
 
Design Principles:
    - Activates during stress, dormant during calm
    - Uses observable stress indicators (VIX, credit, correlation, vol)
    - No HMM dependency for activation — regime-independent crisis detection
    - Uses HMM crisis-type probabilities ONLY for defensive asset selection
      (TLT for deflation, GLD for inflation, SHY as residual)
    - Provides genuine asymmetry: Strategies 1+2 capture upside,
      Strategy 3 captures crisis alpha
    - Daily rebalance for crisis response, weekly for non-crisis positioning
 
Signal Generation:
    Compute stress_score from observable features:
        - VIX percentile (252-day)
        - VIX term structure inversion (VIX / VIX3M)
        - Credit stress (HY OAS z-score)
        - Cross-asset correlation level
        - Volatility acceleration
 
    stress_score ∈ [0, 1] (0 = calm, 1 = extreme stress)
 
    When stress_score > threshold:
        Allocate to defensive assets:
        - TLT weight proportional to crisis-deflation signals
        - GLD weight proportional to crisis-inflation signals
        - SHY as residual (cash safety)
 
    When stress_score < threshold:
        Minimal allocation (mostly cash/SHY)
"""
 
from __future__ import annotations
 
import logging
from typing import Any, Dict, List, Optional
 
import numpy as np
import pandas as pd
 
from strategy_regime import StrategyOutput
 
logger = logging.getLogger("spy_alpha_v8.strategy_defensive")
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
# Stress score activation threshold
# Below this, strategy is mostly in cash
STRESS_THRESHOLD: float = 0.35
 
# Stress score component weights
STRESS_WEIGHTS: Dict[str, float] = {
    "vix_percentile": 0.25,
    "vix_term_inversion": 0.20,
    "credit_stress": 0.20,
    "correlation_stress": 0.15,
    "vol_acceleration": 0.20,
}
 
# Defensive asset allocation bounds
MAX_TLT_DEFENSIVE: float = 0.50
MAX_GLD_DEFENSIVE: float = 0.40
MIN_SHY_DEFENSIVE: float = 0.10
 
# Lookback windows
VIX_PERCENTILE_WINDOW: int = 252
VOL_ACCELERATION_WINDOW: int = 10
CREDIT_ZSCORE_WINDOW: int = 252
CORRELATION_WINDOW: int = 21
 
# Rebalance frequency
REBALANCE_EVERY: int = 5
 
 
# ---------------------------------------------------------------------------
# Stress Score Computation
# ---------------------------------------------------------------------------
 
def compute_stress_components(
    raw_close: pd.DataFrame,
    fred_data: pd.DataFrame,
    vix_term: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute individual stress components, each scaled to [0, 1].
 
    Components:
        vix_percentile:     VIX level percentile over 252-day rolling window
        vix_term_inversion: Degree of VIX term structure inversion (VIX/VIX3M)
        credit_stress:      HY OAS z-score mapped to [0, 1]
        correlation_stress: Cross-asset correlation level
        vol_acceleration:   Rate of change in realized volatility
    """
    components = {}
    returns = raw_close.pct_change()
 
    # ---- VIX Percentile ----
    # Use ^VIX from vix_term if available, otherwise estimate from SPY vol
    if vix_term is not None and not vix_term.empty and "^VIX" in vix_term.columns:
        vix = vix_term["^VIX"]
        components["vix_percentile"] = vix.rolling(
            VIX_PERCENTILE_WINDOW, min_periods=126
        ).rank(pct=True)
    elif "SPY" in returns.columns:
        # Fallback: use realized vol percentile as VIX proxy
        spy_vol = returns["SPY"].rolling(20).std() * np.sqrt(252)
        components["vix_percentile"] = spy_vol.rolling(
            VIX_PERCENTILE_WINDOW, min_periods=126
        ).rank(pct=True)
 
    # ---- VIX Term Structure Inversion ----
    if vix_term is not None and not vix_term.empty:
        if "^VIX" in vix_term.columns and "^VIX3M" in vix_term.columns:
            vix = vix_term["^VIX"]
            vix3m = vix_term["^VIX3M"]
            ratio = vix / vix3m.replace(0, np.nan)
 
            # Map ratio to [0, 1]: ratio < 0.9 → 0 (contango), ratio > 1.2 → 1 (backwardation)
            inversion = (ratio - 0.9) / (1.2 - 0.9)
            components["vix_term_inversion"] = inversion.clip(0, 1)
    else:
        # Fallback: use vol regime change as proxy
        if "SPY" in returns.columns:
            spy_vol_20 = returns["SPY"].rolling(20).std() * np.sqrt(252)
            spy_vol_60 = returns["SPY"].rolling(60).std() * np.sqrt(252)
            ratio = spy_vol_20 / spy_vol_60.replace(0, np.nan)
            inversion = (ratio - 0.9) / (1.2 - 0.9)
            components["vix_term_inversion"] = inversion.clip(0, 1)
 
    # ---- Credit Stress ----
    hy_col = "BAMLH0A0HYM2"
    if hy_col in fred_data.columns:
        hy = fred_data[hy_col]
        mean = hy.rolling(CREDIT_ZSCORE_WINDOW, min_periods=126).mean()
        std = hy.rolling(CREDIT_ZSCORE_WINDOW, min_periods=126).std()
        zscore = (hy - mean) / std.replace(0, np.nan)
 
        # Map z-score to [0, 1]: z < 0 → 0, z > 3 → 1
        components["credit_stress"] = (zscore / 3.0).clip(0, 1)
 
    # ---- Cross-Asset Correlation Stress ----
    # High correlation across asset classes signals crisis
    equity_assets = [a for a in ["SPY", "QQQ", "IWM"] if a in returns.columns]
    rate_assets = [a for a in ["TLT", "IEF"] if a in returns.columns]
    commodity_assets = [a for a in ["GLD", "DBC"] if a in returns.columns]
 
    all_stress_assets = equity_assets + rate_assets + commodity_assets
    if len(all_stress_assets) >= 4:
        # Average absolute pairwise correlation
        corr_series = []
        for i, a1 in enumerate(all_stress_assets):
            for a2 in all_stress_assets[i+1:]:
                if a1 in returns.columns and a2 in returns.columns:
                    pairwise = returns[a1].rolling(CORRELATION_WINDOW).corr(returns[a2]).abs()
                    corr_series.append(pairwise)
 
        if corr_series:
            avg_corr = pd.concat(corr_series, axis=1).mean(axis=1)
 
            # Map to [0, 1]: avg_corr < 0.3 → 0 (normal), avg_corr > 0.7 → 1 (crisis)
            components["correlation_stress"] = ((avg_corr - 0.3) / 0.4).clip(0, 1)
 
    # ---- Volatility Acceleration ----
    if "SPY" in returns.columns:
        spy_vol = returns["SPY"].rolling(VOL_ACCELERATION_WINDOW).std() * np.sqrt(252)
        spy_vol_prev = spy_vol.shift(VOL_ACCELERATION_WINDOW)
 
        # Percentage change in realized vol
        vol_change = (spy_vol - spy_vol_prev) / spy_vol_prev.replace(0, np.nan)
 
        # Map to [0, 1]: change < 0 → 0 (vol declining), change > 1.0 (100% increase) → 1
        components["vol_acceleration"] = (vol_change / 1.0).clip(0, 1)
 
    df = pd.DataFrame(components)
 
    # Log coverage
    for col in df.columns:
        valid = df[col].notna().sum()
        logger.info(f"  Stress component {col}: {valid} valid days")
 
    return df
 
 
def compute_stress_score(
    components: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Compute the composite stress score as a weighted average of components.
 
    Returns a Series with values in [0, 1].
    Missing components are excluded and remaining weights renormalized.
    """
    if weights is None:
        weights = STRESS_WEIGHTS
 
    available = [c for c in weights if c in components.columns]
    if not available:
        logger.warning("No stress components available")
        return pd.Series(0.0, index=components.index)
 
    # Renormalize weights for available components
    total_weight = sum(weights[c] for c in available)
    normalized_weights = {c: weights[c] / total_weight for c in available}
 
    # Weighted average
    score = pd.Series(0.0, index=components.index)
    for comp, w in normalized_weights.items():
        score += components[comp].fillna(0) * w
 
    return score.clip(0, 1)
 
 
# ---------------------------------------------------------------------------
# Deflation/Inflation Signal for Asset Selection
# ---------------------------------------------------------------------------
 
def compute_crisis_type_signal(
    raw_close: pd.DataFrame,
    fred_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine whether stress is deflationary or inflationary in nature.
 
    Uses observable data (not HMM labels) to decide between TLT and GLD:
        - TLT behavior: if TLT is rising during stress → deflation
        - Yield curve: if flattening/inverting → deflation
        - Gold behavior: if GLD is rising during stress → inflation
        - Real rates proxy: high/rising → deflation, low/falling → inflation
 
    Returns DataFrame with columns: deflation_signal, inflation_signal (each [0,1])
    """
    signals = {}
    returns = raw_close.pct_change()
 
    # ---- TLT behavior (21-day) ----
    if "TLT" in returns.columns:
        tlt_ret_21d = returns["TLT"].rolling(21).sum()
        # Positive TLT return → deflation signal
        # Map: -5% → 0, +5% → 1
        signals["tlt_deflation"] = ((tlt_ret_21d + 0.05) / 0.10).clip(0, 1)
 
    # ---- GLD behavior (21-day) ----
    if "GLD" in returns.columns:
        gld_ret_21d = returns["GLD"].rolling(21).sum()
        # Positive GLD return → inflation signal
        signals["gld_inflation"] = ((gld_ret_21d + 0.05) / 0.10).clip(0, 1)
 
    # ---- Yield curve ----
    if "T10Y2Y" in fred_data.columns:
        curve = fred_data["T10Y2Y"]
        # Inverted/flat curve → deflation
        # Map: curve > 2.0 → 0 (steep, inflationary), curve < 0 → 1 (inverted, deflationary)
        signals["curve_deflation"] = ((2.0 - curve) / 2.0).clip(0, 1)
 
    df = pd.DataFrame(signals)
 
    if df.empty:
        return pd.DataFrame({
            "deflation_signal": pd.Series(0.5, index=raw_close.index),
            "inflation_signal": pd.Series(0.5, index=raw_close.index),
        })
 
    # Deflation signal: average of TLT behavior and yield curve
    defl_cols = [c for c in ["tlt_deflation", "curve_deflation"] if c in df.columns]
    infl_cols = [c for c in ["gld_inflation"] if c in df.columns]
 
    deflation = df[defl_cols].mean(axis=1) if defl_cols else pd.Series(0.5, index=df.index)
    inflation = df[infl_cols].mean(axis=1) if infl_cols else pd.Series(0.5, index=df.index)
 
    # Normalize so they sum to ~1 (they represent relative probability)
    total = deflation + inflation
    total = total.replace(0, 1)  # avoid division by zero
 
    return pd.DataFrame({
        "deflation_signal": deflation / total,
        "inflation_signal": inflation / total,
    })
 
 
# ---------------------------------------------------------------------------
# Strategy 3: Defensive / Tail Protection
# ---------------------------------------------------------------------------
 
class DefensiveStrategy:
    """
    Strategy 3: Stress-activated defensive positioning.
 
    Dormant during calm markets (mostly SHY). Activates during stress
    to allocate across TLT (deflation), GLD (inflation), and SHY (cash).
 
    Expected standalone performance:
        - Low or negative Sharpe (cost of protection)
        - Strong positive returns during crises
        - Negatively correlated with Strategies 1+2 during stress
 
    Portfolio-level value:
        - Improves Max DD
        - Provides convexity (asymmetric payoff)
        - Addresses v7's crisis protection problem without HMM label dependency
    """
 
    def __init__(
        self,
        stress_threshold: float = STRESS_THRESHOLD,
        stress_weights: Optional[Dict[str, float]] = None,
        max_tlt: float = MAX_TLT_DEFENSIVE,
        max_gld: float = MAX_GLD_DEFENSIVE,
        min_shy: float = MIN_SHY_DEFENSIVE,
        rebalance_every: int = REBALANCE_EVERY,
    ):
        self.stress_threshold = stress_threshold
        self.stress_weights = stress_weights or STRESS_WEIGHTS
        self.max_tlt = max_tlt
        self.max_gld = max_gld
        self.min_shy = min_shy
        self.rebalance_every = rebalance_every
 
        # Computed during build
        self.stress_components: Optional[pd.DataFrame] = None
        self.stress_score: Optional[pd.Series] = None
        self.crisis_type: Optional[pd.DataFrame] = None
 
    def build(self, snapshot: Dict[str, Any]) -> None:
        """
        Pre-compute stress scores and crisis type signals for the full period.
        """
        from data_pipeline import get_raw_close, get_fred
 
        logger.info("Strategy 3 (Defensive): Building signals...")
 
        raw_close = get_raw_close(snapshot)
        fred_data = get_fred(snapshot)
 
        # Compute stress components (no VIX term structure from snapshot —
        # uses fallback proxies. Live mode will have fresh VIX data.)
        self.stress_components = compute_stress_components(
            raw_close, fred_data, vix_term=None
        )
 
        # Compute composite stress score
        self.stress_score = compute_stress_score(
            self.stress_components, self.stress_weights
        )
 
        # Compute crisis type signals
        self.crisis_type = compute_crisis_type_signal(raw_close, fred_data)
 
        logger.info(
            f"Strategy 3 (Defensive): Built successfully, "
            f"{len(self.stress_score)} days, "
            f"mean stress: {self.stress_score.mean():.3f}, "
            f"days above threshold: {(self.stress_score > self.stress_threshold).sum()}"
        )
 
    def generate_signals(
        self,
        snapshot: Dict[str, Any],
        rebalance_dates: Optional[pd.DatetimeIndex] = None,
    ) -> List[StrategyOutput]:
        """
        Generate strategy signals for each rebalance date.
        """
        if self.stress_score is None:
            raise RuntimeError("Call build() before generate_signals()")
 
        # Determine rebalance dates
        if rebalance_dates is not None:
            available = self.stress_score.index
            dates = rebalance_dates.intersection(available)
        else:
            available = self.stress_score.dropna().index
            start_idx = max(VIX_PERCENTILE_WINDOW, CREDIT_ZSCORE_WINDOW) + 10
            if start_idx >= len(available):
                logger.warning("Insufficient data for defensive signals")
                return []
            dates = available[start_idx::self.rebalance_every]
 
        outputs = []
        for date in dates:
            output = self._generate_single_signal(date)
            if output is not None:
                outputs.append(output)
 
        logger.info(f"Strategy 3: Generated {len(outputs)} signals")
        return outputs
 
    def _generate_single_signal(self, date: pd.Timestamp) -> Optional[StrategyOutput]:
        """Generate a single defensive signal for a given date."""
        if date not in self.stress_score.index:
            return None
 
        stress = self.stress_score.loc[date]
        if pd.isna(stress):
            stress = 0.0
 
        # Get crisis type signals
        if date in self.crisis_type.index:
            deflation_sig = float(self.crisis_type.loc[date, "deflation_signal"])
            inflation_sig = float(self.crisis_type.loc[date, "inflation_signal"])
        else:
            deflation_sig = 0.5
            inflation_sig = 0.5
 
        # ---- Compute weights ----
        weights = {}
 
        if stress > self.stress_threshold:
            # Activated — allocate to defensive assets based on stress level
            # Scale activation intensity by how far above threshold
            intensity = min((stress - self.stress_threshold) / (1.0 - self.stress_threshold), 1.0)
 
            # Defensive budget: proportion of capital going to TLT/GLD
            # At maximum stress, up to (1 - min_shy) goes to TLT/GLD
            defensive_budget = intensity * (1.0 - self.min_shy)
 
            # Split defensive budget between TLT and GLD based on crisis type
            tlt_raw = defensive_budget * deflation_sig
            gld_raw = defensive_budget * inflation_sig
 
            # Apply caps
            tlt_weight = min(tlt_raw, self.max_tlt)
            gld_weight = min(gld_raw, self.max_gld)
 
            # SHY gets the remainder
            shy_weight = max(1.0 - tlt_weight - gld_weight, self.min_shy)
 
            # Normalize to sum to 1.0
            total = tlt_weight + gld_weight + shy_weight
            weights["TLT"] = tlt_weight / total
            weights["GLD"] = gld_weight / total
            weights["SHY"] = shy_weight / total
 
            confidence = intensity
        else:
            # Not activated — mostly cash with minimal positioning
            # Small residual allocations based on crisis type
            residual = 0.05  # 5% in defensive assets even when calm
            weights["TLT"] = residual * deflation_sig
            weights["GLD"] = residual * inflation_sig
            weights["SHY"] = 1.0 - residual
 
            confidence = 0.1
 
        # Clean near-zero weights
        weights = {k: v for k, v in weights.items() if v > 1e-6}
 
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
 
        active_assets = [k for k, v in weights.items() if v > 0.01]
 
        # ---- Stress component detail for metadata ----
        component_detail = {}
        if date in self.stress_components.index:
            for col in self.stress_components.columns:
                val = self.stress_components.loc[date, col]
                if pd.notna(val):
                    component_detail[col] = float(val)
 
        metadata = {
            "date": date.strftime("%Y-%m-%d"),
            "stress_score": float(stress),
            "stress_activated": stress > self.stress_threshold,
            "intensity": float(min((stress - self.stress_threshold) / (1.0 - self.stress_threshold), 1.0)) if stress > self.stress_threshold else 0.0,
            "deflation_signal": float(deflation_sig),
            "inflation_signal": float(inflation_sig),
            "stress_components": component_detail,
        }
 
        return StrategyOutput(
            strategy_name="defensive",
            proposed_weights=weights,
            confidence=confidence,
            active_assets=active_assets,
            strategy_metadata=metadata,
        )
 
    def get_stress_score(self) -> Optional[pd.Series]:
        """Return the full stress score series for diagnostics."""
        return self.stress_score
 
    def get_stress_components(self) -> Optional[pd.DataFrame]:
        """Return the full stress components for diagnostics."""
        return self.stress_components
 
 
# ---------------------------------------------------------------------------
# Standalone Backtest Support
# ---------------------------------------------------------------------------
 
def backtest_defensive_standalone(
    snapshot: Dict[str, Any],
    rebalance_every: int = REBALANCE_EVERY,
) -> pd.DataFrame:
    """
    Run a standalone backtest of the defensive strategy.
 
    Returns a DataFrame with columns:
        - portfolio_return: daily portfolio return
        - spy_return: daily SPY return (benchmark)
        - cumulative: cumulative portfolio return
        - spy_cumulative: cumulative SPY return
        - stress_score: stress level on each day
    """
    from data_pipeline import get_adj_close
 
    adj_close = get_adj_close(snapshot)
 
    # Build strategy
    strategy = DefensiveStrategy(rebalance_every=rebalance_every)
    strategy.build(snapshot)
 
    # Generate signals
    outputs = strategy.generate_signals(snapshot)
 
    if not outputs:
        raise RuntimeError("No signals generated")
 
    signal_dates = [pd.Timestamp(o.strategy_metadata["date"]) for o in outputs]
 
    # Daily returns
    defensive_assets = ["TLT", "GLD", "SHY"]
    available = [a for a in defensive_assets if a in adj_close.columns]
    daily_returns = adj_close[available].pct_change()
    spy_returns = adj_close["SPY"].pct_change() if "SPY" in adj_close.columns else pd.Series(0, index=adj_close.index)
 
    # Walk through signals
    portfolio_returns = []
    current_weights = {}
    signal_idx = 0
 
    for date in daily_returns.index:
        date_str = date.strftime("%Y-%m-%d")
 
        while signal_idx < len(signal_dates) and signal_dates[signal_idx] <= date:
            current_weights = outputs[signal_idx].proposed_weights
            signal_idx += 1
 
        if not current_weights:
            portfolio_returns.append({
                "date": date,
                "portfolio_return": 0.0,
                "spy_return": float(spy_returns.get(date, 0.0)),
            })
            continue
 
        port_ret = 0.0
        for asset, weight in current_weights.items():
            if asset in daily_returns.columns and pd.notna(daily_returns.loc[date, asset]):
                port_ret += weight * daily_returns.loc[date, asset]
 
        # Get stress score for this date
        stress = strategy.stress_score.loc[date] if date in strategy.stress_score.index else 0.0
 
        portfolio_returns.append({
            "date": date,
            "portfolio_return": port_ret,
            "spy_return": float(spy_returns.get(date, 0.0)),
            "stress_score": float(stress) if pd.notna(stress) else 0.0,
        })
 
    result = pd.DataFrame(portfolio_returns).set_index("date")
    result["cumulative"] = (1 + result["portfolio_return"]).cumprod()
    result["spy_cumulative"] = (1 + result["spy_return"]).cumprod()
 
    return result
 
 
def print_defensive_backtest_report(result: pd.DataFrame) -> None:
    """Print performance report for the standalone defensive backtest."""
    returns = result["portfolio_return"].dropna()
    spy_returns = result["spy_return"].dropna()
 
    common = returns.index.intersection(spy_returns.index)
    returns = returns.loc[common]
    spy_returns = spy_returns.loc[common]
 
    # Skip initial zero period
    first_nonzero = returns[returns != 0].index[0] if (returns != 0).any() else returns.index[0]
    returns = returns.loc[first_nonzero:]
    spy_returns = spy_returns.loc[first_nonzero:]
 
    n_years = len(returns) / 252
 
    cum = (1 + returns).cumprod()
    cagr = cum.iloc[-1] ** (1 / n_years) - 1
 
    spy_cum = (1 + spy_returns).cumprod()
    spy_cagr = spy_cum.iloc[-1] ** (1 / n_years) - 1
 
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
 
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (returns.mean() * 252) / downside_vol
 
    cumulative = (1 + returns).cumprod()
    rolling_peak = cumulative.expanding().max()
    drawdown = (cumulative - rolling_peak) / rolling_peak
    max_dd = drawdown.min()
 
    ann_vol = returns.std() * np.sqrt(252)
 
    # Crisis period returns
    crisis_periods = [
        ("2008 Crisis", "2008-09-01", "2009-03-31"),
        ("2020 COVID", "2020-02-15", "2020-04-30"),
        ("2022 Bear", "2022-01-01", "2022-10-31"),
    ]
 
    # Correlation with SPY
    corr_63d = returns.rolling(63).corr(spy_returns)
 
    # Stress activation stats
    if "stress_score" in result.columns:
        stress = result.loc[first_nonzero:, "stress_score"]
        stress_activated_pct = (stress > STRESS_THRESHOLD).mean()
    else:
        stress_activated_pct = 0.0
 
    print(f"\n{'='*60}")
    print(f"STRATEGY 3 (DEFENSIVE) — STANDALONE BACKTEST")
    print(f"{'='*60}")
    print(f"--- Return Summary ---")
    print(f"  Period:              {n_years:.1f} years ({len(returns)} trading days)")
    print(f"  CAGR:                {cagr:>8.1%}    (Benchmark: {spy_cagr:>8.1%})")
    print(f"  Total Return:        {(cum.iloc[-1]-1):>8.1%}    (Benchmark: {(spy_cum.iloc[-1]-1):>8.1%})")
    print(f"--- Risk-Adjusted Metrics ---")
    print(f"  Sharpe Ratio:        {sharpe:>8.2f}")
    print(f"  Sortino Ratio:       {sortino:>8.2f}")
    print(f"--- Risk Metrics ---")
    print(f"  Annualized Vol:      {ann_vol:>8.1%}")
    print(f"  Max Drawdown:        {max_dd:>8.1%}")
    print(f"--- Correlation with SPY ---")
    print(f"  Mean 63d corr:       {corr_63d.mean():>8.3f}")
    print(f"--- Stress Activation ---")
    print(f"  Days activated:      {stress_activated_pct:>8.1%}")
 
    print(f"\n--- Crisis Period Returns ---")
    print(f"  {'Period':<20s} {'Strategy':>10s} {'SPY':>10s} {'Excess':>10s}")
    print(f"  {'-'*50}")
    for name, start, end in crisis_periods:
        mask = (returns.index >= start) & (returns.index <= end)
        if mask.sum() > 0:
            crisis_ret = (1 + returns.loc[mask]).prod() - 1
            spy_crisis_ret = (1 + spy_returns.loc[mask]).prod() - 1
            print(f"  {name:<20s} {crisis_ret:>9.1%} {spy_crisis_ret:>9.1%} {crisis_ret - spy_crisis_ret:>9.1%}")
 
    # Year-by-year
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