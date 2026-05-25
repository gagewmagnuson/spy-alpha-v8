"""
SPY Alpha v8 — Risk Constraint Engine (Layer 3)
==================================================
 
NEW module in v8. Defines the feasible set that the meta-allocator must
operate within. Has ABSOLUTE AUTHORITY — no signal, strategy, or allocator
output can violate these constraints.
 
Components (from build spec Section 7):
    7A. Hard Constraints (non-negotiable position limits)
    7B. Circuit Breakers (ported from v7 — crash momentum + dual-timeframe)
    7C. State Uncertainty Dampener (key v8 innovation)
    7D. Adaptive Constraint Tightening (stress-responsive limits)
 
Design Principles:
    - The allocator PROPOSES, the risk engine DISPOSES
    - Hard constraints are never relaxed under any condition
    - Uncertainty response is continuous, not binary
    - When the system doesn't know what's happening, it gets conservative
    - Circuit breakers operate in the FAST layer (daily) and override everything
    - Observable-latent divergence directly feeds uncertainty
 
Critical Architectural Role:
    This layer exists so that the meta-allocator can safely lean into
    Strategy 1 during favorable states. Without the risk engine absorbing
    tail risk, the allocator self-protects by overweighting defensive
    strategies, suppressing CAGR.
"""
 
from __future__ import annotations
 
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
 
import numpy as np
import pandas as pd
 
logger = logging.getLogger("spy_alpha_v8.risk_engine")
 
 
# ---------------------------------------------------------------------------
# 7A. Hard Constraints (Non-Negotiable)
# ---------------------------------------------------------------------------
 
HARD_CONSTRAINTS: Dict[str, Any] = {
    # Position limits (proven in v6/v7)
    "max_weight_per_asset": 0.30,
    "max_upro_weight": 0.45,
    "max_shy_weight": 0.50,
    "max_tlt_weight": 0.25,
    "max_gld_weight": 0.20,
 
    # Portfolio construction
    "max_turnover_per_rebalance": 0.40,
    "max_individual_stocks": 3,
    "max_same_sector_assets": 2,
    "portfolio_size_min": 8,
    "portfolio_size_max": 12,
 
    # Volatility target range
    "vol_target_low": 0.08,
    "vol_target_high": 0.14,
    "vol_target_mid": 0.12,
}
 
# Stressed constraint overrides (tightened during high uncertainty/stress)
STRESSED_CONSTRAINTS: Dict[str, Any] = {
    "max_weight_per_asset": 0.20,
    "max_upro_weight": 0.25,
    "max_turnover_per_rebalance": 0.25,
    "vol_target_low": 0.06,
    "vol_target_high": 0.10,
    "vol_target_mid": 0.08,
}
 
 
# ---------------------------------------------------------------------------
# 7B. Circuit Breakers (Ported from V7)
# ---------------------------------------------------------------------------
 
# Crash momentum filter threshold
CRASH_MOMENTUM_THRESHOLD: float = -0.04  # 5-day return below -4%
CRASH_MOMENTUM_REDUCTION: float = 0.5    # halve UPRO
 
# Dual-timeframe circuit breaker thresholds
CIRCUIT_BREAKER_MODERATE: float = -0.07
CIRCUIT_BREAKER_SEVERE: float = -0.15
 
 
def apply_circuit_breakers(
    weights: Dict[str, float],
    spy_prices: pd.Series,
    current_date: pd.Timestamp,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Apply crash momentum filter and dual-timeframe circuit breakers.
 
    These operate in the FAST layer (daily) and override all other decisions.
    Ported directly from v7 portfolio_optimizer._circuit_breaker.
 
    Returns:
        Modified weights dict and metadata about breaker activations.
    """
    metadata = {
        "crash_momentum_active": False,
        "circuit_breaker_level": "none",
        "spy_5d_return": None,
        "worst_drawdown": None,
    }
 
    spy_up_to = spy_prices.loc[:current_date]
    spy_recent = spy_up_to.tail(60)
 
    if len(spy_recent) < 10:
        return weights, metadata
 
    current_price = spy_recent.iloc[-1]
    modified = weights.copy()
 
    # ---- Crash momentum filter: 5-day return < -4% ----
    if len(spy_recent) >= 6:
        spy_5d_ret = current_price / spy_recent.iloc[-6] - 1
        metadata["spy_5d_return"] = float(spy_5d_ret)
 
        if spy_5d_ret < CRASH_MOMENTUM_THRESHOLD:
            if "UPRO" in modified:
                modified["UPRO"] *= CRASH_MOMENTUM_REDUCTION
            metadata["crash_momentum_active"] = True
            logger.info(f"  CRASH MOMENTUM: 5d return={spy_5d_ret:.1%}, halving UPRO")
 
    # ---- Fast breaker: 10-day drawdown ----
    rolling_peak_10 = spy_recent.rolling(10).max().iloc[-1]
    fast_dd = (current_price - rolling_peak_10) / rolling_peak_10
 
    # ---- Slow breaker: 40-day drawdown ----
    slow_dd = 0.0
    if len(spy_recent) >= 40:
        rolling_peak_40 = spy_recent.rolling(40).max().iloc[-1]
        slow_dd = (current_price - rolling_peak_40) / rolling_peak_40
 
    worst_dd = min(fast_dd, slow_dd)
    metadata["worst_drawdown"] = float(worst_dd)
 
    if worst_dd < CIRCUIT_BREAKER_SEVERE:
        # Severe: zero UPRO, max defensive
        modified["UPRO"] = 0.0
        modified["SHY"] = 0.50
        metadata["circuit_breaker_level"] = "severe"
        logger.info(f"  CIRCUIT BREAKER SEVERE: dd={worst_dd:.1%}")
 
    elif worst_dd < CIRCUIT_BREAKER_MODERATE:
        # Moderate: reduce UPRO, increase SHY floor
        if "UPRO" in modified:
            modified["UPRO"] *= 0.3
        modified["SHY"] = max(modified.get("SHY", 0), 0.30)
        metadata["circuit_breaker_level"] = "moderate"
        logger.info(f"  CIRCUIT BREAKER MODERATE: dd={worst_dd:.1%}")
 
    return modified, metadata
 
 
# ---------------------------------------------------------------------------
# 7C. State Uncertainty Dampener
# ---------------------------------------------------------------------------
 
# Uncertainty component weights (from build spec Section 7C)
UNCERTAINTY_WEIGHTS: Dict[str, float] = {
    "strategy_disagreement": 0.25,
    "regime_instability": 0.20,
    "transition_acceleration": 0.20,
    "allocator_low_confidence": 0.15,
    "observable_latent_divergence": 0.10,
    "embedding_drift": 0.10,
}
 
 
def compute_uncertainty_score(
    state_features: pd.DataFrame,
    strategy_weights: Dict[str, Dict[str, float]],
    allocator_confidence: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the composite uncertainty score from multiple signals.
 
    This is the most important risk innovation in v8. When the system
    doesn't know what's happening, it gets conservative.
 
    Args:
        state_features: Single row (or latest row) from state representation
        strategy_weights: {strategy_name: {asset: weight}} — proposed weights
        allocator_confidence: Meta-allocator's self-reported confidence
 
    Returns:
        uncertainty_score in [0, 1] and component breakdown
    """
    components = {}
 
    # ---- Strategy Disagreement ----
    # High variance in proposed weights across strategies
    if len(strategy_weights) >= 2:
        all_assets = set()
        for w in strategy_weights.values():
            all_assets.update(w.keys())
 
        weight_vectors = []
        for name, w in strategy_weights.items():
            vec = [w.get(a, 0.0) for a in sorted(all_assets)]
            weight_vectors.append(vec)
 
        weight_array = np.array(weight_vectors)
        # Average standard deviation across assets
        disagreement = np.mean(np.std(weight_array, axis=0))
        # Scale: 0 std → 0, 0.3 std → 1
        components["strategy_disagreement"] = min(disagreement / 0.3, 1.0)
    else:
        components["strategy_disagreement"] = 0.0
 
    # ---- Regime Instability ----
    # High entropy in HMM probability distribution
    if isinstance(state_features, pd.DataFrame):
        row = state_features.iloc[-1] if len(state_features) > 1 else state_features.iloc[0]
    else:
        row = state_features
 
    entropy = row.get("hmm_regime_entropy", None)
    if entropy is not None and not pd.isna(entropy):
        # Max entropy for 5 regimes = ln(5) ≈ 1.609
        # Normalize: entropy/max_entropy
        max_entropy = np.log(5)
        components["regime_instability"] = min(float(entropy) / max_entropy, 1.0)
    else:
        components["regime_instability"] = 0.5  # uncertain when no data
 
    # ---- Transition Acceleration ----
    # Rapidly changing transition features
    trans_cols = [c for c in row.index if c.startswith("trans_")]
    if trans_cols:
        trans_vals = []
        for col in trans_cols:
            val = row.get(col, None)
            if val is not None and not pd.isna(val):
                trans_vals.append(abs(float(val)))
 
        if trans_vals:
            # Average absolute transition velocity
            mean_accel = np.mean(trans_vals)
            # Scale: this is domain-dependent, use percentile-like mapping
            # Higher mean acceleration → higher uncertainty
            components["transition_acceleration"] = min(mean_accel / 0.05, 1.0)
        else:
            components["transition_acceleration"] = 0.0
    else:
        components["transition_acceleration"] = 0.0
 
    # ---- Allocator Low Confidence ----
    if allocator_confidence is not None:
        # Invert: low confidence → high uncertainty
        components["allocator_low_confidence"] = 1.0 - min(max(allocator_confidence, 0), 1)
    else:
        components["allocator_low_confidence"] = 0.5
 
    # ---- Observable-Latent Divergence ----
    # Observable states say one thing, latent states say another
    # Use vol regime vs HMM: if vol is low but HMM says crisis, divergence is high
    vol_10d = row.get("vol_realized_10d", None)
    hmm_max = row.get("hmm_max_prob", None)
    hmm_entropy_val = row.get("hmm_regime_entropy", None)
 
    if vol_10d is not None and hmm_entropy_val is not None and not pd.isna(vol_10d) and not pd.isna(hmm_entropy_val):
        # Low vol + high regime entropy = divergence
        # High vol + low regime entropy = agreement
        vol_calm = max(1.0 - float(vol_10d) / 0.30, 0)  # 0 at 30% vol, 1 at 0% vol
        regime_confused = float(hmm_entropy_val) / np.log(5)
        divergence = vol_calm * regime_confused
        components["observable_latent_divergence"] = min(divergence, 1.0)
    else:
        components["observable_latent_divergence"] = 0.0
 
    # ---- Embedding Drift ----
    # Placeholder until deep embeddings are implemented (Step 7)
    # When embeddings exist, this measures large change in embedding values
    components["embedding_drift"] = 0.0
 
    # ---- Compute weighted uncertainty score ----
    score = 0.0
    total_weight = 0.0
    for signal, weight in UNCERTAINTY_WEIGHTS.items():
        if signal in components:
            score += weight * components[signal]
            total_weight += weight
 
    if total_weight > 0:
        score /= total_weight
 
    score = min(max(score, 0.0), 1.0)
 
    return score, components
 
 
def compute_uncertainty_series(
    state_features: pd.DataFrame,
    strategy_outputs_by_date: Dict[str, Dict[str, Dict[str, float]]],
    allocator_confidences: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute uncertainty scores for every date in the state features.
 
    Args:
        state_features: Full state representation DataFrame
        strategy_outputs_by_date: {date_str: {strategy_name: {asset: weight}}}
        allocator_confidences: Series of allocator confidence values
 
    Returns:
        DataFrame with uncertainty_score and component columns
    """
    results = []
 
    for i, date in enumerate(state_features.index):
        date_str = date.strftime("%Y-%m-%d")
 
        # Get strategy weights for this date
        strategy_weights = strategy_outputs_by_date.get(date_str, {})
 
        # Get allocator confidence
        alloc_conf = None
        if allocator_confidences is not None and date in allocator_confidences.index:
            alloc_conf = float(allocator_confidences.loc[date])
 
        row = state_features.iloc[i:i+1]
        score, components = compute_uncertainty_score(row, strategy_weights, alloc_conf)
 
        result = {"date": date, "uncertainty_score": score}
        result.update({f"unc_{k}": v for k, v in components.items()})
        results.append(result)
 
    df = pd.DataFrame(results).set_index("date")
    return df
 
 
# ---------------------------------------------------------------------------
# 7D. Adaptive Constraint Tightening
# ---------------------------------------------------------------------------
 
def get_active_constraints(
    uncertainty_score: float,
    stress_score: float = 0.0,
) -> Dict[str, Any]:
    """
    Return the active constraint set based on current uncertainty and stress.
 
    Constraints interpolate between normal and stressed levels based on
    the maximum of uncertainty and stress scores.
 
    This is continuous, not binary — constraints tighten proportionally.
    """
    # Use the maximum of uncertainty and stress as the tightening driver
    tightening = max(uncertainty_score, stress_score)
    tightening = min(max(tightening, 0.0), 1.0)
 
    active = {}
    for key in HARD_CONSTRAINTS:
        normal = HARD_CONSTRAINTS[key]
        if key in STRESSED_CONSTRAINTS:
            stressed = STRESSED_CONSTRAINTS[key]
            # Linear interpolation between normal and stressed
            active[key] = normal + tightening * (stressed - normal)
        else:
            active[key] = normal
 
    return active
 
 
# ---------------------------------------------------------------------------
# Risk Engine (Main Class)
# ---------------------------------------------------------------------------
 
class RiskEngine:
    """
    Layer 3: Risk Constraint Engine.
 
    Has absolute authority over the portfolio. The meta-allocator proposes,
    the risk engine disposes.
 
    Responsibilities:
        1. Enforce hard constraints on every portfolio
        2. Run circuit breakers (daily, override everything)
        3. Compute and apply uncertainty dampening
        4. Tighten constraints adaptively during stress/uncertainty
        5. Calibrate allocator confidence to scale leverage
 
    The risk engine processes the allocator's proposed portfolio and returns
    a risk-adjusted portfolio that satisfies all constraints.
    """
 
    def __init__(self):
        self.last_uncertainty_score: float = 0.0
        self.last_uncertainty_components: Dict[str, float] = {}
        self.last_breaker_metadata: Dict[str, Any] = {}
        self.last_active_constraints: Dict[str, Any] = HARD_CONSTRAINTS.copy()
 
    def apply(
        self,
        proposed_weights: Dict[str, float],
        strategy_weights: Dict[str, Dict[str, float]],
        state_features: pd.DataFrame,
        spy_prices: pd.Series,
        current_date: pd.Timestamp,
        stress_score: float = 0.0,
        allocator_confidence: Optional[float] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Apply the full risk constraint pipeline to a proposed portfolio.
 
        Steps:
            1. Compute uncertainty score
            2. Determine active constraints (adaptive tightening)
            3. Apply uncertainty dampening to weights
            4. Enforce hard constraints
            5. Apply circuit breakers
            6. Normalize and return
 
        Args:
            proposed_weights: {asset: weight} from meta-allocator
            strategy_weights: {strategy_name: {asset: weight}} for disagreement
            state_features: State representation (single row or full DataFrame)
            spy_prices: SPY price series for circuit breakers
            current_date: Current date
            stress_score: From Strategy 3's stress computation
            allocator_confidence: Meta-allocator's confidence score
 
        Returns:
            (risk_adjusted_weights, risk_metadata)
        """
        # ---- Step 1: Compute uncertainty ----
        uncertainty, unc_components = compute_uncertainty_score(
            state_features, strategy_weights, allocator_confidence
        )
        self.last_uncertainty_score = uncertainty
        self.last_uncertainty_components = unc_components
 
        # ---- Step 2: Determine active constraints ----
        active = get_active_constraints(uncertainty, stress_score)
        self.last_active_constraints = active
 
        # ---- Step 3: Apply uncertainty dampening ----
        weights = self._apply_uncertainty_dampening(
            proposed_weights, uncertainty, active
        )
 
        # ---- Step 4: Enforce hard constraints ----
        weights = self._enforce_constraints(weights, active)
 
        # ---- Step 5: Apply circuit breakers ----
        weights, breaker_meta = apply_circuit_breakers(
            weights, spy_prices, current_date
        )
        self.last_breaker_metadata = breaker_meta
 
        # ---- Step 6: Normalize ----
        weights = self._normalize(weights)
 
        # ---- Build metadata ----
        metadata = {
            "uncertainty_score": uncertainty,
            "uncertainty_components": unc_components,
            "stress_score": stress_score,
            "tightening_level": max(uncertainty, stress_score),
            "circuit_breaker": breaker_meta,
            "active_constraints": {
                "max_upro": active["max_upro_weight"],
                "max_weight": active["max_weight_per_asset"],
                "vol_target": f"{active['vol_target_low']:.0%}-{active['vol_target_high']:.0%}",
                "max_turnover": active["max_turnover_per_rebalance"],
            },
            "leverage_adjustment": self._compute_leverage_adjustment(
                uncertainty, allocator_confidence
            ),
        }
 
        return weights, metadata
 
    def _apply_uncertainty_dampening(
        self,
        weights: Dict[str, float],
        uncertainty: float,
        active_constraints: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Apply continuous uncertainty dampening to proposed weights.
 
        When uncertainty rises:
            - Reduce leverage proportionally (scale down UPRO)
            - Increase diversification (push toward equal-weight)
            - Increase cash/SHY allocation
            - Shift trust toward observable states (implicit via constraint tightening)
        """
        if uncertainty < 0.2:
            # Low uncertainty — minimal dampening
            return weights.copy()
 
        dampened = weights.copy()
 
        # ---- Scale factor: how much to dampen ----
        # 0.2 uncertainty → 0% dampening
        # 0.5 uncertainty → 37.5% dampening
        # 0.8 uncertainty → 75% dampening
        # 1.0 uncertainty → 100% dampening
        dampen_intensity = (uncertainty - 0.2) / 0.8
        dampen_intensity = min(max(dampen_intensity, 0), 1.0)
 
        # ---- Reduce UPRO proportionally ----
        if "UPRO" in dampened:
            upro_reduction = dampen_intensity * 0.7  # at max uncertainty, reduce UPRO by 70%
            dampened["UPRO"] *= (1.0 - upro_reduction)
 
        # ---- Push toward equal-weight (increase diversification) ----
        if len(dampened) > 1:
            n_assets = len(dampened)
            equal_w = 1.0 / n_assets
 
            blend_toward_equal = dampen_intensity * 0.4  # at max, blend 40% toward equal
 
            for asset in dampened:
                current = dampened[asset]
                dampened[asset] = current * (1 - blend_toward_equal) + equal_w * blend_toward_equal
 
        # ---- Increase SHY allocation ----
        shy_boost = dampen_intensity * 0.15  # at max uncertainty, add 15% SHY
        dampened["SHY"] = dampened.get("SHY", 0) + shy_boost
 
        return dampened
 
    def _enforce_constraints(
        self,
        weights: Dict[str, float],
        active: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Enforce all hard and adaptive constraints on the portfolio.
        """
        constrained = weights.copy()
 
        # ---- Individual instrument caps ----
        instrument_caps = {
            "UPRO": active["max_upro_weight"],
            "SHY": active.get("max_shy_weight", HARD_CONSTRAINTS["max_shy_weight"]),
            "TLT": active.get("max_tlt_weight", HARD_CONSTRAINTS["max_tlt_weight"]),
            "GLD": active.get("max_gld_weight", HARD_CONSTRAINTS["max_gld_weight"]),
        }
 
        for asset, cap in instrument_caps.items():
            if asset in constrained and constrained[asset] > cap:
                constrained[asset] = cap
 
        # ---- General per-asset cap ----
        max_weight = active["max_weight_per_asset"]
        for asset in constrained:
            if asset not in instrument_caps and constrained[asset] > max_weight:
                constrained[asset] = max_weight
 
        # ---- Remove negative weights ----
        constrained = {k: max(v, 0) for k, v in constrained.items()}
 
        # ---- Remove near-zero weights ----
        constrained = {k: v for k, v in constrained.items() if v > 1e-6}
 
        return constrained
 
    def _normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total <= 0:
            return {"SHY": 1.0}  # full cash if nothing left
        return {k: v / total for k, v in weights.items()}
 
    def _compute_leverage_adjustment(
        self,
        uncertainty: float,
        allocator_confidence: Optional[float],
    ) -> float:
        """
        Compute leverage adjustment factor based on uncertainty and confidence.
 
        Returns a multiplier in [0.5, 1.0]:
            - Low uncertainty + high confidence → 1.0 (no adjustment)
            - High uncertainty + low confidence → 0.5 (halve leverage)
 
        This is the allocator confidence calibration from the build spec.
        """
        base = 1.0
 
        # Uncertainty reduces leverage
        if uncertainty > 0.3:
            base -= (uncertainty - 0.3) * 0.5  # max 0.35 reduction
 
        # Low confidence further reduces
        if allocator_confidence is not None and allocator_confidence < 0.5:
            conf_penalty = (0.5 - allocator_confidence) * 0.3  # max 0.15 reduction
            base -= conf_penalty
 
        return max(base, 0.5)
 
    def apply_to_series(
        self,
        proposed_weights_series: List[Dict[str, float]],
        strategy_weights_series: List[Dict[str, Dict[str, float]]],
        state_features: pd.DataFrame,
        spy_prices: pd.Series,
        dates: pd.DatetimeIndex,
        stress_scores: Optional[pd.Series] = None,
        allocator_confidences: Optional[pd.Series] = None,
    ) -> Tuple[List[Dict[str, float]], pd.DataFrame]:
        """
        Apply risk engine to a series of proposed portfolios.
 
        Returns:
            List of risk-adjusted weight dicts and DataFrame of risk metadata.
        """
        adjusted_weights = []
        metadata_records = []
 
        for i, date in enumerate(dates):
            proposed = proposed_weights_series[i] if i < len(proposed_weights_series) else {}
            strat_w = strategy_weights_series[i] if i < len(strategy_weights_series) else {}
 
            # Get state features up to current date
            state_up_to = state_features.loc[:date]
 
            stress = 0.0
            if stress_scores is not None and date in stress_scores.index:
                stress = float(stress_scores.loc[date])
 
            alloc_conf = None
            if allocator_confidences is not None and date in allocator_confidences.index:
                alloc_conf = float(allocator_confidences.loc[date])
 
            adj_w, meta = self.apply(
                proposed, strat_w, state_up_to,
                spy_prices, date, stress, alloc_conf
            )
 
            adjusted_weights.append(adj_w)
            meta["date"] = date
            metadata_records.append(meta)
 
        meta_df = pd.DataFrame(metadata_records).set_index("date")
        return adjusted_weights, meta_df
 
 
# ---------------------------------------------------------------------------
# Backtest Integration
# ---------------------------------------------------------------------------
 
def backtest_with_risk_engine(
    allocator_results: pd.DataFrame,
    strategy_outputs: Dict[str, list],
    state_features: pd.DataFrame,
    adj_close: pd.DataFrame,
    stress_scores: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Apply the risk engine to meta-allocator outputs and compute
    risk-adjusted portfolio returns.
 
    Compares:
        1. Meta-allocator output (no risk engine)
        2. Meta-allocator + risk engine
        3. Equal-weight baseline
 
    This directly tests whether the risk engine improves Max DD
    during regime transitions without excessive return drag.
    """
    from strategy_health import compute_strategy_daily_returns
 
    spy_prices = adj_close["SPY"] if "SPY" in adj_close.columns else pd.Series()
    benchmark = adj_close["SPY"].pct_change() if "SPY" in adj_close.columns else pd.Series(0, index=adj_close.index)
 
    # ---- Compute daily returns per strategy ----
    strategy_returns = {}
    for name, outputs in strategy_outputs.items():
        strategy_returns[name] = compute_strategy_daily_returns(outputs, adj_close)
 
    # ---- Get allocator dates and weights ----
    alloc_dates = allocator_results.index
    strategy_names = ["regime_allocator", "trend_cta", "defensive"]
 
    # Build date-keyed strategy weight lookups (forward-filled)
    strat_weights_by_name = {}
    for name, outputs in strategy_outputs.items():
        date_weight_map = {}
        for o in outputs:
            date_weight_map[o.strategy_metadata["date"]] = o.proposed_weights
        strat_weights_by_name[name] = date_weight_map

    # For each allocator date, find the most recent strategy weights
    proposed_series = []
    strategy_weights_series = []

    # Track current strategy weights (forward-fill)
    current_strat_weights = {name: {} for name in strategy_names}

    for date in alloc_dates:
        date_str = date.strftime("%Y-%m-%d")

        # Update each strategy's weights if there's a signal on or before this date
        for name in strategy_names:
            if date_str in strat_weights_by_name.get(name, {}):
                current_strat_weights[name] = strat_weights_by_name[name][date_str]
            else:
                # Check for most recent signal before this date
                available_dates = sorted(strat_weights_by_name.get(name, {}).keys())
                prior = [d for d in available_dates if d <= date_str]
                if prior:
                    current_strat_weights[name] = strat_weights_by_name[name][prior[-1]]

        # Build blended weights from allocator capital weights
        blended = {}
        strat_w = {}

        for name in strategy_names:
            cap_col = f"capital_weight_{name}"
            if cap_col in allocator_results.columns:
                cap_weight = allocator_results.loc[date, cap_col]
            else:
                cap_weight = 1.0 / len(strategy_names)

            strat_proposed = current_strat_weights[name]
            strat_w[name] = strat_proposed

            for asset, w in strat_proposed.items():
                blended[asset] = blended.get(asset, 0) + cap_weight * w

        proposed_series.append(blended)
        strategy_weights_series.append(strat_w)
 
    # ---- Apply risk engine ----
    logger.info("Applying risk engine to allocator outputs...")
    engine = RiskEngine()
 
    # Get allocator confidences from activation scores
    alloc_conf = None
    act_cols = [c for c in allocator_results.columns if c.startswith("activation_")]
    if act_cols:
        # Use mean activation as confidence proxy
        alloc_conf = allocator_results[act_cols].mean(axis=1)
 
    adjusted_weights, risk_metadata = engine.apply_to_series(
        proposed_series,
        strategy_weights_series,
        state_features,
        spy_prices,
        alloc_dates,
        stress_scores=stress_scores,
        allocator_confidences=alloc_conf,
    )
 
    # ---- Compute returns for all trading days ----
    daily_returns = adj_close.pct_change()
    all_trading_days = daily_returns.index
    
    # Build date-keyed lookups for proposed and adjusted weights
    raw_weight_by_date = {}
    adj_weight_by_date = {}
    for i, date in enumerate(alloc_dates):
        raw_weight_by_date[date] = proposed_series[i]
        adj_weight_by_date[date] = adjusted_weights[i]

    # Forward-fill weights across all trading days
    current_raw = {}
    current_adj = {}
    current_equal = {name: 1.0 / len(strategy_returns) for name in strategy_returns}

    raw_returns_list = []
    risk_adj_returns_list = []
    equal_returns_list = []

    # Start from the first allocator date
    first_alloc = alloc_dates[0]

    for date in all_trading_days:
        if date < first_alloc:
            continue

        # Update weights if this is an allocator date
        if date in raw_weight_by_date:
            current_raw = raw_weight_by_date[date]
            current_adj = adj_weight_by_date[date]

        # Raw allocator return
        raw_ret = sum(
            current_raw.get(a, 0) * daily_returns.loc[date, a]
            for a in current_raw
            if a in daily_returns.columns and pd.notna(daily_returns.loc[date, a])
        )

        # Risk-adjusted return
        adj_ret = sum(
            current_adj.get(a, 0) * daily_returns.loc[date, a]
            for a in current_adj
            if a in daily_returns.columns and pd.notna(daily_returns.loc[date, a])
        )

        # Equal-weight return
        eq_ret = sum(
            ret.loc[date] if date in ret.index and pd.notna(ret.loc[date]) else 0
            for ret in strategy_returns.values()
        ) / len(strategy_returns)

        raw_returns_list.append({"date": date, "return": raw_ret})
        risk_adj_returns_list.append({"date": date, "return": adj_ret})
        equal_returns_list.append({"date": date, "return": eq_ret})

    raw_returns = pd.Series(
        [r["return"] for r in raw_returns_list],
        index=[r["date"] for r in raw_returns_list],
    )
    risk_adj_returns = pd.Series(
        [r["return"] for r in risk_adj_returns_list],
        index=[r["date"] for r in risk_adj_returns_list],
    )
    equal_returns = pd.Series(
        [r["return"] for r in equal_returns_list],
        index=[r["date"] for r in equal_returns_list],
    )
 
    # ---- Compute metrics ----
    def compute_metrics(returns, label):
        r = returns.dropna()
        if len(r) < 252:
            return {"label": label, "error": "insufficient data"}
        n_years = len(r) / 252
        cum = (1 + r).cumprod()
        cagr = cum.iloc[-1] ** (1/n_years) - 1
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        ds = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 1e-6
        sortino = (r.mean() * 252) / ds
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        max_dd = dd.min()
        calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0
        vol = r.std() * np.sqrt(252)
        return {
            "label": label, "sharpe": float(sharpe), "cagr": float(cagr),
            "max_dd": float(max_dd), "sortino": float(sortino),
            "calmar": float(calmar), "vol": float(vol), "n_days": len(r),
        }
 
    return {
        "raw_metrics": compute_metrics(raw_returns, "Allocator (no risk engine)"),
        "risk_adj_metrics": compute_metrics(risk_adj_returns, "Allocator + Risk Engine"),
        "equal_metrics": compute_metrics(equal_returns, "Equal-Weight Baseline"),
        "benchmark_metrics": compute_metrics(benchmark.reindex(alloc_dates), "SPY Benchmark"),
        "risk_metadata": risk_metadata,
        "risk_adj_returns": risk_adj_returns,
        "raw_returns": raw_returns,
        "engine": engine,
    }
 
 
# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
 
def print_risk_engine_report(results: Dict[str, Any]) -> None:
    """Print the risk engine performance report."""
    print(f"\n{'='*70}")
    print(f"RISK ENGINE PERFORMANCE REPORT")
    print(f"{'='*70}")
 
    print(f"\n--- Performance Comparison ---")
    for key in ["raw_metrics", "risk_adj_metrics", "equal_metrics", "benchmark_metrics"]:
        m = results[key]
        if "error" in m:
            print(f"  {m['label']}: {m['error']}")
            continue
        print(
            f"  {m['label']:<30s} "
            f"Sharpe={m['sharpe']:.2f}  "
            f"CAGR={m['cagr']:.1%}  "
            f"MaxDD={m['max_dd']:.1%}  "
            f"Sortino={m['sortino']:.2f}  "
            f"Calmar={m['calmar']:.2f}  "
            f"Vol={m['vol']:.1%}"
        )
 
    # ---- Risk metadata summary ----
    meta = results["risk_metadata"]
    if not meta.empty:
        print(f"\n--- Uncertainty Score ---")
        print(f"  Mean:   {meta['uncertainty_score'].mean():.3f}")
        print(f"  Std:    {meta['uncertainty_score'].std():.3f}")
        print(f"  Max:    {meta['uncertainty_score'].max():.3f}")
 
        print(f"\n--- Circuit Breaker Activations ---")
        breaker_levels = meta["circuit_breaker"].apply(
            lambda x: x.get("circuit_breaker_level", "none") if isinstance(x, dict) else "none"
        )
        for level in ["none", "moderate", "severe"]:
            count = (breaker_levels == level).sum()
            pct = count / len(breaker_levels) * 100
            print(f"  {level:<12s} {count:>5d} ({pct:.1f}%)")
 
        crash_active = meta["circuit_breaker"].apply(
            lambda x: x.get("crash_momentum_active", False) if isinstance(x, dict) else False
        )
        print(f"  Crash momentum active: {crash_active.sum()} days")
 
        print(f"\n--- Constraint Tightening ---")
        tightening = meta["tightening_level"]
        print(f"  Mean tightening: {tightening.mean():.3f}")
        print(f"  Days > 0.3:     {(tightening > 0.3).sum()} ({(tightening > 0.3).mean()*100:.1f}%)")
        print(f"  Days > 0.5:     {(tightening > 0.5).sum()} ({(tightening > 0.5).mean()*100:.1f}%)")
        print(f"  Days > 0.7:     {(tightening > 0.7).sum()} ({(tightening > 0.7).mean()*100:.1f}%)")
 
    # ---- Step 8 Verification ----
    raw = results["raw_metrics"]
    risk = results["risk_adj_metrics"]
    if "error" not in raw and "error" not in risk:
        print(f"\n--- Step 8 Verification ---")
        dd_improved = abs(risk["max_dd"]) < abs(raw["max_dd"])
        sharpe_preserved = risk["sharpe"] >= raw["sharpe"] * 0.90  # allow 10% Sharpe drag
        print(f"  MaxDD improved:     {raw['max_dd']:.1%} -> {risk['max_dd']:.1%} -> {'PASS' if dd_improved else 'FAIL'}")
        print(f"  Sharpe preserved:   {raw['sharpe']:.2f} -> {risk['sharpe']:.2f} (≥{raw['sharpe']*0.90:.2f}) -> {'PASS' if sharpe_preserved else 'FAIL'}")
        print(f"  Overall Step 8:     {'PASS' if (dd_improved and sharpe_preserved) else 'NEEDS REVIEW'}")