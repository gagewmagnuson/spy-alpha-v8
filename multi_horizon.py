"""
SPY Alpha v8 — Multi-Horizon Framework (Step 9)
==================================================
 
NEW module in v8. Separates allocation decisions into three temporal layers,
each with distinct responsibilities and update frequencies.
 
From build spec Section 6D:
 
    Slow Layer (Monthly — every 21 trading days):
        - Sets strategic risk posture
        - Maximum leverage ceiling
        - Overall equity/bond/commodity budget
        - Inputs: macro latent factors, long-term trend state
        - Changes infrequently — provides stability anchor
 
    Medium Layer (Weekly — every 5 trading days):
        - Strategy activation and capital sizing
        - Asset-level weight adjustments within slow layer bounds
        - Where the LightGBM meta-allocator operates
        - Inputs: full state representation + strategy outputs + health metrics
 
    Fast Layer (Daily):
        - Crisis response ONLY
        - Can REDUCE exposure but NEVER increase it
        - Circuit breakers and crash momentum filters (ported from v7)
        - Override authority over the medium layer
        - Inputs: real-time price data, volatility, drawdown
 
Critical Design Principle:
    The fast layer's ability to independently reduce exposure is what
    allows the medium layer to be more aggressive. The allocator can
    safely lean into Strategy 1 during favorable states because the
    fast layer will catch any sudden deterioration.
 
    This directly addresses the CAGR suppression problem: by separating
    crash defense from allocation, each layer can do its job better.
"""
 
from __future__ import annotations
 
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from risk_engine import RiskEngine, get_risk_profile

import numpy as np
import pandas as pd
 
logger = logging.getLogger("spy_alpha_v8.multi_horizon")
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
# Layer update frequencies (trading days)
SLOW_LAYER_FREQUENCY: int = 21    # monthly
MEDIUM_LAYER_FREQUENCY: int = 5   # weekly
FAST_LAYER_FREQUENCY: int = 1     # daily
 
# Slow layer budget bounds
SLOW_EQUITY_BUDGET_MIN: float = 0.20
SLOW_EQUITY_BUDGET_MAX: float = 0.85
SLOW_BOND_BUDGET_MIN: float = 0.05
SLOW_BOND_BUDGET_MAX: float = 0.50
SLOW_COMMODITY_BUDGET_MIN: float = 0.00
SLOW_COMMODITY_BUDGET_MAX: float = 0.25
 
# Slow layer leverage ceiling range
SLOW_LEVERAGE_CEILING_MIN: float = 0.60   # most conservative
SLOW_LEVERAGE_CEILING_MAX: float = 1.30   # most aggressive
 
# Fast layer — can only reduce, never increase
FAST_LAYER_MAX_REDUCTION: float = 0.50  # can reduce total exposure by up to 50%
 
 
# ---------------------------------------------------------------------------
# Asset Classification
# ---------------------------------------------------------------------------
 
EQUITY_ASSETS = {
    "SPY", "QQQ", "IWM", "VWO", "UPRO",
    "XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU", "XLC",
    "SMH", "XBI", "XME",
    "AAPL", "NVDA", "MSFT", "AMZN", "TSLA", "META", "GOOGL", "JPM", "LLY", "UNH", "XOM", "CAT",
}
 
BOND_ASSETS = {"TLT", "IEF", "SHY", "HYG"}
 
COMMODITY_ASSETS = {"GLD", "DBC"}
 
 
def classify_weights(
    weights: Dict[str, float],
) -> Dict[str, float]:
    """Classify portfolio weights into equity/bond/commodity buckets."""
    equity = sum(w for a, w in weights.items() if a in EQUITY_ASSETS)
    bond = sum(w for a, w in weights.items() if a in BOND_ASSETS)
    commodity = sum(w for a, w in weights.items() if a in COMMODITY_ASSETS)
    other = sum(w for a, w in weights.items()
                if a not in EQUITY_ASSETS and a not in BOND_ASSETS and a not in COMMODITY_ASSETS)
 
    return {
        "equity": equity,
        "bond": bond,
        "commodity": commodity,
        "other": other,
    }
 
 
# ---------------------------------------------------------------------------
# Slow Layer — Strategic Risk Posture
# ---------------------------------------------------------------------------
 
@dataclass
class SlowLayerState:
    """Output of the slow layer — strategic bounds for the medium layer."""
    date: pd.Timestamp
    leverage_ceiling: float
    equity_budget: Tuple[float, float]    # (min, max)
    bond_budget: Tuple[float, float]
    commodity_budget: Tuple[float, float]
    risk_posture: str                      # "aggressive", "balanced", "defensive"
    metadata: Dict[str, Any] = field(default_factory=dict)
 
 
def compute_slow_layer(
    state_features: pd.DataFrame,
    current_date: pd.Timestamp,
) -> SlowLayerState:
    """
    Compute the slow layer strategic risk posture.
 
    Uses long-lookback indicators to set strategic bounds:
        - Macro trend (yield curve, financial conditions)
        - Long-term trend strength (252-day)
        - Volatility regime (60-day)
        - Regime stability (entropy trend)
 
    Updates monthly (every 21 trading days). Changes infrequently
    to provide a stability anchor.
    """
    # Get the most recent state
    state_up_to = state_features.loc[:current_date]
    if state_up_to.empty:
        return _default_slow_state(current_date)
 
    latest = state_up_to.iloc[-1]
 
    # ---- Assess macro environment ----
    # Gather available indicators
    scores = []
 
    # Long-term trend strength
    trend_252 = latest.get("trend_spy_dist_ma_252d", None)
    if trend_252 is not None and not pd.isna(trend_252):
        # Positive distance from 252d MA = bullish
        # Map: -10% → 0, 0% → 0.5, +10% → 1.0
        trend_score = (float(trend_252) + 0.10) / 0.20
        scores.append(("trend", min(max(trend_score, 0), 1)))
 
    # Trend breadth
    breadth = latest.get("trend_breadth_avg", None)
    if breadth is not None and not pd.isna(breadth):
        scores.append(("breadth", float(breadth)))
 
    # Volatility regime (inverted: low vol = favorable)
    vol_60d = latest.get("vol_realized_60d", None)
    if vol_60d is not None and not pd.isna(vol_60d):
        # Map: 30% vol → 0, 10% vol → 1.0
        vol_score = (0.30 - float(vol_60d)) / 0.20
        scores.append(("vol", min(max(vol_score, 0), 1)))
 
    # Regime stability (low entropy = stable)
    entropy = latest.get("hmm_regime_entropy", None)
    if entropy is not None and not pd.isna(entropy):
        max_entropy = np.log(5)
        stability = 1.0 - float(entropy) / max_entropy
        scores.append(("stability", min(max(stability, 0), 1)))
 
    # Macro conditions (yield curve)
    t10y2y = latest.get("macro_t10y2y_level", None)
    if t10y2y is not None and not pd.isna(t10y2y):
        # Positive spread = healthy, inverted = concerning
        # Map: -1% → 0, 0% → 0.4, +2% → 1.0
        macro_score = (float(t10y2y) + 1.0) / 3.0
        scores.append(("macro", min(max(macro_score, 0), 1)))
 
    # ---- Compute composite strategic score ----
    if scores:
        composite = np.mean([s[1] for s in scores])
    else:
        composite = 0.5  # neutral when no data
 
    # ---- Map composite to risk posture ----
    if composite > 0.65:
        risk_posture = "aggressive"
    elif composite > 0.40:
        risk_posture = "balanced"
    else:
        risk_posture = "defensive"
 
    # ---- Set strategic bounds based on posture ----
    # Leverage ceiling: scales with composite
    leverage_ceiling = (
        SLOW_LEVERAGE_CEILING_MIN
        + composite * (SLOW_LEVERAGE_CEILING_MAX - SLOW_LEVERAGE_CEILING_MIN)
    )
 
    # Equity budget: wider range when bullish
    eq_min = SLOW_EQUITY_BUDGET_MIN + composite * 0.15  # 0.20 → 0.35
    eq_max = SLOW_EQUITY_BUDGET_MAX
 
    # Bond budget: wider range when defensive
    bond_min = SLOW_BOND_BUDGET_MIN
    bond_max = SLOW_BOND_BUDGET_MAX - composite * 0.20  # 0.50 → 0.30
 
    # Commodity budget
    comm_min = SLOW_COMMODITY_BUDGET_MIN
    comm_max = SLOW_COMMODITY_BUDGET_MAX
 
    return SlowLayerState(
        date=current_date,
        leverage_ceiling=leverage_ceiling,
        equity_budget=(eq_min, eq_max),
        bond_budget=(bond_min, bond_max),
        commodity_budget=(comm_min, comm_max),
        risk_posture=risk_posture,
        metadata={
            "composite_score": float(composite),
            "component_scores": {name: float(val) for name, val in scores},
        },
    )
 
 
def _default_slow_state(date: pd.Timestamp) -> SlowLayerState:
    """Return default balanced slow layer state."""
    return SlowLayerState(
        date=date,
        leverage_ceiling=1.0,
        equity_budget=(SLOW_EQUITY_BUDGET_MIN, SLOW_EQUITY_BUDGET_MAX),
        bond_budget=(SLOW_BOND_BUDGET_MIN, SLOW_BOND_BUDGET_MAX),
        commodity_budget=(SLOW_COMMODITY_BUDGET_MIN, SLOW_COMMODITY_BUDGET_MAX),
        risk_posture="balanced",
        metadata={"composite_score": 0.5},
    )

# ---------------------------------------------------------------------------
# Conditional Weighting Layer (Priority 1 — CAGR Recovery)
# ---------------------------------------------------------------------------

def apply_conditional_weighting(
    capital_weights: Dict[str, float],
    posture_strength: float,
    uncertainty: float,
    stress: float,
    max_boost: float = 0.45,
    s2_share: float = 0.5,
    profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Adjust meta-allocator capital weights based on continuous market conditions.

    When conditions are jointly favorable (aggressive posture, low uncertainty,
    low stress), increase Strategy 1's capital allocation at the expense of
    Strategies 2 and 3. When any condition deteriorates, the boost decays
    smoothly toward zero.

    This layer sits AFTER the allocator output and BEFORE strategy blending.
    It modifies capital weights, NOT direct asset allocations.

    The multiplicative favorable_score ensures the boost only fires when
    ALL three inputs are favorable — any single bad signal suppresses it.

    Args:
        capital_weights: {strategy_name: weight} from allocator, sums to 1.0
        posture_strength: slow layer composite_score in [0, 1]
        uncertainty: uncertainty score in [0, 1]
        stress: stress score in [0, 1]
        max_boost: maximum S1 weight increase (default 0.25)
        s2_share: fraction of boost taken from S2 vs S3 (default 0.5)
        profile: risk profile dict (can override max_boost)

    Returns:
        (adjusted_weights, diagnostics)
    """
    if profile and "conditional_max_boost" in profile:
        max_boost = profile["conditional_max_boost"]

    # ---- Compute favorable score (multiplicative) ----
    # All three must be favorable for full boost
    favorable_score = posture_strength * (1.0 - uncertainty) * (1.0 - stress)
    favorable_score = max(min(favorable_score, 1.0), 0.0)

    # ---- Compute S1 boost ----
    s1_boost = max_boost * favorable_score

    # ---- Get base weights ----
    s1_key = "regime_allocator"
    s2_key = "trend_cta"
    s3_key = "defensive"

    base_s1 = capital_weights.get(s1_key, 1.0 / 3)
    base_s2 = capital_weights.get(s2_key, 1.0 / 3)
    base_s3 = capital_weights.get(s3_key, 1.0 / 3)

    # ---- Apply boost: add to S1, subtract from S2/S3 proportionally ----
    adj_s1 = base_s1 + s1_boost

    # Split the reduction between S2 and S3
    s2_reduction = s1_boost * s2_share
    s3_reduction = s1_boost * (1.0 - s2_share)

    adj_s2 = max(base_s2 - s2_reduction, 0.05)  # floor at 5%
    adj_s3 = max(base_s3 - s3_reduction, 0.05)  # floor at 5%

    # ---- Normalize to sum to 1.0 ----
    total = adj_s1 + adj_s2 + adj_s3
    adjusted = {
        s1_key: adj_s1 / total,
        s2_key: adj_s2 / total,
        s3_key: adj_s3 / total,
    }

    # ---- Diagnostics ----
    diagnostics = {
        "favorable_score": float(favorable_score),
        "s1_boost": float(s1_boost),
        "posture_strength": float(posture_strength),
        "uncertainty": float(uncertainty),
        "stress": float(stress),
        "base_s1": float(base_s1),
        "adjusted_s1": float(adjusted[s1_key]),
        "base_s2": float(base_s2),
        "adjusted_s2": float(adjusted[s2_key]),
        "base_s3": float(base_s3),
        "adjusted_s3": float(adjusted[s3_key]),
    }

    return adjusted, diagnostics 
 
# ---------------------------------------------------------------------------
# Medium Layer — Strategy Activation & Capital Sizing
# ---------------------------------------------------------------------------
 
def apply_slow_layer_bounds(
    weights: Dict[str, float],
    slow_state: SlowLayerState,
) -> Dict[str, float]:
    """
    Constrain the medium layer's proposed weights within the slow layer's
    strategic bounds.
 
    Enforces:
        - Equity/bond/commodity budget ranges
        - Leverage ceiling
 
    Does NOT change the relative allocation within each bucket —
    only scales buckets to fit within bounds.
    """
    if not weights:
        return weights
 
    bounded = weights.copy()
    buckets = classify_weights(bounded)
 
    # ---- Enforce equity budget ----
    eq_min, eq_max = slow_state.equity_budget
    if buckets["equity"] > eq_max and buckets["equity"] > 0:
        scale = eq_max / buckets["equity"]
        for asset in bounded:
            if asset in EQUITY_ASSETS:
                bounded[asset] *= scale
 
    elif buckets["equity"] < eq_min and buckets["equity"] > 0:
        # Don't force equity UP — only cap it
        pass
 
    # ---- Enforce bond budget ----
    bond_min, bond_max = slow_state.bond_budget
    if buckets["bond"] > bond_max and buckets["bond"] > 0:
        scale = bond_max / buckets["bond"]
        for asset in bounded:
            if asset in BOND_ASSETS:
                bounded[asset] *= scale
 
    # ---- Enforce commodity budget ----
    comm_min, comm_max = slow_state.commodity_budget
    if buckets["commodity"] > comm_max and buckets["commodity"] > 0:
        scale = comm_max / buckets["commodity"]
        for asset in bounded:
            if asset in COMMODITY_ASSETS:
                bounded[asset] *= scale
 
    # ---- Enforce leverage ceiling ----
    total = sum(bounded.values())
    if total > slow_state.leverage_ceiling:
        scale = slow_state.leverage_ceiling / total
        bounded = {k: v * scale for k, v in bounded.items()}
 
    # ---- Normalize ----
    total = sum(bounded.values())
    if total > 0:
        bounded = {k: v / total for k, v in bounded.items()}
 
    return bounded
 
 
# ---------------------------------------------------------------------------
# Fast Layer — Crisis Response
# ---------------------------------------------------------------------------
 
@dataclass
class FastLayerAction:
    """Output of the fast layer — reduction applied to medium layer output."""
    date: pd.Timestamp
    reduction_factor: float         # 1.0 = no change, 0.5 = reduce by 50%
    cash_increase: float            # additional SHY weight
    circuit_breaker_level: str      # "none", "moderate", "severe"
    crash_momentum_active: bool
    override_active: bool           # True if fast layer modified the portfolio
    metadata: Dict[str, Any] = field(default_factory=dict)
 
 
def compute_fast_layer(
    weights: Dict[str, float],
    spy_prices: pd.Series,
    current_date: pd.Timestamp,
    state_features: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, float], FastLayerAction]:
    """
    Fast layer crisis response. Operates DAILY.
 
    CRITICAL RULE: Can REDUCE exposure but NEVER increase it.
 
    This layer has override authority over the medium layer.
    It processes the medium layer's output and applies emergency
    reductions when crisis conditions are detected.
 
    Inputs: real-time price data, volatility, drawdown.
    """
    from risk_engine import apply_circuit_breakers
 
    # ---- Apply circuit breakers (ported from v7) ----
    adjusted, breaker_meta = apply_circuit_breakers(
        weights, spy_prices, current_date
    )
 
    override_active = False
    reduction_factor = 1.0
    cash_increase = 0.0
    breaker_level = breaker_meta.get("circuit_breaker_level", "none")
    crash_active = breaker_meta.get("crash_momentum_active", False)
 
    # ---- Additional fast-layer volatility check ----
    # If intraday-scale vol is extremely elevated, further reduce
    spy_up_to = spy_prices.loc[:current_date]
    if len(spy_up_to) >= 10:
        recent_returns = spy_up_to.pct_change().tail(10).dropna()
        if len(recent_returns) >= 5:
            recent_vol = recent_returns.std() * np.sqrt(252)
 
            # Extremely elevated vol (>40% annualized) → additional reduction
            if recent_vol > 0.40:
                vol_reduction = min((recent_vol - 0.40) / 0.30, 0.5)  # max 50% additional
                reduction_factor = 1.0 - vol_reduction
 
                # Scale down all risky assets, increase SHY
                for asset in list(adjusted.keys()):
                    if asset != "SHY":
                        adjusted[asset] *= reduction_factor
 
                cash_increase = vol_reduction * 0.3
                adjusted["SHY"] = adjusted.get("SHY", 0) + cash_increase
 
    # ---- Detect if any changes were made ----
    if breaker_level != "none" or crash_active or reduction_factor < 1.0:
        override_active = True
 
    # ---- CRITICAL: Ensure we never INCREASED any position ----
    for asset in adjusted:
        if asset in weights:
            # Fast layer can only reduce, never increase
            if adjusted[asset] > weights[asset] and asset != "SHY":
                adjusted[asset] = weights[asset]
 
    # ---- Normalize ----
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
 
    action = FastLayerAction(
        date=current_date,
        reduction_factor=reduction_factor,
        cash_increase=cash_increase,
        circuit_breaker_level=breaker_level,
        crash_momentum_active=crash_active,
        override_active=override_active,
        metadata=breaker_meta,
    )
 
    return adjusted, action
 
 
# ---------------------------------------------------------------------------
# Multi-Horizon Coordinator
# ---------------------------------------------------------------------------
 
class MultiHorizonCoordinator:
    """
    Coordinates the three temporal layers into a single portfolio decision.
 
    Flow:
        1. Slow layer sets strategic bounds (monthly)
        2. Medium layer proposes allocation within bounds (weekly)
        3. Fast layer applies crisis reductions (daily)
        4. Risk engine enforces absolute constraints (always)
 
    The coordinator maintains state across layers and ensures
    proper interaction between temporal scales.
    """
 
    def __init__(self, profile: Optional[str] = None):
        self.profile_name = profile or "balanced"
        self.profile = get_risk_profile(self.profile_name) if profile else None
        self.current_slow_state: Optional[SlowLayerState] = None
        self.last_slow_update: Optional[pd.Timestamp] = None
        self.last_medium_update: Optional[pd.Timestamp] = None
        self.medium_weights: Dict[str, float] = {}
 
    def process(
        self,
        proposed_weights: Dict[str, float],
        state_features: pd.DataFrame,
        spy_prices: pd.Series,
        current_date: pd.Timestamp,
        strategy_weights: Optional[Dict[str, Dict[str, float]]] = None,
        stress_score: float = 0.0,
        allocator_confidence: Optional[float] = None,
        force_slow_update: bool = False,
        force_medium_update: bool = False,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Process a portfolio proposal through all three temporal layers.
 
        Args:
            proposed_weights: Raw weights from meta-allocator
            state_features: Full state representation
            spy_prices: SPY price series
            current_date: Current date
            strategy_weights: Per-strategy proposed weights (for uncertainty)
            stress_score: From defensive strategy
            allocator_confidence: Meta-allocator confidence
            force_slow_update: Force slow layer recomputation
            force_medium_update: Force medium layer update
 
        Returns:
            (final_weights, metadata)
        """
        metadata = {
            "slow_layer": {},
            "medium_layer": {},
            "fast_layer": {},
        }
 
        # ---- Slow Layer (monthly) ----
        should_update_slow = (
            force_slow_update
            or self.current_slow_state is None
            or self.last_slow_update is None
            or self._trading_days_since(self.last_slow_update, current_date, state_features.index)
            >= SLOW_LAYER_FREQUENCY
        )
 
        if should_update_slow:
            self.current_slow_state = compute_slow_layer(state_features, current_date)
            self.last_slow_update = current_date
            logger.debug(
                f"Slow layer updated: posture={self.current_slow_state.risk_posture}, "
                f"leverage_ceiling={self.current_slow_state.leverage_ceiling:.2f}"
            )
 
        metadata["slow_layer"] = {
            "risk_posture": self.current_slow_state.risk_posture,
            "leverage_ceiling": self.current_slow_state.leverage_ceiling,
            "composite_score": self.current_slow_state.metadata.get("composite_score", 0.5),
            "updated": should_update_slow,
        }
 
        # ---- Medium Layer (weekly) ----
        should_update_medium = (
            force_medium_update
            or self.last_medium_update is None
            or self._trading_days_since(self.last_medium_update, current_date, state_features.index)
            >= MEDIUM_LAYER_FREQUENCY
        )
 
        if should_update_medium:
            # ---- Apply conditional weighting to capital weights ----
            # This adjusts how much capital each strategy receives
            # based on posture strength, uncertainty, and stress.
            # Must happen BEFORE strategy blending.
            if strategy_weights:
                from risk_engine import compute_uncertainty_score
                unc_score, _ = compute_uncertainty_score(
                    state_features, strategy_weights, allocator_confidence
                )
                posture_strength = self.current_slow_state.metadata.get("composite_score", 0.5)

                # Extract base capital weights from proposed (if embedded)
                # For live mode, proposed_weights already has strategies blended,
                # so we re-blend with adjusted capital weights
                risk_profile = self.profile
                adjusted_cap, cw_diag = apply_conditional_weighting(
                    {name: 1.0 / len(strategy_weights) for name in strategy_weights},
                    posture_strength, unc_score, stress_score,
                    profile=risk_profile,
                )

                # Re-blend using adjusted capital weights
                reblended = {}
                for name, strat_w in strategy_weights.items():
                    cap_w = adjusted_cap.get(name, 1.0 / len(strategy_weights))
                    for asset, w in strat_w.items():
                        reblended[asset] = reblended.get(asset, 0) + cap_w * w

                proposed_weights = reblended
                metadata["conditional_weighting"] = cw_diag

            # Apply posture-aware equity/defensive scaling
            adjusted_proposed = self._apply_posture_bias(
                proposed_weights, self.current_slow_state, profile=self.profile
            )
            # Apply slow layer bounds to adjusted weights
            self.medium_weights = apply_slow_layer_bounds(
                adjusted_proposed, self.current_slow_state
            )
            self.last_medium_update = current_date
        # Otherwise hold existing medium weights
 
        metadata["medium_layer"] = {
            "updated": should_update_medium,
            "n_assets": len(self.medium_weights),
        }
 
        # ---- Apply Risk Engine (uncertainty dampening + constraints) ----
        from risk_engine import RiskEngine
        engine = RiskEngine(profile=self.profile_name)
        risk_adjusted, risk_meta = engine.apply(
            self.medium_weights,
            strategy_weights or {},
            state_features,
            spy_prices,
            current_date,
            stress_score=stress_score,
            allocator_confidence=allocator_confidence,
        )
 
        # ---- Fast Layer (daily — always runs) ----
        final_weights, fast_action = compute_fast_layer(
            risk_adjusted, spy_prices, current_date, state_features
        )
 
        metadata["fast_layer"] = {
            "override_active": fast_action.override_active,
            "reduction_factor": fast_action.reduction_factor,
            "circuit_breaker": fast_action.circuit_breaker_level,
            "crash_momentum": fast_action.crash_momentum_active,
        }
 
        metadata["risk_engine"] = {
            "uncertainty_score": risk_meta.get("uncertainty_score", 0),
            "tightening_level": risk_meta.get("tightening_level", 0),
        }
 
        return final_weights, metadata
 
    def _apply_posture_bias(
        self,
        weights: Dict[str, float],
        slow_state: SlowLayerState,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Adjust proposed weights based on the slow layer's strategic posture.

        When posture is aggressive:
            - Scale up equity assets (where Strategy 1 dominates)
            - Scale down defensive assets
            - This allows more of Strategy 1's alpha to pass through

        When posture is defensive:
            - Scale down equity, scale up bonds/cash
            - Protective behavior

        When balanced:
            - No adjustment (pass through as-is)

        This is the key mechanism for CAGR recovery: the slow layer
        identifies favorable environments, and this bias lets the
        portfolio express more conviction in those periods.
        """
        if slow_state.risk_posture == "balanced":
            return weights.copy()

        adjusted = weights.copy()
        composite = slow_state.metadata.get("composite_score", 0.5)

        if slow_state.risk_posture == "aggressive":
            # Scale up equity, scale down defensive
            # Intensity based on composite score (0.65-1.0 range for aggressive)
            intensity = min((composite - 0.65) / 0.35, 1.0)  # 0 at threshold, 1 at max
            intensity = max(intensity, 0)

            boost_factor = profile.get("posture_equity_boost", 0.30) if profile else 0.30
            cut_factor = profile.get("posture_defensive_cut", 0.25) if profile else 0.25
            equity_boost = 1.0 + intensity * boost_factor
            defensive_cut = 1.0 - intensity * cut_factor

            for asset in adjusted:
                if asset in EQUITY_ASSETS:
                    adjusted[asset] *= equity_boost
                elif asset in BOND_ASSETS and asset != "SHY":
                    adjusted[asset] *= defensive_cut
                elif asset == "SHY":
                    adjusted[asset] *= defensive_cut

        elif slow_state.risk_posture == "defensive":
            # Scale down equity, scale up defensive
            intensity = min((0.40 - composite) / 0.40, 1.0)  # 0 at threshold, 1 at min
            intensity = max(intensity, 0)

            cut_factor = profile.get("posture_defensive_cut", 0.25) if profile else 0.25
            boost_factor = profile.get("posture_equity_boost", 0.20) if profile else 0.20
            equity_cut = 1.0 - intensity * cut_factor
            defensive_boost = 1.0 + intensity * boost_factor

            for asset in adjusted:
                if asset in EQUITY_ASSETS:
                    adjusted[asset] *= equity_cut
                elif asset in BOND_ASSETS or asset == "SHY":
                    adjusted[asset] *= defensive_boost

        # Renormalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def _trading_days_since(
        self,
        last_date: pd.Timestamp,
        current_date: pd.Timestamp,
        trading_index: pd.DatetimeIndex,
    ) -> int:
        """Count trading days between two dates."""
        mask = (trading_index > last_date) & (trading_index <= current_date)
        return int(mask.sum())
 
 
# ---------------------------------------------------------------------------
# Backtest Integration
# ---------------------------------------------------------------------------
 
def backtest_multi_horizon(
    allocator_results: pd.DataFrame,
    strategy_outputs: Dict[str, list],
    state_features: pd.DataFrame,
    adj_close: pd.DataFrame,
    stress_scores: Optional[pd.Series] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Backtest the full multi-horizon framework.
 
    Compares:
        1. Allocator + risk engine (no multi-horizon — Step 8 result)
        2. Allocator + multi-horizon + risk engine (Step 9)
        3. Equal-weight baseline
 
    Tests whether temporal separation improves stability metrics.
    """
    from strategy_health import compute_strategy_daily_returns
 
    spy_prices = adj_close["SPY"] if "SPY" in adj_close.columns else pd.Series()
    benchmark = adj_close["SPY"].pct_change() if "SPY" in adj_close.columns else pd.Series(0, index=adj_close.index)
    daily_returns = adj_close.pct_change()
 
    # ---- Compute strategy returns ----
    strategy_returns = {}
    for name, outputs in strategy_outputs.items():
        strategy_returns[name] = compute_strategy_daily_returns(outputs, adj_close)
 
    strategy_names = ["regime_allocator", "trend_cta", "defensive"]
 
    # ---- Build forward-filled strategy weight lookups ----
    strat_weights_by_name = {}
    for name, outputs in strategy_outputs.items():
        date_weight_map = {}
        for o in outputs:
            date_weight_map[o.strategy_metadata["date"]] = o.proposed_weights
        strat_weights_by_name[name] = date_weight_map
 
    # ---- Build allocator-date blended weights (forward-filled) ----
    alloc_dates = allocator_results.index
    first_alloc = alloc_dates[0]
 
    current_strat_weights = {name: {} for name in strategy_names}
 
    proposed_by_date = {}
    strat_w_by_date = {}
    conditional_diagnostics = {}

    # Pre-compute uncertainty series for conditional weighting
    from risk_engine import compute_uncertainty_score

    for date in alloc_dates:
        date_str = date.strftime("%Y-%m-%d")

        for name in strategy_names:
            available = sorted(strat_weights_by_name.get(name, {}).keys())
            prior = [d for d in available if d <= date_str]
            if prior:
                current_strat_weights[name] = strat_weights_by_name[name][prior[-1]]

        # ---- Read base capital weights from allocator ----
        base_cap_weights = {}
        for name in strategy_names:
            cap_col = f"capital_weight_{name}"
            base_cap_weights[name] = allocator_results.loc[date, cap_col] if cap_col in allocator_results.columns else 1.0 / len(strategy_names)

        # ---- Compute conditional weighting inputs ----
        # Posture strength: from slow layer composite score
        state_up_to = state_features.loc[:date]
        slow_state = compute_slow_layer(state_up_to, date)
        posture_strength = slow_state.metadata.get("composite_score", 0.5)

        # Uncertainty: compute from state + strategy weights
        unc_score, _ = compute_uncertainty_score(
            state_up_to, current_strat_weights
        )

        # Stress: from defensive strategy
        stress_val = 0.0
        if stress_scores is not None and date in stress_scores.index:
            stress_val = float(stress_scores.loc[date]) if not pd.isna(stress_scores.loc[date]) else 0.0

        # ---- Apply conditional weighting ----
        risk_profile = get_risk_profile(profile) if profile else None
        adjusted_cap_weights, cw_diag = apply_conditional_weighting(
            base_cap_weights, posture_strength, unc_score, stress_val,
            profile=risk_profile,
        )
        conditional_diagnostics[date] = cw_diag

        # ---- Blend strategy weights using adjusted capital weights ----
        blended = {}
        strat_w = {}
        for name in strategy_names:
            cap_weight = adjusted_cap_weights.get(name, 1.0 / len(strategy_names))

            strat_proposed = current_strat_weights[name]
            strat_w[name] = strat_proposed

            for asset, w in strat_proposed.items():
                blended[asset] = blended.get(asset, 0) + cap_weight * w

        proposed_by_date[date] = blended
        strat_w_by_date[date] = strat_w
 
    # ---- Run multi-horizon on every trading day ----
    coordinator = MultiHorizonCoordinator(profile=profile)
 
    mh_returns_list = []
    raw_returns_list = []
    equal_returns_list = []
    mh_metadata_list = []
 
    current_proposed = {}
    current_strat_w = {}
 
    alloc_conf = None
    act_cols = [c for c in allocator_results.columns if c.startswith("activation_")]
    if act_cols:
        alloc_conf_series = allocator_results[act_cols].mean(axis=1)
 
    for date in daily_returns.index:
        if date < first_alloc:
            continue
 
        # Update proposed weights if this is an allocator date
        if date in proposed_by_date:
            current_proposed = proposed_by_date[date]
            current_strat_w = strat_w_by_date[date]
 
        if not current_proposed:
            continue
 
        # Get stress score
        stress = 0.0
        if stress_scores is not None and date in stress_scores.index:
            stress = float(stress_scores.loc[date]) if not pd.isna(stress_scores.loc[date]) else 0.0
 
        # Get allocator confidence
        conf = None
        if alloc_conf is not None and date in alloc_conf_series.index:
            conf = float(alloc_conf_series.loc[date])
 
        # Process through multi-horizon coordinator
        final_w, meta = coordinator.process(
            proposed_weights=current_proposed,
            state_features=state_features.loc[:date],
            spy_prices=spy_prices,
            current_date=date,
            strategy_weights=current_strat_w,
            stress_score=stress,
            allocator_confidence=conf,
        )
 
        # Compute multi-horizon return
        mh_ret = sum(
            final_w.get(a, 0) * daily_returns.loc[date, a]
            for a in final_w
            if a in daily_returns.columns and pd.notna(daily_returns.loc[date, a])
        )
 
        # Raw allocator return (no multi-horizon)
        raw_ret = sum(
            current_proposed.get(a, 0) * daily_returns.loc[date, a]
            for a in current_proposed
            if a in daily_returns.columns and pd.notna(daily_returns.loc[date, a])
        )
 
        # Equal weight return
        eq_ret = sum(
            ret.loc[date] if date in ret.index and pd.notna(ret.loc[date]) else 0
            for ret in strategy_returns.values()
        ) / len(strategy_returns)
 
        mh_returns_list.append({"date": date, "return": mh_ret})
        raw_returns_list.append({"date": date, "return": raw_ret})
        equal_returns_list.append({"date": date, "return": eq_ret})
        mh_metadata_list.append({"date": date, **meta})
 
    # ---- Build result series ----
    mh_returns = pd.Series(
        [r["return"] for r in mh_returns_list],
        index=[r["date"] for r in mh_returns_list],
    )
    raw_returns = pd.Series(
        [r["return"] for r in raw_returns_list],
        index=[r["date"] for r in raw_returns_list],
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
        kurtosis = r.kurtosis()
        return {
            "label": label, "sharpe": float(sharpe), "cagr": float(cagr),
            "max_dd": float(max_dd), "sortino": float(sortino),
            "calmar": float(calmar), "vol": float(vol),
            "kurtosis": float(kurtosis), "n_days": len(r),
        }
 
    # ---- Analyze multi-horizon metadata ----
    mh_meta_df = pd.DataFrame(mh_metadata_list).set_index("date")
 
    # Extract slow layer posture distribution
    slow_postures = mh_meta_df["slow_layer"].apply(
        lambda x: x.get("risk_posture", "unknown") if isinstance(x, dict) else "unknown"
    )
 
    # Extract fast layer override frequency
    fast_overrides = mh_meta_df["fast_layer"].apply(
        lambda x: x.get("override_active", False) if isinstance(x, dict) else False
    )
 
    # ---- Conditional weighting diagnostics ----
    cw_diag_df = pd.DataFrame.from_dict(conditional_diagnostics, orient="index")
    cw_diag_df.index.name = "date"

    # Compute conditional weighting summary by posture bucket
    cw_summary = {}
    if not cw_diag_df.empty:
        # Map each date to its posture
        cw_diag_df["posture"] = slow_postures.reindex(cw_diag_df.index).ffill()

        for posture in ["aggressive", "balanced", "defensive"]:
            mask = cw_diag_df["posture"] == posture
            if mask.any():
                subset = cw_diag_df.loc[mask]
                cw_summary[posture] = {
                    "count": int(mask.sum()),
                    "mean_favorable_score": float(subset["favorable_score"].mean()),
                    "mean_s1_boost": float(subset["s1_boost"].mean()),
                    "mean_adj_s1": float(subset["adjusted_s1"].mean()),
                    "mean_adj_s2": float(subset["adjusted_s2"].mean()),
                    "mean_adj_s3": float(subset["adjusted_s3"].mean()),
                }

    return {
        "mh_metrics": compute_metrics(mh_returns, "Multi-Horizon"),
        "raw_metrics": compute_metrics(raw_returns, "Allocator Only"),
        "equal_metrics": compute_metrics(equal_returns, "Equal-Weight"),
        "benchmark_metrics": compute_metrics(benchmark.reindex(mh_returns.index), "SPY Benchmark"),
        "mh_returns": mh_returns,
        "raw_returns": raw_returns,
        "mh_metadata": mh_meta_df,
        "slow_posture_distribution": slow_postures.value_counts().to_dict(),
        "fast_override_rate": float(fast_overrides.mean()),
        "fast_override_days": int(fast_overrides.sum()),
        "conditional_weighting_diagnostics": cw_diag_df,
        "conditional_weighting_summary": cw_summary,
    }
 
 
# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
 
def print_multi_horizon_report(results: Dict[str, Any]) -> None:
    """Print the multi-horizon framework performance report."""
    print(f"\n{'='*70}")
    print(f"MULTI-HORIZON FRAMEWORK REPORT")
    print(f"{'='*70}")
 
    print(f"\n--- Performance Comparison ---")
    for key in ["mh_metrics", "raw_metrics", "equal_metrics", "benchmark_metrics"]:
        m = results[key]
        if "error" in m:
            print(f"  {m['label']}: {m['error']}")
            continue
        kurtosis_str = f"  Kurt={m.get('kurtosis', 0):.1f}" if "kurtosis" in m else ""
        print(
            f"  {m['label']:<20s} "
            f"Sharpe={m['sharpe']:.2f}  "
            f"CAGR={m['cagr']:.1%}  "
            f"MaxDD={m['max_dd']:.1%}  "
            f"Sortino={m['sortino']:.2f}  "
            f"Calmar={m['calmar']:.2f}  "
            f"Vol={m['vol']:.1%}"
            f"{kurtosis_str}"
        )
 
    print(f"\n--- Slow Layer Posture Distribution ---")
    for posture, count in sorted(results["slow_posture_distribution"].items()):
        total = sum(results["slow_posture_distribution"].values())
        print(f"  {posture:<15s} {count:>5d} ({count/total*100:.1f}%)")
 
    print(f"\n--- Fast Layer Activity ---")
    print(f"  Override days:  {results['fast_override_days']} ({results['fast_override_rate']*100:.1f}%)")
 
    # ---- Step 9 Verification ----
    mh = results["mh_metrics"]
    raw = results["raw_metrics"]
    if "error" not in mh and "error" not in raw:
        print(f"\n--- Step 9 Verification ---")
        kurtosis_improved = mh.get("kurtosis", 99) < raw.get("kurtosis", 99)
        dd_improved = abs(mh["max_dd"]) < abs(raw["max_dd"])
        sharpe_preserved = mh["sharpe"] >= raw["sharpe"] * 0.90
 
        print(f"  Kurtosis reduced:   {raw.get('kurtosis', 0):.1f} -> {mh.get('kurtosis', 0):.1f} -> {'PASS' if kurtosis_improved else 'FAIL'}")
        print(f"  MaxDD improved:     {raw['max_dd']:.1%} -> {mh['max_dd']:.1%} -> {'PASS' if dd_improved else 'FAIL'}")
        print(f"  Sharpe preserved:   {raw['sharpe']:.2f} -> {mh['sharpe']:.2f} (>={raw['sharpe']*0.90:.2f}) -> {'PASS' if sharpe_preserved else 'FAIL'}")
        print(f"  Overall Step 9:     {'PASS' if (dd_improved and sharpe_preserved) else 'NEEDS REVIEW'}")

    # ---- Conditional Weighting Diagnostics ----
    cw_summary = results.get("conditional_weighting_summary", {})
    if cw_summary:
        print(f"\n--- Conditional Weighting Layer ---")
        print(f"  {'Posture':<15s} {'Count':>6s} {'FavScore':>10s} {'S1 Boost':>10s} {'Adj S1':>10s} {'Adj S2':>10s} {'Adj S3':>10s}")
        for posture in ["aggressive", "balanced", "defensive"]:
            if posture in cw_summary:
                s = cw_summary[posture]
                print(
                    f"  {posture:<15s} {s['count']:>6d} "
                    f"{s['mean_favorable_score']:>10.3f} "
                    f"{s['mean_s1_boost']:>10.3f} "
                    f"{s['mean_adj_s1']:>9.1%} "
                    f"{s['mean_adj_s2']:>9.1%} "
                    f"{s['mean_adj_s3']:>9.1%}"
                )

        # Overall dispersion check
        cw_diag = results.get("conditional_weighting_diagnostics", pd.DataFrame())
        if not cw_diag.empty:
            print(f"\n  Overall S1 allocation: {cw_diag['adjusted_s1'].mean():.1%} "
                  f"+/- {cw_diag['adjusted_s1'].std():.1%} "
                  f"(base: {cw_diag['base_s1'].mean():.1%})")
            print(f"  Favorable score:       {cw_diag['favorable_score'].mean():.3f} "
                  f"+/- {cw_diag['favorable_score'].std():.3f} "
                  f"[{cw_diag['favorable_score'].min():.3f} — {cw_diag['favorable_score'].max():.3f}]")