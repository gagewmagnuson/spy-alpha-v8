"""
SPY Alpha v8 — Strategy 1: Regime Allocator
==============================================
 
Wraps the v7 core pipeline (HMM → forecaster → selector → optimizer)
in the standardized v8 strategy interface.
 
V8 Modifications from v7:
    - Label-driven overlay REMOVED (no risk_on_prob → UPRO/SHY sizing)
    - UPRO and SHY participate in normal selection and rank-based allocation
    - Strategy outputs proposed weights WITHOUT leverage/defense decisions
    - Regime probabilities are passed through as metadata for the state representation
    - The meta-allocator decides how much to trust this strategy
 
What is ported unchanged:
    - feature_engineering.py (PCA pipeline, ~400 features → 20 components)
    - regime_model.py (5-regime Gaussian HMM, walk-forward, two-step label alignment)
    - return_forecaster.py (hierarchical Layer 1 ETF + Layer 2 stock alpha)
    - asset_selector.py (dynamic 8-11 asset selection with constraints)
    - portfolio_optimizer.py (rank-based allocation, vol targeting, smoothing)
 
Walk-forward configuration (unchanged, DO NOT MODIFY):
    - Training window: 756 days (~3 years)
    - Retrain frequency: 63 days (~quarterly)
    - Purge gap: 5 days
    - Rebalance: every 5 trading days (weekly)
"""
 
from __future__ import annotations
 
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
 
import numpy as np
import pandas as pd
 
from data_pipeline import get_raw_close, get_adj_close, get_fred
from feature_engineering import FeatureEngine
from regime_model import (
    RegimeModel,
    walk_forward_regimes,
    smooth_probabilities,
)
from return_forecaster import ReturnForecaster
from asset_selector import AssetSelector
from portfolio_optimizer import PortfolioOptimizer
 
logger = logging.getLogger("spy_alpha_v8.strategy_regime")
 
 
# ---------------------------------------------------------------------------
# Configuration (all proven v7 values — DO NOT MODIFY)
# ---------------------------------------------------------------------------
 
DEFAULT_N_REGIMES = 5
DEFAULT_N_COMPONENTS = 20
DEFAULT_TRAIN_WINDOW = 756      # ~3 years
DEFAULT_RETRAIN_EVERY = 63      # ~quarterly
DEFAULT_PURGE_DAYS = 5
DEFAULT_FORECAST_HORIZON = 21   # ~1 month
DEFAULT_REBALANCE_EVERY = 5     # weekly
 
 
# ---------------------------------------------------------------------------
# Standardized Strategy Output
# ---------------------------------------------------------------------------
 
@dataclass
class StrategyOutput:
    """
    Standardized output format for all v8 strategies.
 
    Every strategy produces this identical structure so the
    meta-allocator can compare them uniformly.
    """
    strategy_name: str
    proposed_weights: Dict[str, float]      # asset → weight (sum to 1.0)
    confidence: float                       # 0.0 to 1.0
    active_assets: List[str]
    strategy_metadata: Dict[str, Any] = field(default_factory=dict)
 
 
# ---------------------------------------------------------------------------
# Strategy 1: Regime Allocator
# ---------------------------------------------------------------------------
 
class RegimeAllocatorStrategy:
    """
    Strategy 1: Regime-based multi-asset allocator.
 
    Wraps the full v7 pipeline and exposes it through the standardized
    StrategyOutput interface. The meta-allocator treats this as one of
    three strategy signals.
 
    The strategy runs its own internal walk-forward HMM, generates
    forecasts, selects assets, and optimizes weights. The output is
    the proposed portfolio — the meta-allocator decides how much
    capital to give it.
    """
 
    def __init__(
        self,
        n_regimes: int = DEFAULT_N_REGIMES,
        n_components: int = DEFAULT_N_COMPONENTS,
        train_window: int = DEFAULT_TRAIN_WINDOW,
        retrain_every: int = DEFAULT_RETRAIN_EVERY,
        purge_days: int = DEFAULT_PURGE_DAYS,
        forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
        rebalance_every: int = DEFAULT_REBALANCE_EVERY,
        include_stocks: bool = True,
    ):
        self.n_regimes = n_regimes
        self.n_components = n_components
        self.train_window = train_window
        self.retrain_every = retrain_every
        self.purge_days = purge_days
        self.forecast_horizon = forecast_horizon
        self.rebalance_every = rebalance_every
        self.include_stocks = include_stocks
 
        # Pipeline components (initialized during build)
        self.feature_engine: Optional[FeatureEngine] = None
        self.regime_model: Optional[RegimeModel] = None
        self.forecaster: Optional[ReturnForecaster] = None
        self.selector: Optional[AssetSelector] = None
        self.optimizer: Optional[PortfolioOptimizer] = None
 
        # Walk-forward results (populated after build)
        self.smoothed_probs: Optional[pd.DataFrame] = None
        self.raw_probs: Optional[pd.DataFrame] = None
        self.wf_result = None
        self.obs_features: Optional[pd.DataFrame] = None
        self.sector_features: Optional[pd.DataFrame] = None
 
    def build(self, snapshot: Dict[str, Any]) -> None:
        """
        Build all pipeline components from a frozen snapshot.
 
        This runs the full walk-forward HMM training, builds features,
        and initializes all downstream components. Called once during
        setup, not on every rebalance.
        """
        logger.info("Strategy 1 (Regime Allocator): Building pipeline...")
 
        # ---- Feature engineering ----
        self.feature_engine = FeatureEngine(
            reduce_dims=True, n_components=self.n_components
        )
        self.obs_features = self.feature_engine.build_observation_features(snapshot)
        self.sector_features = self.feature_engine.build_sector_features(snapshot)
 
        logger.info(
            f"  Features: observation {self.obs_features.shape}, "
            f"sector {self.sector_features.shape}"
        )
 
        # ---- Walk-forward regime model ----
        spy_close = get_raw_close(snapshot, ["SPY"])["SPY"]
        tlt_close_df = get_raw_close(snapshot, ["TLT"])
        tlt_close = tlt_close_df["TLT"] if "TLT" in tlt_close_df.columns else None
 
        self.regime_model = RegimeModel(n_regimes=self.n_regimes)
        wf_result = walk_forward_regimes(
            self.obs_features,
            self.regime_model,
            train_window=self.train_window,
            retrain_every=self.retrain_every,
            purge_days=self.purge_days,
            spy_close=spy_close,
            tlt_close=tlt_close,
        )
 
        self.smoothed_probs = wf_result.probabilities
        self.raw_probs = wf_result.raw_probabilities
        self.wf_result = wf_result
 
        logger.info(
            f"  Regime model: {len(self.smoothed_probs)} out-of-sample days, "
            f"{len(wf_result.window_info)} windows"
        )
 
        # ---- Initialize downstream components ----
        self.forecaster = ReturnForecaster(horizon=self.forecast_horizon)
        self.selector = AssetSelector()
        self.optimizer = PortfolioOptimizer()
 
        logger.info("Strategy 1 (Regime Allocator): Pipeline built successfully")
 
    def generate_signals(
        self,
        snapshot: Dict[str, Any],
    ) -> List[StrategyOutput]:
        """
        Generate strategy signals for every rebalance date in the backtest period.
 
        Returns a list of StrategyOutput, one per rebalance date, aligned
        with the walk-forward regime probability dates.
        """
        if self.smoothed_probs is None:
            raise RuntimeError("Call build() before generate_signals()")
 
        adj_close = get_adj_close(snapshot)
        raw_close = get_raw_close(snapshot)
 
        # ---- Generate forecasts ----
        forecasts = self.forecaster.generate_forecast_series(
            self.smoothed_probs,
            self.sector_features,
            adj_close,
            raw_close,
            include_stocks=self.include_stocks,
        )
 
        # ---- Run selections ----
        selections = []
        for fc in forecasts:
            sel = self.selector.select(
                forecast=fc, raw_close=raw_close, adj_close=adj_close
            )
            selections.append(sel)
 
        # ---- Optimize allocations ----
        allocations = self.optimizer.optimize_series(forecasts, selections, adj_close)
 
        # ---- Convert to standardized StrategyOutput ----
        outputs = []
        for alloc, fc, sel in zip(allocations, forecasts, selections):
            output = self._to_strategy_output(alloc, fc, sel)
            outputs.append(output)
 
        logger.info(
            f"Strategy 1: Generated {len(outputs)} signals "
            f"({outputs[0].strategy_metadata.get('date', '?')} → "
            f"{outputs[-1].strategy_metadata.get('date', '?')})"
        )
 
        return outputs
 
    def get_regime_probabilities(self) -> Optional[pd.DataFrame]:
        """
        Return the walk-forward regime probabilities for the state representation.
 
        These feed into the state representation as latent features.
        They are NOT used for allocation decisions within this strategy
        (the overlay has been removed).
        """
        return self.smoothed_probs
 
    def get_allocations_for_backtest(
        self,
        snapshot: Dict[str, Any],
    ):
        """
        Return raw PortfolioAllocation objects for backtesting.
 
        This bypasses the StrategyOutput wrapper and returns the native
        v7 allocation objects that backtest_engine.py expects. Used for
        standalone Strategy 1 backtesting only.
        """
        if self.smoothed_probs is None:
            raise RuntimeError("Call build() before get_allocations_for_backtest()")
 
        adj_close = get_adj_close(snapshot)
        raw_close = get_raw_close(snapshot)
 
        forecasts = self.forecaster.generate_forecast_series(
            self.smoothed_probs,
            self.sector_features,
            adj_close,
            raw_close,
            include_stocks=self.include_stocks,
        )
 
        selections = []
        for fc in forecasts:
            sel = self.selector.select(
                forecast=fc, raw_close=raw_close, adj_close=adj_close
            )
            selections.append(sel)
 
        allocations = self.optimizer.optimize_series(forecasts, selections, adj_close)
 
        return allocations, forecasts, selections
 
    def _to_strategy_output(self, alloc, forecast, selection) -> StrategyOutput:
        """Convert a PortfolioAllocation to the standardized StrategyOutput."""
        # Extract weights as dict, filtering near-zero
        weights = {}
        for asset, weight in alloc.weights.items():
            if abs(weight) > 1e-6:
                weights[asset] = float(weight)
 
        # Confidence: use the forecast's overall_confidence
        confidence = float(forecast.overall_confidence)
 
        # Active assets: those with non-trivial weight
        active = [a for a, w in weights.items() if w > 0.01]
 
        # Regime probabilities for metadata
        regime_probs = {}
        for regime in ["Bull", "Slowdown", "Crisis-Deflation", "Crisis-Inflation", "Inflation"]:
            if regime in forecast.regime_probs:
                regime_probs[regime] = float(forecast.regime_probs[regime])
 
        metadata = {
            "date": forecast.forecast_date,
            "dominant_regime": forecast.dominant_regime,
            "regime_conviction": float(forecast.regime_conviction),
            "regime_probs": regime_probs,
            "risk_on_prob": float(alloc.risk_on_prob),
            "upro_weight": float(alloc.upro_weight),
            "shy_weight": float(alloc.shy_weight),
            "tlt_weight": float(alloc.tlt_weight),
            "gld_weight": float(alloc.gld_weight),
            "target_volatility": float(alloc.target_volatility),
            "realized_volatility": float(alloc.realized_volatility),
            "turnover": float(alloc.turnover),
            "n_stocks": alloc.n_stocks,
            "n_selected": len(selection.selected_assets),
            "selected_assets": selection.selected_assets,
        }
 
        return StrategyOutput(
            strategy_name="regime_allocator",
            proposed_weights=weights,
            confidence=confidence,
            active_assets=active,
            strategy_metadata=metadata,
        )
 
 
# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
 
def print_strategy_output(output: StrategyOutput) -> None:
    """Pretty-print a single strategy output."""
    print(f"\n{'='*60}")
    print(f"STRATEGY: {output.strategy_name}")
    print(f"{'='*60}")
    print(f"  Date:       {output.strategy_metadata.get('date', '?')}")
    print(f"  Confidence: {output.confidence:.3f}")
    print(f"  Regime:     {output.strategy_metadata.get('dominant_regime', '?')} "
          f"({output.strategy_metadata.get('regime_conviction', 0):.1%})")
    print(f"  Active:     {len(output.active_assets)} assets")
    print()
 
    print(f"  {'Asset':<8s} {'Weight':>8s}")
    print(f"  {'-'*16}")
    for asset in sorted(output.proposed_weights, key=output.proposed_weights.get, reverse=True):
        w = output.proposed_weights[asset]
        if w > 0.005:
            marker = " ←" if asset in ("UPRO", "SHY", "TLT", "GLD") else ""
            print(f"  {asset:<8s} {w:>7.1%}{marker}")
 
 
def summarize_strategy_outputs(outputs: List[StrategyOutput]) -> Dict[str, Any]:
    """Summarize a series of strategy outputs for diagnostics."""
    if not outputs:
        return {}
 
    n = len(outputs)
 
    # Collect all weights across all outputs
    all_weights = pd.DataFrame([o.proposed_weights for o in outputs]).fillna(0)
    confidences = [o.confidence for o in outputs]
 
    # Asset frequency
    asset_freq = {}
    for o in outputs:
        for asset in o.active_assets:
            asset_freq[asset] = asset_freq.get(asset, 0) + 1
 
    # Regime distribution
    regime_dist = {}
    for o in outputs:
        regime = o.strategy_metadata.get("dominant_regime", "Unknown")
        regime_dist[regime] = regime_dist.get(regime, 0) + 1
 
    return {
        "n_signals": n,
        "mean_confidence": float(np.mean(confidences)),
        "std_confidence": float(np.std(confidences)),
        "mean_n_active": float(np.mean([len(o.active_assets) for o in outputs])),
        "mean_weights": {col: float(all_weights[col].mean()) for col in all_weights.columns
                         if all_weights[col].mean() > 0.005},
        "asset_frequency": {k: f"{v/n:.1%}" for k, v in
                           sorted(asset_freq.items(), key=lambda x: -x[1])[:15]},
        "regime_distribution": regime_dist,
    }