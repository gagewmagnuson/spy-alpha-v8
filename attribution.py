"""
SPY Alpha v8 — Attribution & Signal Decomposition (Layer 4)
=============================================================
 
NEW module in v8. Decomposes every portfolio decision into:
    - State assessment (what does the system think is happening?)
    - Strategy contributions (which strategy drove what?)
    - Risk modifications (what did the risk engine change?)
    - Final portfolio with confidence
 
From build spec Section 8:
    - Every signal decomposes into: state assessment, strategy contributions,
      risk modifications
    - Track record verifiable via prediction_history.csv
    - Clean JSON output suitable for API/dashboard consumption
    - Explainability overlay: internally complex, externally simple attribution
"""
 
from __future__ import annotations
 
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
 
import numpy as np
import pandas as pd
 
logger = logging.getLogger("spy_alpha_v8.attribution")
 
 
# ---------------------------------------------------------------------------
# Signal Attribution
# ---------------------------------------------------------------------------
 
def build_daily_signal(
    date: pd.Timestamp,
    final_weights: Dict[str, float],
    state_features: pd.Series,
    strategy_outputs: Dict[str, Any],
    allocator_weights: Dict[str, float],
    risk_metadata: Dict[str, Any],
    multi_horizon_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the complete daily signal output with full attribution.
 
    Matches the build spec Section 8A format (latest_prediction.json).
    """
    # ---- State Assessment ----
    state_assessment = _build_state_assessment(state_features)
 
    # ---- Strategy Signals ----
    strategy_signals = _build_strategy_signals(strategy_outputs, allocator_weights)
 
    # ---- Risk Modifications ----
    risk_modifications = _build_risk_modifications(risk_metadata, multi_horizon_metadata)
 
    # ---- Portfolio ----
    portfolio = _build_portfolio_summary(final_weights, risk_metadata)
 
    # ---- Attribution ----
    attribution = _build_attribution(
        strategy_outputs, allocator_weights, risk_metadata
    )
 
    signal = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "forecast_date": date.strftime("%Y-%m-%d"),
        "state_assessment": state_assessment,
        "strategy_signals": strategy_signals,
        "risk_modifications": risk_modifications,
        "portfolio": portfolio,
        "attribution": attribution,
    }
 
    return signal
 
 
def _build_state_assessment(state_features: pd.Series) -> Dict[str, Any]:
    """Extract key state descriptors for the signal output."""
    assessment = {}
 
    # Volatility regime
    vol_10d = state_features.get("vol_realized_10d", None)
    if vol_10d is not None and not pd.isna(vol_10d):
        vol = float(vol_10d)
        if vol < 0.10:
            assessment["volatility_regime"] = "low_vol"
        elif vol < 0.20:
            assessment["volatility_regime"] = "normal_vol"
        else:
            assessment["volatility_regime"] = "high_vol"
        assessment["realized_vol_10d"] = round(vol, 4)
 
    # Trend strength
    breadth = state_features.get("trend_breadth_avg", None)
    if breadth is not None and not pd.isna(breadth):
        assessment["trend_strength"] = round(float(breadth), 3)
 
    # HMM regime
    hmm_max = state_features.get("hmm_max_prob", None)
    if hmm_max is not None and not pd.isna(hmm_max):
        assessment["hmm_conviction"] = round(float(hmm_max), 3)
 
    # Find dominant regime
    regime_cols = {
        "hmm_bull_prob": "Bull",
        "hmm_slowdown_prob": "Slowdown",
        "hmm_crisis_deflation_prob": "Crisis-Deflation",
        "hmm_crisis_inflation_prob": "Crisis-Inflation",
        "hmm_inflation_prob": "Inflation",
    }
    max_prob = 0
    dominant = "Unknown"
    for col, name in regime_cols.items():
        val = state_features.get(col, None)
        if val is not None and not pd.isna(val) and float(val) > max_prob:
            max_prob = float(val)
            dominant = name
    assessment["hmm_dominant_regime"] = dominant
 
    # Macro conditions
    t10y2y = state_features.get("macro_t10y2y_level", None)
    if t10y2y is not None and not pd.isna(t10y2y):
        t = float(t10y2y)
        if t < 0:
            assessment["macro_conditions"] = "inverted_curve"
        elif t < 1.0:
            assessment["macro_conditions"] = "flat_curve"
        else:
            assessment["macro_conditions"] = "normal_curve"
 
    # Entropy / uncertainty from HMM
    entropy = state_features.get("hmm_regime_entropy", None)
    if entropy is not None and not pd.isna(entropy):
        assessment["regime_entropy"] = round(float(entropy), 3)
 
    return assessment
 
 
def _build_strategy_signals(
    strategy_outputs: Dict[str, Any],
    allocator_weights: Dict[str, float],
) -> Dict[str, Any]:
    """Build strategy signal summary."""
    signals = {}
 
    for name, output in strategy_outputs.items():
        if output is None:
            continue
 
        signal = {
            "activation": round(float(allocator_weights.get(name, 0.33)), 3),
        }
 
        if hasattr(output, "confidence"):
            signal["confidence"] = round(float(output.confidence), 3)
 
        if hasattr(output, "active_assets"):
            signal["n_active_assets"] = len(output.active_assets)
 
        meta = output.strategy_metadata if hasattr(output, "strategy_metadata") else {}
 
        if name == "regime_allocator":
            signal["regime_call"] = meta.get("dominant_regime", "Unknown")
            signal["top_assets"] = list(output.proposed_weights.keys())[:3] if hasattr(output, "proposed_weights") else []
 
        elif name == "trend_cta":
            signal["trend_direction"] = meta.get("trend_direction", "unknown")
            signal["assets_in_uptrend"] = meta.get("assets_in_uptrend", 0)
 
        elif name == "defensive":
            signal["stress_score"] = round(float(meta.get("stress_score", 0)), 3)
            signal["stress_activated"] = meta.get("stress_activated", False)
 
        signals[name] = signal
 
    return signals
 
 
def _build_risk_modifications(
    risk_metadata: Dict[str, Any],
    mh_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Build risk modification summary."""
    modifications = {}

    # Uncertainty dampening
    modifications["uncertainty_score"] = round(
        float(risk_metadata.get("uncertainty_score", 0)), 3
    )

    # Constraint tightening
    modifications["tightening_level"] = round(
        float(risk_metadata.get("tightening_level", 0)), 3
    )

    # Circuit breaker
    breaker = risk_metadata.get("circuit_breaker", {})
    if isinstance(breaker, dict):
        modifications["circuit_breaker"] = breaker.get("circuit_breaker_level", "none")
        modifications["crash_momentum"] = breaker.get("crash_momentum_active", False)

    # Multi-horizon layers
    slow = mh_metadata.get("slow_layer", {})
    fast = mh_metadata.get("fast_layer", {})

    if isinstance(slow, dict):
        modifications["strategic_posture"] = slow.get("risk_posture", "unknown")
        modifications["composite_score"] = round(
            float(slow.get("composite_score", 0.5)), 3
        )
        modifications["leverage_ceiling"] = round(
            float(slow.get("leverage_ceiling", 1.0)), 2
        )

    if isinstance(fast, dict):
        modifications["fast_layer_override"] = fast.get("override_active", False)

    # Conditional weighting telemetry
    cw = mh_metadata.get("conditional_weighting", {})
    if cw:
        modifications["favorable_score"] = round(float(cw.get("favorable_score", 0)), 3)
        modifications["s1_boost"] = round(float(cw.get("s1_boost", 0)), 3)
        modifications["adjusted_s1_weight"] = round(float(cw.get("adjusted_s1", 0)), 3)
        modifications["adjusted_s2_weight"] = round(float(cw.get("adjusted_s2", 0)), 3)
        modifications["adjusted_s3_weight"] = round(float(cw.get("adjusted_s3", 0)), 3)
        modifications["stress_input"] = round(float(cw.get("stress", 0)), 3)

    return modifications
 
 
def _build_portfolio_summary(
    final_weights: Dict[str, float],
    risk_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Build portfolio summary."""
    # Sort weights descending
    sorted_weights = dict(sorted(final_weights.items(), key=lambda x: -x[1]))
 
    portfolio = {
        "selected_assets": [a for a, w in sorted_weights.items() if w > 0.01],
        "weights": {a: round(w, 4) for a, w in sorted_weights.items() if w > 0.005},
        "n_assets": len([w for w in final_weights.values() if w > 0.01]),
        "upro_weight": round(final_weights.get("UPRO", 0), 4),
        "shy_weight": round(final_weights.get("SHY", 0), 4),
        "tlt_weight": round(final_weights.get("TLT", 0), 4),
        "gld_weight": round(final_weights.get("GLD", 0), 4),
    }
 
    return portfolio
 
 
def _build_attribution(
    strategy_outputs: Dict[str, Any],
    allocator_weights: Dict[str, float],
    risk_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build attribution breakdown showing how much each component
    contributed to the final portfolio.
    """
    total_weight = sum(allocator_weights.values()) if allocator_weights else 1.0
    if total_weight == 0:
        total_weight = 1.0
 
    attribution = {}
    for name, cap_weight in allocator_weights.items():
        pct = cap_weight / total_weight * 100
        attribution[f"{name}_contribution"] = f"{pct:.0f}%"
 
    # Risk layer modification estimate
    tightening = risk_metadata.get("tightening_level", 0)
    attribution["risk_layer_modification"] = f"-{tightening * 5:.0f}%"
 
    uncertainty = risk_metadata.get("uncertainty_score", 0)
    attribution["uncertainty_reduction"] = f"-{uncertainty * 3:.0f}%"
 
    return attribution
 
 
# ---------------------------------------------------------------------------
# Prediction History Tracking
# ---------------------------------------------------------------------------
 
class PredictionTracker:
    """
    Tracks daily predictions and realized outcomes.
 
    Maintains prediction_history.csv with all fields from v7 plus:
        - Strategy activation scores
        - Strategy health metrics
        - Uncertainty score
        - Attribution breakdown
        - Risk layer modifications
        - Allocator confidence
    """
 
    def __init__(self, signals_dir: Path = Path("signals")):
        self.signals_dir = signals_dir
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.signals_dir / "prediction_history.csv"
 
    def save_signal(self, signal: Dict[str, Any]) -> Path:
        """Save the latest signal as JSON."""
        path = self.signals_dir / "latest_prediction.json"
        with open(path, "w") as f:
            json.dump(signal, f, indent=2, default=str)
        logger.info(f"Signal saved to {path}")
        return path
 
    def append_to_history(
        self,
        signal: Dict[str, Any],
        realized_spy_return: Optional[float] = None,
        realized_portfolio_return: Optional[float] = None,
    ) -> None:
        """Append a prediction record to the history CSV."""
        record = {
            "date": signal.get("forecast_date", ""),
            "generated_at": signal.get("generated_at", ""),
        }
 
        # Flatten state assessment
        state = signal.get("state_assessment", {})
        for k, v in state.items():
            record[f"state_{k}"] = v
 
        # Flatten strategy signals
        strategies = signal.get("strategy_signals", {})
        for name, sig in strategies.items():
            for k, v in sig.items():
                record[f"strat_{name}_{k}"] = v
 
        # Risk modifications
        risk = signal.get("risk_modifications", {})
        for k, v in risk.items():
            record[f"risk_{k}"] = v
 
        # Portfolio
        portfolio = signal.get("portfolio", {})
        record["n_assets"] = portfolio.get("n_assets", 0)
        record["upro_weight"] = portfolio.get("upro_weight", 0)
        record["shy_weight"] = portfolio.get("shy_weight", 0)
 
        # Top 5 weights
        weights = portfolio.get("weights", {})
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])[:5]
        for i, (asset, w) in enumerate(sorted_w):
            record[f"top{i+1}_asset"] = asset
            record[f"top{i+1}_weight"] = w
 
        # Attribution
        attribution = signal.get("attribution", {})
        for k, v in attribution.items():
            record[f"attr_{k}"] = v
 
        # Realized returns (filled in later when available)
        record["realized_spy_return"] = realized_spy_return
        record["realized_portfolio_return"] = realized_portfolio_return
 
        # Append to CSV
        df = pd.DataFrame([record])
        if self.history_path.exists():
            df.to_csv(self.history_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.history_path, index=False)
 
    def load_history(self) -> pd.DataFrame:
        """Load the full prediction history."""
        if not self.history_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.history_path)
 
 
# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
 
def print_signal(signal: Dict[str, Any]) -> None:
    """Pretty-print a daily signal."""
    print(f"\n{'='*60}")
    print(f"DAILY SIGNAL — {signal.get('forecast_date', '?')}")
    print(f"{'='*60}")
 
    # State
    state = signal.get("state_assessment", {})
    print(f"\n--- State Assessment ---")
    for k, v in state.items():
        print(f"  {k:<30s} {v}")
 
    # Strategies
    strategies = signal.get("strategy_signals", {})
    print(f"\n--- Strategy Signals ---")
    for name, sig in strategies.items():
        print(f"  {name}:")
        for k, v in sig.items():
            print(f"    {k:<25s} {v}")
 
    # Risk
    risk = signal.get("risk_modifications", {})
    print(f"\n--- Risk Modifications ---")
    for k, v in risk.items():
        print(f"  {k:<30s} {v}")
 
    # Portfolio
    portfolio = signal.get("portfolio", {})
    print(f"\n--- Portfolio ---")
    print(f"  Assets: {portfolio.get('n_assets', 0)}")
    weights = portfolio.get("weights", {})
    for asset, w in sorted(weights.items(), key=lambda x: -x[1]):
        if w > 0.01:
            marker = " *" if asset in ("UPRO", "SHY", "TLT", "GLD") else ""
            print(f"    {asset:<8s} {w:>7.1%}{marker}")
 
    # Attribution
    attribution = signal.get("attribution", {})
    print(f"\n--- Attribution ---")
    for k, v in attribution.items():
        print(f"  {k:<35s} {v}")