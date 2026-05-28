"""
SPY Alpha v8 — Main Orchestrator
===================================
 
Wires all v8 layers together into a unified pipeline:
 
    Layer 0: State Representation
    Layer 1: Strategy Layer (3 independent strategies)
    Layer 2: Meta-Allocator (LightGBM)
    Layer 3: Risk Constraint Engine
    Multi-Horizon Framework (slow/medium/fast)
    Layer 4: Attribution & Signal Output
 
Modes:
    backtest   — Full walk-forward backtest on frozen snapshot
    daily      — Live inference with fresh data
    snapshot   — Create/manage data snapshots
    signal     — Generate and display latest signal
 
Usage:
    python spy_alpha_v8.py backtest --snapshot baseline_v7
    python spy_alpha_v8.py daily --snapshot baseline_v7 --fred-key KEY
    python spy_alpha_v8.py snapshot --name my_snapshot --fred-key KEY
"""
 
from __future__ import annotations
 
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
 
import numpy as np
import pandas as pd
 
# ---- V8 Modules ----
from data_pipeline import (
    SnapshotManager,
    get_adj_close,
    get_raw_close,
    get_fred,
    fetch_daily_live,
    fetch_stress_data,
)
from feature_engineering import FeatureEngine
from state_representation import StateRepresentationBuilder
from strategy_regime import RegimeAllocatorStrategy
from strategy_trend import TrendCTAStrategy
from strategy_defensive import DefensiveStrategy
from meta_allocator import backtest_meta_allocator
from multi_horizon import (
    MultiHorizonCoordinator,
    backtest_multi_horizon,
    print_multi_horizon_report,
)
from risk_engine import RiskEngine
from attribution import (
    build_daily_signal,
    PredictionTracker,
    print_signal,
)
 
logger = logging.getLogger("spy_alpha_v8")
 
 
# ---------------------------------------------------------------------------
# Full Backtest Pipeline
# ---------------------------------------------------------------------------
 
def run_backtest(
    snapshot_name: str = "baseline_v7",
    extended_snapshot_name: str = "baseline_v7_extended",
    fred_api_key: Optional[str] = None,
    data_dir: Optional[str] = None,
    profile: str = "balanced",
) -> Dict[str, Any]:
    """
    Run the full v8 walk-forward backtest.
 
    Pipeline:
        1. Load snapshots
        2. Build all three strategies
        3. Build state representation
        4. Train meta-allocator (walk-forward on extended dataset)
        5. Apply multi-horizon framework (slow/medium/fast layers)
        6. Compute performance metrics
        7. Generate attribution summary
    """
    mgr = SnapshotManager(data_dir=data_dir)
 
    # ---- Load data ----
    logger.info(f"Loading snapshot: {snapshot_name}")
    snap = mgr.load_snapshot(snapshot_name)
    adj_close = get_adj_close(snap)
    raw_close = get_raw_close(snap)
    fred_data = get_fred(snap)
 
    # Load extended snapshot if available
    snap_ext = None
    try:
        snap_ext = mgr.load_snapshot(extended_snapshot_name)
        logger.info(f"Extended snapshot loaded: {extended_snapshot_name}")
    except FileNotFoundError:
        logger.warning(f"Extended snapshot '{extended_snapshot_name}' not found — proceeding without")
 
    # ---- Build Strategy 1: Regime Allocator ----
    logger.info("Building Strategy 1 (Regime Allocator)...")
    s1 = RegimeAllocatorStrategy()
    s1.build(snap)
    s1_outputs = s1.generate_signals(snap)
    regime_probs = s1.get_regime_probabilities()
 
    # ---- Build Strategy 2: Trend/CTA ----
    logger.info("Building Strategy 2 (Trend/CTA)...")
    s2 = TrendCTAStrategy()
    s2.build(snap)
 
    # Align to S1 rebalance dates
    s1_dates = pd.DatetimeIndex([
        pd.Timestamp(o.strategy_metadata["date"]) for o in s1_outputs
    ])
    s2_outputs = s2.generate_signals(snap, rebalance_dates=s1_dates)
 
    # ---- Build Strategy 3: Defensive ----
    logger.info("Building Strategy 3 (Defensive)...")
    s3 = DefensiveStrategy()
    s3.build(snap)
    s3_outputs = s3.generate_signals(snap, rebalance_dates=s1_dates)
    stress_scores = s3.get_stress_score()
 
    # ---- Build State Representation ----
    logger.info("Building state representation...")
    builder = StateRepresentationBuilder()
    state = builder.build(raw_close, fred_data, regime_probs=regime_probs)
 
    # ---- Strategy outputs dict ----
    strategy_outputs = {
        "regime_allocator": s1_outputs,
        "trend_cta": s2_outputs,
        "defensive": s3_outputs,
    }
 
    # ---- Train Meta-Allocator ----
    logger.info("Training meta-allocator (walk-forward)...")
    alloc_result = backtest_meta_allocator(
        snap, state, strategy_outputs, adj_close,
        extended_snapshot=snap_ext,
    )
 
    # ---- Run Multi-Horizon Backtest ----
    logger.info("Running multi-horizon backtest...")
    mh_results = backtest_multi_horizon(
        alloc_result["allocator_results"],
        strategy_outputs,
        state,
        adj_close,
        stress_scores=stress_scores,
        profile=profile,
    )
 
    # ---- Print Results ----
    print_multi_horizon_report(mh_results)
 
    # ---- Print comparison with v7 baseline ----
    mh = mh_results["mh_metrics"]
    if "error" not in mh:
        print(f"\n{'='*70}")
        print(f"V8 FULL PIPELINE vs BENCHMARKS")
        print(f"{'='*70}")
        print(f"  {'Metric':<20s} {'V8 Full':>10s} {'V7 Baseline':>12s} {'SPY':>10s}")
        print(f"  {'-'*52}")
        print(f"  {'Sharpe':<20s} {mh['sharpe']:>10.2f} {'1.29':>12s} {'0.87':>10s}")
        print(f"  {'CAGR':<20s} {mh['cagr']:>9.1%} {'27.0%':>12s} {'~14%':>10s}")
        print(f"  {'Max DD':<20s} {mh['max_dd']:>9.1%} {'-19.2%':>12s} {'-33.7%':>10s}")
        print(f"  {'Sortino':<20s} {mh['sortino']:>10.2f} {'1.71':>12s} {'~1.1':>10s}")
        print(f"  {'Calmar':<20s} {mh['calmar']:>10.2f} {'1.41':>12s} {'~0.4':>10s}")
 
    # ---- Save Backtest Artifacts ----
    backtests_dir = Path("backtests")
    backtests_dir.mkdir(parents=True, exist_ok=True)
 
    # 1. Performance metrics (JSON)
    metrics_record = {
        "v8_full_pipeline": mh_results["mh_metrics"],
        "allocator_only": mh_results["raw_metrics"],
        "equal_weight": mh_results["equal_metrics"],
        "spy_benchmark": mh_results["benchmark_metrics"],
        "slow_posture_distribution": mh_results["slow_posture_distribution"],
        "fast_override_rate": mh_results["fast_override_rate"],
        "fast_override_days": mh_results["fast_override_days"],
        "allocator_capital_weights": alloc_result.get("capital_weight_stats", {}),
        "snapshot_used": snapshot_name,
        "extended_snapshot_used": extended_snapshot_name if snap_ext else None,
    }
    with open(backtests_dir / "backtest_results.json", "w") as f:
        json.dump(metrics_record, f, indent=2, default=str)
    logger.info(f"Backtest metrics saved: backtests/backtest_results.json")
 
    # 2. Equity curve (CSV)
    mh_returns = mh_results["mh_returns"]
    spy_returns = mh_results.get("benchmark_metrics", {})
    benchmark_rets = adj_close["SPY"].pct_change().reindex(mh_returns.index)
 
    equity_df = pd.DataFrame({
        "portfolio_return": mh_returns,
        "spy_return": benchmark_rets,
        "portfolio_cumulative": (1 + mh_returns).cumprod(),
        "spy_cumulative": (1 + benchmark_rets.fillna(0)).cumprod(),
    })
    equity_df.index.name = "date"
    equity_df.to_csv(backtests_dir / "equity_curve.csv")
    logger.info(f"Equity curve saved: backtests/equity_curve.csv")
 
    # 3. Allocation history (CSV)
    alloc_df = alloc_result["allocator_results"].copy()
    # Add multi-horizon metadata if available
    mh_meta = mh_results.get("mh_metadata", pd.DataFrame())
    if not mh_meta.empty:
        # Extract slow layer posture
        if "slow_layer" in mh_meta.columns:
            alloc_posture = mh_meta["slow_layer"].apply(
                lambda x: x.get("risk_posture", "unknown") if isinstance(x, dict) else "unknown"
            )
            alloc_leverage = mh_meta["slow_layer"].apply(
                lambda x: x.get("leverage_ceiling", 1.0) if isinstance(x, dict) else 1.0
            )
        else:
            alloc_posture = pd.Series("unknown", index=mh_meta.index)
            alloc_leverage = pd.Series(1.0, index=mh_meta.index)
 
        if "fast_layer" in mh_meta.columns:
            fast_override = mh_meta["fast_layer"].apply(
                lambda x: x.get("override_active", False) if isinstance(x, dict) else False
            )
        else:
            fast_override = pd.Series(False, index=mh_meta.index)
 
        # Merge with allocator results on common dates
        mh_extra = pd.DataFrame({
            "slow_posture": alloc_posture,
            "leverage_ceiling": alloc_leverage,
            "fast_override": fast_override,
        })
        # Reindex to allocator dates
        mh_extra = mh_extra.reindex(alloc_df.index, method="ffill")
        for col in mh_extra.columns:
            if col not in alloc_df.columns:
                alloc_df[col] = mh_extra[col]
 
    alloc_df.index.name = "date"
    alloc_df.to_csv(backtests_dir / "allocation_history.csv")
    logger.info(f"Allocation history saved: backtests/allocation_history.csv")
 
    # 4. Equity curve plot (PNG)
    _plot_backtest(equity_df, alloc_result, mh_results, backtests_dir / "equity_curve.png")
    logger.info(f"Equity curve plot saved: backtests/equity_curve.png")
 
    print(f"\n--- Artifacts Saved ---")
    print(f"  backtests/backtest_results.json")
    print(f"  backtests/equity_curve.csv")
    print(f"  backtests/allocation_history.csv")
    print(f"  backtests/equity_curve.png")
 
    return {
        "snapshot": snap,
        "strategies": {"s1": s1, "s2": s2, "s3": s3},
        "strategy_outputs": strategy_outputs,
        "state": state,
        "allocator_result": alloc_result,
        "multi_horizon_results": mh_results,
        "stress_scores": stress_scores,
    }
 
 
# ---------------------------------------------------------------------------
# Backtest Visualization
# ---------------------------------------------------------------------------
 
def _plot_backtest(
    equity_df: pd.DataFrame,
    alloc_result: Dict[str, Any],
    mh_results: Dict[str, Any],
    save_path: Path,
) -> None:
    """
    Generate a 5-panel backtest visualization:
        Panel 1: Equity curve (portfolio vs SPY, log scale)
        Panel 2: Drawdown comparison
        Panel 3: Strategy capital allocation weights
        Panel 4: Slow layer posture + fast layer overrides
        Panel 5: Uncertainty / tightening level
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
 
    mh = mh_results["mh_metrics"]
    if "error" in mh:
        logger.warning("Cannot plot — insufficient metrics")
        return
 
    fig, axes = plt.subplots(
        5, 1, figsize=(16, 18),
        gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1, 1], "hspace": 0.08},
        sharex=True,
    )
 
    # ---- Panel 1: Equity Curve ----
    ax = axes[0]
    ax.plot(equity_df.index, equity_df["portfolio_cumulative"],
            color="#2ecc71", linewidth=1.2,
            label=f"V8 Portfolio (Sharpe: {mh['sharpe']:.2f})")
    ax.plot(equity_df.index, equity_df["spy_cumulative"],
            color="#3498db", linewidth=1.0, alpha=0.7, label="SPY")
    ax.set_ylabel("Cumulative Value ($1 start)", fontsize=11)
    ax.set_title(
        f"SPY Alpha v8 — Full Pipeline Backtest: "
        f"CAGR {mh['cagr']:.1%} | Sharpe {mh['sharpe']:.2f} | "
        f"Max DD {mh['max_dd']:.1%} | Calmar {mh['calmar']:.2f}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
 
    # ---- Panel 2: Drawdown ----
    ax = axes[1]
    port_cum = equity_df["portfolio_cumulative"]
    port_peak = port_cum.expanding().max()
    port_dd = (port_cum - port_peak) / port_peak
 
    spy_cum = equity_df["spy_cumulative"]
    spy_peak = spy_cum.expanding().max()
    spy_dd = (spy_cum - spy_peak) / spy_peak
 
    ax.fill_between(port_dd.index, port_dd.values,
                    color="#e74c3c", alpha=0.4, label=f"Portfolio DD (max: {mh['max_dd']:.1%})")
    ax.plot(spy_dd.index, spy_dd.values,
            color="#3498db", linewidth=0.8, alpha=0.5, label="SPY DD")
    ax.set_ylabel("Drawdown", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
 
    # ---- Panel 3: Strategy Capital Allocation ----
    ax = axes[2]
    alloc_df = alloc_result["allocator_results"]
    strategy_names = ["regime_allocator", "trend_cta", "defensive"]
    colors = {"regime_allocator": "#2ecc71", "trend_cta": "#3498db", "defensive": "#e74c3c"}
 
    for name in strategy_names:
        col = f"capital_weight_{name}"
        if col in alloc_df.columns:
            ax.fill_between(alloc_df.index, 0, alloc_df[col],
                          alpha=0.0)  # invisible, just for stacking
            ax.plot(alloc_df.index, alloc_df[col],
                   color=colors.get(name, "#999"), linewidth=0.8,
                   label=f"{name} ({alloc_df[col].mean():.0%} avg)")
 
    ax.set_ylabel("Capital Weight", fontsize=11)
    ax.set_ylim(0, 0.7)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
 
    # ---- Panel 4: Slow Layer Posture ----
    ax = axes[3]
    mh_meta = mh_results.get("mh_metadata", pd.DataFrame())
    if not mh_meta.empty and "slow_layer" in mh_meta.columns:
        posture_map = {"aggressive": 1.0, "balanced": 0.5, "defensive": 0.0}
        posture_colors = {"aggressive": "#2ecc71", "balanced": "#f39c12", "defensive": "#e74c3c"}
 
        posture_series = mh_meta["slow_layer"].apply(
            lambda x: x.get("risk_posture", "balanced") if isinstance(x, dict) else "balanced"
        )
        posture_numeric = posture_series.map(posture_map).fillna(0.5)
 
        ax.fill_between(posture_numeric.index, posture_numeric.values,
                       step="post", alpha=0.4, color="#2ecc71")
        ax.set_ylabel("Posture", fontsize=11)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["Defensive", "Balanced", "Aggressive"], fontsize=9)
 
        # Overlay fast layer overrides
        if "fast_layer" in mh_meta.columns:
            fast_overrides = mh_meta["fast_layer"].apply(
                lambda x: x.get("override_active", False) if isinstance(x, dict) else False
            )
            override_dates = fast_overrides[fast_overrides].index
            for d in override_dates:
                ax.axvline(d, color="#e74c3c", alpha=0.15, linewidth=0.5)
 
        ax.grid(True, alpha=0.3)
        ax.legend(["Strategic Posture", "Fast Layer Override"], loc="upper right", fontsize=9)
 
    # ---- Panel 5: Uncertainty / Tightening ----
    ax = axes[4]
    if not mh_meta.empty and "risk_engine" in mh_meta.columns:
        uncertainty = mh_meta["risk_engine"].apply(
            lambda x: x.get("uncertainty_score", 0) if isinstance(x, dict) else 0
        )
        tightening = mh_meta["risk_engine"].apply(
            lambda x: x.get("tightening_level", 0) if isinstance(x, dict) else 0
        )
 
        ax.fill_between(uncertainty.index, uncertainty.values,
                       alpha=0.4, color="#9b59b6", label="Uncertainty")
        ax.plot(tightening.index, tightening.values,
               color="#e74c3c", linewidth=0.8, alpha=0.7, label="Tightening")
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Date", fontsize=11)
    ax.grid(True, alpha=0.3)
 
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
 
 
# ---------------------------------------------------------------------------
# Daily Live Pipeline
# ---------------------------------------------------------------------------
 
def run_daily(
    snapshot_name: str = "baseline_v7",
    fred_api_key: Optional[str] = None,
    data_dir: Optional[str] = None,
    save_signal: bool = True,
    profile: str = "balanced",
) -> Dict[str, Any]:
    """
    Run the daily live inference pipeline.
 
    Uses the frozen snapshot for model training and fresh data for inference.
    Generates a signal with full attribution.
    """
    import os
    if fred_api_key:
        os.environ["FRED_API_KEY"] = fred_api_key
 
    mgr = SnapshotManager(data_dir=data_dir)
 
    # ---- Load snapshot for model training ----
    logger.info(f"Loading snapshot: {snapshot_name}")
    snap = mgr.load_snapshot(snapshot_name)
 
    # ---- Build strategies on snapshot ----
    logger.info("Building strategies on snapshot data...")
    s1 = RegimeAllocatorStrategy()
    s1.build(snap)
 
    s2 = TrendCTAStrategy()
    s2.build(snap)
 
    s3 = DefensiveStrategy()
    s3.build(snap)
 
    # ---- Fetch fresh data ----
    logger.info("Fetching fresh data for inference...")
    live_data = fetch_daily_live(fred_api_key=fred_api_key)
 
    # ---- Build state representation from fresh data ----
    logger.info("Building state representation from fresh data...")
    raw_close_live = live_data["raw_prices"]
    if isinstance(raw_close_live.columns, pd.MultiIndex):
        raw_close_live = raw_close_live["Close"]
 
    fred_live = live_data["fred_data"]
    stress_fred = live_data.get("stress_fred", None)
    vix_term = live_data.get("vix_term", None)
 
    regime_probs = s1.get_regime_probabilities()
 
    builder = StateRepresentationBuilder()
    state = builder.build(
        raw_close_live, fred_live,
        regime_probs=regime_probs,
        stress_fred=stress_fred,
        vix_term=vix_term,
    )
 
    # ---- Generate strategy signals for latest date ----
    # Use the last available signals from each strategy
    s1_outputs = s1.generate_signals(snap)
    s1_dates = pd.DatetimeIndex([
        pd.Timestamp(o.strategy_metadata["date"]) for o in s1_outputs
    ])
    s2_outputs = s2.generate_signals(snap, rebalance_dates=s1_dates)
    s3_outputs = s3.generate_signals(snap, rebalance_dates=s1_dates)
 
    latest_s1 = s1_outputs[-1] if s1_outputs else None
    latest_s2 = s2_outputs[-1] if s2_outputs else None
    latest_s3 = s3_outputs[-1] if s3_outputs else None
 
    # ---- Get allocator weights ----
    # Use equal weights as default (allocator needs full backtest to train)
    allocator_weights = {
        "regime_allocator": 0.35,
        "trend_cta": 0.35,
        "defensive": 0.30,
    }
 
    # ---- Build proposed blended weights ----
    blended = {}
    strategy_outputs_latest = {}
    for name, output, cap_w in [
        ("regime_allocator", latest_s1, allocator_weights["regime_allocator"]),
        ("trend_cta", latest_s2, allocator_weights["trend_cta"]),
        ("defensive", latest_s3, allocator_weights["defensive"]),
    ]:
        if output is None:
            continue
        strategy_outputs_latest[name] = output
        for asset, w in output.proposed_weights.items():
            blended[asset] = blended.get(asset, 0) + cap_w * w
 
 
    # ---- Process through multi-horizon coordinator ----
    # This is the SAME path as the backtest:
    #   posture bias -> slow layer bounds -> risk engine -> fast layer
    adj_close = get_adj_close(snap)
    spy_prices = adj_close["SPY"] if "SPY" in adj_close.columns else pd.Series()
    current_date = state.index[-1]

    stress = s3.get_stress_score()
    stress_val = float(stress.iloc[-1]) if stress is not None and len(stress) > 0 else 0.0

    strategy_w = {name: o.proposed_weights for name, o in strategy_outputs_latest.items()}

    coordinator = MultiHorizonCoordinator(profile=profile)
    final_weights, mh_meta = coordinator.process(
        proposed_weights=blended,
        state_features=state,
        spy_prices=spy_prices,
        current_date=current_date,
        strategy_weights=strategy_w,
        stress_score=stress_val,
        force_slow_update=True,
        force_medium_update=True,
    )
 
    # ---- Enforce portfolio constraints (match backtest behavior) ----
    # Step 1: Remove de minimis allocations (< 2%)
    final_weights = {k: v for k, v in final_weights.items() if v >= 0.02 or k == "SHY"}

    # Step 2: Renormalize
    total = sum(final_weights.values())
    if total > 0:
        final_weights = {k: v / total for k, v in final_weights.items()}

    # Step 3: If still above 12 assets, remove lowest until at cap
    MAX_PORTFOLIO_SIZE = 12
    while len(final_weights) > MAX_PORTFOLIO_SIZE:
        # Find lowest weight asset (never remove SHY)
        removable = {k: v for k, v in final_weights.items() if k != "SHY"}
        if not removable:
            break
        lowest = min(removable, key=removable.get)
        del final_weights[lowest]
        # Renormalize
        total = sum(final_weights.values())
        if total > 0:
            final_weights = {k: v / total for k, v in final_weights.items()}

    # ---- Get risk metadata from coordinator's last engine run ----
    risk_meta = mh_meta.get("risk_engine", {})

    # ---- Use conditional-weighted capital allocations for attribution ----
    cw = mh_meta.get("conditional_weighting", {})
    if cw:
        allocator_weights = {
            "regime_allocator": cw.get("adjusted_s1", allocator_weights.get("regime_allocator", 0.35)),
            "trend_cta": cw.get("adjusted_s2", allocator_weights.get("trend_cta", 0.35)),
            "defensive": cw.get("adjusted_s3", allocator_weights.get("defensive", 0.30)),
        }

    # ---- Build signal with attribution ----
    latest_state = state.iloc[-1]

    signal = build_daily_signal(
        date=current_date,
        final_weights=final_weights,
        state_features=latest_state,
        strategy_outputs=strategy_outputs_latest,
        allocator_weights=allocator_weights,
        risk_metadata=risk_meta,
        multi_horizon_metadata=mh_meta,
    )
 
    # ---- Print and save ----
    print_signal(signal)
 
    if save_signal:
        tracker = PredictionTracker()
        signal_path = tracker.save_signal(signal)
        tracker.append_to_history(signal)
        logger.info(f"Signal saved to {signal_path}")
 
    return {
        "signal": signal,
        "final_weights": final_weights,
        "risk_metadata": risk_meta,
        "mh_metadata": mh_meta,
    }
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
 
    parser = argparse.ArgumentParser(
        description="SPY Alpha v8 — Hierarchical Adaptive Multi-Strategy Allocator"
    )
    parser.add_argument("--fred-key", type=str, default=None, help="FRED API key")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
 
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
 
    # ---- Backtest ----
    bt_parser = subparsers.add_parser("backtest", help="Run full walk-forward backtest")
    bt_parser.add_argument("--snapshot", type=str, default="baseline_v7", help="Snapshot name")
    bt_parser.add_argument("--extended", type=str, default="baseline_v7_extended", help="Extended snapshot")
    bt_parser.add_argument("--profile", type=str, default="balanced", choices=["aggressive", "balanced", "defensive"], help="Risk profile")
 
    # ---- Daily ----
    daily_parser = subparsers.add_parser("daily", help="Run daily live inference")
    daily_parser.add_argument("--snapshot", type=str, default="baseline_v7", help="Snapshot name")
    daily_parser.add_argument("--profile", type=str, default="balanced", choices=["aggressive", "balanced", "defensive"], help="Risk profile")
    daily_parser.add_argument("--no-save", action="store_true", help="Don't save signal")

    # ---- Execute ----
    exec_parser = subparsers.add_parser("execute", help="Execute rebalance via Alpaca")
    exec_parser.add_argument("--dry-run", action="store_true", help="Compute orders without submitting")
    exec_parser.add_argument("--signal", type=str, default="signals/latest_prediction.json", help="Signal file path")
    exec_parser.add_argument("--summary", action="store_true", help="Show current portfolio summary only")
 
    # ---- Snapshot ----
    snap_parser = subparsers.add_parser("snapshot", help="Create/manage snapshots")
    snap_parser.add_argument("--action", choices=["create", "list", "compare"], default="list")
    snap_parser.add_argument("--name", type=str, help="Snapshot name")
    snap_parser.add_argument("--compare-to", type=str, help="Second snapshot for comparison")
    snap_parser.add_argument("--start", type=str, default="2005-01-01")
    snap_parser.add_argument("--end", type=str, default=None)
    snap_parser.add_argument("--overwrite", action="store_true")
 
    args = parser.parse_args()
 
    if args.command is None:
        parser.print_help()
        sys.exit(1)
 
    # Set FRED key in environment
    if args.fred_key:
        import os
        os.environ["FRED_API_KEY"] = args.fred_key
 
    if args.command == "backtest":
        run_backtest(
            snapshot_name=args.snapshot,
            extended_snapshot_name=args.extended,
            fred_api_key=args.fred_key,
            data_dir=args.data_dir,
            profile=args.profile,
        )
 
    elif args.command == "daily":
        run_daily(
            snapshot_name=args.snapshot,
            fred_api_key=args.fred_key,
            data_dir=args.data_dir,
            save_signal=not args.no_save,
            profile=args.profile,
        )
 
    elif args.command == "snapshot":
        mgr = SnapshotManager(data_dir=args.data_dir)
 
        if args.action == "create":
            if not args.name:
                print("Error: --name required for snapshot creation")
                sys.exit(1)
            path = mgr.create_snapshot(
                name=args.name,
                start=args.start,
                end=args.end,
                fred_api_key=args.fred_key,
                overwrite=args.overwrite,
            )
            print(f"\nSnapshot created: {path}")
 
        elif args.action == "list":
            snapshots = mgr.list_snapshots()
            if not snapshots:
                print("\nNo snapshots found.")
            else:
                print(f"\n{len(snapshots)} snapshot(s):\n")
                for s in snapshots:
                    print(
                        f"  {s['snapshot_name']:30s}  "
                        f"{s.get('actual_start', '?')} -> {s.get('actual_end', '?')}  "
                        f"({s.get('trading_days', '?')} days)"
                    )
 
        elif args.action == "compare":
            if not args.name or not args.compare_to:
                print("Error: --name and --compare-to required")
                sys.exit(1)
            report = mgr.compare_snapshots(args.name, args.compare_to)
            print(json.dumps(report, indent=2))
 

    elif args.command == "execute":
        import os
        if args.fred_key:
            os.environ["FRED_API_KEY"] = args.fred_key
        os.environ.setdefault("ALPACA_API_KEY", "")
        os.environ.setdefault("ALPACA_SECRET_KEY", "")

        from execution_engine import AlpacaExecutor, print_execution_report

        executor = AlpacaExecutor()

        if args.summary:
            summary = executor.get_portfolio_summary()
            print(f"\nPortfolio Value: ${summary['portfolio_value']:,.2f}")
            print(f"Cash: ${summary['cash']:,.2f}")
            print(f"Positions: {summary['n_positions']}")
            for h in summary["holdings"]:
                print(f"  {h['symbol']:<8s} ${h['market_value']:>10,.2f} ({h['weight']:>6.1%})  P&L: {h['unrealized_plpc']:>+.1%}")
        else:
            report = executor.execute_rebalance(
                signal_path=Path(args.signal),
                dry_run=args.dry_run,
            )
            print_execution_report(report)

if __name__ == "__main__":
    main()