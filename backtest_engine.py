"""
SPY Alpha v7 — Backtest Engine & Performance Analytics
========================================================

Walk-forward backtesting with 5-regime attribution, dynamic asset tracking,
and asset rotation analysis.

Changes from v6:
    - 5-regime attribution (Bull, Slowdown, Crisis-Deflation, Crisis-Inflation, Inflation)
    - Dynamic asset universe tracking (which assets held over time)
    - Stock vs ETF selection rate tracking
    - Asset rotation heatmap
    - Extended weight panel for TLT/GLD overlays
    - All 27+ performance metrics preserved
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("spy_alpha_v8.backtest_engine")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TRANSACTION_COST_BPS = 5
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04


# ---------------------------------------------------------------------------
# Backtest Result Container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for complete backtest output."""
    metrics: Dict[str, float]

    equity_curve: pd.Series
    benchmark_curve: pd.Series
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series

    weight_history: pd.DataFrame
    regime_history: pd.Series
    upro_history: pd.Series
    shy_history: pd.Series
    tlt_history: pd.Series
    gld_history: pd.Series
    turnover_history: pd.Series

    drawdown_series: pd.Series
    benchmark_drawdown: pd.Series

    regime_metrics: Dict[str, Dict[str, float]]
    asset_frequency: Dict[str, int]          # how often each asset selected
    n_stocks_history: pd.Series              # stocks held per rebalance

    config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core Backtest Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Walk-forward backtest engine for v7's dynamic asset universe."""

    def __init__(
        self,
        transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
        risk_free_rate: float = RISK_FREE_RATE,
        benchmark_ticker: str = "SPY",
    ):
        self.transaction_cost_bps = transaction_cost_bps
        self.risk_free_rate = risk_free_rate
        self.benchmark_ticker = benchmark_ticker

    def run(
        self,
        allocations: List[Any],
        adj_close: pd.DataFrame,
    ) -> BacktestResult:
        """Execute the backtest."""
        if not allocations:
            raise ValueError("No allocations provided for backtest")

        logger.info(f"Running backtest: {len(allocations)} allocation periods")

        # ---- Extract allocation history ----
        alloc_dates = [pd.Timestamp(a.allocation_date) for a in allocations]
        weight_records = []
        regime_records = []
        upro_records, shy_records, tlt_records, gld_records = [], [], [], []
        turnover_records = []
        n_stocks_records = []
        asset_freq: Dict[str, int] = {}

        for a in allocations:
            dt = pd.Timestamp(a.allocation_date)
            weight_records.append({"date": dt, **a.weights.to_dict()})
            regime_records.append({"date": dt, "regime": a.dominant_regime})
            upro_records.append({"date": dt, "upro": a.upro_weight})
            shy_records.append({"date": dt, "shy": a.shy_weight})
            tlt_records.append({"date": dt, "tlt": a.tlt_weight})
            gld_records.append({"date": dt, "gld": a.gld_weight})
            turnover_records.append({"date": dt, "turnover": a.turnover})
            n_stocks_records.append({"date": dt, "n_stocks": getattr(a, "n_stocks", 0)})

            for asset in a.selected_assets:
                asset_freq[asset] = asset_freq.get(asset, 0) + 1

        weight_history = pd.DataFrame(weight_records).set_index("date").fillna(0)
        regime_history = pd.Series({r["date"]: r["regime"] for r in regime_records}, name="regime")
        upro_history = pd.Series({r["date"]: r["upro"] for r in upro_records}, name="upro")
        shy_history = pd.Series({r["date"]: r["shy"] for r in shy_records}, name="shy")
        tlt_history = pd.Series({r["date"]: r["tlt"] for r in tlt_records}, name="tlt")
        gld_history = pd.Series({r["date"]: r["gld"] for r in gld_records}, name="gld")
        turnover_history = pd.Series({r["date"]: r["turnover"] for r in turnover_records}, name="turnover")
        n_stocks_history = pd.Series({r["date"]: r["n_stocks"] for r in n_stocks_records}, name="n_stocks")

        # ---- Compute daily returns ----
        daily_returns = adj_close.pct_change().dropna(how="all")

        portfolio_returns = self._compute_portfolio_returns(
            weight_history, daily_returns, alloc_dates
        )

        tc_series = self._compute_transaction_costs(turnover_history, alloc_dates, portfolio_returns.index)
        portfolio_returns_net = portfolio_returns - tc_series

        if self.benchmark_ticker in daily_returns.columns:
            benchmark_returns = daily_returns[self.benchmark_ticker].reindex(
                portfolio_returns_net.index
            ).fillna(0)
        else:
            benchmark_returns = pd.Series(0, index=portfolio_returns_net.index)

        equity_curve = (1 + portfolio_returns_net).cumprod()
        benchmark_curve = (1 + benchmark_returns).cumprod()

        drawdown_series = self._compute_drawdown(equity_curve)
        benchmark_drawdown = self._compute_drawdown(benchmark_curve)

        metrics = self._compute_metrics(
            portfolio_returns_net, benchmark_returns, equity_curve, benchmark_curve
        )

        regime_metrics = self._regime_attribution(
            portfolio_returns_net, benchmark_returns, regime_history, alloc_dates
        )

        logger.info(
            f"Backtest complete: Sharpe={metrics['sharpe']:.2f}, "
            f"Total Return={metrics['total_return']:.1%}, "
            f"Max DD={metrics['max_drawdown']:.1%}"
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            portfolio_returns=portfolio_returns_net,
            benchmark_returns=benchmark_returns,
            weight_history=weight_history,
            regime_history=regime_history,
            upro_history=upro_history,
            shy_history=shy_history,
            tlt_history=tlt_history,
            gld_history=gld_history,
            turnover_history=turnover_history,
            drawdown_series=drawdown_series,
            benchmark_drawdown=benchmark_drawdown,
            regime_metrics=regime_metrics,
            asset_frequency=asset_freq,
            n_stocks_history=n_stocks_history,
        )

    # ------------------------------------------------------------------
    # Portfolio Return Computation
    # ------------------------------------------------------------------

    def _compute_portfolio_returns(
        self,
        weight_history: pd.DataFrame,
        daily_returns: pd.DataFrame,
        alloc_dates: List[pd.Timestamp],
    ) -> pd.Series:
        """Compute daily portfolio returns using allocation weights."""
        start_date = alloc_dates[0]
        all_dates = daily_returns.index[daily_returns.index >= start_date]

        portfolio_returns = pd.Series(0.0, index=all_dates, name="portfolio")
        tickers_in_weights = weight_history.columns.tolist()

        for i in range(len(alloc_dates)):
            period_start = alloc_dates[i]

            if i + 1 < len(alloc_dates):
                period_end = alloc_dates[i + 1]
                period_mask = (all_dates >= period_start) & (all_dates < period_end)
            else:
                period_mask = all_dates >= period_start

            period_dates = all_dates[period_mask]
            if len(period_dates) == 0:
                continue

            weights = weight_history.loc[alloc_dates[i]]

            for ticker in tickers_in_weights:
                w = weights.get(ticker, 0)
                if w == 0 or ticker not in daily_returns.columns:
                    continue

                ticker_ret = daily_returns[ticker].reindex(period_dates).fillna(0)

                # For UPRO, use actual UPRO returns if available
                if ticker == "UPRO" and "UPRO" in daily_returns.columns:
                    ticker_ret = daily_returns["UPRO"].reindex(period_dates).fillna(0)
                elif ticker == "UPRO" and "SPY" in daily_returns.columns:
                    spy_ret = daily_returns["SPY"].reindex(period_dates).fillna(0)
                    ticker_ret = spy_ret * 3.0

                portfolio_returns.loc[period_dates] += w * ticker_ret

        return portfolio_returns

    def _compute_transaction_costs(
        self,
        turnover_history: pd.Series,
        alloc_dates: List[pd.Timestamp],
        all_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """Compute daily transaction cost series."""
        tc = pd.Series(0.0, index=all_dates)
        cost_rate = self.transaction_cost_bps / 10_000

        for date in alloc_dates:
            if date in turnover_history.index and date in tc.index:
                turnover = turnover_history[date]
                tc[date] = turnover * cost_rate

        return tc

    # ------------------------------------------------------------------
    # Performance Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        port_returns: pd.Series,
        bench_returns: pd.Series,
        equity_curve: pd.Series,
        bench_curve: pd.Series,
    ) -> Dict[str, float]:
        """Compute comprehensive performance metrics (27+)."""
        n_days = len(port_returns)
        n_years = n_days / TRADING_DAYS_PER_YEAR

        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        bench_total = float(bench_curve.iloc[-1] / bench_curve.iloc[0] - 1)
        cagr = float((1 + total_return) ** (1 / max(n_years, 0.01)) - 1)
        bench_cagr = float((1 + bench_total) ** (1 / max(n_years, 0.01)) - 1)

        ann_vol = float(port_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        bench_vol = float(bench_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        downside_returns = port_returns[port_returns < 0]
        downside_vol = float(
            downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        ) if len(downside_returns) > 0 else 0.001

        daily_rf = self.risk_free_rate / TRADING_DAYS_PER_YEAR
        excess_returns = port_returns - daily_rf

        sharpe = float(
            excess_returns.mean() / max(port_returns.std(), 1e-8) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        sortino = float(
            excess_returns.mean() / max(downside_vol / np.sqrt(TRADING_DAYS_PER_YEAR), 1e-8)
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        ) if downside_vol > 0 else 0.0

        dd = self._compute_drawdown(equity_curve)
        max_drawdown = float(dd.min())
        bench_dd = self._compute_drawdown(bench_curve)
        bench_max_dd = float(bench_dd.min())

        avg_drawdown = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
        max_dd_duration = self._max_drawdown_duration(equity_curve)

        calmar = float(cagr / abs(max_drawdown)) if max_drawdown != 0 else 0.0

        win_rate = float((port_returns > 0).mean())

        gross_profits = port_returns[port_returns > 0].sum()
        gross_losses = abs(port_returns[port_returns < 0].sum())
        profit_factor = float(gross_profits / max(gross_losses, 1e-8))

        var_95 = float(np.percentile(port_returns, 5))
        cvar_95 = float(port_returns[port_returns <= var_95].mean()) if (port_returns <= var_95).any() else var_95
        skewness = float(port_returns.skew())
        kurtosis = float(port_returns.kurt())

        excess_vs_bench = cagr - bench_cagr
        tracking_error = float(
            (port_returns - bench_returns).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        information_ratio = float(excess_vs_bench / max(tracking_error, 1e-8))

        cov_matrix = np.cov(port_returns.values, bench_returns.values)
        beta = float(cov_matrix[0, 1] / max(cov_matrix[1, 1], 1e-8))
        alpha_ann = float(cagr - (self.risk_free_rate + beta * (bench_cagr - self.risk_free_rate)))

        return {
            "total_return": total_return,
            "cagr": cagr,
            "benchmark_total_return": bench_total,
            "benchmark_cagr": bench_cagr,
            "excess_return": excess_vs_bench,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "information_ratio": information_ratio,
            "annualized_volatility": ann_vol,
            "benchmark_volatility": bench_vol,
            "downside_volatility": downside_vol,
            "max_drawdown": max_drawdown,
            "benchmark_max_drawdown": bench_max_dd,
            "avg_drawdown": avg_drawdown,
            "max_drawdown_duration_days": max_dd_duration,
            "beta": beta,
            "alpha_annualized": alpha_ann,
            "tracking_error": tracking_error,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "var_95_daily": var_95,
            "cvar_95_daily": cvar_95,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "n_trading_days": n_days,
            "n_years": round(n_years, 2),
        }

    def _compute_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        peak = equity_curve.expanding().max()
        return (equity_curve - peak) / peak

    def _max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        peak = equity_curve.expanding().max()
        in_drawdown = equity_curve < peak
        max_duration = 0
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        return max_duration

    # ------------------------------------------------------------------
    # Regime Attribution (5-regime support)
    # ------------------------------------------------------------------

    def _regime_attribution(
        self,
        port_returns: pd.Series,
        bench_returns: pd.Series,
        regime_history: pd.Series,
        alloc_dates: List[pd.Timestamp],
    ) -> Dict[str, Dict[str, float]]:
        """Break down performance by regime (supports 4 or 5 regimes)."""
        daily_regime = pd.Series(index=port_returns.index, dtype=str)

        for i in range(len(alloc_dates)):
            start = alloc_dates[i]
            end = alloc_dates[i + 1] if i + 1 < len(alloc_dates) else port_returns.index[-1]
            if start in regime_history.index:
                regime = regime_history[start]
                mask = (port_returns.index >= start) & (port_returns.index <= end)
                daily_regime[mask] = regime

        daily_regime = daily_regime.dropna()

        results = {}
        for regime in daily_regime.unique():
            if pd.isna(regime):
                continue

            mask = daily_regime == regime
            r_port = port_returns[mask]
            r_bench = bench_returns.reindex(r_port.index).fillna(0)

            n_days = len(r_port)
            if n_days < 5:
                continue

            ann_ret = float(r_port.mean() * TRADING_DAYS_PER_YEAR)
            ann_vol = float(r_port.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            bench_ann_ret = float(r_bench.mean() * TRADING_DAYS_PER_YEAR)

            sharpe = float(
                (r_port.mean() - self.risk_free_rate / TRADING_DAYS_PER_YEAR)
                / max(r_port.std(), 1e-8) * np.sqrt(TRADING_DAYS_PER_YEAR)
            )

            results[regime] = {
                "n_days": n_days,
                "pct_of_total": n_days / len(port_returns),
                "annualized_return": ann_ret,
                "annualized_volatility": ann_vol,
                "sharpe": sharpe,
                "benchmark_return": bench_ann_ret,
                "excess_return": ann_ret - bench_ann_ret,
                "win_rate": float((r_port > 0).mean()),
            }

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, result: BacktestResult) -> None:
        """Print comprehensive backtest report."""
        m = result.metrics

        print(f"\n{'='*70}")
        print(f"SPY ALPHA v7 — BACKTEST PERFORMANCE REPORT")
        print(f"{'='*70}")

        print(f"\n--- Return Summary ---")
        print(f"  Period:              {m['n_years']:.1f} years ({m['n_trading_days']} trading days)")
        print(f"  Total Return:        {m['total_return']:>8.1%}    (Benchmark: {m['benchmark_total_return']:>8.1%})")
        print(f"  CAGR:                {m['cagr']:>8.1%}    (Benchmark: {m['benchmark_cagr']:>8.1%})")
        print(f"  Excess Return:       {m['excess_return']:>8.1%}")

        print(f"\n--- Risk-Adjusted Metrics ---")
        print(f"  Sharpe Ratio:        {m['sharpe']:>8.2f}")
        print(f"  Sortino Ratio:       {m['sortino']:>8.2f}")
        print(f"  Calmar Ratio:        {m['calmar']:>8.2f}")
        print(f"  Information Ratio:   {m['information_ratio']:>8.2f}")

        print(f"\n--- Risk Metrics ---")
        print(f"  Annualized Vol:      {m['annualized_volatility']:>8.1%}    (Benchmark: {m['benchmark_volatility']:>8.1%})")
        print(f"  Downside Vol:        {m['downside_volatility']:>8.1%}")
        print(f"  Max Drawdown:        {m['max_drawdown']:>8.1%}    (Benchmark: {m['benchmark_max_drawdown']:>8.1%})")
        print(f"  Avg Drawdown:        {m['avg_drawdown']:>8.1%}")
        print(f"  Max DD Duration:     {m['max_drawdown_duration_days']:>5d} days")

        print(f"\n--- Alpha / Beta ---")
        print(f"  Beta:                {m['beta']:>8.2f}")
        print(f"  Alpha (ann):         {m['alpha_annualized']:>8.1%}")
        print(f"  Tracking Error:      {m['tracking_error']:>8.1%}")

        print(f"\n--- Trade Statistics ---")
        print(f"  Win Rate:            {m['win_rate']:>8.1%}")
        print(f"  Profit Factor:       {m['profit_factor']:>8.2f}")
        print(f"  VaR (95%, daily):    {m['var_95_daily']:>8.2%}")
        print(f"  CVaR (95%, daily):   {m['cvar_95_daily']:>8.2%}")
        print(f"  Skewness:            {m['skewness']:>8.2f}")
        print(f"  Kurtosis:            {m['kurtosis']:>8.2f}")

        # ---- Regime attribution ----
        print(f"\n--- Regime Attribution ---")
        print(f"  {'Regime':<20s} {'Days':>6s} {'% Time':>7s} {'Ann Ret':>8s} {'Sharpe':>7s} {'vs Bench':>9s}")
        print(f"  {'-'*57}")
        for regime, stats in sorted(result.regime_metrics.items()):
            print(
                f"  {regime:<20s} {stats['n_days']:>6d} "
                f"{stats['pct_of_total']:>6.1%} "
                f"{stats['annualized_return']:>7.1%} "
                f"{stats['sharpe']:>7.2f} "
                f"{stats['excess_return']:>+8.1%}"
            )

        # ---- Overlay usage ----
        print(f"\n--- Overlay Instrument Usage ---")
        print(f"  Mean UPRO weight:    {result.upro_history.mean():>8.1%}")
        print(f"  Max UPRO weight:     {result.upro_history.max():>8.1%}")
        print(f"  Mean SHY weight:     {result.shy_history.mean():>8.1%}")
        print(f"  Mean TLT weight:     {result.tlt_history.mean():>8.1%}")
        print(f"  Mean GLD weight:     {result.gld_history.mean():>8.1%}")
        print(f"  Mean turnover:       {result.turnover_history.mean():>8.1%}")
        print(f"  Mean stocks held:    {result.n_stocks_history.mean():>8.1f}")

        # ---- Asset frequency ----
        print(f"\n--- Asset Selection Frequency (top 15) ---")
        n_periods = len(result.upro_history)
        sorted_freq = sorted(result.asset_frequency.items(), key=lambda x: -x[1])
        for asset, count in sorted_freq[:15]:
            pct = count / max(n_periods, 1)
            print(f"  {asset:<8s} {count:>5d} ({pct:>5.1%})")

        print(f"\n{'='*70}")

    def plot_equity_curve(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None,
        figsize: tuple = (16, 16),
    ) -> None:
        """Plot equity curve, drawdown, regime, and overlay weights."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(
            5, 1, figsize=figsize,
            gridspec_kw={"height_ratios": [3, 1.5, 1, 1, 1], "hspace": 0.08},
            sharex=True,
        )

        m = result.metrics

        regime_colors = {
            "Bull": "#2ecc71", "Slowdown": "#f39c12",
            "Crisis": "#e74c3c", "Crisis-Deflation": "#e74c3c",
            "Crisis-Inflation": "#c0392b", "Inflation": "#9b59b6",
        }

        # ---- Panel 1: Equity Curve ----
        ax = axes[0]
        ax.plot(result.equity_curve.index, result.equity_curve.values,
                color="#2ecc71", linewidth=1.2, label=f"Portfolio (Sharpe: {m['sharpe']:.2f})")
        ax.plot(result.benchmark_curve.index, result.benchmark_curve.values,
                color="#3498db", linewidth=1.0, alpha=0.7, label="SPY")
        ax.set_ylabel("Cumulative Value ($1 start)", fontsize=11)
        ax.set_title(
            f"SPY Alpha v7 — Backtest: CAGR {m['cagr']:.1%} | "
            f"Sharpe {m['sharpe']:.2f} | Max DD {m['max_drawdown']:.1%}",
            fontsize=13, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        dates = sorted(result.regime_history.index)
        for i in range(len(dates)):
            start = dates[i]
            end = dates[i + 1] if i + 1 < len(dates) else result.equity_curve.index[-1]
            regime = result.regime_history[start]
            color = regime_colors.get(regime, "#cccccc")
            ax.axvspan(start, end, alpha=0.08, color=color, linewidth=0)

        # ---- Panel 2: Drawdown ----
        ax = axes[1]
        ax.fill_between(result.drawdown_series.index, result.drawdown_series.values,
                        color="#e74c3c", alpha=0.4, label="Portfolio DD")
        ax.plot(result.benchmark_drawdown.index, result.benchmark_drawdown.values,
                color="#3498db", linewidth=0.8, alpha=0.5, label="SPY DD")
        ax.set_ylabel("Drawdown", fontsize=11)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # ---- Panel 3: UPRO / SHY weights ----
        ax = axes[2]
        ax.fill_between(result.upro_history.index, result.upro_history.values,
                        step="post", color="#2ecc71", alpha=0.5, label="UPRO")
        ax.fill_between(result.shy_history.index, result.shy_history.values,
                        step="post", color="#3498db", alpha=0.5, label="SHY")
        ax.set_ylabel("Weight", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # ---- Panel 4: TLT / GLD weights ----
        ax = axes[3]
        ax.fill_between(result.tlt_history.index, result.tlt_history.values,
                        step="post", color="#e67e22", alpha=0.5, label="TLT")
        ax.fill_between(result.gld_history.index, result.gld_history.values,
                        step="post", color="#f1c40f", alpha=0.5, label="GLD")
        ax.set_ylabel("Weight", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # ---- Panel 5: Turnover ----
        ax = axes[4]
        ax.bar(result.turnover_history.index, result.turnover_history.values,
               width=15, color="#9b59b6", alpha=0.6, label="Turnover")
        ax.axhline(result.turnover_history.mean(), color="#e74c3c", linestyle="--",
                    alpha=0.6, label=f"Mean: {result.turnover_history.mean():.1%}")
        ax.set_ylabel("Turnover", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def save_report(self, result: BacktestResult, path: str) -> Path:
        """Save backtest results to JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "metrics": result.metrics,
            "regime_metrics": result.regime_metrics,
            "asset_frequency": result.asset_frequency,
            "equity_curve_start": float(result.equity_curve.iloc[0]),
            "equity_curve_end": float(result.equity_curve.iloc[-1]),
            "date_range": {
                "start": str(result.equity_curve.index[0].date()),
                "end": str(result.equity_curve.index[-1].date()),
            },
            "overlay_stats": {
                "upro_mean": float(result.upro_history.mean()),
                "upro_max": float(result.upro_history.max()),
                "shy_mean": float(result.shy_history.mean()),
                "tlt_mean": float(result.tlt_history.mean()),
                "gld_mean": float(result.gld_history.mean()),
            },
            "turnover_stats": {
                "mean": float(result.turnover_history.mean()),
                "max": float(result.turnover_history.max()),
            },
        }

        with open(output, "w") as f:
            json.dump(record, f, indent=2, default=str)

        logger.info(f"Report saved: {output}")
        return output