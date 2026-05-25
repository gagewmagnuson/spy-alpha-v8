"""
SPY Alpha v7 — Asset Selector
================================

NEW module in v7. Separates asset selection ("what to hold") from
portfolio optimization ("how much to hold").

Selection Pipeline (every rebalance):
    1. Score all ~33 assets using forecaster expected returns
    2. Risk-adjust scores (Sharpe-like: return / volatility)
    3. Apply correlation penalty (prevents XLK + SMH + NVDA + MSFT)
    4. Apply hard constraints (max stocks, max same-sector, min macro)
    5. Select top 8-11 assets

UPRO and SHY are NOT part of the selection pipeline — they are overlay
instruments sized by regime logic in the portfolio optimizer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_pipeline import (
    TIER1_MACRO,
    TIER4_STOCKS,
    STOCK_SECTOR_MAP,
)
from return_forecaster import AssetForecast

logger = logging.getLogger("spy_alpha_v8.asset_selector")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Selection constraints
MAX_PORTFOLIO_SIZE = 11
MIN_PORTFOLIO_SIZE = 8
MAX_INDIVIDUAL_STOCKS = 3
MAX_SAME_SECTOR_ASSETS = 2   # e.g., max 2 of {XLK, SMH, NVDA, MSFT, AAPL}
MIN_MACRO_ASSETS = 1          # At least 1 of {SPY, QQQ, IWM, VWO, TLT, GLD}

# Overlay instruments — excluded from selection, sized separately
# V8: UPRO and SHY are no longer overlay instruments — they participate
# in normal selection. Leverage and defense are handled by the meta-allocator
# and risk constraint layer.
OVERLAY_INSTRUMENTS: set = set()

# Macro core assets (eligible for min_macro constraint)
MACRO_ASSETS = {"SPY", "QQQ", "IWM", "VWO", "TLT", "GLD"}

# Correlation penalty coefficient
CORRELATION_PENALTY = 0.2

# Sector grouping for same-sector constraint
# Maps each asset to its "sector group" for concentration limiting
SECTOR_GROUP_MAP: Dict[str, str] = {
    # Technology group
    "XLK": "tech", "SMH": "tech", "AAPL": "tech", "NVDA": "tech", "MSFT": "tech",
    # Consumer Discretionary group
    "XLY": "cons_disc", "AMZN": "cons_disc", "TSLA": "cons_disc",
    # Communication group
    "XLC": "comm", "META": "comm", "GOOGL": "comm",
    # Financials group
    "XLF": "financials", "JPM": "financials",
    # Healthcare group
    "XLV": "healthcare", "XBI": "healthcare", "LLY": "healthcare", "UNH": "healthcare",
    # Energy group
    "XLE": "energy", "XOM": "energy",
    # Industrials group
    "XLI": "industrials", "CAT": "industrials",
    # Others — each its own group
    "XLB": "materials", "XME": "metals_mining",
    "XLRE": "real_estate", "XLU": "utilities", "XLP": "cons_staples",
    # Macro assets — each its own group
    "SPY": "spy", "QQQ": "qqq", "IWM": "iwm", "VWO": "vwo",
    "TLT": "tlt", "GLD": "gld",
}


# ---------------------------------------------------------------------------
# Selection Result
# ---------------------------------------------------------------------------

@dataclass
class AssetSelection:
    """Container for asset selection output."""
    selected_assets: List[str]
    scores: pd.Series                  # risk-adjusted scores for all candidates
    adjusted_scores: pd.Series         # scores after correlation penalty
    n_stocks: int                      # how many individual stocks selected
    n_etfs: int                        # how many ETFs selected
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Asset Selector
# ---------------------------------------------------------------------------

class AssetSelector:
    """
    Selects top 8-11 assets from the ~33 asset universe.

    V8: UPRO and SHY now participate in normal selection.
    Leverage and defense decisions are deferred to the meta-allocator
    and risk constraint layer.
    """

    def __init__(
        self,
        max_size: int = MAX_PORTFOLIO_SIZE,
        min_size: int = MIN_PORTFOLIO_SIZE,
        max_stocks: int = MAX_INDIVIDUAL_STOCKS,
        max_same_sector: int = MAX_SAME_SECTOR_ASSETS,
        min_macro: int = MIN_MACRO_ASSETS,
        corr_penalty: float = CORRELATION_PENALTY,
        vol_lookback: int = 63,
    ):
        self.max_size = max_size
        self.min_size = min_size
        self.max_stocks = max_stocks
        self.max_same_sector = max_same_sector
        self.min_macro = min_macro
        self.corr_penalty = corr_penalty
        self.vol_lookback = vol_lookback

    def select(
        self,
        forecast: AssetForecast,
        raw_close: pd.DataFrame,
        adj_close: pd.DataFrame,
    ) -> AssetSelection:
        """
        Run the full selection pipeline.

        Steps:
            1. Score all assets (expected return from forecaster)
            2. Risk-adjust (return / volatility)
            3. Correlation penalty
            4. Constraint enforcement
            5. Select top 8-11
        """
        # Get all candidate assets (exclude overlays)
        candidates = [
            t for t in forecast.expected_returns.index
            if t not in OVERLAY_INSTRUMENTS and pd.notna(forecast.expected_returns[t])
        ]

        if not candidates:
            logger.warning("No candidate assets for selection")
            return AssetSelection(
                selected_assets=[], scores=pd.Series(dtype=float),
                adjusted_scores=pd.Series(dtype=float),
                n_stocks=0, n_etfs=0,
            )

        # ---- Step 1: Raw scores ----
        raw_scores = forecast.expected_returns[candidates].copy()

        # ---- Step 2: Risk-adjust ----
        risk_adjusted = self._risk_adjust(raw_scores, adj_close, candidates)

        # ---- Step 3: Correlation penalty ----
        adjusted = self._apply_correlation_penalty(risk_adjusted, adj_close, candidates)

        # ---- Step 4 & 5: Constrained selection ----
        selected = self._constrained_select(adjusted, candidates)

        n_stocks = sum(1 for t in selected if t in TIER4_STOCKS)
        n_etfs = len(selected) - n_stocks

        logger.info(
            f"Selected {len(selected)} assets: {n_etfs} ETFs + {n_stocks} stocks"
        )
        logger.info(f"  Selected: {selected}")

        return AssetSelection(
            selected_assets=selected,
            scores=risk_adjusted,
            adjusted_scores=adjusted,
            n_stocks=n_stocks,
            n_etfs=n_etfs,
            metadata={
                "n_candidates": len(candidates),
                "max_score": float(adjusted.max()),
                "min_score": float(adjusted.min()),
            },
        )

    def _risk_adjust(
        self,
        scores: pd.Series,
        adj_close: pd.DataFrame,
        candidates: List[str],
    ) -> pd.Series:
        """Risk-adjust scores: expected_return / volatility."""
        risk_adjusted = scores.copy()

        for ticker in candidates:
            if ticker not in adj_close.columns:
                continue

            returns = adj_close[ticker].pct_change().dropna().tail(self.vol_lookback)
            if len(returns) < 20:
                continue

            vol = returns.std() * np.sqrt(252)
            if vol > 0:
                risk_adjusted[ticker] = scores[ticker] / vol

        return risk_adjusted

    def _apply_correlation_penalty(
        self,
        scores: pd.Series,
        adj_close: pd.DataFrame,
        candidates: List[str],
    ) -> pd.Series:
        """
        Penalize assets highly correlated with higher-scoring assets.

        This prevents selecting XLK + SMH + NVDA + MSFT simultaneously.
        """
        adjusted = scores.copy()

        # Sort by score descending — process highest first
        sorted_assets = scores.sort_values(ascending=False).index.tolist()

        # Compute correlation matrix
        available = [t for t in candidates if t in adj_close.columns]
        if len(available) < 2:
            return adjusted

        returns = adj_close[available].pct_change().dropna().tail(self.vol_lookback)
        if len(returns) < 20:
            return adjusted

        corr_matrix = returns.corr()

        # Track "already selected" in order of score
        selected_so_far = []
        for asset in sorted_assets:
            if asset not in corr_matrix.index:
                continue

            if not selected_so_far:
                selected_so_far.append(asset)
                continue

            # Average correlation with already-selected assets
            corrs = []
            for sel in selected_so_far:
                if sel in corr_matrix.columns and asset in corr_matrix.index:
                    corrs.append(abs(corr_matrix.loc[asset, sel]))

            if corrs:
                avg_corr = np.mean(corrs)
                adjusted[asset] -= self.corr_penalty * avg_corr

            selected_so_far.append(asset)

        return adjusted

    def _constrained_select(
        self,
        scores: pd.Series,
        candidates: List[str],
    ) -> List[str]:
        """
        Select top assets while respecting constraints:
            - Max 3 individual stocks
            - Max 2 from same sector group
            - At least 1 macro asset
            - Portfolio size 8-11
        """
        sorted_assets = scores.sort_values(ascending=False).index.tolist()

        selected = []
        stock_count = 0
        sector_counts: Dict[str, int] = {}
        has_macro = False

        for asset in sorted_assets:
            if len(selected) >= self.max_size:
                break

            # Stock constraint
            if asset in TIER4_STOCKS:
                if stock_count >= self.max_stocks:
                    continue
                stock_count += 1

            # Same-sector constraint
            sector_group = SECTOR_GROUP_MAP.get(asset, asset)
            current_count = sector_counts.get(sector_group, 0)
            if current_count >= self.max_same_sector:
                continue
            sector_counts[sector_group] = current_count + 1

            # Track macro
            if asset in MACRO_ASSETS:
                has_macro = True

            selected.append(asset)

        # Ensure minimum macro asset
        if not has_macro and len(selected) > 0:
            # Find best macro asset not yet selected
            for asset in sorted_assets:
                if asset in MACRO_ASSETS and asset not in selected:
                    # Replace the worst selected asset
                    if len(selected) >= self.min_size:
                        selected[-1] = asset
                    else:
                        selected.append(asset)
                    break

        # Ensure minimum portfolio size
        if len(selected) < self.min_size:
            for asset in sorted_assets:
                if asset not in selected and len(selected) < self.min_size:
                    selected.append(asset)

        return selected


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_selection(selection: AssetSelection) -> None:
    """Pretty-print asset selection."""
    print(f"\n{'='*60}")
    print(f"ASSET SELECTION")
    print(f"{'='*60}")
    print(f"  Selected: {len(selection.selected_assets)} assets "
          f"({selection.n_etfs} ETFs + {selection.n_stocks} stocks)")
    print(f"\n  {'Asset':<8s} {'Score':>10s} {'Selected':>10s}")
    print(f"  {'-'*28}")

    for ticker in selection.adjusted_scores.sort_values(ascending=False).index:
        score = selection.adjusted_scores[ticker]
        is_selected = "  ✓" if ticker in selection.selected_assets else ""
        marker = " ★" if ticker in TIER4_STOCKS else ""
        print(f"  {ticker:<8s} {score:>9.4f} {is_selected}{marker}")