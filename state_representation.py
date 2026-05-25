"""
SPY Alpha v8 — Multi-State Representation
============================================
 
NEW module in v8. Builds a rich, multi-dimensional view of market conditions
from observable data, latent models, and transition dynamics.
 
Design Principles:
    - No single state model controls decisions
    - Observable states (higher trust) vs latent states (supplementary)
    - Transition features capture regime change dynamics (velocity/acceleration)
    - Both raw levels and transformed versions preserved for critical features
    - Feature groups labeled for explainability, pruning, and uncertainty dampening
    - Deep embeddings placeholder until Step 7
 
Data Sources:
    - Frozen snapshot: raw prices, FRED macro (for HMM features)
    - Fresh pull: VIX term structure (^VIX, ^VIX3M, ^SKEW), NFCI, STLFSI4
    - HMM regime probabilities: fed in as latent features, NOT used for allocation
 
Key Constraint:
    - This module does NOT make allocation decisions
    - It provides state information to the meta-allocator
    - The meta-allocator decides what to do with it
"""
 
from __future__ import annotations
 
import logging
from typing import Any, Dict, List, Optional, Tuple
 
import numpy as np
import pandas as pd
 
logger = logging.getLogger("spy_alpha_v8.state_representation")
 
 
# ---------------------------------------------------------------------------
# Feature Group Registry
# ---------------------------------------------------------------------------
 
FEATURE_GROUPS: Dict[str, List[str]] = {
    "volatility": [],
    "cross_asset_stress": [],
    "macro": [],
    "trend": [],
    "vrp": [],
    "latent_hmm": [],
    "latent_macro_pca": [],
    "latent_embeddings": [],
    "transition": [],
}
 
# Risk-on assets for trend breadth calculations
RISK_ON_ASSETS: List[str] = ["SPY", "QQQ", "IWM", "VWO", "XLK", "XLF", "XLY", "XLE", "XLI", "SMH"]
 
# Equity basket for cross-basket correlation
EQUITY_BASKET: List[str] = ["SPY", "QQQ", "IWM"]
 
# Rate basket for cross-basket correlation
RATE_BASKET: List[str] = ["TLT", "IEF", "SHY"]
 
# Commodity basket for cross-basket correlation
COMMODITY_BASKET: List[str] = ["GLD", "DBC"]
 
 
# ---------------------------------------------------------------------------
# Observable State Builders
# ---------------------------------------------------------------------------
 
def build_volatility_features(
    raw_close: pd.DataFrame,
    vix_term: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build volatility regime features from SPY prices and VIX term structure.
 
    Features:
        - Realized volatility (10d, 60d)
        - Vol-of-vol (rolling std of daily vol changes)
        - VIX level percentile (252-day rolling rank)
        - VIX term structure ratio (VIX / VIX3M), z-scored
        - VRP: VIX minus realized vol (20d, 60d)
        - Term structure slope: VIX3M minus VIX
    """
    features = {}
 
    # ---- Realized volatility from SPY ----
    if "SPY" in raw_close.columns:
        spy = raw_close["SPY"].dropna()
        spy_returns = spy.pct_change()
 
        # 10-day and 60-day realized vol (annualized)
        features["vol_realized_10d"] = spy_returns.rolling(10).std() * np.sqrt(252)
        features["vol_realized_60d"] = spy_returns.rolling(60).std() * np.sqrt(252)
 
        # 20-day realized vol (for VRP calculation)
        vol_20d = spy_returns.rolling(20).std() * np.sqrt(252)
        features["vol_realized_20d"] = vol_20d
 
        # Vol-of-vol: rolling std of daily changes in 10d realized vol
        vol_10d = features["vol_realized_10d"]
        features["vol_of_vol"] = vol_10d.diff().rolling(21).std()
 
    # ---- VIX term structure features ----
    if vix_term is not None and not vix_term.empty:
        # VIX level and percentile
        if "^VIX" in vix_term.columns:
            vix = vix_term["^VIX"]
            features["vix_level"] = vix
            features["vix_percentile_252d"] = vix.rolling(252, min_periods=126).rank(pct=True)
 
            # VIX z-score
            vix_mean = vix.rolling(252, min_periods=126).mean()
            vix_std = vix.rolling(252, min_periods=126).std()
            features["vix_zscore_252d"] = (vix - vix_mean) / vix_std.replace(0, np.nan)
 
        # VIX / VIX3M ratio (term structure)
        if "^VIX" in vix_term.columns and "^VIX3M" in vix_term.columns:
            vix = vix_term["^VIX"]
            vix3m = vix_term["^VIX3M"]
 
            ratio = vix / vix3m.replace(0, np.nan)
            features["vix_term_ratio"] = ratio
 
            # Z-score the ratio against 252-day history
            ratio_mean = ratio.rolling(252, min_periods=126).mean()
            ratio_std = ratio.rolling(252, min_periods=126).std()
            features["vix_term_ratio_zscore"] = (ratio - ratio_mean) / ratio_std.replace(0, np.nan)
 
            # Term structure slope: VIX3M - VIX
            # Positive = contango (normal), negative = backwardation (stress)
            features["vix_term_slope"] = vix3m - vix
 
        # VRP features: implied minus realized
        if "^VIX" in vix_term.columns and "vol_realized_20d" in features:
            vix = vix_term["^VIX"]
            # VIX is in percentage points, realized vol is a decimal — align units
            vix_decimal = vix / 100.0
            features["vrp_20d"] = vix_decimal - features["vol_realized_20d"]
            if "vol_realized_60d" in features:
                features["vrp_60d"] = vix_decimal - features["vol_realized_60d"]
 
        # SKEW features
        if "^SKEW" in vix_term.columns:
            skew = vix_term["^SKEW"]
            skew_mean = skew.rolling(252, min_periods=126).mean()
            skew_std = skew.rolling(252, min_periods=126).std()
            features["skew_level"] = skew
            features["skew_zscore_252d"] = (skew - skew_mean) / skew_std.replace(0, np.nan)
 
    df = pd.DataFrame(features)
 
    # Register feature names
    FEATURE_GROUPS["volatility"] = [c for c in df.columns if not c.startswith("vrp_")]
    FEATURE_GROUPS["vrp"] = [c for c in df.columns if c.startswith("vrp_")]
 
    return df
 
 
def build_cross_asset_stress_features(
    raw_close: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build cross-asset stress features from price data.
 
    Features:
        - Credit-equity divergence: HYG return vs SPY return (21d)
        - Rate-equity divergence: TLT return vs SPY return (21d)
        - Cross-basket correlation (equity/rates/commodity) at 21d and 63d
    """
    features = {}
 
    returns = raw_close.pct_change()
 
    # ---- Credit-equity divergence ----
    if "HYG" in returns.columns and "SPY" in returns.columns:
        hyg_ret_21d = returns["HYG"].rolling(21).sum()
        spy_ret_21d = returns["SPY"].rolling(21).sum()
        features["credit_equity_divergence_21d"] = hyg_ret_21d - spy_ret_21d
 
    # ---- Rate-equity divergence ----
    if "TLT" in returns.columns and "SPY" in returns.columns:
        tlt_ret_21d = returns["TLT"].rolling(21).sum()
        spy_ret_21d = returns["SPY"].rolling(21).sum()
        features["rate_equity_divergence_21d"] = tlt_ret_21d - spy_ret_21d
 
    # ---- Cross-basket correlations ----
    # Average pairwise correlation between baskets
    for window in [21, 63]:
        # Equity vs Rates
        eq_rate_corrs = []
        for eq in EQUITY_BASKET:
            for rt in RATE_BASKET:
                if eq in returns.columns and rt in returns.columns:
                    corr = returns[eq].rolling(window).corr(returns[rt])
                    eq_rate_corrs.append(corr)
        if eq_rate_corrs:
            features[f"corr_equity_rates_{window}d"] = pd.concat(eq_rate_corrs, axis=1).mean(axis=1)
 
        # Equity vs Commodities
        eq_comm_corrs = []
        for eq in EQUITY_BASKET:
            for cm in COMMODITY_BASKET:
                if eq in returns.columns and cm in returns.columns:
                    corr = returns[eq].rolling(window).corr(returns[cm])
                    eq_comm_corrs.append(corr)
        if eq_comm_corrs:
            features[f"corr_equity_commodities_{window}d"] = pd.concat(eq_comm_corrs, axis=1).mean(axis=1)
 
        # All-asset correlation (average pairwise across all baskets)
        all_assets = [a for a in EQUITY_BASKET + RATE_BASKET + COMMODITY_BASKET
                      if a in returns.columns]
        if len(all_assets) >= 3:
            rolling_corrs = []
            for i, a1 in enumerate(all_assets):
                for a2 in all_assets[i+1:]:
                    rolling_corrs.append(returns[a1].rolling(window).corr(returns[a2]))
            if rolling_corrs:
                features[f"corr_cross_basket_{window}d"] = pd.concat(rolling_corrs, axis=1).mean(axis=1)
 
    df = pd.DataFrame(features)
    FEATURE_GROUPS["cross_asset_stress"] = list(df.columns)
    return df
 
 
def build_macro_features(
    fred_data: pd.DataFrame,
    stress_fred: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build macro condition features from FRED data.
 
    Core FRED (from snapshot): T10Y2Y, T10Y3M, BAMLH0A0HYM2
    Stress FRED (fresh pull): NFCI, STLFSI4
 
    Features include raw levels and z-scored versions.
    """
    features = {}
 
    # ---- Yield curve features ----
    for series in ["T10Y2Y", "T10Y3M"]:
        if series in fred_data.columns:
            data = fred_data[series]
            features[f"macro_{series.lower()}_level"] = data
 
            # Z-score against 252-day rolling history
            mean = data.rolling(252, min_periods=126).mean()
            std = data.rolling(252, min_periods=126).std()
            features[f"macro_{series.lower()}_zscore"] = (data - mean) / std.replace(0, np.nan)
 
    # ---- Credit stress: HY OAS ----
    hy_col = "BAMLH0A0HYM2"
    if hy_col in fred_data.columns:
        hy = fred_data[hy_col]
        features["macro_hy_oas_level"] = hy
 
        mean = hy.rolling(252, min_periods=126).mean()
        std = hy.rolling(252, min_periods=126).std()
        features["macro_hy_oas_zscore"] = (hy - mean) / std.replace(0, np.nan)
 
    # ---- Financial conditions (from fresh stress data) ----
    if stress_fred is not None and not stress_fred.empty:
        for series in ["NFCI", "STLFSI4"]:
            if series in stress_fred.columns:
                data = stress_fred[series]
                features[f"macro_{series.lower()}_level"] = data
 
                mean = data.rolling(252, min_periods=126).mean()
                std = data.rolling(252, min_periods=126).std()
                features[f"macro_{series.lower()}_zscore"] = (data - mean) / std.replace(0, np.nan)
 
    df = pd.DataFrame(features)
    FEATURE_GROUPS["macro"] = list(df.columns)
    return df
 
 
def build_trend_features(
    raw_close: pd.DataFrame,
    assets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build trend persistence features across the observation universe.
 
    Features:
        - Per-asset trend score: weighted average of above/below 50d/100d/200d MAs
        - Fraction of risk-on assets above each MA threshold
        - Cross-asset momentum breadth (average trend score)
        - Multi-timeframe trend strength (SPY-specific)
    """
    features = {}
 
    if assets is None:
        assets = RISK_ON_ASSETS
 
    available = [a for a in assets if a in raw_close.columns]
    if not available:
        return pd.DataFrame()
 
    # ---- Per-asset trend scores ----
    ma_windows = [50, 100, 200]
    ma_weights = [0.30, 0.35, 0.35]  # from build spec
 
    # Track above/below MA for breadth calculations
    above_ma = {w: pd.DataFrame() for w in ma_windows}
 
    for asset in available:
        price = raw_close[asset].dropna()
 
        asset_signals = []
        for window, weight in zip(ma_windows, ma_weights):
            ma = price.rolling(window, min_periods=window).mean()
            signal = (price > ma).astype(float)
            above_ma[window][asset] = signal
            asset_signals.append(signal * weight)
 
        # Trend score per asset: weighted average of MA signals
        trend_score = sum(asset_signals)
        features[f"trend_score_{asset}"] = trend_score
 
    # ---- Breadth features: fraction of risk-on assets above each MA ----
    for window in ma_windows:
        if not above_ma[window].empty:
            breadth = above_ma[window].mean(axis=1)
            features[f"trend_breadth_{window}d"] = breadth
 
    # ---- Cross-asset momentum breadth (average trend score) ----
    trend_cols = [f"trend_score_{a}" for a in available if f"trend_score_{a}" in features]
    if trend_cols:
        trend_df = pd.DataFrame({c: features[c] for c in trend_cols})
        features["trend_breadth_avg"] = trend_df.mean(axis=1)
 
    # ---- SPY multi-timeframe trend strength ----
    if "SPY" in raw_close.columns:
        spy = raw_close["SPY"].dropna()
        for window in [21, 63, 126, 252]:
            ma = spy.rolling(window, min_periods=window).mean()
            features[f"trend_spy_dist_ma_{window}d"] = (spy - ma) / ma
 
    df = pd.DataFrame(features)
    FEATURE_GROUPS["trend"] = list(df.columns)
    return df
 
 
# ---------------------------------------------------------------------------
# Latent State Builders
# ---------------------------------------------------------------------------
 
def build_latent_hmm_features(
    regime_probs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Package HMM regime probabilities as latent state features.
 
    Input is a DataFrame with columns: Bull, Slowdown, Crisis-Deflation,
    Crisis-Inflation, Inflation — one row per trading day.
 
    These are used as FEATURES for the meta-allocator, NOT for direct
    allocation decisions. This is the critical v8 architectural change.
    """
    features = {}
 
    regime_names = ["Bull", "Slowdown", "Crisis-Deflation", "Crisis-Inflation", "Inflation"]
 
    for regime in regime_names:
        if regime in regime_probs.columns:
            col_name = f"hmm_{regime.lower().replace('-', '_')}_prob"
            features[col_name] = regime_probs[regime]
 
    # Regime entropy: high entropy = uncertain regime classification
    probs = regime_probs[[r for r in regime_names if r in regime_probs.columns]]
    if not probs.empty:
        # Shannon entropy: -sum(p * log(p))
        # Clip to avoid log(0)
        clipped = probs.clip(lower=1e-10)
        entropy = -(clipped * np.log(clipped)).sum(axis=1)
        features["hmm_regime_entropy"] = entropy
 
        # Max probability (dominant regime conviction)
        features["hmm_max_prob"] = probs.max(axis=1)
 
    df = pd.DataFrame(features)
    FEATURE_GROUPS["latent_hmm"] = list(df.columns)
    return df
 
 
def build_latent_macro_pca_features(
    fred_data: pd.DataFrame,
    n_components: int = 5,
) -> pd.DataFrame:
    """
    Build macro latent factors via PCA on FRED features.
 
    Separate PCA from the HMM's feature engineering PCA — this operates
    on the raw FRED series directly, not on the observation feature matrix.
 
    Used as supplementary latent context for the meta-allocator.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
 
    if fred_data.empty:
        return pd.DataFrame()
 
    # Forward-fill and drop columns with too many NaNs
    filled = fred_data.ffill()
    valid_cols = [c for c in filled.columns if filled[c].notna().mean() > 0.7]
 
    if len(valid_cols) < n_components:
        logger.warning(
            f"Only {len(valid_cols)} valid FRED columns, need {n_components} for PCA. "
            f"Reducing components."
        )
        n_components = max(1, len(valid_cols))
 
    clean = filled[valid_cols].dropna()
    if len(clean) < 252:
        logger.warning(f"Only {len(clean)} clean rows for macro PCA, need at least 252")
        return pd.DataFrame()
 
    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean)
 
    # PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)
 
    col_names = [f"macro_pc_{i:02d}" for i in range(n_components)]
    df = pd.DataFrame(components, index=clean.index, columns=col_names)
 
    explained = pca.explained_variance_ratio_
    logger.info(
        f"Macro PCA: {n_components} components explain "
        f"{sum(explained):.1%} of variance "
        f"({', '.join(f'{v:.1%}' for v in explained[:3])}...)"
    )
 
    FEATURE_GROUPS["latent_macro_pca"] = col_names
    return df
 
 
def build_deep_embedding_placeholder(
    index: pd.DatetimeIndex,
    n_dims: int = 16,
) -> pd.DataFrame:
    """
    Placeholder for deep temporal embeddings (built in Step 7).
 
    Returns a constant zero vector so downstream code can be built and
    tested without the actual embedding model. The meta-allocator will
    learn to ignore these until they contain real information.
    """
    col_names = [f"embedding_{i:02d}" for i in range(n_dims)]
    df = pd.DataFrame(
        np.zeros((len(index), n_dims)),
        index=index,
        columns=col_names,
    )
 
    FEATURE_GROUPS["latent_embeddings"] = col_names
    return df
 
 
# ---------------------------------------------------------------------------
# Transition Feature Builder
# ---------------------------------------------------------------------------
 
def build_transition_features(
    state_df: pd.DataFrame,
    regime_probs: Optional[pd.DataFrame] = None,
    velocity_window: int = 5,
) -> pd.DataFrame:
    """
    Build transition/acceleration features from state variables.
 
    Markets fail through acceleration, not levels. These features capture
    the rate of change in state variables, which often leads regime transitions.
 
    Features:
        - Regime probability velocity (5-day change in each HMM probability)
        - Volatility acceleration (5-day change in 10-day realized vol)
        - Correlation acceleration (5-day change in cross-basket correlation)
        - Macro deterioration speed (5-day change in NFCI/STLFSI z-scores)
        - Credit stress velocity (5-day change in HY OAS z-score)
        - Trend breakdown speed (5-day change in MA breadth score)
    """
    features = {}
 
    # ---- Regime probability velocity ----
    if regime_probs is not None and not regime_probs.empty:
        regime_names = ["Bull", "Slowdown", "Crisis-Deflation", "Crisis-Inflation", "Inflation"]
        for regime in regime_names:
            if regime in regime_probs.columns:
                col_name = f"trans_hmm_{regime.lower().replace('-', '_')}_velocity"
                features[col_name] = regime_probs[regime].diff(velocity_window)
 
    # ---- Volatility acceleration ----
    if "vol_realized_10d" in state_df.columns:
        features["trans_vol_acceleration"] = state_df["vol_realized_10d"].diff(velocity_window)
 
    # ---- Correlation acceleration ----
    for col in state_df.columns:
        if col.startswith("corr_cross_basket_"):
            window_str = col.split("_")[-1]  # e.g., "21d"
            features[f"trans_corr_acceleration_{window_str}"] = state_df[col].diff(velocity_window)
 
    # ---- Macro deterioration speed ----
    macro_velocity_cols = [
        "macro_nfci_zscore",
        "macro_stlfsi4_zscore",
    ]
    for col in macro_velocity_cols:
        if col in state_df.columns:
            short_name = col.replace("macro_", "").replace("_zscore", "")
            features[f"trans_{short_name}_velocity"] = state_df[col].diff(velocity_window)
 
    # ---- Credit stress velocity ----
    if "macro_hy_oas_zscore" in state_df.columns:
        features["trans_credit_stress_velocity"] = state_df["macro_hy_oas_zscore"].diff(velocity_window)
 
    # ---- Trend breakdown speed ----
    if "trend_breadth_avg" in state_df.columns:
        features["trans_trend_breakdown_speed"] = state_df["trend_breadth_avg"].diff(velocity_window)
 
    for col in state_df.columns:
        if col.startswith("trend_breadth_") and col != "trend_breadth_avg":
            window_str = col.split("_")[-1]  # e.g., "50d"
            features[f"trans_breadth_{window_str}_velocity"] = state_df[col].diff(velocity_window)
 
    # ---- VIX acceleration ----
    if "vix_level" in state_df.columns:
        features["trans_vix_velocity"] = state_df["vix_level"].diff(velocity_window)
 
    df = pd.DataFrame(features)
    FEATURE_GROUPS["transition"] = list(df.columns)
    return df
 
 
# ---------------------------------------------------------------------------
# State Representation Builder (Main Class)
# ---------------------------------------------------------------------------
 
class StateRepresentationBuilder:
    """
    Builds the complete multi-state representation for each trading day.
 
    Combines:
        - Observable states (volatility, stress, macro, trend, VRP)
        - Latent states (HMM probabilities, macro PCA, deep embeddings)
        - Transition features (velocity/acceleration of state variables)
 
    Output is a single DataFrame where each row is a trading day and
    each column is a state feature. The meta-allocator consumes this directly.
    """
 
    def __init__(
        self,
        macro_pca_components: int = 5,
        embedding_dims: int = 16,
        velocity_window: int = 5,
    ):
        self.macro_pca_components = macro_pca_components
        self.embedding_dims = embedding_dims
        self.velocity_window = velocity_window
 
    def build(
        self,
        raw_close: pd.DataFrame,
        fred_data: pd.DataFrame,
        regime_probs: Optional[pd.DataFrame] = None,
        stress_fred: Optional[pd.DataFrame] = None,
        vix_term: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build the complete state representation.
 
        Args:
            raw_close: Raw (unadjusted) close prices for observation universe
            fred_data: Core FRED data from snapshot (for HMM features)
            regime_probs: HMM regime probability DataFrame (optional, None until
                         regime model runs)
            stress_fred: Fresh NFCI/STLFSI4 data (optional)
            vix_term: Fresh VIX/VIX3M/SKEW data (optional)
 
        Returns:
            DataFrame with all state features, aligned to common date index
        """
        logger.info("Building state representation...")
 
        # ---- Observable States ----
        logger.info("  Building volatility features...")
        vol_features = build_volatility_features(raw_close, vix_term)
 
        logger.info("  Building cross-asset stress features...")
        stress_features = build_cross_asset_stress_features(raw_close)
 
        logger.info("  Building macro features...")
        macro_features = build_macro_features(fred_data, stress_fred)
 
        logger.info("  Building trend features...")
        trend_features = build_trend_features(raw_close)
 
        # ---- Combine observable features ----
        observable_parts = []
        for part in [vol_features, stress_features, macro_features, trend_features]:
            if not part.empty:
                observable_parts.append(part)
 
        if not observable_parts:
            raise ValueError("No observable features could be built — check input data")
 
        observable = pd.concat(observable_parts, axis=1)
 
        # ---- Latent States ----
        latent_parts = []
 
        # HMM regime probabilities
        if regime_probs is not None and not regime_probs.empty:
            logger.info("  Building latent HMM features...")
            hmm_features = build_latent_hmm_features(regime_probs)
            latent_parts.append(hmm_features)
 
        # Macro PCA
        if not fred_data.empty:
            logger.info("  Building latent macro PCA features...")
            macro_pca = build_latent_macro_pca_features(
                fred_data, n_components=self.macro_pca_components
            )
            if not macro_pca.empty:
                latent_parts.append(macro_pca)
 
        # Deep embeddings (placeholder)
        logger.info("  Adding deep embedding placeholder...")
        embedding_placeholder = build_deep_embedding_placeholder(
            observable.index, n_dims=self.embedding_dims
        )
        latent_parts.append(embedding_placeholder)
 
        # ---- Combine all pre-transition features ----
        all_parts = [observable]
        for part in latent_parts:
            if not part.empty:
                all_parts.append(part)
 
        state_df = pd.concat(all_parts, axis=1)
 
        # ---- Transition Features ----
        logger.info("  Building transition features...")
        transition = build_transition_features(
            state_df, regime_probs, velocity_window=self.velocity_window
        )
        if not transition.empty:
            state_df = pd.concat([state_df, transition], axis=1)
 
        # ---- Final cleanup ----
        # Replace inf with NaN
        state_df = state_df.replace([np.inf, -np.inf], np.nan)
 
        # Log summary
        n_features = state_df.shape[1]
        n_days = state_df.shape[0]
        nan_rate = state_df.isnull().mean().mean()
 
        logger.info(
            f"State representation built: {n_days} days × {n_features} features, "
            f"NaN rate: {nan_rate:.1%}"
        )
 
        # Log feature group sizes
        total_registered = 0
        for group_name, group_cols in FEATURE_GROUPS.items():
            present = [c for c in group_cols if c in state_df.columns]
            if present:
                logger.info(f"    {group_name}: {len(present)} features")
                total_registered += len(present)
 
        unregistered = n_features - total_registered
        if unregistered > 0:
            logger.warning(f"    {unregistered} features not in any group")
 
        return state_df
 
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return the current feature group registry."""
        return {k: list(v) for k, v in FEATURE_GROUPS.items()}
 
    def get_observable_features(self, state_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only observable (higher trust) features from state representation.
 
        Used when uncertainty is high and latent features should be downweighted.
        """
        observable_groups = ["volatility", "cross_asset_stress", "macro", "trend", "vrp"]
        obs_cols = []
        for group in observable_groups:
            obs_cols.extend([c for c in FEATURE_GROUPS.get(group, []) if c in state_df.columns])
        return state_df[obs_cols] if obs_cols else pd.DataFrame(index=state_df.index)
 
    def get_latent_features(self, state_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only latent (supplementary) features from state representation.
 
        These should be downweighted during high uncertainty.
        """
        latent_groups = ["latent_hmm", "latent_macro_pca", "latent_embeddings"]
        lat_cols = []
        for group in latent_groups:
            lat_cols.extend([c for c in FEATURE_GROUPS.get(group, []) if c in state_df.columns])
        return state_df[lat_cols] if lat_cols else pd.DataFrame(index=state_df.index)
 
    def get_transition_features(self, state_df: pd.DataFrame) -> pd.DataFrame:
        """Extract only transition features from state representation."""
        trans_cols = [c for c in FEATURE_GROUPS.get("transition", []) if c in state_df.columns]
        return state_df[trans_cols] if trans_cols else pd.DataFrame(index=state_df.index)
 
 
# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
 
def print_state_summary(state_df: pd.DataFrame) -> None:
    """Print a summary of the state representation."""
    print(f"\n{'='*60}")
    print(f"STATE REPRESENTATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Shape: {state_df.shape[0]} days × {state_df.shape[1]} features")
    print(f"  Date range: {state_df.index[0].date()} → {state_df.index[-1].date()}")
    print(f"  Overall NaN rate: {state_df.isnull().mean().mean():.2%}")
    print()
 
    print(f"  {'Group':<25s} {'Features':>8s} {'NaN Rate':>10s}")
    print(f"  {'-'*43}")
 
    total_registered = 0
    for group_name, group_cols in FEATURE_GROUPS.items():
        present = [c for c in group_cols if c in state_df.columns]
        if present:
            group_nan = state_df[present].isnull().mean().mean()
            print(f"  {group_name:<25s} {len(present):>8d} {group_nan:>9.2%}")
            total_registered += len(present)
 
    unregistered = state_df.shape[1] - total_registered
    if unregistered > 0:
        print(f"  {'(unregistered)':<25s} {unregistered:>8d}")
 
    print()
 
    # Latest state snapshot
    latest = state_df.iloc[-1]
    print(f"  Latest state ({state_df.index[-1].date()}):")
 
    key_features = [
        "vol_realized_10d", "vol_realized_60d", "vix_level", "vix_percentile_252d",
        "vix_term_ratio", "vrp_20d", "trend_breadth_avg",
        "macro_hy_oas_level", "macro_nfci_level", "macro_stlfsi4_level",
        "hmm_regime_entropy", "hmm_max_prob",
    ]
 
    for feat in key_features:
        if feat in latest.index and pd.notna(latest[feat]):
            print(f"    {feat:<35s} {latest[feat]:>10.4f}")