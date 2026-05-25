"""
SPY Alpha v8 — Feature Engineering
====================================
 
Ported from v6 with minimal changes (observation universe ticker list only).
Feature computations are IDENTICAL to v6 — DO NOT modify.
 
CRITICAL WARNING (from v6 experiments):
    Every attempt to modify the feature space destabilized the HMM.
    The expanded observation universe (EFA, EWJ) naturally creates new
    features via per-instrument computation. Do NOT add new feature types.
 
Changes from v6:
    - Observation universe expanded: 12 → 14 assets (added EFA, EWJ)
    - Sector features now include Tier 3 thematic ETFs (SMH, XBI, XME)
    - All feature computation functions are UNCHANGED
 
Computes all features for regime detection and sector allocation.
 
CRITICAL DESIGN RULE (from v5 post-mortem):
    ALL technical, momentum, trend, drawdown, consistency, and cross-sectional
    features are calculated EXCLUSIVELY on raw (unadjusted) Close prices.
    Adjusted prices are NEVER used here — they are reserved solely for return
    calculations in the backtester.
 
    Why: v5 used auto_adjust=True, which retroactively rewrites the entire
    price history on every dividend payment. This caused a 4,475% → 595%
    return collapse overnight with zero code changes. Raw prices are stable
    by definition and eliminate this failure mode entirely.
 
Feature Categories:
    1. Technical      — SMA ratios, Bollinger bands, RSI, ATR, VWAP ratio
    2. Momentum       — Multi-horizon returns, rate of change, return acceleration
    3. Trend          — SMA slopes, trend consistency, multi-timeframe alignment
    4. Drawdown       — Current drawdown from rolling max, drawdown duration
    5. Consistency    — Rolling Sharpe, return skew, hit rate
    6. Cross-sectional — Relative strength, correlation regime, breadth, dispersion
    7. Macro (FRED)   — Yield curve, credit spreads, labor market, inflation, sentiment
 
Usage:
    from data_pipeline import SnapshotManager, get_raw_close, get_fred
    from feature_engineering import FeatureEngine
 
    mgr = SnapshotManager()
    snap = mgr.load_snapshot("baseline_2026")
 
    engine = FeatureEngine()
    obs_features = engine.build_observation_features(snap)
    sector_features = engine.build_sector_features(snap)
 
    # Or build everything at once
    all_features = engine.build_all(snap)
"""
 
from __future__ import annotations
 
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple
 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
 
from data_pipeline import (
    FRED_SERIES,
    OBSERVATION_UNIVERSE,
    TRADING_UNIVERSE,
    TIER2_SECTORS,
    TIER3_THEMATIC,
    get_fred,
    get_raw_close,
    get_raw_ohlcv,
)
 
logger = logging.getLogger("spy_alpha_v8.feature_engineering")
 
# Suppress pandas PerformanceWarning for chained operations
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
# Standard rolling windows used across feature categories
WINDOWS_SHORT = [5, 10, 21]         # ~1wk, 2wk, 1mo
WINDOWS_MEDIUM = [42, 63]           # ~2mo, 3mo
WINDOWS_LONG = [126, 252]           # ~6mo, 1yr
ALL_WINDOWS = WINDOWS_SHORT + WINDOWS_MEDIUM + WINDOWS_LONG
 
# Minimum observations required before a feature is considered valid
MIN_LOOKBACK = 252  # 1 year — ensures longest rolling window is fully populated
 
 
# ---------------------------------------------------------------------------
# Individual Feature Computations (all on RAW close prices)
# ---------------------------------------------------------------------------
 
def compute_technical_features(
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    volume: Optional[pd.Series] = None,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Technical indicator features from raw price data.
 
    All computed on UNADJUSTED prices. Ratios and normalized indicators
    are inherently scale-invariant, so raw vs adjusted doesn't matter
    for the indicator value — but using raw ensures the input data
    itself is stable across runs.
 
    Parameters
    ----------
    close : pd.Series
        Raw close prices for a single instrument.
    high : pd.Series, optional
        Raw high prices (for ATR).
    low : pd.Series, optional
        Raw low prices (for ATR).
    volume : pd.Series, optional
        Volume (for VWAP ratio).
    prefix : str
        Column name prefix (e.g., "SPY_").
 
    Returns
    -------
    pd.DataFrame
        Technical features indexed by date.
    """
    features = {}
    p = prefix
 
    # ---- SMA ratios (price relative to moving average) ----
    for w in ALL_WINDOWS:
        sma = close.rolling(w, min_periods=w).mean()
        features[f"{p}sma_ratio_{w}d"] = close / sma - 1.0
 
    # ---- Exponential MA ratios ----
    for w in WINDOWS_SHORT + WINDOWS_MEDIUM:
        ema = close.ewm(span=w, min_periods=w).mean()
        features[f"{p}ema_ratio_{w}d"] = close / ema - 1.0
 
    # ---- Bollinger Band position (where price sits within bands) ----
    for w in [21, 63]:
        sma = close.rolling(w, min_periods=w).mean()
        std = close.rolling(w, min_periods=w).std()
        # Normalized position: 0 = lower band, 1 = upper band
        bb_pos = (close - (sma - 2 * std)) / (4 * std)
        features[f"{p}bb_position_{w}d"] = bb_pos
        # Bandwidth as pct of SMA (volatility proxy)
        features[f"{p}bb_width_{w}d"] = (4 * std) / sma
 
    # ---- RSI (Relative Strength Index) ----
    for w in [14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(w, min_periods=w).mean()
        loss = (-delta.clip(upper=0)).rolling(w, min_periods=w).mean()
        rs = gain / loss.replace(0, np.nan)
        features[f"{p}rsi_{w}d"] = 100 - (100 / (1 + rs))
 
    # ---- ATR (Average True Range) as pct of close ----
    if high is not None and low is not None:
        for w in [14, 21]:
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(w, min_periods=w).mean()
            features[f"{p}atr_pct_{w}d"] = atr / close
 
    # ---- VWAP ratio (volume-weighted average price) ----
    if volume is not None:
        for w in [21, 63]:
            vwap = (close * volume).rolling(w, min_periods=w).sum() / \
                   volume.rolling(w, min_periods=w).sum()
            features[f"{p}vwap_ratio_{w}d"] = close / vwap - 1.0
 
    return pd.DataFrame(features, index=close.index)
 
 
def compute_momentum_features(
    close: pd.Series,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Momentum and return features from raw close prices.
 
    Parameters
    ----------
    close : pd.Series
        Raw close prices.
    prefix : str
        Column name prefix.
 
    Returns
    -------
    pd.DataFrame
        Momentum features.
    """
    features = {}
    p = prefix
 
    # ---- Simple returns over multiple horizons ----
    for w in ALL_WINDOWS:
        features[f"{p}return_{w}d"] = close.pct_change(w)
 
    # ---- Log returns (better statistical properties) ----
    log_close = np.log(close)
    for w in [5, 21, 63]:
        features[f"{p}log_return_{w}d"] = log_close.diff(w)
 
    # ---- Rate of change (ROC) ----
    for w in [10, 21, 63]:
        features[f"{p}roc_{w}d"] = (close - close.shift(w)) / close.shift(w)
 
    # ---- Return acceleration (change in momentum) ----
    ret_21 = close.pct_change(21)
    ret_63 = close.pct_change(63)
    features[f"{p}return_accel_21_63"] = ret_21 - ret_63 / 3  # normalized to same horizon
 
    # ---- Momentum quality: return relative to volatility ----
    for w in [21, 63]:
        ret = close.pct_change(w)
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(w, min_periods=w).std() * np.sqrt(w)
        features[f"{p}momentum_quality_{w}d"] = ret / vol.replace(0, np.nan)
 
    return pd.DataFrame(features, index=close.index)
 
 
def compute_trend_features(
    close: pd.Series,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Trend strength and direction features from raw close prices.
 
    Parameters
    ----------
    close : pd.Series
        Raw close prices.
    prefix : str
        Column name prefix.
 
    Returns
    -------
    pd.DataFrame
        Trend features.
    """
    features = {}
    p = prefix
 
    # ---- SMA slope (normalized by price level) ----
    for w in [21, 63, 126, 252]:
        sma = close.rolling(w, min_periods=w).mean()
        slope = sma.diff(5) / sma  # 5-day change in SMA, normalized
        features[f"{p}sma_slope_{w}d"] = slope
 
    # ---- Trend consistency (fraction of positive daily returns) ----
    daily_ret = close.pct_change()
    for w in [21, 63, 126]:
        features[f"{p}trend_consistency_{w}d"] = daily_ret.rolling(
            w, min_periods=w
        ).apply(lambda x: (x > 0).mean(), raw=True)
 
    # ---- Multi-timeframe trend alignment ----
    sma_21 = close.rolling(21, min_periods=21).mean()
    sma_63 = close.rolling(63, min_periods=63).mean()
    sma_126 = close.rolling(126, min_periods=126).mean()
    sma_252 = close.rolling(252, min_periods=252).mean()
 
    # Count how many SMAs price is above (0-4 scale)
    above_count = (
        (close > sma_21).astype(float) +
        (close > sma_63).astype(float) +
        (close > sma_126).astype(float) +
        (close > sma_252).astype(float)
    )
    features[f"{p}trend_alignment"] = above_count / 4.0
 
    # SMA ordering: bullish = short > medium > long
    features[f"{p}sma_order_21_63"] = (sma_21 / sma_63 - 1.0)
    features[f"{p}sma_order_63_126"] = (sma_63 / sma_126 - 1.0)
    features[f"{p}sma_order_126_252"] = (sma_126 / sma_252 - 1.0)
 
    # ---- Linear regression slope + R² (trend strength and quality) ----
    # Slope tells you trend direction/magnitude; R² tells you how clean it is.
    # A high slope with low R² is a noisy trend — the model shouldn't trust it.
    # This was missing in the original build and flagged during review.
    for w in [63, 126, 252]:
        x = np.arange(w, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
 
        def _lr_slope(y, _x=x, _x_mean=x_mean, _x_var=x_var, _w=w):
            if len(y) < _w or np.isnan(y).any():
                return np.nan
            y_mean = y.mean()
            return ((_x - _x_mean) * (y - y_mean)).sum() / _x_var
 
        def _lr_r2(y, _x=x, _x_mean=x_mean, _x_var=x_var, _w=w):
            if len(y) < _w or np.isnan(y).any():
                return np.nan
            y_mean = y.mean()
            ss_xy = ((_x - _x_mean) * (y - y_mean)).sum()
            slope = ss_xy / _x_var
            y_pred = y_mean + slope * (_x - _x_mean)
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - y_mean) ** 2).sum()
            if ss_tot == 0:
                return np.nan
            return 1.0 - ss_res / ss_tot
 
        slope = close.rolling(w, min_periods=w).apply(_lr_slope, raw=True)
        features[f"{p}lr_slope_{w}d"] = slope / close  # normalize by price level
 
        r2 = close.rolling(w, min_periods=w).apply(_lr_r2, raw=True)
        features[f"{p}trend_r2_{w}d"] = r2  # 0–1 scale, higher = cleaner trend
 
    return pd.DataFrame(features, index=close.index)
 
 
def compute_drawdown_features(
    close: pd.Series,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Drawdown features from raw close prices.
 
    Parameters
    ----------
    close : pd.Series
        Raw close prices.
    prefix : str
        Column name prefix.
 
    Returns
    -------
    pd.DataFrame
        Drawdown features.
    """
    features = {}
    p = prefix
 
    # ---- Current drawdown from rolling highs ----
    for w in [63, 126, 252]:
        rolling_max = close.rolling(w, min_periods=1).max()
        dd = (close - rolling_max) / rolling_max
        features[f"{p}drawdown_{w}d"] = dd
 
    # ---- Drawdown from all-time (expanding) high ----
    expanding_max = close.expanding(min_periods=1).max()
    features[f"{p}drawdown_ath"] = (close - expanding_max) / expanding_max
 
    # ---- Drawdown duration (days since last rolling high) ----
    for w in [126, 252]:
        rolling_max = close.rolling(w, min_periods=1).max()
        at_high = (close >= rolling_max * 0.999).astype(float)  # within 0.1% of high
 
        # Count days since last high
        duration = at_high.copy()
        for i in range(1, len(duration)):
            if duration.iloc[i] == 0:
                duration.iloc[i] = duration.iloc[i - 1] + 1
            else:
                duration.iloc[i] = 0
        features[f"{p}dd_duration_{w}d"] = duration / w  # normalize by window
 
    return pd.DataFrame(features, index=close.index)
 
 
def compute_consistency_features(
    close: pd.Series,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Return consistency and quality features from raw close prices.
 
    Parameters
    ----------
    close : pd.Series
        Raw close prices.
    prefix : str
        Column name prefix.
 
    Returns
    -------
    pd.DataFrame
        Consistency features.
    """
    features = {}
    p = prefix
    daily_ret = close.pct_change()
 
    # ---- Rolling Sharpe ratio (annualized) ----
    for w in [21, 63, 126]:
        roll_mean = daily_ret.rolling(w, min_periods=w).mean()
        roll_std = daily_ret.rolling(w, min_periods=w).std()
        features[f"{p}rolling_sharpe_{w}d"] = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252)
 
    # ---- Rolling return skewness ----
    for w in [63, 126]:
        features[f"{p}return_skew_{w}d"] = daily_ret.rolling(w, min_periods=w).skew()
 
    # ---- Rolling return kurtosis ----
    for w in [63, 126]:
        features[f"{p}return_kurtosis_{w}d"] = daily_ret.rolling(w, min_periods=w).kurt()
 
    # ---- Hit rate (fraction of positive return days) ----
    for w in [21, 63]:
        features[f"{p}hit_rate_{w}d"] = daily_ret.rolling(
            w, min_periods=w
        ).apply(lambda x: (x > 0).mean(), raw=True)
 
    # ---- Rolling realized volatility ----
    for w in [21, 63, 126]:
        features[f"{p}realized_vol_{w}d"] = daily_ret.rolling(
            w, min_periods=w
        ).std() * np.sqrt(252)
 
    # ---- Volatility ratio (short-term vs long-term vol) ----
    vol_21 = daily_ret.rolling(21, min_periods=21).std()
    vol_63 = daily_ret.rolling(63, min_periods=63).std()
    features[f"{p}vol_ratio_21_63"] = vol_21 / vol_63.replace(0, np.nan)
 
    return pd.DataFrame(features, index=close.index)
 
 
# ---------------------------------------------------------------------------
# Cross-Sectional Features (computed across multiple instruments)
# ---------------------------------------------------------------------------
 
def compute_cross_sectional_features(
    close_df: pd.DataFrame,
    prefix: str = "xsec_",
) -> pd.DataFrame:
    """
    Cross-sectional features computed across the observation universe.
 
    These capture relative positioning and inter-market dynamics that
    single-instrument features miss. Genuine macro regime shifts manifest
    across multiple asset classes simultaneously — this is why v6 uses
    a broad observation universe instead of SPY alone.
 
    Parameters
    ----------
    close_df : pd.DataFrame
        Raw close prices for all observation universe instruments.
        Columns = ticker names.
    prefix : str
        Column name prefix.
 
    Returns
    -------
    pd.DataFrame
        Cross-sectional features indexed by date.
    """
    features = {}
    p = prefix
    daily_ret = close_df.pct_change()
 
    # ---- Cross-sectional return dispersion ----
    for w in [21, 63]:
        rolling_ret = close_df.pct_change(w)
        features[f"{p}return_dispersion_{w}d"] = rolling_ret.std(axis=1)
 
    # ---- Breadth: fraction of assets with positive momentum ----
    for w in [21, 63]:
        rolling_ret = close_df.pct_change(w)
        features[f"{p}breadth_{w}d"] = (rolling_ret > 0).mean(axis=1)
 
    # ---- Average cross-asset correlation (correlation regime) ----
    for w in [63, 126]:
        # Rolling pairwise correlation average
        corr_avg = daily_ret.rolling(w, min_periods=w).corr().groupby(level=0).apply(
            lambda x: x.values[np.triu_indices(len(x), k=1)].mean()
            if len(x) > 1 else np.nan
        )
        features[f"{p}avg_correlation_{w}d"] = corr_avg
 
    # ---- Equity vs bond relative strength ----
    equity_tickers = [t for t in ["SPY", "QQQ", "IWM"] if t in close_df.columns]
    bond_tickers = [t for t in ["TLT", "IEF", "SHY"] if t in close_df.columns]
 
    if equity_tickers and bond_tickers:
        for w in [21, 63]:
            eq_ret = close_df[equity_tickers].pct_change(w).mean(axis=1)
            bd_ret = close_df[bond_tickers].pct_change(w).mean(axis=1)
            features[f"{p}equity_vs_bond_{w}d"] = eq_ret - bd_ret
 
    # ---- Risk-on vs risk-off spread ----
    risk_on = [t for t in ["SPY", "QQQ", "HYG"] if t in close_df.columns]
    risk_off = [t for t in ["TLT", "GLD", "SHY"] if t in close_df.columns]
 
    if risk_on and risk_off:
        for w in [21, 63]:
            on_ret = close_df[risk_on].pct_change(w).mean(axis=1)
            off_ret = close_df[risk_off].pct_change(w).mean(axis=1)
            features[f"{p}risk_on_off_{w}d"] = on_ret - off_ret
 
    # ---- DM vs EM equity spread ----
    dm = [t for t in ["SPY", "QQQ"] if t in close_df.columns]
    em = [t for t in ["VEA", "VWO"] if t in close_df.columns]
 
    if dm and em:
        for w in [21, 63]:
            dm_ret = close_df[dm].pct_change(w).mean(axis=1)
            em_ret = close_df[em].pct_change(w).mean(axis=1)
            features[f"{p}dm_vs_em_{w}d"] = dm_ret - em_ret
 
    # ---- VIX-specific features (if available) ----
    vix_col = None
    for candidate in ["^VIX", "VIX", "VIXY"]:
        if candidate in close_df.columns:
            vix_col = candidate
            break
 
    if vix_col is not None:
        vix = close_df[vix_col]
 
        # VIX level z-score
        for w in [63, 252]:
            vix_mean = vix.rolling(w, min_periods=w).mean()
            vix_std = vix.rolling(w, min_periods=w).std()
            features[f"{p}vix_zscore_{w}d"] = (vix - vix_mean) / vix_std.replace(0, np.nan)
 
        # VIX term structure proxy: current vs rolling average
        vix_ma_21 = vix.rolling(21, min_periods=21).mean()
        features[f"{p}vix_vs_ma21"] = vix / vix_ma_21 - 1.0
 
        # VIX rate of change
        features[f"{p}vix_roc_5d"] = vix.pct_change(5)
        features[f"{p}vix_roc_21d"] = vix.pct_change(21)
 
    return pd.DataFrame(features, index=close_df.index)
 
 
# ---------------------------------------------------------------------------
# Macro (FRED) Features
# ---------------------------------------------------------------------------
 
def compute_macro_features(
    fred_df: pd.DataFrame,
    prefix: str = "macro_",
) -> pd.DataFrame:
    """
    Macro features derived from FRED economic data.
 
    Parameters
    ----------
    fred_df : pd.DataFrame
        FRED data from snapshot (forward-filled to business day frequency).
    prefix : str
        Column name prefix.
 
    Returns
    -------
    pd.DataFrame
        Macro features indexed by date.
    """
    if fred_df.empty:
        logger.warning("Empty FRED DataFrame — macro features will be unavailable.")
        return pd.DataFrame()
 
    features = {}
    p = prefix
 
    # ---- Yield curve features ----
    for series_id in ["T10Y2Y", "T10Y3M"]:
        if series_id in fred_df.columns:
            s = fred_df[series_id]
            features[f"{p}{series_id}_level"] = s
            features[f"{p}{series_id}_chg_21d"] = s.diff(21)
            features[f"{p}{series_id}_chg_63d"] = s.diff(63)
 
            # Z-score of level
            for w in [126, 252]:
                s_mean = s.rolling(w, min_periods=w).mean()
                s_std = s.rolling(w, min_periods=w).std()
                features[f"{p}{series_id}_zscore_{w}d"] = (s - s_mean) / s_std.replace(0, np.nan)
 
    # ---- Credit spread features ----
    hy_col = "BAMLH0A0HYM2"
    if hy_col in fred_df.columns:
        hy = fred_df[hy_col]
        features[f"{p}hy_oas_level"] = hy
        features[f"{p}hy_oas_chg_21d"] = hy.diff(21)
        features[f"{p}hy_oas_chg_63d"] = hy.diff(63)
 
        for w in [126, 252]:
            hy_mean = hy.rolling(w, min_periods=w).mean()
            hy_std = hy.rolling(w, min_periods=w).std()
            features[f"{p}hy_oas_zscore_{w}d"] = (hy - hy_mean) / hy_std.replace(0, np.nan)
 
    # ---- Labor market features ----
    if "UNRATE" in fred_df.columns:
        ur = fred_df["UNRATE"]
        features[f"{p}unrate_level"] = ur
        features[f"{p}unrate_chg_3m"] = ur.diff(63)  # ~3 months
        features[f"{p}unrate_chg_12m"] = ur.diff(252)
 
    if "ICSA" in fred_df.columns:
        icsa = fred_df["ICSA"]
        features[f"{p}claims_level"] = icsa
        # Year-over-year pct change
        features[f"{p}claims_yoy"] = icsa.pct_change(252)
        # 4-week moving average (smoothed)
        icsa_ma4 = icsa.rolling(20, min_periods=10).mean()  # ~4 weeks
        features[f"{p}claims_ma4w_chg"] = icsa_ma4.pct_change(63)
 
    # ---- Inflation features ----
    if "CPIAUCSL" in fred_df.columns:
        cpi = fred_df["CPIAUCSL"]
        features[f"{p}cpi_yoy"] = cpi.pct_change(252)  # year-over-year
        features[f"{p}cpi_mom"] = cpi.pct_change(21)    # month-over-month proxy
        # CPI acceleration
        cpi_yoy = cpi.pct_change(252)
        features[f"{p}cpi_accel"] = cpi_yoy.diff(63)
 
    # ---- Fed funds rate features ----
    if "FEDFUNDS" in fred_df.columns:
        ff = fred_df["FEDFUNDS"]
        features[f"{p}fedfunds_level"] = ff
        features[f"{p}fedfunds_chg_63d"] = ff.diff(63)
        features[f"{p}fedfunds_chg_252d"] = ff.diff(252)
 
    # ---- Consumer sentiment ----
    if "UMCSENT" in fred_df.columns:
        sent = fred_df["UMCSENT"]
        features[f"{p}sentiment_level"] = sent
        features[f"{p}sentiment_chg_63d"] = sent.diff(63)
 
        for w in [126, 252]:
            s_mean = sent.rolling(w, min_periods=w).mean()
            s_std = sent.rolling(w, min_periods=w).std()
            features[f"{p}sentiment_zscore_{w}d"] = (sent - s_mean) / s_std.replace(0, np.nan)
 
    return pd.DataFrame(features, index=fred_df.index)
 
 
# ---------------------------------------------------------------------------
# Feature Engine (main orchestrator)
# ---------------------------------------------------------------------------
 
class FeatureEngine:
    """
    Orchestrates feature computation for regime detection and sector allocation.
 
    Consumes snapshot data from SnapshotManager and produces clean feature
    matrices with proper NaN handling for assets with different start dates.
 
    Parameters
    ----------
    min_lookback : int
        Minimum number of observations before features are considered valid.
        Default 252 (1 year) ensures the longest rolling windows are populated.
    nan_threshold : float
        Maximum fraction of NaN values allowed in a feature column.
        Columns exceeding this threshold are dropped with a warning.
    reduce_dims : bool
        If True, apply correlation filtering + PCA to observation features.
        Enabled by default because 400+ raw features destabilize the HMM.
        Set False for exploratory analysis of individual features.
    n_components : int
        Number of PCA components to retain. Default 20 targets the sweet
        spot between information retention and HMM parameter count.
    correlation_threshold : float
        Drop one feature from each pair with pairwise |correlation| above
        this threshold. Applied before PCA to remove pure redundancy.
    """
 
    def __init__(
        self,
        min_lookback: int = MIN_LOOKBACK,
        nan_threshold: float = 0.3,
        reduce_dims: bool = True,
        n_components: int = 20,
        correlation_threshold: float = 0.95,
    ):
        self.min_lookback = min_lookback
        self.nan_threshold = nan_threshold
        self.reduce_dims = reduce_dims
        self.n_components = n_components
        self.correlation_threshold = correlation_threshold
 
        # Fitted PCA and scaler stored here for transform reuse (e.g., daily mode)
        self._pca: Optional[PCA] = None
        self._scaler: Optional[StandardScaler] = None
        self._retained_cols: Optional[List[str]] = None
 
    def build_observation_features(
        self,
        snapshot: Dict[str, Any],
        tickers: Optional[List[str]] = None,
        reduce: Optional[bool] = None,
        fit_pca: bool = True,
    ) -> pd.DataFrame:
        """
        Build features for the observation universe (used by regime model).
 
        Computes per-instrument technical features for each observation
        universe asset, plus cross-sectional and macro features.
 
        Parameters
        ----------
        snapshot : dict
            Loaded snapshot from SnapshotManager.load_snapshot().
        tickers : list of str, optional
            Override observation universe tickers.
        reduce : bool or None
            Whether to apply dimensionality reduction. None = use engine default
            (self.reduce_dims). Set explicitly to override per-call.
        fit_pca : bool
            If True, fit PCA on this data (training/backtest mode).
            If False, reuse previously fitted PCA (daily inference mode).
 
        Returns
        -------
        pd.DataFrame
            Feature matrix indexed by date. Columns = feature names (raw)
            or PC_00..PC_N (if reduced).
        """
        tickers = tickers or OBSERVATION_UNIVERSE
        logger.info(f"Building observation features for {len(tickers)} instruments")
 
        raw_close = get_raw_close(snapshot, tickers)
        available_tickers = list(raw_close.columns)
        logger.info(f"  Available tickers: {available_tickers}")
 
        all_features = []
 
        # ---- Per-instrument features for key observation assets ----
        # Focus on the most important instruments to keep dimensionality manageable
        key_instruments = [t for t in ["SPY", "QQQ", "TLT", "HYG", "GLD"] if t in available_tickers]
 
        for ticker in key_instruments:
            prefix = f"{ticker}_"
            close = raw_close[ticker].dropna()
 
            if len(close) < self.min_lookback:
                logger.warning(
                    f"  {ticker}: only {len(close)} observations "
                    f"(need {self.min_lookback}), skipping"
                )
                continue
 
            # Get full OHLCV for ATR and VWAP
            try:
                ohlcv = get_raw_ohlcv(snapshot, ticker)
                high = ohlcv["High"] if "High" in ohlcv.columns else None
                low = ohlcv["Low"] if "Low" in ohlcv.columns else None
                volume = ohlcv["Volume"] if "Volume" in ohlcv.columns else None
            except (KeyError, Exception):
                high, low, volume = None, None, None
 
            feats = pd.concat([
                compute_technical_features(close, high, low, volume, prefix),
                compute_momentum_features(close, prefix),
                compute_trend_features(close, prefix),
                compute_drawdown_features(close, prefix),
                compute_consistency_features(close, prefix),
            ], axis=1)
 
            all_features.append(feats)
            logger.info(f"  {ticker}: {feats.shape[1]} features")
 
        # ---- Cross-sectional features ----
        xsec = compute_cross_sectional_features(raw_close)
        if not xsec.empty:
            all_features.append(xsec)
            logger.info(f"  Cross-sectional: {xsec.shape[1]} features")
 
        # ---- Macro features ----
        fred_df = get_fred(snapshot)
        macro = compute_macro_features(fred_df)
        if not macro.empty:
            all_features.append(macro)
            logger.info(f"  Macro (FRED): {macro.shape[1]} features")
 
        # ---- Combine and clean ----
        if not all_features:
            raise RuntimeError("No features computed — check data availability")
 
        combined = pd.concat(all_features, axis=1)
        combined = self._clean_features(combined, "observation")
 
        # ---- Dimensionality reduction (for HMM stability) ----
        should_reduce = reduce if reduce is not None else self.reduce_dims
        if should_reduce:
            combined = self.reduce_dimensions(combined, fit=fit_pca)
 
        return combined
 
    def build_sector_features(
        self,
        snapshot: Dict[str, Any],
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build features for the trading universe sectors.
 
        Used by the return forecaster and portfolio optimizer for
        sector-level allocation decisions.
 
        Parameters
        ----------
        snapshot : dict
            Loaded snapshot.
        tickers : list of str, optional
            Override trading universe tickers.
 
        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame: (date, ticker) → features.
        """
        # ETFs that need sector-level features for the forecaster
        # Tier 2 (sectors) + Tier 3 (thematic) — excludes overlay instruments (UPRO, SHY)
        # and Tier 1 macro assets that are handled by the observation universe
        sector_etfs = TIER2_SECTORS + TIER3_THEMATIC
        tickers = tickers or sector_etfs
 
        raw_close = get_raw_close(snapshot, tickers)
        available_tickers = list(raw_close.columns)
        logger.info(f"Building sector features for {len(available_tickers)} ETFs")
 
        sector_frames = []
 
        for ticker in available_tickers:
            close = raw_close[ticker].dropna()
 
            if len(close) < self.min_lookback:
                logger.warning(f"  {ticker}: insufficient data, skipping")
                continue
 
            try:
                ohlcv = get_raw_ohlcv(snapshot, ticker)
                high = ohlcv.get("High")
                low = ohlcv.get("Low")
                volume = ohlcv.get("Volume")
            except (KeyError, Exception):
                high, low, volume = None, None, None
 
            feats = pd.concat([
                compute_technical_features(close, high, low, volume, prefix=""),
                compute_momentum_features(close, prefix=""),
                compute_trend_features(close, prefix=""),
                compute_drawdown_features(close, prefix=""),
                compute_consistency_features(close, prefix=""),
            ], axis=1)
 
            feats["ticker"] = ticker
            sector_frames.append(feats)
 
        if not sector_frames:
            raise RuntimeError("No sector features computed — check data availability")
 
        combined = pd.concat(sector_frames, axis=0)
        combined = combined.set_index("ticker", append=True)
        combined.index.names = ["date", "ticker"]
 
        # Clean per-ticker
        for ticker in combined.index.get_level_values("ticker").unique():
            mask = combined.index.get_level_values("ticker") == ticker
            ticker_data = combined.loc[mask]
            # Drop columns that are all NaN for this ticker
            all_nan_cols = ticker_data.columns[ticker_data.isnull().all()]
            if len(all_nan_cols) > 0:
                logger.warning(f"  {ticker}: {len(all_nan_cols)} all-NaN columns")
 
        logger.info(f"Sector features: {combined.shape}")
        return combined
 
    def build_all(
        self,
        snapshot: Dict[str, Any],
        reduce: Optional[bool] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Build all feature sets from a snapshot.
 
        Parameters
        ----------
        snapshot : dict
            Loaded snapshot.
        reduce : bool or None
            Override dimensionality reduction for observation features.
            None = use engine default (self.reduce_dims).
 
        Returns
        -------
        dict with keys:
            'observation'     : pd.DataFrame — Features for regime detection
            'observation_raw' : pd.DataFrame — Raw features before reduction
                                (only present if reduce=True, useful for analysis)
            'sector'          : pd.DataFrame — Features for sector allocation
            'metadata'        : dict — Feature computation summary
        """
        # Build raw observation features first (no reduction)
        obs_raw = self.build_observation_features(snapshot, reduce=False)
 
        # Apply reduction if requested
        should_reduce = reduce if reduce is not None else self.reduce_dims
        if should_reduce:
            obs = self.reduce_dimensions(obs_raw, fit=True)
        else:
            obs = obs_raw
 
        sector = self.build_sector_features(snapshot)
 
        metadata = {
            "observation_features": list(obs.columns),
            "observation_shape": list(obs.shape),
            "observation_date_range": [
                str(obs.index[0].date()),
                str(obs.index[-1].date()),
            ],
            "sector_shape": list(sector.shape),
            "sector_tickers": list(
                sector.index.get_level_values("ticker").unique()
            ),
            "observation_nan_pct": float(obs.isnull().mean().mean()),
            "dimensionality_reduction": should_reduce,
        }
 
        # Add PCA diagnostics if reduction was applied
        if should_reduce and self._pca is not None:
            metadata["pca_n_components"] = self._pca.n_components_
            metadata["pca_explained_variance_total"] = float(
                self._pca.explained_variance_ratio_.sum()
            )
            metadata["pca_explained_variance_per_component"] = [
                round(float(v), 4) for v in self._pca.explained_variance_ratio_
            ]
            metadata["raw_features_before_reduction"] = obs_raw.shape[1]
            metadata["features_after_corr_filter"] = len(self._retained_cols) if self._retained_cols else None
 
        logger.info(
            f"All features built: observation {obs.shape}, sector {sector.shape}"
        )
 
        result = {
            "observation": obs,
            "sector": sector,
            "metadata": metadata,
        }
 
        # Include raw features for analysis when reduction is active
        if should_reduce:
            result["observation_raw"] = obs_raw
 
        return result
 
    def _clean_features(
        self,
        df: pd.DataFrame,
        label: str,
    ) -> pd.DataFrame:
        """
        Clean feature DataFrame: trim lookback NaNs, drop bad columns,
        forward-fill remaining gaps.
 
        Parameters
        ----------
        df : pd.DataFrame
            Raw feature DataFrame.
        label : str
            Label for logging.
 
        Returns
        -------
        pd.DataFrame
            Cleaned feature DataFrame.
        """
        initial_shape = df.shape
        initial_nans = df.isnull().sum().sum()
 
        # ---- Drop columns with excessive NaNs ----
        nan_frac = df.isnull().mean()
        bad_cols = nan_frac[nan_frac > self.nan_threshold].index.tolist()
        if bad_cols:
            logger.warning(
                f"  [{label}] Dropping {len(bad_cols)} columns with "
                f">{self.nan_threshold:.0%} NaN: {bad_cols[:5]}..."
            )
            df = df.drop(columns=bad_cols)
 
        # ---- Trim leading NaN rows (lookback period) ----
        # Find the first row where at least 80% of features are non-NaN
        non_nan_frac = df.notna().mean(axis=1)
        valid_mask = non_nan_frac >= 0.8
        if valid_mask.any():
            first_valid = valid_mask.idxmax()
            df = df.loc[first_valid:]
 
        # ---- Forward-fill remaining NaNs (handles staggered asset starts) ----
        remaining_nans = df.isnull().sum().sum()
        if remaining_nans > 0:
            df = df.ffill().bfill()  # ffill first, then bfill for any leading gaps
            final_nans = df.isnull().sum().sum()
            if final_nans > 0:
                # Last resort: fill with 0 for any remaining NaNs
                logger.warning(
                    f"  [{label}] {final_nans} NaNs remain after ffill/bfill, filling with 0"
                )
                df = df.fillna(0)
 
        # ---- Replace infinities ----
        inf_count = np.isinf(df.values).sum()
        if inf_count > 0:
            logger.warning(f"  [{label}] Replacing {inf_count} inf values with NaN → ffill")
            df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
 
        logger.info(
            f"  [{label}] Cleaned: {initial_shape} → {df.shape}, "
            f"NaNs: {initial_nans} → {df.isnull().sum().sum()}"
        )
 
        return df
 
    def reduce_dimensions(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Reduce feature dimensionality for HMM stability.
 
        Two-stage process:
            1. Correlation filter: drop one feature from each pair with
               |correlation| > threshold. Removes pure redundancy cheaply.
            2. PCA: project remaining features into n_components principal
               components. Reduces parameter count from O(n²) to O(k²)
               for the HMM's covariance matrices.
 
        The fitted scaler, PCA, and retained column list are stored on the
        engine instance so that daily live data can be transformed consistently
        using transform mode (fit=False).
 
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned feature matrix (output of _clean_features).
        fit : bool
            If True, fit scaler + PCA on this data (training mode).
            If False, reuse previously fitted objects (inference mode).
 
        Returns
        -------
        pd.DataFrame
            Reduced feature matrix. Columns = PC_00, PC_01, ... PC_{n-1}.
        """
        if not fit:
            # ---- Transform mode: reuse fitted objects ----
            if self._pca is None or self._scaler is None or self._retained_cols is None:
                raise RuntimeError(
                    "reduce_dimensions called with fit=False but no fitted PCA exists. "
                    "Call with fit=True first (on training data)."
                )
            # Apply same column filter
            available_cols = [c for c in self._retained_cols if c in df.columns]
            missing_cols = [c for c in self._retained_cols if c not in df.columns]
            if missing_cols:
                logger.warning(
                    f"  [reduce_dims] {len(missing_cols)} training columns missing "
                    f"in new data, filling with 0: {missing_cols[:5]}..."
                )
                for c in missing_cols:
                    df[c] = 0.0
            df_filtered = df[self._retained_cols]
 
            # Apply same scaling and PCA transform
            scaled = self._scaler.transform(df_filtered)
            components = self._pca.transform(scaled)
 
            pc_cols = [f"PC_{i:02d}" for i in range(components.shape[1])]
            return pd.DataFrame(components, index=df_filtered.index, columns=pc_cols)
 
        # ---- Fit mode: learn correlation filter, scaler, and PCA ----
        logger.info(
            f"  [reduce_dims] Starting: {df.shape[1]} features, "
            f"target {self.n_components} components"
        )
 
        # Stage 1: Correlation-based filtering
        df_filtered = self._filter_correlated(df)
 
        # Stage 2: Standardize
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_filtered)
 
        # Stage 3: PCA
        n_comp = min(self.n_components, df_filtered.shape[1], df_filtered.shape[0])
        pca = PCA(n_components=n_comp, random_state=42)
        components = pca.fit_transform(scaled)
 
        # Store fitted objects for reuse
        self._scaler = scaler
        self._pca = pca
        self._retained_cols = list(df_filtered.columns)
 
        # Log variance explained
        cum_var = pca.explained_variance_ratio_.cumsum()
        logger.info(
            f"  [reduce_dims] PCA: {n_comp} components explain "
            f"{cum_var[-1]:.1%} of variance"
        )
        logger.info(
            f"  [reduce_dims] Variance by component: "
            f"{[f'{v:.1%}' for v in pca.explained_variance_ratio_[:5]]}..."
        )
        logger.info(
            f"  [reduce_dims] Final: {df.shape[1]} → {df_filtered.shape[1]} "
            f"(corr filter) → {n_comp} (PCA)"
        )
 
        pc_cols = [f"PC_{i:02d}" for i in range(n_comp)]
        return pd.DataFrame(components, index=df_filtered.index, columns=pc_cols)
 
    def _reduce_block(
        self,
        df: pd.DataFrame,
        n_components: int,
        label: str,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Reduce a feature block independently via correlation filter + PCA.
 
        Parameters
        ----------
        df : pd.DataFrame
            Feature block to reduce.
        n_components : int
            Number of PCA components to produce.
        label : str
            Label for column naming (e.g., 'tech' → TC_00, 'macro' → MC_00).
        fit : bool
            If True, fit new PCA. If False, reuse stored objects.
 
        Returns
        -------
        pd.DataFrame
            Reduced features with labeled columns.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
 
        prefix = label[0].upper() + "C"  # TC_ for tech, MC_ for macro
 
        if not fit:
            # Reuse stored objects
            key = f"_pca_{label}"
            scaler_key = f"_scaler_{label}"
            cols_key = f"_cols_{label}"
 
            pca = getattr(self, key, None)
            scaler = getattr(self, scaler_key, None)
            retained = getattr(self, cols_key, None)
 
            if pca is None or scaler is None or retained is None:
                raise RuntimeError(f"No fitted PCA for block '{label}'. Fit first.")
 
            available = [c for c in retained if c in df.columns]
            for c in retained:
                if c not in df.columns:
                    df[c] = 0.0
            df_filtered = df[retained]
 
            scaled = scaler.transform(df_filtered)
            components = pca.transform(scaled)
            pc_cols = [f"{prefix}_{i:02d}" for i in range(components.shape[1])]
            return pd.DataFrame(components, index=df_filtered.index, columns=pc_cols)
 
        # Fit mode
        df_filtered = self._filter_correlated(df)
 
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_filtered)
 
        n_comp = min(n_components, df_filtered.shape[1], df_filtered.shape[0])
        pca = PCA(n_components=n_comp, random_state=42)
        components = pca.fit_transform(scaled)
 
        # Store fitted objects
        setattr(self, f"_pca_{label}", pca)
        setattr(self, f"_scaler_{label}", scaler)
        setattr(self, f"_cols_{label}", list(df_filtered.columns))
 
        var_explained = pca.explained_variance_ratio_.sum()
        logger.info(
            f"  [{label}] PCA: {df.shape[1]} → {df_filtered.shape[1]} "
            f"(corr filter) → {n_comp} components ({var_explained:.1%} variance)"
        )
 
        pc_cols = [f"{prefix}_{i:02d}" for i in range(n_comp)]
        return pd.DataFrame(components, index=df_filtered.index, columns=pc_cols)
 
    def _filter_correlated(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Drop one feature from each highly correlated pair.
 
        Uses a greedy approach: for each pair with |corr| > threshold,
        drop the feature that has higher average correlation with all
        other features (i.e., the more redundant one).
 
        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix.
 
        Returns
        -------
        pd.DataFrame
            Filtered feature matrix with redundant columns removed.
        """
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
 
        # Find columns to drop
        to_drop = set()
        for col in upper.columns:
            high_corr = upper.index[upper[col] > self.correlation_threshold].tolist()
            if high_corr:
                # Drop the one with higher mean correlation to everything else
                mean_corr_col = corr[col].mean()
                for hc in high_corr:
                    if hc not in to_drop:
                        mean_corr_hc = corr[hc].mean()
                        if mean_corr_col > mean_corr_hc:
                            to_drop.add(col)
                        else:
                            to_drop.add(hc)
 
        if to_drop:
            logger.info(
                f"  [reduce_dims] Correlation filter: dropping {len(to_drop)} features "
                f"with |corr| > {self.correlation_threshold}"
            )
 
        return df.drop(columns=list(to_drop))
 
    def get_feature_names(
        self,
        snapshot: Dict[str, Any],
        universe: str = "observation",
    ) -> List[str]:
        """
        Get feature names without computing full features.
 
        Useful for inspecting what features will be generated.
 
        Parameters
        ----------
        snapshot : dict
            Loaded snapshot.
        universe : str
            'observation' or 'sector'.
 
        Returns
        -------
        list of str
            Feature column names.
        """
        if universe == "observation":
            feats = self.build_observation_features(snapshot)
        else:
            feats = self.build_sector_features(snapshot)
        return list(feats.columns)
 
 
# ---------------------------------------------------------------------------
# Walk-Forward Feature Slicing
# ---------------------------------------------------------------------------
 
def get_walkforward_slice(
    features: pd.DataFrame,
    train_end: str,
    train_lookback_days: int = 756,  # ~3 years
    purge_days: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract train/test slices for walk-forward validation with purging.
 
    The purge gap between train and test prevents information leakage
    from overlapping rolling windows.
 
    Parameters
    ----------
    features : pd.DataFrame
        Full feature matrix (from build_observation_features).
    train_end : str
        Last date of training period (YYYY-MM-DD).
    train_lookback_days : int
        Number of trading days for training window.
    purge_days : int
        Gap between train and test to prevent leakage.
 
    Returns
    -------
    train_features : pd.DataFrame
        Training slice.
    test_features : pd.DataFrame
        Test slice (starts purge_days after train_end).
    """
    train_end_dt = pd.Timestamp(train_end)
 
    # Training window
    train_start_dt = train_end_dt - pd.offsets.BDay(train_lookback_days)
    train_mask = (features.index >= train_start_dt) & (features.index <= train_end_dt)
    train_features = features.loc[train_mask]
 
    # Test window (after purge gap)
    test_start_dt = train_end_dt + pd.offsets.BDay(purge_days)
    test_mask = features.index >= test_start_dt
    test_features = features.loc[test_mask]
 
    return train_features, test_features
 
 
# ---------------------------------------------------------------------------
# CLI / Example Usage
# ---------------------------------------------------------------------------
 
def main():
    """Example usage and smoke test for feature engineering."""
    import argparse
 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
 
    parser = argparse.ArgumentParser(
        description="SPY Alpha v8 — Feature Engineering"
    )
    parser.add_argument("--snapshot", type=str, required=True, help="Snapshot name to load.")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory path.")
    parser.add_argument(
        "--universe",
        choices=["observation", "sector", "all"],
        default="all",
        help="Which feature set to build.",
    )
    args = parser.parse_args()
 
    from data_pipeline import SnapshotManager
 
    mgr = SnapshotManager(data_dir=args.data_dir)
    snap = mgr.load_snapshot(args.snapshot)
 
    engine = FeatureEngine()
 
    if args.universe in ("observation", "all"):
        obs = engine.build_observation_features(snap)
        print(f"\n{'='*60}")
        print(f"Observation Features")
        print(f"{'='*60}")
        print(f"  Shape:       {obs.shape}")
        print(f"  Date range:  {obs.index[0].date()} → {obs.index[-1].date()}")
        print(f"  NaN pct:     {obs.isnull().mean().mean():.4%}")
        print(f"  Features:    {list(obs.columns[:10])}...")
        print(f"\n  Sample (last 3 rows, first 5 cols):")
        print(obs.iloc[-3:, :5].to_string())
 
    if args.universe in ("sector", "all"):
        sector = engine.build_sector_features(snap)
        print(f"\n{'='*60}")
        print(f"Sector Features")
        print(f"{'='*60}")
        print(f"  Shape:       {sector.shape}")
        tickers = sector.index.get_level_values("ticker").unique()
        print(f"  Tickers:     {list(tickers)}")
        print(f"  NaN pct:     {sector.isnull().mean().mean():.4%}")
 
    if args.universe == "all":
        all_feats = engine.build_all(snap)
        print(f"\n{'='*60}")
        print(f"Feature Summary")
        print(f"{'='*60}")
        for k, v in all_feats["metadata"].items():
            print(f"  {k}: {v}")
 
    print("\n✓ Feature engineering complete")
 
 
if __name__ == "__main__":
    main()