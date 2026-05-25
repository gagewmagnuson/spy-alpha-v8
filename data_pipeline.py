"""
SPY Alpha v8 — Data Pipeline & Caching System
================================================
 
Foundational data layer implementing frozen-snapshot architecture.
 
Changes from v7:
    - Merged extended dataset support (index proxies, extended macro) from deep learner repo
    - Added stress data fetching (NFCI, STLFSI4) for state representation
    - Added VIX term structure tickers (^VIX3M, ^SKEW) to fresh data pulls
    - Version tag updated to v8.0
    - Environment variable updated to SPY_V8_DATA_DIR
 
Preserved from v7 (non-negotiable):
    - Raw (unadjusted) prices for ALL feature engineering
    - Adjusted close used ONLY for return calculations in the backtester
    - Frozen Parquet snapshots with MD5 checksums
    - Only daily mode pulls fresh data for live predictions
    - Full metadata stored with every snapshot
    - All asset universe definitions unchanged
 
Preserved from v7 deep learner repo:
    - Index proxy mappings (QQQ→NASDAQCOM, TLT→DGS20, etc.)
    - Extended macro series (22 FRED series back to 1970)
    - Proxy feature matrix builder
    - Extended feature matrix builder (price + macro + cross-domain)
"""
 
from __future__ import annotations
 
import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
 
import pandas as pd
import yfinance as yf
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
logger = logging.getLogger("spy_alpha_v8.data_pipeline")
 
DEFAULT_DATA_DIR = Path("data")
 
# ---- Asset Universes (unchanged from v7) ----
 
# Observation Universe (14 assets — regime detection only)
OBSERVATION_UNIVERSE: List[str] = [
    # Equities
    "SPY", "QQQ", "IWM", "VEA", "VWO",
    # Developed International
    "EFA", "EWJ",
    # Rates
    "TLT", "IEF", "SHY",
    # Credit
    "HYG",
    # Commodities / Inflation
    "GLD", "DBC",
    # Volatility
    "^VIX",
]
 
# Trading Universe — Tier 1: Macro Core (always in selection pool)
TIER1_MACRO: List[str] = [
    "SPY", "QQQ", "IWM", "VWO", "TLT", "GLD", "UPRO", "SHY",
]
 
# Trading Universe — Tier 2: Sector ETFs
TIER2_SECTORS: List[str] = [
    "XLK", "XLV", "XLF", "XLY", "XLP",
    "XLE", "XLI", "XLB", "XLRE", "XLU", "XLC",
]
 
# Trading Universe — Tier 3: Thematic ETFs
TIER3_THEMATIC: List[str] = [
    "SMH", "XBI", "XME",
]
 
# Trading Universe — Tier 4: Elite Individual Stocks
TIER4_STOCKS: List[str] = [
    "AAPL", "NVDA", "MSFT", "AMZN", "TSLA",
    "META", "GOOGL", "JPM", "LLY", "UNH", "XOM", "CAT",
]
 
# Stock-to-sector parent mapping (for Layer 2 alpha)
STOCK_SECTOR_MAP: Dict[str, str] = {
    "AAPL": "XLK", "NVDA": "XLK", "MSFT": "XLK",
    "AMZN": "XLY", "TSLA": "XLY",
    "META": "XLC", "GOOGL": "XLC",
    "JPM": "XLF",
    "LLY": "XLV", "UNH": "XLV",
    "XOM": "XLE",
    "CAT": "XLI",
}
 
# Full trading universe (all tiers)
TRADING_UNIVERSE: List[str] = TIER1_MACRO + TIER2_SECTORS + TIER3_THEMATIC + TIER4_STOCKS
 
# Combined unique list for data pulls
ALL_TICKERS: List[str] = sorted(set(OBSERVATION_UNIVERSE + TRADING_UNIVERSE))
 
# ---- FRED Series for HMM features (unchanged from v7 — DO NOT MODIFY) ----
 
FRED_SERIES: Dict[str, str] = {
    "T10Y2Y":  "10Y-2Y Treasury Spread",
    "T10Y3M":  "10Y-3M Treasury Spread",
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",
    "UNRATE":  "Unemployment Rate",
    "CPIAUCSL": "CPI All Urban Consumers",
    "FEDFUNDS": "Federal Funds Rate",
    "UMCSENT": "U of Michigan Consumer Sentiment",
    "ICSA":    "Initial Jobless Claims",
}
 
# ---- Stress FRED Series (NEW in v8 — pulled fresh, NOT in snapshot) ----
 
STRESS_FRED_SERIES: Dict[str, str] = {
    "NFCI":    "Chicago Fed National Financial Conditions Index",
    "STLFSI4": "St. Louis Fed Financial Stress Index",
}
 
# ---- VIX Term Structure Tickers (NEW in v8 — pulled fresh via yfinance) ----
 
VIX_TERM_TICKERS: List[str] = ["^VIX", "^VIX3M", "^SKEW"]
 
# ---- Extended History Index Proxies (ported from deep learner repo) ----
 
INDEX_PROXIES: Dict[str, Dict[str, str]] = {
    "QQQ": {"series": "NASDAQCOM", "type": "price", "description": "NASDAQ Composite"},
    "TLT": {"series": "DGS20", "type": "yield", "description": "20-Year Treasury Yield"},
    "SHY": {"series": "DGS1", "type": "yield", "description": "1-Year Treasury Yield"},
    "VIX": {"series": "VIXCLS", "type": "price", "description": "CBOE VIX Index"},
}
 
# ---- Extended Macro Series (ported from deep learner repo) ----
 
EXTENDED_MACRO_SERIES: Dict[str, Dict[str, str]] = {
    # Yield Curve (1970+)
    "DGS10": {"category": "rates", "description": "10-Year Treasury Yield"},
    "DGS2": {"category": "rates", "description": "2-Year Treasury Yield"},
    "DGS30": {"category": "rates", "description": "30-Year Treasury Yield"},
    "T10Y2Y": {"category": "yield_curve", "description": "10Y-2Y Spread"},
    "T10Y3M": {"category": "yield_curve", "description": "10Y-3M Spread"},
    "FEDFUNDS": {"category": "rates", "description": "Fed Funds Rate"},
    "DPRIME": {"category": "rates", "description": "Bank Prime Rate"},
 
    # Credit Stress (1983+)
    "DAAA": {"category": "credit", "description": "Moodys AAA Corporate Yield"},
    "DBAA": {"category": "credit", "description": "Moodys BAA Corporate Yield"},
    "AAA10Y": {"category": "credit", "description": "AAA-10Y Spread"},
    "BAA10Y": {"category": "credit", "description": "BAA-10Y Spread"},
    "TEDRATE": {"category": "credit", "description": "TED Spread"},
 
    # Inflation & Real Economy (1970+)
    "CPIAUCSL": {"category": "inflation", "description": "CPI All Urban Consumers"},
    "UNRATE": {"category": "labor", "description": "Unemployment Rate"},
    "ICSA": {"category": "labor", "description": "Initial Jobless Claims"},
    "INDPRO": {"category": "production", "description": "Industrial Production"},
    "M2SL": {"category": "liquidity", "description": "M2 Money Supply"},
    "PERMIT": {"category": "housing", "description": "Building Permits"},
    "UMCSENT": {"category": "sentiment", "description": "Consumer Sentiment"},
 
    # Currency & Dollar (1971+)
    "DEXJPUS": {"category": "currency", "description": "USD/JPY Exchange Rate"},
    "DTWEXM": {"category": "currency", "description": "Trade-Weighted Dollar Major"},
 
    # Commodities (1986+)
    "DCOILWTICO": {"category": "commodities", "description": "WTI Crude Oil Price"},
 
    # Volatility (1990+)
    "VIXCLS": {"category": "volatility", "description": "CBOE VIX Index"},
}
 
DEFAULT_START_DATE = "2005-01-01"
DEFAULT_END_DATE_SENTINEL = "today"
 
 
# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
 
def _compute_checksum(df: pd.DataFrame) -> str:
    """Compute a deterministic MD5 checksum of a DataFrame."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()
 
 
def _now_utc_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()
 
 
# ---------------------------------------------------------------------------
# Core Data Fetchers
# ---------------------------------------------------------------------------
 
def fetch_yfinance_raw(
    tickers: List[str],
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch OHLCV data from yfinance using auto_adjust=False.
 
    Returns two DataFrames:
        raw_prices : DataFrame with MultiIndex columns (field, ticker)
        adj_close  : DataFrame with single-level columns (ticker)
 
    All prices are UNADJUSTED (raw) except adj_close which is used
    ONLY for return calculations. This is non-negotiable — see v5 post-mortem.
    """
    if end is None or end == DEFAULT_END_DATE_SENTINEL:
        end = datetime.now().strftime("%Y-%m-%d")
 
    logger.info(f"Fetching yfinance data for {len(tickers)} tickers: {start} → {end}")
 
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        threads=True,
        progress=False,
    )
 
    if data.empty:
        raise RuntimeError("yfinance returned empty DataFrame — check tickers and date range.")
 
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data["Adj Close"].copy()
        raw_fields = ["Open", "High", "Low", "Close", "Volume"]
        raw_prices = data[raw_fields].copy()
    else:
        adj_close = data[["Adj Close"]].copy()
        adj_close.columns = [tickers[0]]
        raw_prices = data[["Open", "High", "Low", "Close", "Volume"]].copy()
        raw_prices.columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], [tickers[0]]]
        )
 
    raw_prices = raw_prices.dropna(how="all")
    adj_close = adj_close.loc[raw_prices.index]
 
    # Log data availability per ticker for stock-specific monitoring
    if isinstance(raw_prices.columns, pd.MultiIndex):
        close_df = raw_prices["Close"]
        for ticker in TIER4_STOCKS:
            if ticker in close_df.columns:
                valid = close_df[ticker].notna().sum()
                first_valid = close_df[ticker].first_valid_index()
                logger.info(f"  {ticker}: {valid} valid days, starts {first_valid}")
 
    logger.info(
        f"Fetched {len(raw_prices)} trading days, "
        f"{raw_prices.isnull().sum().sum()} total NaN cells in raw prices"
    )
 
    return raw_prices, adj_close
 
 
def fetch_fred_series(
    series_ids: Optional[Dict[str, str]] = None,
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch macro series from FRED.
 
    Uses fredapi if available, otherwise returns empty DataFrame.
    """
    if series_ids is None:
        series_ids = FRED_SERIES
 
    if end is None or end == DEFAULT_END_DATE_SENTINEL:
        end = datetime.now().strftime("%Y-%m-%d")
 
    api_key = api_key or os.environ.get("FRED_API_KEY")
 
    if not api_key:
        logger.warning(
            "No FRED API key found (set FRED_API_KEY env var). "
            "Returning empty FRED DataFrame — macro features will be unavailable."
        )
        return pd.DataFrame()
 
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
 
        frames = {}
        for sid, name in series_ids.items():
            try:
                s = fred.get_series(sid, observation_start=start, observation_end=end)
                frames[sid] = s
                logger.info(f"  FRED {sid} ({name}): {len(s)} observations")
            except Exception as e:
                logger.warning(f"  FRED {sid} ({name}) failed: {e}")
 
        if not frames:
            logger.warning("No FRED series fetched successfully.")
            return pd.DataFrame()
 
        df = pd.DataFrame(frames)
        df = df.asfreq("B").ffill()
        return df
 
    except ImportError:
        logger.warning("fredapi not installed. Returning empty FRED DataFrame.")
        return pd.DataFrame()
 
 
# ---------------------------------------------------------------------------
# Stress Data Fetcher (NEW in v8)
# ---------------------------------------------------------------------------
 
def fetch_stress_data(
    api_key: Optional[str] = None,
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch fresh stress/state data for v8 state representation.
 
    Returns dict with:
        'stress_fred'  : DataFrame with NFCI and STLFSI4
        'vix_term'     : DataFrame with ^VIX, ^VIX3M, ^SKEW
 
    This data is pulled FRESH at runtime and is NOT stored in snapshots.
    It feeds into the state representation layer, not the HMM.
    """
    result = {
        "stress_fred": pd.DataFrame(),
        "vix_term": pd.DataFrame(),
    }
 
    # ---- FRED stress indicators ----
    stress_fred = fetch_fred_series(
        series_ids=STRESS_FRED_SERIES,
        start=start,
        end=end,
        api_key=api_key,
    )
    result["stress_fred"] = stress_fred
    if not stress_fred.empty:
        logger.info(
            f"Stress FRED data: {len(stress_fred)} days, "
            f"columns: {list(stress_fred.columns)}"
        )
 
    # ---- VIX term structure from yfinance ----
    if end is None or end == DEFAULT_END_DATE_SENTINEL:
        end = datetime.now().strftime("%Y-%m-%d")
 
    try:
        vix_data = yf.download(
            VIX_TERM_TICKERS,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
 
        if not vix_data.empty:
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_close = vix_data["Close"].copy()
            else:
                vix_close = vix_data[["Close"]].copy()
                vix_close.columns = [VIX_TERM_TICKERS[0]]
 
            vix_close = vix_close.dropna(how="all")
            result["vix_term"] = vix_close
            logger.info(
                f"VIX term structure data: {len(vix_close)} days, "
                f"columns: {list(vix_close.columns)}"
            )
    except Exception as e:
        logger.warning(f"VIX term structure fetch failed: {e}")
 
    return result
 
 
# ---------------------------------------------------------------------------
# Extended Dataset Fetchers (ported from deep learner repo)
# ---------------------------------------------------------------------------
 
def fetch_index_proxies(
    api_key: Optional[str] = None,
    start: str = "1970-01-01",
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch extended history index proxies from FRED for meta-allocator training.
 
    Returns dict mapping ETF ticker to DataFrame with columns:
        - price: raw index level (or synthetic price for yields)
        - returns: daily returns (aligned across proxies)
 
    Yield series (TLT, SHY) are converted to synthetic price series
    using duration-based approximation: daily_return ≈ -duration * yield_change
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
 
    if not api_key:
        logger.warning("No FRED API key — cannot fetch index proxies")
        return {}
 
    if end is None or end == DEFAULT_END_DATE_SENTINEL:
        end = datetime.now().strftime("%Y-%m-%d")
 
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
    except ImportError:
        logger.warning("fredapi not installed — cannot fetch index proxies")
        return {}
 
    proxies = {}
 
    for etf_ticker, config in INDEX_PROXIES.items():
        series_id = config["series"]
        series_type = config["type"]
 
        try:
            raw = fred.get_series(series_id, observation_start=start, observation_end=end)
            raw = raw.dropna()
 
            if len(raw) < 252:
                logger.warning(f"  Proxy {etf_ticker} ({series_id}): only {len(raw)} observations, skipping")
                continue
 
            if series_type == "yield":
                # Convert yield to synthetic bond price returns
                # Approximation: daily_return ≈ -modified_duration * daily_yield_change
                # Duration approximation: TLT ~ 17 years, SHY ~ 1 year
                duration = 17.0 if etf_ticker == "TLT" else 1.0
                yield_change = raw.diff() / 100  # daily change in yield (decimal)
                daily_returns = -duration * yield_change
                daily_returns = daily_returns.fillna(0)
 
                # Build synthetic price from returns
                synthetic_price = (1 + daily_returns).cumprod() * 100
 
                df = pd.DataFrame({
                    "price": synthetic_price,
                    "returns": daily_returns,
                    "raw_yield": raw,
                }, index=raw.index)
            else:
                # Price series — compute returns directly
                daily_returns = raw.pct_change().fillna(0)
 
                df = pd.DataFrame({
                    "price": raw,
                    "returns": daily_returns,
                }, index=raw.index)
 
            df.index = pd.to_datetime(df.index)
            df = df.asfreq("B").ffill()
 
            proxies[etf_ticker] = df
            logger.info(
                f"  Proxy {etf_ticker} ({series_id}): {len(df)} days, "
                f"{df.index[0].date()} → {df.index[-1].date()}"
            )
 
        except Exception as e:
            logger.warning(f"  Proxy {etf_ticker} ({series_id}) failed: {e}")
 
    # ---- Fetch SPY and GLD proxies from yfinance (better long history) ----
    try:
        yf_proxies = {
            "SPY": "^GSPC",   # S&P 500 index back to 1927
            "GLD": "GC=F",    # Gold futures back to ~1979
        }
 
        for etf_ticker, yf_ticker in yf_proxies.items():
            try:
                data = yf.download(yf_ticker, start=start, end=end, auto_adjust=False, progress=False)
                if len(data) < 252:
                    logger.warning(f"  Proxy {etf_ticker} ({yf_ticker}): only {len(data)} days, skipping")
                    continue
 
                price = data["Close"].squeeze()
                daily_returns = price.pct_change().fillna(0)
 
                df = pd.DataFrame({
                    "price": price,
                    "returns": daily_returns,
                }, index=price.index)
 
                df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz else pd.to_datetime(df.index)
 
                proxies[etf_ticker] = df
                logger.info(
                    f"  Proxy {etf_ticker} ({yf_ticker} via yfinance): {len(df)} days, "
                    f"{df.index[0].date()} → {df.index[-1].date()}"
                )
            except Exception as e:
                logger.warning(f"  Proxy {etf_ticker} ({yf_ticker}) failed: {e}")
    except ImportError:
        logger.warning("yfinance not available for proxy fetch")
 
    return proxies
 
 
def build_proxy_feature_matrix(
    proxies: Dict[str, pd.DataFrame],
    lookback_windows: List[int] = [5, 10, 21, 63, 126, 252],
) -> pd.DataFrame:
    """
    Build a feature matrix from index proxies for meta-allocator training.
 
    Uses raw returns — NOT PCA. The meta-allocator and deep embeddings
    perform their own feature extraction.
 
    Assets are introduced as they become available — earlier proxies
    provide data from their start date, later proxies have NaN until
    their data begins. This preserves maximum history.
 
    Features per proxy:
        - Returns at multiple horizons
        - Rolling volatility at multiple windows
        - Rolling Sharpe ratio
        - Drawdown from rolling peak
        - Cross-asset return correlations
    """
    if not proxies:
        return pd.DataFrame()
 
    # Normalize all indexes
    for ticker in proxies:
        proxies[ticker].index = pd.to_datetime(proxies[ticker].index).normalize()
 
    # Use the EARLIEST start and LATEST end across all proxies
    earliest_start = min(df.index[0] for df in proxies.values())
    latest_end = max(df.index[-1] for df in proxies.values())
 
    # Build a master date index from the proxy with the longest history
    longest_proxy = max(proxies.values(), key=lambda df: len(df))
    master_index = longest_proxy.loc[earliest_start:latest_end].index
 
    features = {}
 
    for ticker, df in proxies.items():
        # Reindex to master — NaN before this proxy's start date
        returns = df["returns"].reindex(master_index)
        price = df["price"].reindex(master_index)
 
        for window in lookback_windows:
            # Rolling returns
            features[f"{ticker}_ret_{window}d"] = returns.rolling(window).sum()
 
            # Rolling volatility (annualized)
            features[f"{ticker}_vol_{window}d"] = returns.rolling(window).std() * (252 ** 0.5)
 
            # Rolling Sharpe (annualized)
            roll_mean = returns.rolling(window).mean() * 252
            roll_std = returns.rolling(window).std() * (252 ** 0.5)
            features[f"{ticker}_sharpe_{window}d"] = roll_mean / roll_std.replace(0, float("nan"))
 
        # Drawdown from rolling peak
        rolling_peak = price.expanding().max()
        features[f"{ticker}_drawdown"] = (price - rolling_peak) / rolling_peak
 
        # Distance from 52-week high
        peak_252 = price.rolling(252).max()
        features[f"{ticker}_dist_52w_high"] = (price - peak_252) / peak_252
 
    # Cross-asset features: rolling correlations between key pairs
    key_pairs = [("SPY", "TLT"), ("SPY", "GLD"), ("SPY", "VIX"), ("TLT", "GLD")]
    for t1, t2 in key_pairs:
        if t1 in proxies and t2 in proxies:
            r1 = proxies[t1]["returns"].reindex(master_index)
            r2 = proxies[t2]["returns"].reindex(master_index)
            features[f"{t1}_{t2}_corr_63d"] = r1.rolling(63).corr(r2)
            features[f"{t1}_{t2}_corr_252d"] = r1.rolling(252).corr(r2)
 
    feature_df = pd.DataFrame(features)
 
    # Drop rows where ALL features are NaN, but keep rows with partial data
    feature_df = feature_df.dropna(how="all")
 
    logger.info(
        f"Proxy feature matrix: {feature_df.shape[0]} days × {feature_df.shape[1]} features, "
        f"{feature_df.index[0].date()} → {feature_df.index[-1].date()}, "
        f"NaN rate: {feature_df.isnull().mean().mean():.1%}"
    )
 
    return feature_df
 
 
def fetch_extended_macro(
    api_key: Optional[str] = None,
    start: str = "1970-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch all extended macro series from FRED and build into a single DataFrame.
 
    Series are forward-filled to daily frequency. Monthly/weekly series
    are interpolated to daily. Each series starts when it becomes available —
    earlier dates are NaN.
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.warning("No FRED API key — cannot fetch extended macro")
        return pd.DataFrame()
 
    if end is None or end == DEFAULT_END_DATE_SENTINEL:
        end = datetime.now().strftime("%Y-%m-%d")
 
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
    except ImportError:
        logger.warning("fredapi not installed")
        return pd.DataFrame()
 
    series_data = {}
    for sid, config in EXTENDED_MACRO_SERIES.items():
        try:
            data = fred.get_series(sid, observation_start=start, observation_end=end)
            data = data.dropna()
            if len(data) > 0:
                series_data[sid] = data
                logger.info(f"  Extended macro {sid}: {len(data)} obs, {data.index[0].date()} → {data.index[-1].date()}")
        except Exception as e:
            logger.warning(f"  Extended macro {sid} failed: {e}")
 
    if not series_data:
        return pd.DataFrame()
 
    # Combine into single DataFrame, resample to business daily
    df = pd.DataFrame(series_data)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("B")
    df = df.ffill()
 
    logger.info(f"Extended macro: {df.shape[0]} days × {df.shape[1]} series")
    return df
 
 
def build_extended_feature_matrix(
    proxy_prices: Dict[str, pd.DataFrame],
    macro_data: pd.DataFrame,
    lookback_windows: List[int] = [5, 10, 21, 63, 126, 252],
) -> pd.DataFrame:
    """
    Build a comprehensive feature matrix combining proxy price features
    and macro-causal features for the meta-allocator.
 
    Features include:
        - Price-based: returns, vol, Sharpe, drawdown (from proxy data)
        - Yield curve: level, slope, curvature, velocity of change
        - Credit: spreads, momentum, stress indicators
        - Macro: inflation momentum, employment trends, liquidity
        - Cross-asset: divergences, lead-lag signals
        - Volatility: level, term structure (when available)
    """
    features = {}
 
    # ---- Price-based features (from proxy data) ----
    # Normalize all indexes
    for ticker in proxy_prices:
        proxy_prices[ticker].index = pd.to_datetime(proxy_prices[ticker].index).normalize()
 
    earliest_start = min(df.index[0] for df in proxy_prices.values())
    latest_end = max(df.index[-1] for df in proxy_prices.values())
    longest_proxy = max(proxy_prices.values(), key=lambda df: len(df))
    master_index = longest_proxy.loc[earliest_start:latest_end].index
 
    for ticker, df in proxy_prices.items():
        returns = df["returns"].reindex(master_index)
        price = df["price"].reindex(master_index)
 
        for window in lookback_windows:
            features[f"{ticker}_ret_{window}d"] = returns.rolling(window).sum()
            features[f"{ticker}_vol_{window}d"] = returns.rolling(window).std() * (252 ** 0.5)
            roll_mean = returns.rolling(window).mean() * 252
            roll_std = returns.rolling(window).std() * (252 ** 0.5)
            features[f"{ticker}_sharpe_{window}d"] = roll_mean / roll_std.replace(0, float("nan"))
 
        rolling_peak = price.expanding().max()
        features[f"{ticker}_drawdown"] = (price - rolling_peak) / rolling_peak
        peak_252 = price.rolling(252).max()
        features[f"{ticker}_dist_52w_high"] = (price - peak_252) / peak_252
 
    # Cross-asset price features
    key_pairs = [("SPY", "TLT"), ("SPY", "GLD"), ("SPY", "VIX"), ("TLT", "GLD")]
    for t1, t2 in key_pairs:
        if t1 in proxy_prices and t2 in proxy_prices:
            r1 = proxy_prices[t1]["returns"].reindex(master_index)
            r2 = proxy_prices[t2]["returns"].reindex(master_index)
            features[f"{t1}_{t2}_corr_63d"] = r1.rolling(63).corr(r2)
            features[f"{t1}_{t2}_corr_252d"] = r1.rolling(252).corr(r2)
 
    # ---- Macro features ----
    if not macro_data.empty:
        macro_aligned = macro_data.reindex(master_index).ffill()
 
        # Yield curve features
        if "DGS10" in macro_aligned.columns:
            features["yield_10y"] = macro_aligned["DGS10"]
            features["yield_10y_chg_21d"] = macro_aligned["DGS10"].diff(21)
            features["yield_10y_chg_63d"] = macro_aligned["DGS10"].diff(63)
            features["yield_10y_chg_252d"] = macro_aligned["DGS10"].diff(252)
 
        if "DGS2" in macro_aligned.columns:
            features["yield_2y"] = macro_aligned["DGS2"]
            features["yield_2y_chg_21d"] = macro_aligned["DGS2"].diff(21)
 
        if "DGS10" in macro_aligned.columns and "DGS2" in macro_aligned.columns:
            spread = macro_aligned["DGS10"] - macro_aligned["DGS2"]
            features["curve_10y2y"] = spread
            features["curve_10y2y_chg_21d"] = spread.diff(21)
            features["curve_10y2y_chg_63d"] = spread.diff(63)
            features["curve_steepening"] = spread.diff(5)  # short-term velocity
 
        if "T10Y2Y" in macro_aligned.columns:
            features["t10y2y"] = macro_aligned["T10Y2Y"]
            features["t10y2y_velocity"] = macro_aligned["T10Y2Y"].diff(5)
            features["t10y2y_acceleration"] = macro_aligned["T10Y2Y"].diff(5).diff(5)
 
        if "T10Y3M" in macro_aligned.columns:
            features["t10y3m"] = macro_aligned["T10Y3M"]
            features["t10y3m_velocity"] = macro_aligned["T10Y3M"].diff(5)
 
        if "DGS30" in macro_aligned.columns and "DGS10" in macro_aligned.columns:
            features["curve_30y10y"] = macro_aligned["DGS30"] - macro_aligned["DGS10"]
 
        if "FEDFUNDS" in macro_aligned.columns:
            features["fed_funds"] = macro_aligned["FEDFUNDS"]
            features["fed_funds_chg_63d"] = macro_aligned["FEDFUNDS"].diff(63)
            if "DGS10" in macro_aligned.columns:
                features["real_rate_proxy"] = macro_aligned["DGS10"] - macro_aligned["FEDFUNDS"]
 
        # Credit stress features
        if "AAA10Y" in macro_aligned.columns:
            features["aaa_spread"] = macro_aligned["AAA10Y"]
            features["aaa_spread_chg_21d"] = macro_aligned["AAA10Y"].diff(21)
 
        if "BAA10Y" in macro_aligned.columns:
            features["baa_spread"] = macro_aligned["BAA10Y"]
            features["baa_spread_chg_21d"] = macro_aligned["BAA10Y"].diff(21)
            features["baa_spread_chg_63d"] = macro_aligned["BAA10Y"].diff(63)
 
        if "BAA10Y" in macro_aligned.columns and "AAA10Y" in macro_aligned.columns:
            features["credit_stress"] = macro_aligned["BAA10Y"] - macro_aligned["AAA10Y"]
            features["credit_stress_chg_21d"] = features["credit_stress"].diff(21)
 
        if "TEDRATE" in macro_aligned.columns:
            features["ted_spread"] = macro_aligned["TEDRATE"]
            features["ted_spread_chg_21d"] = macro_aligned["TEDRATE"].diff(21)
 
        # Inflation features
        if "CPIAUCSL" in macro_aligned.columns:
            cpi = macro_aligned["CPIAUCSL"]
            features["cpi_yoy"] = cpi.pct_change(252)  # year-over-year change
            features["cpi_mom"] = cpi.pct_change(21)    # month-over-month
            features["cpi_acceleration"] = cpi.pct_change(252).diff(63)
 
        # Labor market
        if "UNRATE" in macro_aligned.columns:
            features["unemployment"] = macro_aligned["UNRATE"]
            features["unemployment_chg_63d"] = macro_aligned["UNRATE"].diff(63)
            features["unemployment_chg_252d"] = macro_aligned["UNRATE"].diff(252)
 
        if "ICSA" in macro_aligned.columns:
            features["initial_claims"] = macro_aligned["ICSA"]
            features["claims_4w_avg"] = macro_aligned["ICSA"].rolling(20).mean()  # ~4 weeks
            features["claims_chg_63d"] = macro_aligned["ICSA"].pct_change(63)
 
        # Liquidity
        if "M2SL" in macro_aligned.columns:
            features["m2_yoy"] = macro_aligned["M2SL"].pct_change(252)
            features["m2_mom"] = macro_aligned["M2SL"].pct_change(21)
 
        # Production
        if "INDPRO" in macro_aligned.columns:
            features["indpro_yoy"] = macro_aligned["INDPRO"].pct_change(252)
            features["indpro_mom"] = macro_aligned["INDPRO"].pct_change(21)
 
        # Sentiment
        if "UMCSENT" in macro_aligned.columns:
            features["consumer_sentiment"] = macro_aligned["UMCSENT"]
            features["sentiment_chg_63d"] = macro_aligned["UMCSENT"].diff(63)
 
        # Currency
        if "DEXJPUS" in macro_aligned.columns:
            features["usdjpy"] = macro_aligned["DEXJPUS"]
            features["usdjpy_chg_21d"] = macro_aligned["DEXJPUS"].pct_change(21)
            features["usdjpy_chg_63d"] = macro_aligned["DEXJPUS"].pct_change(63)
 
        if "DTWEXM" in macro_aligned.columns:
            features["dollar_index"] = macro_aligned["DTWEXM"]
            features["dollar_chg_21d"] = macro_aligned["DTWEXM"].pct_change(21)
            features["dollar_chg_63d"] = macro_aligned["DTWEXM"].pct_change(63)
 
        # Commodities
        if "DCOILWTICO" in macro_aligned.columns:
            features["oil_price"] = macro_aligned["DCOILWTICO"]
            features["oil_chg_21d"] = macro_aligned["DCOILWTICO"].pct_change(21)
            features["oil_chg_63d"] = macro_aligned["DCOILWTICO"].pct_change(63)
            features["oil_vol_21d"] = macro_aligned["DCOILWTICO"].pct_change().rolling(21).std() * (252 ** 0.5)
 
        # VIX features
        if "VIXCLS" in macro_aligned.columns:
            features["vix"] = macro_aligned["VIXCLS"]
            features["vix_chg_5d"] = macro_aligned["VIXCLS"].diff(5)
            features["vix_chg_21d"] = macro_aligned["VIXCLS"].diff(21)
            features["vix_percentile_252d"] = macro_aligned["VIXCLS"].rolling(252).rank(pct=True)
 
        # ---- Cross-domain divergence features ----
        # Equity-Bond divergence
        if "SPY" in proxy_prices and "DGS10" in macro_aligned.columns:
            spy_ret_21 = proxy_prices["SPY"]["returns"].reindex(master_index).rolling(21).sum()
            yield_chg_21 = macro_aligned["DGS10"].diff(21)
            features["equity_bond_divergence"] = spy_ret_21 + yield_chg_21 * 0.1  # both rising = unusual
 
        # Equity-Credit divergence
        if "SPY" in proxy_prices and "BAA10Y" in macro_aligned.columns:
            spy_ret_21 = proxy_prices["SPY"]["returns"].reindex(master_index).rolling(21).sum()
            credit_chg = macro_aligned["BAA10Y"].diff(21)
            features["equity_credit_divergence"] = spy_ret_21 + credit_chg  # equity up + spreads widening = warning
 
        # Equity-VIX divergence
        if "SPY" in proxy_prices and "VIXCLS" in macro_aligned.columns:
            spy_ret_21 = proxy_prices["SPY"]["returns"].reindex(master_index).rolling(21).sum()
            vix_chg = macro_aligned["VIXCLS"].diff(21)
            features["equity_vix_divergence"] = spy_ret_21 + vix_chg / 100  # equity up + VIX up = unstable
 
        # Oil-Equity divergence (inflation signal)
        if "SPY" in proxy_prices and "DCOILWTICO" in macro_aligned.columns:
            spy_ret_63 = proxy_prices["SPY"]["returns"].reindex(master_index).rolling(63).sum()
            oil_ret_63 = macro_aligned["DCOILWTICO"].pct_change(63)
            features["oil_equity_divergence"] = oil_ret_63 - spy_ret_63  # oil up + equity down = inflation regime
 
    feature_df = pd.DataFrame(features)
    feature_df = feature_df.dropna(how="all")
 
    n_features = feature_df.shape[1]
    nan_rate = feature_df.isnull().mean().mean()
 
    logger.info(
        f"Extended feature matrix: {feature_df.shape[0]} days × {n_features} features, "
        f"{feature_df.index[0].date()} → {feature_df.index[-1].date()}, "
        f"NaN rate: {nan_rate:.1%}"
    )
 
    return feature_df
 
 
# ---------------------------------------------------------------------------
# Snapshot Manager
# ---------------------------------------------------------------------------
 
class SnapshotManager:
    """
    Manages frozen data snapshots for reproducible backtesting.
 
    Each snapshot is a directory containing:
        raw_prices.parquet       — Unadjusted OHLCV (for features)
        adj_close.parquet        — Adjusted close (for returns only)
        fred_data.parquet        — FRED macro series (core 8 for HMM)
        proxy_returns.parquet    — Extended proxy returns (1970+)
        proxy_prices.parquet     — Extended proxy prices (1970+)
        proxy_features.parquet   — Proxy feature matrix
        extended_macro.parquet   — Extended FRED macro (22 series, 1970+)
        extended_features.parquet — Combined extended feature matrix
        metadata.json            — Source, date range, checksums, ticker mappings
    """
 
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir or os.environ.get("SPY_V8_DATA_DIR", DEFAULT_DATA_DIR))
        self.snapshots_dir = self.data_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SnapshotManager initialized: {self.snapshots_dir}")
 
    def _snapshot_path(self, name: str) -> Path:
        return self.snapshots_dir / name
 
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots with metadata."""
        snapshots = []
        if not self.snapshots_dir.exists():
            return snapshots
 
        for d in sorted(self.snapshots_dir.iterdir()):
            if d.is_dir() and (d / "metadata.json").exists():
                with open(d / "metadata.json") as f:
                    meta = json.load(f)
                meta["snapshot_name"] = d.name
                snapshots.append(meta)
 
        return snapshots
 
    def create_snapshot(
        self,
        name: str,
        tickers: Optional[List[str]] = None,
        start: str = DEFAULT_START_DATE,
        end: Optional[str] = None,
        fred_api_key: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Pull fresh data and freeze it as a named snapshot.
 
        Creates both the core snapshot (for HMM) and extended dataset
        (for meta-allocator training and deep embeddings).
 
        Handles stock-specific data considerations:
        - Individual stocks may not have full history (graceful NaN handling)
        - Ticker changes are recorded in metadata
        """
        snap_dir = self._snapshot_path(name)
 
        if snap_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Snapshot '{name}' already exists at {snap_dir}. "
                f"Use overwrite=True to replace."
            )
 
        snap_dir.mkdir(parents=True, exist_ok=True)
        tickers = tickers or ALL_TICKERS
 
        # ---- Fetch market data ----
        raw_prices, adj_close = fetch_yfinance_raw(tickers, start=start, end=end)
 
        # ---- Fetch FRED data ----
        fred_data = fetch_fred_series(start=start, end=end, api_key=fred_api_key)
 
        # ---- Save core data as Parquet ----
        raw_prices.to_parquet(snap_dir / "raw_prices.parquet")
        adj_close.to_parquet(snap_dir / "adj_close.parquet")
        if not fred_data.empty:
            fred_data.to_parquet(snap_dir / "fred_data.parquet")
 
        # ---- Fetch and save index proxies for meta-allocator ----
        proxies = fetch_index_proxies(api_key=fred_api_key, start="1970-01-01", end=end)
        if proxies:
            proxy_returns = pd.DataFrame({
                ticker: df["returns"] for ticker, df in proxies.items()
            })
            proxy_prices_df = pd.DataFrame({
                ticker: df["price"] for ticker, df in proxies.items()
            })
            proxy_features = build_proxy_feature_matrix(proxies)
 
            proxy_returns.to_parquet(snap_dir / "proxy_returns.parquet")
            proxy_prices_df.to_parquet(snap_dir / "proxy_prices.parquet")
            proxy_features.to_parquet(snap_dir / "proxy_features.parquet")
 
            logger.info(
                f"Index proxies saved: {len(proxies)} series, "
                f"features {proxy_features.shape}"
            )
 
        # ---- Fetch and save extended macro features ----
        extended_macro = fetch_extended_macro(api_key=fred_api_key, start="1970-01-01", end=end)
        if not extended_macro.empty and proxies:
            extended_features = build_extended_feature_matrix(proxies, extended_macro)
            extended_features.to_parquet(snap_dir / "extended_features.parquet")
            extended_macro.to_parquet(snap_dir / "extended_macro.parquet")
            logger.info(f"Extended features saved: {extended_features.shape}")
 
        # ---- Read back for checksums (critical: round-trip matching) ----
        raw_prices_saved = pd.read_parquet(snap_dir / "raw_prices.parquet")
        adj_close_saved = pd.read_parquet(snap_dir / "adj_close.parquet")
        fred_saved = pd.read_parquet(snap_dir / "fred_data.parquet") if not fred_data.empty else fred_data
 
        # ---- Catalog stock data availability ----
        stock_availability = {}
        if isinstance(raw_prices.columns, pd.MultiIndex) and "Close" in raw_prices.columns.get_level_values(0):
            close_df = raw_prices["Close"]
            for ticker in TIER4_STOCKS:
                if ticker in close_df.columns:
                    valid_count = int(close_df[ticker].notna().sum())
                    first = close_df[ticker].first_valid_index()
                    last = close_df[ticker].last_valid_index()
                    stock_availability[ticker] = {
                        "valid_days": valid_count,
                        "first_date": str(first.date()) if first is not None else None,
                        "last_date": str(last.date()) if last is not None else None,
                    }
 
        # ---- Build metadata ----
        actual_end = raw_prices.index[-1].strftime("%Y-%m-%d")
        actual_start = raw_prices.index[0].strftime("%Y-%m-%d")
 
        metadata = {
            "created_at": _now_utc_iso(),
            "version": "v8.0",
            "tickers": tickers,
            "observation_universe": OBSERVATION_UNIVERSE,
            "trading_universe": TRADING_UNIVERSE,
            "tier1_macro": TIER1_MACRO,
            "tier2_sectors": TIER2_SECTORS,
            "tier3_thematic": TIER3_THEMATIC,
            "tier4_stocks": TIER4_STOCKS,
            "stock_sector_map": STOCK_SECTOR_MAP,
            "stock_availability": stock_availability,
            "requested_start": start,
            "requested_end": end or "today",
            "actual_start": actual_start,
            "actual_end": actual_end,
            "trading_days": len(raw_prices),
            "raw_prices_checksum": _compute_checksum(raw_prices_saved),
            "adj_close_checksum": _compute_checksum(adj_close_saved),
            "fred_checksum": _compute_checksum(fred_saved) if not fred_data.empty else None,
            "fred_series": list(fred_data.columns) if not fred_data.empty else [],
            "raw_prices_shape": list(raw_prices.shape),
            "adj_close_shape": list(adj_close.shape),
            "fred_shape": list(fred_data.shape) if not fred_data.empty else [0, 0],
            "nan_summary": {
                "raw_prices_total_nans": int(raw_prices.isnull().sum().sum()),
                "adj_close_total_nans": int(adj_close.isnull().sum().sum()),
                "fred_total_nans": int(fred_data.isnull().sum().sum()) if not fred_data.empty else 0,
            },
        }
 
        with open(snap_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
 
        logger.info(
            f"Snapshot '{name}' created: {metadata['trading_days']} days, "
            f"{actual_start} → {actual_end}, {len(tickers)} tickers"
        )
        return snap_dir
 
    def load_snapshot(
        self,
        name: str,
        verify_checksums: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a frozen snapshot for backtesting.
 
        Returns dict with:
            'raw_prices', 'adj_close', 'fred_data',
            'proxy_features', 'proxy_returns', 'proxy_prices',
            'extended_features', 'extended_macro',
            'metadata'
        """
        snap_dir = self._snapshot_path(name)
        if not snap_dir.exists():
            raise FileNotFoundError(f"Snapshot '{name}' not found at {snap_dir}")
 
        with open(snap_dir / "metadata.json") as f:
            metadata = json.load(f)
 
        raw_prices = pd.read_parquet(snap_dir / "raw_prices.parquet")
        adj_close = pd.read_parquet(snap_dir / "adj_close.parquet")
 
        fred_path = snap_dir / "fred_data.parquet"
        fred_data = pd.read_parquet(fred_path) if fred_path.exists() else pd.DataFrame()
 
        # Extended dataset files (may not exist in older snapshots)
        proxy_features_path = snap_dir / "proxy_features.parquet"
        proxy_returns_path = snap_dir / "proxy_returns.parquet"
        proxy_prices_path = snap_dir / "proxy_prices.parquet"
 
        proxy_features = pd.read_parquet(proxy_features_path) if proxy_features_path.exists() else pd.DataFrame()
        proxy_returns = pd.read_parquet(proxy_returns_path) if proxy_returns_path.exists() else pd.DataFrame()
        proxy_prices = pd.read_parquet(proxy_prices_path) if proxy_prices_path.exists() else pd.DataFrame()
 
        extended_features_path = snap_dir / "extended_features.parquet"
        extended_macro_path = snap_dir / "extended_macro.parquet"
 
        extended_features = pd.read_parquet(extended_features_path) if extended_features_path.exists() else pd.DataFrame()
        extended_macro = pd.read_parquet(extended_macro_path) if extended_macro_path.exists() else pd.DataFrame()
 
        if verify_checksums:
            raw_ck = _compute_checksum(raw_prices)
            adj_ck = _compute_checksum(adj_close)
 
            if raw_ck != metadata.get("raw_prices_checksum"):
                raise RuntimeError(
                    f"CHECKSUM MISMATCH: raw_prices in snapshot '{name}' has been modified! "
                    f"Data integrity compromised — do not use this snapshot."
                )
            if adj_ck != metadata.get("adj_close_checksum"):
                raise RuntimeError(
                    f"CHECKSUM MISMATCH: adj_close in snapshot '{name}' has been modified! "
                    f"Data integrity compromised — do not use this snapshot."
                )
 
            if not fred_data.empty and metadata.get("fred_checksum"):
                fred_ck = _compute_checksum(fred_data)
                if fred_ck != metadata["fred_checksum"]:
                    logger.warning(
                        f"FRED checksum mismatch in snapshot '{name}'."
                    )
 
            logger.info(f"Snapshot '{name}' loaded and verified — checksums OK")
 
        return {
            "raw_prices": raw_prices,
            "adj_close": adj_close,
            "fred_data": fred_data,
            "proxy_features": proxy_features,
            "proxy_returns": proxy_returns,
            "proxy_prices": proxy_prices,
            "extended_features": extended_features,
            "extended_macro": extended_macro,
            "metadata": metadata,
        }
 
    def compare_snapshots(self, name_a: str, name_b: str) -> Dict[str, Any]:
        """Compare two snapshots to detect data drift."""
        snap_a = self.load_snapshot(name_a, verify_checksums=False)
        snap_b = self.load_snapshot(name_b, verify_checksums=False)
 
        report: Dict[str, Any] = {
            "snapshot_a": name_a,
            "snapshot_b": name_b,
            "compared_at": _now_utc_iso(),
        }
 
        report["shape_match"] = {
            "raw_prices": snap_a["raw_prices"].shape == snap_b["raw_prices"].shape,
            "adj_close": snap_a["adj_close"].shape == snap_b["adj_close"].shape,
        }
 
        common_idx = snap_a["raw_prices"].index.intersection(snap_b["raw_prices"].index)
        report["overlapping_days"] = len(common_idx)
 
        if len(common_idx) == 0:
            report["error"] = "No overlapping dates between snapshots"
            return report
 
        raw_a = snap_a["raw_prices"].loc[common_idx]
        raw_b = snap_b["raw_prices"].loc[common_idx]
        raw_diff = (raw_a - raw_b).abs()
 
        report["raw_prices_drift"] = {
            "max_abs_diff": float(raw_diff.max().max()),
            "mean_abs_diff": float(raw_diff.mean().mean()),
            "identical": bool((raw_diff == 0).all().all()),
            "changed_cells": int((raw_diff > 0).sum().sum()),
            "total_cells": int(raw_diff.size),
        }
 
        common_adj_idx = snap_a["adj_close"].index.intersection(snap_b["adj_close"].index)
        if len(common_adj_idx) > 0:
            adj_a = snap_a["adj_close"].loc[common_adj_idx]
            adj_b = snap_b["adj_close"].loc[common_adj_idx]
            adj_diff = (adj_a - adj_b).abs()
 
            report["adj_close_drift"] = {
                "max_abs_diff": float(adj_diff.max().max()),
                "mean_abs_diff": float(adj_diff.mean().mean()),
                "identical": bool((adj_diff == 0).all().all()),
                "changed_cells": int((adj_diff > 0).sum().sum()),
                "total_cells": int(adj_diff.size),
            }
 
        return report
 
 
# ---------------------------------------------------------------------------
# Convenience Accessors
# ---------------------------------------------------------------------------
 
def get_raw_close(snapshot: Dict[str, Any], tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extract raw (unadjusted) Close prices from a snapshot.
 
    This is the PRIMARY price source for all feature engineering.
    NEVER use adjusted prices for features.
    """
    raw = snapshot["raw_prices"]
 
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
 
    if tickers is not None:
        available = [t for t in tickers if t in close.columns]
        missing = [t for t in tickers if t not in close.columns]
        if missing:
            logger.warning(f"Tickers not in snapshot: {missing}")
        close = close[available]
 
    return close
 
 
def get_adj_close(snapshot: Dict[str, Any], tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extract adjusted close prices from a snapshot.
 
    ⚠️  USE ONLY FOR RETURN CALCULATIONS IN THE BACKTESTER.
    ⚠️  NEVER use this for feature engineering.
    """
    adj = snapshot["adj_close"]
 
    if tickers is not None:
        available = [t for t in tickers if t in adj.columns]
        missing = [t for t in tickers if t not in adj.columns]
        if missing:
            logger.warning(f"Tickers not in snapshot: {missing}")
        adj = adj[available]
 
    return adj
 
 
def get_raw_ohlcv(snapshot: Dict[str, Any], ticker: str) -> pd.DataFrame:
    """Extract full raw OHLCV for a single ticker from a snapshot."""
    raw = snapshot["raw_prices"]
    fields = ["Open", "High", "Low", "Close", "Volume"]
 
    if isinstance(raw.columns, pd.MultiIndex):
        df = pd.DataFrame({f: raw[(f, ticker)] for f in fields if (f, ticker) in raw.columns})
    else:
        df = raw[fields].copy()
 
    return df
 
 
def get_fred(snapshot: Dict[str, Any], series: Optional[List[str]] = None) -> pd.DataFrame:
    """Extract FRED macro data from a snapshot."""
    fred = snapshot["fred_data"]
    if fred.empty:
        return fred
    if series is not None:
        available = [s for s in series if s in fred.columns]
        fred = fred[available]
    return fred
 
 
# ---------------------------------------------------------------------------
# Daily Mode (Live Inference)
# ---------------------------------------------------------------------------
 
def fetch_daily_live(
    tickers: Optional[List[str]] = None,
    fred_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pull fresh data for daily live inference.
 
    This is the ONLY function that should pull fresh data.
    All backtesting must use frozen snapshots.
 
    Returns core data dict plus stress data for state representation.
    """
    tickers = tickers or ALL_TICKERS
 
    raw_prices, adj_close = fetch_yfinance_raw(tickers)
    fred_data = fetch_fred_series(api_key=fred_api_key)
 
    # Fetch stress data for v8 state representation
    stress_data = fetch_stress_data(api_key=fred_api_key)
 
    metadata = {
        "mode": "daily_live",
        "pulled_at": _now_utc_iso(),
        "tickers": tickers,
        "trading_days": len(raw_prices),
        "actual_start": raw_prices.index[0].strftime("%Y-%m-%d"),
        "actual_end": raw_prices.index[-1].strftime("%Y-%m-%d"),
    }
 
    logger.info(f"Daily live pull: {metadata['trading_days']} days through {metadata['actual_end']}")
 
    return {
        "raw_prices": raw_prices,
        "adj_close": adj_close,
        "fred_data": fred_data,
        "stress_fred": stress_data["stress_fred"],
        "vix_term": stress_data["vix_term"],
        "metadata": metadata,
    }
 
 
# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------
 
def main():
    """Command-line interface for snapshot management."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
 
    parser = argparse.ArgumentParser(
        description="SPY Alpha v8 — Data Pipeline & Snapshot Manager"
    )
    parser.add_argument(
        "--action",
        choices=["snapshot", "load", "daily", "list", "compare"],
        required=True,
    )
    parser.add_argument("--name", type=str, help="Snapshot name.")
    parser.add_argument("--compare-to", type=str, help="Second snapshot name.")
    parser.add_argument("--start", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--fred-key", type=str, default=None)
 
    args = parser.parse_args()
    mgr = SnapshotManager(data_dir=args.data_dir)
 
    if args.action == "snapshot":
        if not args.name:
            parser.error("--name is required for snapshot creation")
        path = mgr.create_snapshot(
            name=args.name,
            start=args.start,
            end=args.end,
            fred_api_key=args.fred_key,
            overwrite=args.overwrite,
        )
        print(f"\n✓ Snapshot created: {path}")
 
    elif args.action == "load":
        if not args.name:
            parser.error("--name is required for snapshot loading")
        snap = mgr.load_snapshot(args.name)
        meta = snap["metadata"]
        print(f"\n✓ Snapshot '{args.name}' loaded successfully")
        print(f"  Trading days:  {meta['trading_days']}")
        print(f"  Date range:    {meta['actual_start']} → {meta['actual_end']}")
        print(f"  Tickers:       {len(meta['tickers'])}")
        print(f"  Raw prices:    {meta['raw_prices_shape']}")
 
    elif args.action == "list":
        snapshots = mgr.list_snapshots()
        if not snapshots:
            print("\nNo snapshots found.")
        else:
            print(f"\n{len(snapshots)} snapshot(s):\n")
            for s in snapshots:
                print(
                    f"  {s['snapshot_name']:30s}  "
                    f"{s.get('actual_start', '?')} → {s.get('actual_end', '?')}  "
                    f"({s.get('trading_days', '?')} days)"
                )
 
    elif args.action == "compare":
        if not args.name or not args.compare_to:
            parser.error("--name and --compare-to required")
        report = mgr.compare_snapshots(args.name, args.compare_to)
        print(json.dumps(report, indent=2))
 
 
if __name__ == "__main__":
    main()