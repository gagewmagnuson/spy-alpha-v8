"""
SPY Alpha v7 — Regime Model
=============================

Gaussian Hidden Markov Model for macro regime detection with 5-regime support.

Changes from v6:
    - 5 regimes: Bull, Slowdown, Crisis-Deflation, Crisis-Inflation, Inflation
    - Two-step label alignment: SPY returns (Step 1) + TLT behavior (Step 2)
    - Crisis type distinction solves the v6 TLT/GLD allocation failure
    - All proven parameters preserved (smoothing, penalties, windows)
    - Backward compatible: still supports 4-regime mode for baseline comparison
 
Gaussian Hidden Markov Model for macro regime detection with full
probability outputs, exponential smoothing, and transition penalties.
 
Design Principles (from v5 post-mortem):
    - HMM instead of GMM: v5's GMM rediscovered clusters independently at
      each walk-forward window with no temporal awareness. Small data
      perturbations caused cluster centers to shift by 2.53 in standardized
      space, producing wildly different regime labels from identical code.
      HMMs explicitly model regime transitions and persistence, making them
      inherently more stable.
    - Probability smoothing: Raw HMM posteriors can oscillate rapidly near
      decision boundaries. Exponential smoothing + transition penalties
      target effective regime duration of ~20-40 trading days.
    - No binary thresholds: v5's Control 3/4 used hard thresholds that
      broke when data distributions shifted. All outputs are continuous
      probability distributions.
    - Walk-forward retraining: The model is retrained on expanding or
      rolling windows, never peeking at future data.
 
Usage:
    from data_pipeline import SnapshotManager
    from feature_engineering import FeatureEngine
    from regime_model import RegimeModel, RegimeDiagnostics
 
    mgr = SnapshotManager()
    snap = mgr.load_snapshot("baseline_2026")
 
    engine = FeatureEngine(reduce_dims=True, n_components=20)
    features = engine.build_observation_features(snap)
 
    model = RegimeModel(n_regimes=4)
    model.fit(features)
    probs = model.predict_proba(features)
    smoothed = model.smooth_probabilities(probs)
 
    diag = RegimeDiagnostics(model)
    report = diag.full_report(features, smoothed)
"""
 
from __future__ import annotations
 
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
 
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
 
logger = logging.getLogger("spy_alpha_v8.regime_model")
 
# Suppress convergence warnings during walk-forward (expected for short windows)
warnings.filterwarnings("ignore", category=DeprecationWarning)
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
# Default regime labels — v7 uses 5 regimes to distinguish crisis types
DEFAULT_REGIME_LABELS_4 = ["Bull", "Slowdown", "Crisis", "Inflation"]
DEFAULT_REGIME_LABELS_5 = ["Bull", "Slowdown", "Crisis-Deflation", "Crisis-Inflation", "Inflation"]
DEFAULT_REGIME_LABELS = DEFAULT_REGIME_LABELS_5  # v7 default
 
# Smoothing defaults targeting 20-40 day effective regime duration
DEFAULT_SMOOTHING_ALPHA = 0.18       # EMA decay factor (lower = smoother)
DEFAULT_TRANSITION_PENALTY = 0.10    # Cost of switching regimes per step
 
 
# ---------------------------------------------------------------------------
# Regime Label Alignment
# ---------------------------------------------------------------------------
 
def align_regime_labels(
    model: GaussianHMM,
    features: np.ndarray,
    reference_means: Optional[np.ndarray] = None,
    spy_returns: Optional[np.ndarray] = None,
    tlt_returns: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Resolve HMM label permutation ambiguity using economic meaning.

    Two-Step Method (v7, for 5 regimes):
        Step 1: Sort by average SPY return during each state's dominant periods.
                Highest return = Bull (0), second = Slowdown (1).
                Three lowest-return states proceed to Step 2.
        Step 2: Among the three remaining states, use TLT behavior to distinguish:
                Highest TLT return = Crisis-Deflation (bonds rally)
                Lowest TLT return = Crisis-Inflation (bonds fall)
                Remaining = Inflation

    For 4 regimes (backward compatible):
        Sort all states by SPY return descending: Bull, Slowdown, Inflation, Crisis.

    Fallback: if spy_returns unavailable, match to reference means or sort by PC_00.
    """
    means = model.means_
    n_states = means.shape[0]

    if reference_means is not None:
        from scipy.optimize import linear_sum_assignment
        cost = np.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(n_states):
                cost[i, j] = np.linalg.norm(means[i] - reference_means[j])
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = np.zeros(n_states, dtype=int)
        perm[row_ind] = col_ind
        return perm

    if spy_returns is not None and len(spy_returns) == len(features):
        state_sequence = model.predict(features)
        state_returns = []

        for state in range(n_states):
            mask = state_sequence == state
            if mask.sum() > 0:
                avg_ret = spy_returns[mask].mean()
            else:
                avg_ret = 0.0
            state_returns.append(avg_ret)

        if n_states == 5 and tlt_returns is not None and len(tlt_returns) == len(features):
            # ---- Two-Step Method for 5 regimes ----

            # Step 1: Identify Bull and Slowdown by highest SPY returns
            sorted_by_spy = np.argsort(state_returns)[::-1]  # descending
            bull_state = sorted_by_spy[0]
            slowdown_state = sorted_by_spy[1]
            remaining_states = sorted_by_spy[2:]  # 3 lowest-return states

            # Step 2: Distinguish crisis types using TLT behavior
            remaining_tlt_returns = []
            for state in remaining_states:
                mask = state_sequence == state
                if mask.sum() > 0:
                    avg_tlt = tlt_returns[mask].mean()
                else:
                    avg_tlt = 0.0
                remaining_tlt_returns.append(avg_tlt)

            # Sort remaining by TLT return
            tlt_order = np.argsort(remaining_tlt_returns)[::-1]  # descending TLT
            crisis_deflation_state = remaining_states[tlt_order[0]]  # highest TLT (bonds rally)
            crisis_inflation_state = remaining_states[tlt_order[2]]  # lowest TLT (bonds fall)
            inflation_state = remaining_states[tlt_order[1]]          # middle

            # Build permutation: perm[old_label] = new_label
            # Order: Bull=0, Slowdown=1, Crisis-Deflation=2, Crisis-Inflation=3, Inflation=4
            inv_perm = np.zeros(n_states, dtype=int)
            inv_perm[bull_state] = 0
            inv_perm[slowdown_state] = 1
            inv_perm[crisis_deflation_state] = 2
            inv_perm[crisis_inflation_state] = 3
            inv_perm[inflation_state] = 4

            return inv_perm

        else:
            # ---- Standard Method for 4 regimes (or 5 without TLT) ----
            # Sort descending: highest return = Bull (0), lowest = Crisis (N-1)
            sorted_states = np.argsort(state_returns)[::-1]
            inv_perm = np.zeros(n_states, dtype=int)
            inv_perm[sorted_states] = np.arange(n_states)
            return inv_perm

    # Fallback: sort by PC_00
    pc0_means = means[:, 0]
    perm = np.argsort(-pc0_means)
    inv_perm = np.zeros_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    return inv_perm
 
 
# ---------------------------------------------------------------------------
# Probability Smoothing
# ---------------------------------------------------------------------------
 
def smooth_probabilities(
    raw_probs: pd.DataFrame,
    alpha: float = DEFAULT_SMOOTHING_ALPHA,
    transition_penalty: float = DEFAULT_TRANSITION_PENALTY,
) -> pd.DataFrame:
    """
    Apply exponential smoothing with transition penalties to raw posteriors.
 
    Two-stage smoothing:
        1. Exponential moving average on probabilities (reduces oscillation).
        2. Transition penalty: after EMA, penalize probability mass that
           would cause a regime switch, encouraging persistence.
 
    The combination targets effective regime duration of ~20-40 trading days.
    v5's regimes oscillated daily near the 50% boundary — this prevents that.
 
    Parameters
    ----------
    raw_probs : pd.DataFrame
        Raw posterior probabilities from HMM. Columns = regime names,
        rows = dates. Each row sums to 1.
    alpha : float
        EMA decay factor. Lower = smoother. Default 0.12.
        Effective half-life ≈ -1/ln(1-alpha) ≈ 8 days at 0.12.
    transition_penalty : float
        Fraction of probability mass penalized for switching.
        Applied to non-dominant regimes after EMA. Default 0.15.
 
    Returns
    -------
    pd.DataFrame
        Smoothed probabilities (rows still sum to 1).
    """
    values = raw_probs.values.copy()
    n_steps, n_regimes = values.shape
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
 
    for t in range(1, n_steps):
        # Stage 1: EMA
        ema = alpha * values[t] + (1 - alpha) * smoothed[t - 1]
 
        # Stage 2: Transition penalty
        prev_dominant = np.argmax(smoothed[t - 1])
        penalty = np.ones(n_regimes) * (1 - transition_penalty)
        penalty[prev_dominant] = 1.0  # no penalty for staying in current regime
        penalized = ema * penalty
 
        # Re-normalize to sum to 1
        total = penalized.sum()
        if total > 0:
            smoothed[t] = penalized / total
        else:
            smoothed[t] = smoothed[t - 1]
 
    return pd.DataFrame(smoothed, index=raw_probs.index, columns=raw_probs.columns)
 
 
# ---------------------------------------------------------------------------
# Core Regime Model
# ---------------------------------------------------------------------------
 
class RegimeModel:
    """
    Gaussian HMM regime detection model with walk-forward support.
 
    Parameters
    ----------
    n_regimes : int
        Number of hidden states. Default 4 (Bull, Slowdown, Crisis, Inflation).
    regime_labels : list of str, optional
        Human-readable names for each regime.
    covariance_type : str
        HMM covariance type. 'full' captures cross-feature correlations
        but needs more data. 'diag' is more stable with fewer observations.
        Default 'full' since PCA reduction keeps dimensionality at ~20.
    n_iter : int
        Maximum EM iterations per fit.
    random_state : int
        Random seed for reproducibility.
    smoothing_alpha : float
        EMA decay for probability smoothing.
    transition_penalty : float
        Penalty for regime switches in smoothing.
    min_train_days : int
        Minimum training window size. Fits with fewer days are skipped.
    """
 
    def __init__(
        self,
        n_regimes: int = 5,
        regime_labels: Optional[List[str]] = None,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
        smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
        transition_penalty: float = DEFAULT_TRANSITION_PENALTY,
        min_train_days: int = 504,  # ~2 years
    ):
        self.n_regimes = n_regimes
        if regime_labels is not None:
            self.regime_labels = regime_labels
        elif n_regimes == 5:
            self.regime_labels = DEFAULT_REGIME_LABELS_5
        elif n_regimes == 4:
            self.regime_labels = DEFAULT_REGIME_LABELS_4
        else:
            self.regime_labels = [f"Regime_{i}" for i in range(n_regimes)]
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.smoothing_alpha = smoothing_alpha
        self.transition_penalty = transition_penalty
        self.min_train_days = min_train_days
 
        self._hmm: Optional[GaussianHMM] = None
        self._label_perm: Optional[np.ndarray] = None
        self._reference_means: Optional[np.ndarray] = None
        self._fit_info: Dict[str, Any] = {}
        self._spy_returns: Optional[np.ndarray] = None
        self._tlt_returns: Optional[np.ndarray] = None
 
    @property
    def is_fitted(self) -> bool:
        return self._hmm is not None
 
    def _create_hmm(self) -> GaussianHMM:
        """Create a fresh GaussianHMM instance with configured parameters."""
        return GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
 
    def fit(
        self,
        features: pd.DataFrame,
        set_reference: bool = True,
        spy_returns: Optional[pd.Series] = None,
        tlt_returns: Optional[pd.Series] = None,
    ) -> "RegimeModel":
        """
        Fit the HMM on observation features.
 
        Parameters
        ----------
        features : pd.DataFrame
            PCA-reduced observation features (from FeatureEngine).
            Expected columns: PC_00, PC_01, ..., PC_N.
        set_reference : bool
            If True, store fitted means as reference for future label
            alignment. Set True for the initial fit, False for walk-forward
            refits that should align to the original labeling.
 
        Returns
        -------
        self
        """
        X = features.values.astype(np.float64)
        # Store SPY and TLT returns for return-based label alignment
        if spy_returns is not None:
            self._spy_returns = spy_returns.reindex(features.index).pct_change().fillna(0).values
        elif self._spy_returns is None:
            self._spy_returns = None
        if tlt_returns is not None:
            self._tlt_returns = tlt_returns.reindex(features.index).pct_change().fillna(0).values
        elif self._tlt_returns is None:
            self._tlt_returns = None
        n_samples, n_features = X.shape
 
        if n_samples < self.min_train_days:
            raise ValueError(
                f"Training data has {n_samples} days, minimum is {self.min_train_days}. "
                f"Provide more data or reduce min_train_days."
            )
 
        logger.info(
            f"Fitting HMM: {n_samples} days × {n_features} features, "
            f"{self.n_regimes} regimes, cov={self.covariance_type}"
        )
 
        hmm = self._create_hmm()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm.fit(X)

        # Align labels using SPY returns if available
        if self._reference_means is not None and not set_reference:
            perm = align_regime_labels(hmm, X, self._reference_means)
        else:
            perm = align_regime_labels(hmm, X, spy_returns=self._spy_returns, tlt_returns=self._tlt_returns)
 
        self._hmm = hmm
        self._label_perm = perm
 
        if set_reference:
            # Store aligned means as reference for future walk-forward refits
            aligned_means = hmm.means_[np.argsort(perm)]
            self._reference_means = aligned_means
 
        # Store fit diagnostics
        self._fit_info = {
            "n_samples": n_samples,
            "n_features": n_features,
            "converged": hmm.monitor_.converged,
            "n_iterations": hmm.monitor_.iter,
            "log_likelihood": float(hmm.score(X)),
            "train_start": str(features.index[0].date()),
            "train_end": str(features.index[-1].date()),
        }
 
        logger.info(
            f"HMM fit complete: converged={hmm.monitor_.converged}, "
            f"iterations={hmm.monitor_.iter}, "
            f"log_likelihood={self._fit_info['log_likelihood']:.1f}"
        )
 
        return self
 
    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute raw posterior regime probabilities.
 
        Parameters
        ----------
        features : pd.DataFrame
            Observation features (same format as training data).
 
        Returns
        -------
        pd.DataFrame
            Columns = regime labels, rows = dates.
            Each row sums to 1.0.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
 
        X = features.values.astype(np.float64)
        raw_probs = self._hmm.predict_proba(X)
 
        # Apply label permutation for consistent ordering
        aligned_probs = raw_probs[:, np.argsort(self._label_perm)]
 
        return pd.DataFrame(
            aligned_probs,
            index=features.index,
            columns=self.regime_labels,
        )
 
    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.Series:
        """
        Predict most likely regime (hard assignment).
 
        Note: The spec prefers soft probability-weighted allocation.
        Use predict_proba() for the primary signal. This method exists
        for diagnostics and visualization only.
 
        Parameters
        ----------
        features : pd.DataFrame
            Observation features.
 
        Returns
        -------
        pd.Series
            Regime labels indexed by date.
        """
        probs = self.predict_proba(features)
        regime_idx = probs.values.argmax(axis=1)
        labels = [self.regime_labels[i] for i in regime_idx]
        return pd.Series(labels, index=features.index, name="regime")
 
    def smooth_probabilities(
        self,
        raw_probs: pd.DataFrame,
        alpha: Optional[float] = None,
        penalty: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Apply probability smoothing with instance-configured parameters.
 
        Parameters
        ----------
        raw_probs : pd.DataFrame
            Raw posteriors from predict_proba().
        alpha : float, optional
            Override smoothing alpha.
        penalty : float, optional
            Override transition penalty.
 
        Returns
        -------
        pd.DataFrame
            Smoothed probabilities.
        """
        return smooth_probabilities(
            raw_probs,
            alpha=alpha or self.smoothing_alpha,
            transition_penalty=penalty or self.transition_penalty,
        )
 
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the learned regime transition probability matrix.
 
        Returns
        -------
        pd.DataFrame
            Transition matrix with aligned regime labels.
            transmat[i, j] = P(regime_j at t+1 | regime_i at t).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
 
        raw_transmat = self._hmm.transmat_
        # Apply label permutation
        order = np.argsort(self._label_perm)
        aligned = raw_transmat[np.ix_(order, order)]
 
        return pd.DataFrame(
            aligned,
            index=self.regime_labels,
            columns=self.regime_labels,
        )
 
    def get_regime_means(self) -> pd.DataFrame:
        """
        Get the fitted regime means in feature space.
 
        Returns
        -------
        pd.DataFrame
            Rows = regimes, columns = features.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
 
        order = np.argsort(self._label_perm)
        means = self._hmm.means_[order]
 
        return pd.DataFrame(means, index=self.regime_labels)
 
    @property
    def fit_info(self) -> Dict[str, Any]:
        """Return diagnostics from the most recent fit."""
        return self._fit_info.copy()
 
 
# ---------------------------------------------------------------------------
# Walk-Forward Engine
# ---------------------------------------------------------------------------
 
@dataclass
class WalkForwardResult:
    """Container for walk-forward backtest results."""
    probabilities: pd.DataFrame          # Smoothed out-of-sample probabilities
    raw_probabilities: pd.DataFrame      # Raw out-of-sample probabilities
    regimes: pd.Series                   # Hard regime assignments
    window_info: List[Dict[str, Any]]    # Per-window fit diagnostics
    transition_matrices: List[pd.DataFrame]  # Per-window transition matrices
 
 
def walk_forward_regimes(
    features: pd.DataFrame,
    model: RegimeModel,
    train_window: int = 756,
    retrain_every: int = 63,
    min_test_days: int = 1,
    expanding: bool = False,
    purge_days: int = 5,
    spy_close: Optional[pd.Series] = None,
    tlt_close: Optional[pd.Series] = None,
) -> WalkForwardResult:
    """
    Walk-forward regime detection with periodic retraining.
 
    Trains the HMM on a rolling (or expanding) window, then predicts
    out-of-sample on the next segment. The first fit sets the reference
    label alignment; subsequent refits align to it.
 
    Parameters
    ----------
    features : pd.DataFrame
        Full PCA-reduced feature matrix (entire history).
    model : RegimeModel
        Configured but unfitted RegimeModel instance.
    train_window : int
        Number of trading days in each training window (rolling mode).
    retrain_every : int
        Number of days between retrains.
    min_test_days : int
        Minimum test days per window. Windows with fewer are skipped.
    expanding : bool
        If True, training window expands from the start of data.
        If False, uses a rolling window of fixed size.
    purge_days : int
        Gap between train and test to prevent leakage.
 
    Returns
    -------
    WalkForwardResult
        Contains smoothed probabilities, raw probabilities, hard regimes,
        per-window diagnostics, and transition matrices.
    """
    dates = features.index
    n_total = len(dates)
    logger.info(
        f"Walk-forward: {n_total} days, train_window={train_window}, "
        f"retrain_every={retrain_every}, expanding={expanding}"
    )
 
    all_raw_probs = []
    window_info = []
    transition_matrices = []
    is_first_fit = True
 
    # Generate retrain points
    retrain_points = list(range(train_window, n_total - min_test_days, retrain_every))
 
    if not retrain_points:
        raise ValueError(
            f"Not enough data for walk-forward: {n_total} days with "
            f"train_window={train_window}. Need at least {train_window + min_test_days} days."
        )
 
    logger.info(f"  {len(retrain_points)} retrain windows")
 
    for i, split_idx in enumerate(retrain_points):
        # Define train window
        if expanding:
            train_start = 0
        else:
            train_start = max(0, split_idx - train_window)
 
        train_end = split_idx
        test_start = split_idx + purge_days
 
        # Define test window end (next retrain point or end of data)
        if i + 1 < len(retrain_points):
            test_end = retrain_points[i + 1] + purge_days
        else:
            test_end = n_total
 
        # Bounds check
        if test_start >= n_total:
            break
        test_end = min(test_end, n_total)
 
        train_data = features.iloc[train_start:train_end]
        test_data = features.iloc[test_start:test_end]
 
        if len(test_data) < min_test_days:
            continue
 
        # Fit model
        try:
            model.fit(train_data, set_reference=is_first_fit, spy_returns=spy_close, tlt_returns=tlt_close)
 
            if is_first_fit:
                is_first_fit = False
 
            # Predict on test window
            raw_probs = model.predict_proba(test_data)
            all_raw_probs.append(raw_probs)
 
            # Store diagnostics
            info = model.fit_info.copy()
            info["window_idx"] = i
            info["train_days"] = len(train_data)
            info["test_days"] = len(test_data)
            info["test_start"] = str(test_data.index[0].date())
            info["test_end"] = str(test_data.index[-1].date())
            window_info.append(info)
 
            transition_matrices.append(model.get_transition_matrix())
 
        except Exception as e:
            logger.warning(f"  Window {i} failed: {e}")
            info = {"window_idx": i, "error": str(e)}
            window_info.append(info)
            continue
 
    if not all_raw_probs:
        raise RuntimeError("All walk-forward windows failed. Check data quality.")
 
    # Combine out-of-sample probabilities
    combined_raw = pd.concat(all_raw_probs)
 
    # Handle any overlapping dates (keep first prediction — earliest retrain)
    combined_raw = combined_raw[~combined_raw.index.duplicated(keep="first")]
    combined_raw = combined_raw.sort_index()
 
    # Apply smoothing
    combined_smoothed = model.smooth_probabilities(combined_raw)
 
    # Hard regime assignment from smoothed probabilities
    regime_idx = combined_smoothed.values.argmax(axis=1)
    regimes = pd.Series(
        [model.regime_labels[i] for i in regime_idx],
        index=combined_smoothed.index,
        name="regime",
    )
 
    logger.info(
        f"Walk-forward complete: {len(combined_smoothed)} out-of-sample days, "
        f"{len(window_info)} windows"
    )
 
    return WalkForwardResult(
        probabilities=combined_smoothed,
        raw_probabilities=combined_raw,
        regimes=regimes,
        window_info=window_info,
        transition_matrices=transition_matrices,
    )
 
 
# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
 
class RegimeDiagnostics:
    """
    Comprehensive diagnostic reporting for regime model outputs.
 
    v5 had minimal diagnostics, which meant hours of manual investigation
    when performance collapsed. These diagnostics make problems visible
    immediately.
 
    Parameters
    ----------
    model : RegimeModel
        Fitted regime model (for transition matrix, means, etc.).
    """
 
    def __init__(self, model: RegimeModel):
        self.model = model
 
    def conviction_metrics(
        self,
        probs: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Compute conviction statistics from regime probabilities.
 
        High conviction (max probability >> 0.5) means the model is
        confident in its regime call. Low conviction (max probability
        near 1/n_regimes) means the model is uncertain — the portfolio
        should shrink toward equal-weight.
 
        v5's degraded run had mean max conviction of 0.51 (essentially
        random for 2 regimes). This metric would have caught that instantly.
 
        Parameters
        ----------
        probs : pd.DataFrame
            Smoothed regime probabilities.
 
        Returns
        -------
        dict
            Conviction statistics.
        """
        max_prob = probs.max(axis=1)
 
        return {
            "mean_max_conviction": float(max_prob.mean()),
            "median_max_conviction": float(max_prob.median()),
            "min_max_conviction": float(max_prob.min()),
            "max_max_conviction": float(max_prob.max()),
            "std_max_conviction": float(max_prob.std()),
            "pct_above_60": float((max_prob > 0.6).mean()),
            "pct_above_70": float((max_prob > 0.7).mean()),
            "pct_above_80": float((max_prob > 0.8).mean()),
            "pct_below_40": float((max_prob < 0.4).mean()),
        }
 
    def regime_distribution(
        self,
        probs: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Compute time-weighted regime distribution.
 
        Uses the full probability distribution, not hard assignments.
        Shows what fraction of time the model spent in each regime
        (probability-weighted).
 
        Parameters
        ----------
        probs : pd.DataFrame
            Smoothed regime probabilities.
 
        Returns
        -------
        dict
            Regime label → fraction of time.
        """
        return {col: float(probs[col].mean()) for col in probs.columns}
 
    def regime_durations(
        self,
        probs: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Compute effective regime duration statistics.
 
        Target: 20-40 days mean duration (from spec).
        If too short → increase smoothing_alpha or transition_penalty.
        If too long → decrease them.
 
        Parameters
        ----------
        probs : pd.DataFrame
            Smoothed regime probabilities.
 
        Returns
        -------
        dict
            Duration statistics per regime and overall.
        """
        dominant = probs.idxmax(axis=1)
        regimes = dominant.values
 
        # Compute run lengths
        durations_by_regime = {label: [] for label in self.model.regime_labels}
        current_regime = regimes[0]
        current_length = 1
 
        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_length += 1
            else:
                durations_by_regime[current_regime].append(current_length)
                current_regime = regimes[i]
                current_length = 1
        durations_by_regime[current_regime].append(current_length)
 
        all_durations = []
        per_regime = {}
        for label, durs in durations_by_regime.items():
            if durs:
                all_durations.extend(durs)
                per_regime[label] = {
                    "count": len(durs),
                    "mean_days": float(np.mean(durs)),
                    "median_days": float(np.median(durs)),
                    "min_days": int(np.min(durs)),
                    "max_days": int(np.max(durs)),
                }
            else:
                per_regime[label] = {"count": 0}
 
        overall_mean = float(np.mean(all_durations)) if all_durations else 0
        in_target = 20 <= overall_mean <= 40
 
        return {
            "overall_mean_duration": overall_mean,
            "overall_median_duration": float(np.median(all_durations)) if all_durations else 0,
            "target_range": "20-40 days",
            "within_target": in_target,
            "per_regime": per_regime,
        }
 
    def transition_analysis(self) -> Dict[str, Any]:
        """
        Analyze the learned transition matrix.
 
        High diagonal values = persistent regimes (good).
        Low diagonal values = frequent switching (bad — likely noisy).
 
        Returns
        -------
        dict
            Transition matrix analysis.
        """
        transmat = self.model.get_transition_matrix()
        diag = np.diag(transmat.values)
 
        return {
            "transition_matrix": transmat.to_dict(),
            "diagonal_persistence": {
                label: float(diag[i])
                for i, label in enumerate(self.model.regime_labels)
            },
            "mean_persistence": float(diag.mean()),
            "min_persistence": float(diag.min()),
            "implied_mean_duration": {
                label: float(1 / (1 - diag[i])) if diag[i] < 1 else float("inf")
                for i, label in enumerate(self.model.regime_labels)
            },
        }
 
    def stability_metrics(
        self,
        probs: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Measure probability stability over time.
 
        Unstable probabilities (large daily swings) indicate the model
        is sensitive to small input changes — exactly the v5 failure mode.
 
        Parameters
        ----------
        probs : pd.DataFrame
            Smoothed regime probabilities.
 
        Returns
        -------
        dict
            Stability metrics.
        """
        daily_change = probs.diff().abs()
 
        return {
            "mean_daily_prob_change": float(daily_change.mean().mean()),
            "max_daily_prob_change": float(daily_change.max().max()),
            "per_regime_mean_change": {
                col: float(daily_change[col].mean())
                for col in probs.columns
            },
            "per_regime_max_change": {
                col: float(daily_change[col].max())
                for col in probs.columns
            },
        }
 
    def smoothing_impact(
        self,
        raw_probs: pd.DataFrame,
        smoothed_probs: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Quantify the effect of probability smoothing.
 
        Shows how much smoothing changed the raw posteriors.
        Large changes suggest the raw model is noisy.
 
        Parameters
        ----------
        raw_probs : pd.DataFrame
            Raw HMM posteriors.
        smoothed_probs : pd.DataFrame
            Smoothed probabilities.
 
        Returns
        -------
        dict
            Smoothing impact metrics.
        """
        # Align indices
        common = raw_probs.index.intersection(smoothed_probs.index)
        raw = raw_probs.loc[common]
        smooth = smoothed_probs.loc[common]
 
        diff = (raw - smooth).abs()
 
        # Regime agreement rate
        raw_dominant = raw.idxmax(axis=1)
        smooth_dominant = smooth.idxmax(axis=1)
        agreement = (raw_dominant == smooth_dominant).mean()
 
        return {
            "mean_abs_change": float(diff.mean().mean()),
            "max_abs_change": float(diff.max().max()),
            "regime_agreement_rate": float(agreement),
            "per_regime_mean_change": {
                col: float(diff[col].mean()) for col in diff.columns
            },
        }
 
    def full_report(
        self,
        features: pd.DataFrame,
        smoothed_probs: pd.DataFrame,
        raw_probs: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive diagnostic report.
 
        Parameters
        ----------
        features : pd.DataFrame
            Observation features.
        smoothed_probs : pd.DataFrame
            Smoothed regime probabilities.
        raw_probs : pd.DataFrame, optional
            Raw HMM posteriors (for smoothing impact analysis).
 
        Returns
        -------
        dict
            Complete diagnostic report.
        """
        report = {
            "fit_info": self.model.fit_info,
            "conviction": self.conviction_metrics(smoothed_probs),
            "regime_distribution": self.regime_distribution(smoothed_probs),
            "regime_durations": self.regime_durations(smoothed_probs),
            "transition_analysis": self.transition_analysis(),
            "stability": self.stability_metrics(smoothed_probs),
        }
 
        if raw_probs is not None:
            report["smoothing_impact"] = self.smoothing_impact(
                raw_probs, smoothed_probs
            )
 
        return report
 
 
def print_diagnostic_report(report: Dict[str, Any]) -> None:
    """Pretty-print a diagnostic report to console."""
    print("\n" + "=" * 70)
    print("REGIME MODEL DIAGNOSTIC REPORT")
    print("=" * 70)
 
    # Fit info
    fi = report.get("fit_info", {})
    print(f"\n--- Fit Info ---")
    print(f"  Training period:  {fi.get('train_start', '?')} → {fi.get('train_end', '?')}")
    print(f"  Training days:    {fi.get('n_samples', '?')}")
    print(f"  Features:         {fi.get('n_features', '?')}")
    print(f"  Converged:        {fi.get('converged', '?')}")
    print(f"  Log-likelihood:   {fi.get('log_likelihood', '?')}")
 
    # Conviction
    cv = report.get("conviction", {})
    print(f"\n--- Conviction ---")
    print(f"  Mean max conviction:  {cv.get('mean_max_conviction', 0):.3f}")
    print(f"  Median:               {cv.get('median_max_conviction', 0):.3f}")
    print(f"  % above 0.7:          {cv.get('pct_above_70', 0):.1%}")
    print(f"  % above 0.8:          {cv.get('pct_above_80', 0):.1%}")
    print(f"  % below 0.4:          {cv.get('pct_below_40', 0):.1%}")
 
    # Regime distribution
    rd = report.get("regime_distribution", {})
    print(f"\n--- Regime Distribution (probability-weighted) ---")
    for label, frac in rd.items():
        print(f"  {label:15s}: {frac:.1%}")
 
    # Regime durations
    dur = report.get("regime_durations", {})
    print(f"\n--- Regime Durations ---")
    print(f"  Overall mean:    {dur.get('overall_mean_duration', 0):.1f} days")
    print(f"  Target range:    {dur.get('target_range', '?')}")
    print(f"  Within target:   {dur.get('within_target', '?')}")
    per_regime = dur.get("per_regime", {})
    for label, stats in per_regime.items():
        if stats.get("count", 0) > 0:
            print(
                f"  {label:15s}: {stats['count']} episodes, "
                f"mean {stats['mean_days']:.1f}d, "
                f"range [{stats['min_days']}-{stats['max_days']}]d"
            )
 
    # Transition analysis
    ta = report.get("transition_analysis", {})
    print(f"\n--- Transition Persistence ---")
    print(f"  Mean self-transition: {ta.get('mean_persistence', 0):.3f}")
    for label, p in ta.get("diagonal_persistence", {}).items():
        imp_dur = ta.get("implied_mean_duration", {}).get(label, 0)
        print(f"  {label:15s}: P(stay)={p:.3f}, implied duration={imp_dur:.1f}d")
 
    # Stability
    st = report.get("stability", {})
    print(f"\n--- Probability Stability ---")
    print(f"  Mean daily change:  {st.get('mean_daily_prob_change', 0):.4f}")
    print(f"  Max daily change:   {st.get('max_daily_prob_change', 0):.4f}")
 
    # Smoothing impact
    si = report.get("smoothing_impact", {})
    if si:
        print(f"\n--- Smoothing Impact ---")
        print(f"  Mean abs change:         {si.get('mean_abs_change', 0):.4f}")
        print(f"  Regime agreement rate:   {si.get('regime_agreement_rate', 0):.1%}")
 
    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_regime_probabilities(
    probs: pd.DataFrame,
    spy_close: Optional[pd.Series] = None,
    title: str = "SPY Alpha v6 — Regime Probabilities",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12),
) -> None:
    """
    Plot regime probabilities over time with SPY price and historical events.

    Parameters
    ----------
    probs : pd.DataFrame
        Smoothed regime probabilities (columns = regime labels).
    spy_close : pd.Series, optional
        SPY raw close prices (plotted on top panel). If None, top panel is skipped.
    title : str
        Plot title.
    save_path : str, optional
        File path to save the figure. If None, displays interactively.
    figsize : tuple
        Figure size (width, height).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # ---- Historical events for context ----
    events = [
        ("2007-10-09", "Pre-GFC Peak"),
        ("2008-09-15", "Lehman Brothers"),
        ("2009-03-09", "GFC Bottom"),
        ("2010-05-06", "Flash Crash"),
        ("2011-08-05", "US Downgrade / Euro Crisis"),
        ("2015-08-24", "China Devaluation"),
        ("2018-02-05", "Volmageddon"),
        ("2018-12-24", "Fed Tightening Selloff"),
        ("2020-02-19", "Pre-COVID Peak"),
        ("2020-03-23", "COVID Bottom"),
        ("2021-11-08", "Inflation Surge Begins"),
        ("2022-01-03", "Fed Pivot / Rate Hikes"),
        ("2022-06-16", "Bear Market Low"),
        ("2022-10-12", "2022 Bottom"),
        ("2023-03-10", "SVB Collapse"),
        ("2024-08-05", "Yen Carry Unwind"),
    ]

    # Filter events to data range
    date_min, date_max = probs.index[0], probs.index[-1]
    events = [
        (d, label) for d, label in events
        if date_min <= pd.Timestamp(d) <= date_max
    ]

    # ---- Color scheme ----
    regime_colors = {
        "Bull": "#2ecc71",              # green
        "Slowdown": "#f39c12",          # amber
        "Crisis": "#e74c3c",            # red (v6 compat)
        "Crisis-Deflation": "#e74c3c",  # red
        "Crisis-Inflation": "#c0392b",  # dark red
        "Inflation": "#9b59b6",         # purple
    }
    # Fallback for custom regime names
    default_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#3498db", "#1abc9c"]

    n_panels = 3 if spy_close is not None else 2
    height_ratios = [2, 2, 1.2] if spy_close is not None else [2, 1.2]

    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
        sharex=True,
    )

    ax_idx = 0

    # ---- Panel 1: SPY price with regime-colored background ----
    if spy_close is not None:
        ax = axes[ax_idx]
        ax_idx += 1

        # Align SPY to probability dates
        common_idx = spy_close.index.intersection(probs.index)
        spy_aligned = spy_close.loc[common_idx]

        ax.plot(spy_aligned.index, spy_aligned.values, color="#2c3e50", linewidth=0.8, alpha=0.9)
        ax.set_ylabel("SPY Price (Raw)", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Color background by dominant regime
        dominant = probs.loc[common_idx].idxmax(axis=1)
        prev_regime = dominant.iloc[0]
        start = common_idx[0]

        for i in range(1, len(common_idx)):
            curr_regime = dominant.iloc[i]
            if curr_regime != prev_regime or i == len(common_idx) - 1:
                color = regime_colors.get(prev_regime, "#cccccc")
                ax.axvspan(start, common_idx[i], alpha=0.15, color=color, linewidth=0)
                start = common_idx[i]
                prev_regime = curr_regime

        # Mark events
        for date_str, label in events:
            dt = pd.Timestamp(date_str)
            if dt in spy_aligned.index or (spy_aligned.index[0] <= dt <= spy_aligned.index[-1]):
                ax.axvline(dt, color="#7f8c8d", linestyle="--", alpha=0.5, linewidth=0.7)
                y_pos = ax.get_ylim()[1] * 0.98
                ax.text(
                    dt, y_pos, f" {label}", fontsize=7, rotation=45,
                    va="top", ha="left", color="#2c3e50", alpha=0.8,
                )

        ax.grid(True, alpha=0.3)

    # ---- Panel 2: Regime probabilities (stacked area) ----
    ax = axes[ax_idx]
    ax_idx += 1

    cols = list(probs.columns)
    colors = [regime_colors.get(c, default_colors[i % len(default_colors)]) for i, c in enumerate(cols)]

    ax.stackplot(
        probs.index, *[probs[c].values for c in cols],
        labels=cols, colors=colors, alpha=0.7,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Regime Probability", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, ncol=len(cols), framealpha=0.8)

    # Mark events
    for date_str, label in events:
        dt = pd.Timestamp(date_str)
        ax.axvline(dt, color="#7f8c8d", linestyle="--", alpha=0.5, linewidth=0.7)

    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Max conviction over time ----
    ax = axes[ax_idx]

    max_prob = probs.max(axis=1)
    ax.fill_between(max_prob.index, max_prob.values, alpha=0.4, color="#3498db")
    ax.plot(max_prob.index, max_prob.values, color="#2980b9", linewidth=0.8)
    ax.axhline(0.7, color="#e74c3c", linestyle=":", alpha=0.6, label="0.7 threshold")
    ax.axhline(0.5, color="#f39c12", linestyle=":", alpha=0.6, label="0.5 threshold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Max Conviction", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
 
 
# ---------------------------------------------------------------------------
# CLI / Example Usage
# ---------------------------------------------------------------------------
 
def main():
    """Example usage and smoke test for regime model."""
    import argparse
 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
 
    parser = argparse.ArgumentParser(
        description="SPY Alpha v6 — Regime Model"
    )
    parser.add_argument("--snapshot", type=str, required=True, help="Snapshot name.")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory.")
    parser.add_argument("--n-regimes", type=int, default=4, help="Number of regimes.")
    parser.add_argument(
        "--mode",
        choices=["fit", "walkforward"],
        default="fit",
        help="Run mode: single fit or walk-forward.",
    )
    parser.add_argument("--n-components", type=int, default=20, help="PCA components.")
    args = parser.parse_args()
 
    from data_pipeline import SnapshotManager
    from feature_engineering import FeatureEngine
 
    # Load data
    mgr = SnapshotManager(data_dir=args.data_dir)
    snap = mgr.load_snapshot(args.snapshot)
 
    # Build features (with PCA reduction)
    engine = FeatureEngine(reduce_dims=True, n_components=args.n_components)
    features = engine.build_observation_features(snap)
    print(f"\nFeatures: {features.shape}")
    print(f"Date range: {features.index[0].date()} → {features.index[-1].date()}")
 
    model = RegimeModel(n_regimes=args.n_regimes)
 
    if args.mode == "fit":
        # Single fit on all data
        model.fit(features)
        raw_probs = model.predict_proba(features)
        smoothed = model.smooth_probabilities(raw_probs)
 
        print(f"\nSmoothed probabilities sample (last 5 days):")
        print(smoothed.tail().to_string(float_format="{:.3f}".format))
 
        diag = RegimeDiagnostics(model)
        report = diag.full_report(features, smoothed, raw_probs)
        print_diagnostic_report(report)
 
    elif args.mode == "walkforward":
        # Walk-forward
        result = walk_forward_regimes(features, model)
 
        print(f"\nWalk-forward results:")
        print(f"  Out-of-sample days: {len(result.probabilities)}")
        print(f"  Windows: {len(result.window_info)}")
 
        print(f"\nSmoothed probabilities sample (last 5 days):")
        print(result.probabilities.tail().to_string(float_format="{:.3f}".format))
 
        # Refit on full data for diagnostics
        model.fit(features, set_reference=False)
        diag = RegimeDiagnostics(model)
        report = diag.full_report(features, result.probabilities, result.raw_probabilities)
        print_diagnostic_report(report)
 
        # Walk-forward specific diagnostics
        print(f"\n--- Walk-Forward Window Summary ---")
        for w in result.window_info[:5]:
            print(
                f"  Window {w.get('window_idx', '?')}: "
                f"train={w.get('n_samples', '?')}d, "
                f"test={w.get('test_days', '?')}d, "
                f"converged={w.get('converged', '?')}, "
                f"LL={w.get('log_likelihood', 0):.0f}"
            )
        if len(result.window_info) > 5:
            print(f"  ... ({len(result.window_info) - 5} more windows)")
 
 
if __name__ == "__main__":
    main()