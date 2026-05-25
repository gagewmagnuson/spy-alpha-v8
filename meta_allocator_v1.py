"""
SPY Alpha v8 — Meta-Allocator (LightGBM Phase 1)
===================================================
 
NEW module in v8. The brain of the system — decides how much capital
each strategy receives based on the current market state and strategy
performance.
 
Architecture (from build spec Section 6):
    Stage 1: Strategy Activation
        - For each strategy, compute activation score ∈ [0, 1]
        - Answers: "Should this strategy matter right now?"
        - Smooth activation with exponential decay (α = 0.3)
        - Minimum holding period (5 trading days)
 
    Stage 2: Capital Sizing
        - Given active strategies, allocate capital proportionally
        - Inputs: activation scores + strategy proposed weights + risk constraints
        - Output: blended portfolio weights
 
Model:
    - LightGBM (Phase 1) — tabular, heterogeneous, nonlinear, interaction-heavy
    - Ensemble of 5 models with different random seeds
    - Maximum tree depth: 6
    - L2 regularization
    - Feature importance pruning (< 1% importance removed)
 
Training:
    - Walk-forward validation
    - Training window: 2520 days (~10 years)
    - Retrain frequency: 126 days (~semi-annually)
    - Purge gap: 21 days
    - Tail-aware reward function
 
    Initial implementation trains on 2005-present ETF data.
    Extended dataset (1970+) integration deferred until architecture validated.
 
Constraints:
    - Activations are soft (continuous), never binary
    - Transition penalties prevent rapid oscillation
    - Allocator confidence calibration scales leverage
"""
 
from __future__ import annotations
 
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
 
import numpy as np
import pandas as pd
 
logger = logging.getLogger("spy_alpha_v8.meta_allocator")
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
# Walk-forward parameters
TRAIN_WINDOW: int = 1260          # ~5 years (ETF-era; revert to 2520 for extended dataset)
RETRAIN_EVERY: int = 63           # ~quarterly
PURGE_GAP: int = 21              # longer purge for allocation decisions
 
# Activation smoothing
ACTIVATION_SMOOTHING_ALPHA: float = 0.3    # exponential decay
MIN_HOLDING_PERIOD: int = 5                # trading days
 
# LightGBM hyperparameters
LGBM_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,       # L1 regularization
    "reg_lambda": 1.0,      # L2 regularization
    "min_child_samples": 20,
    "verbose": -1,
}
 
# Ensemble
N_ENSEMBLE: int = 5
 
# Feature importance pruning threshold
IMPORTANCE_THRESHOLD: float = 0.01
 
# Reward function weights (from build spec)
REWARD_WEIGHTS: Dict[str, float] = {
    "differential_sharpe": 1.0,
    "sortino_component": 0.5,
    "drawdown_penalty": 2.0,
    "turnover_penalty": 0.3,
    "tail_risk_penalty": 1.0,
}
 
# Strategy names
STRATEGY_NAMES: List[str] = ["regime_allocator", "trend_cta", "defensive"]
 
# Activation bounds
MIN_ACTIVATION: float = 0.05    # never fully deactivate
MAX_ACTIVATION: float = 0.95    # never fully activate
 
 
# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------
 
def compute_tail_aware_reward(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    turnover: float,
    window: int = 63,
) -> float:
    """
    Compute the tail-aware reward for a portfolio over a window.
 
    reward = differential_sharpe
           + 0.5 * sortino_component
           - 2.0 * drawdown_penalty
           - 0.3 * turnover_penalty
           - 1.0 * tail_risk_penalty
 
    This reward function prioritizes:
        1. Risk-adjusted returns (Sharpe, Sortino)
        2. Drawdown avoidance (heavy penalty)
        3. Low turnover
        4. Tail risk control (kurtosis/extreme losses)
    """
    if len(portfolio_returns) < 21:
        return 0.0
 
    port = portfolio_returns.dropna()
    bench = benchmark_returns.reindex(port.index).dropna()
 
    if len(port) < 21:
        return 0.0
 
    # ---- Differential Sharpe ----
    port_sharpe = port.mean() / port.std() * np.sqrt(252) if port.std() > 0 else 0
    bench_sharpe = bench.mean() / bench.std() * np.sqrt(252) if bench.std() > 0 else 0
    diff_sharpe = port_sharpe - bench_sharpe
 
    # ---- Sortino Component ----
    downside = port[port < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (port.mean() * 252) / downside_vol if downside_vol > 0 else 0
    sortino_component = min(sortino / 2.0, 2.0)  # cap contribution
 
    # ---- Drawdown Penalty ----
    cum = (1 + port).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    max_dd = abs(dd.min())
    drawdown_penalty = max(max_dd - 0.10, 0)  # penalty kicks in above 10% DD
 
    # ---- Turnover Penalty ----
    turnover_penalty = max(turnover - 0.20, 0)  # penalty above 20% turnover
 
    # ---- Tail Risk Penalty ----
    if len(port) > 10:
        kurtosis = port.kurtosis()
        tail_penalty = max(kurtosis - 3.0, 0) / 10.0  # excess kurtosis, scaled
    else:
        tail_penalty = 0.0
 
    # ---- Combine ----
    w = REWARD_WEIGHTS
    reward = (
        w["differential_sharpe"] * diff_sharpe
        + w["sortino_component"] * sortino_component
        - w["drawdown_penalty"] * drawdown_penalty
        - w["turnover_penalty"] * turnover_penalty
        - w["tail_risk_penalty"] * tail_penalty
    )
 
    return float(reward)
 
 
# ---------------------------------------------------------------------------
# Activation Smoothing
# ---------------------------------------------------------------------------
 
def smooth_activations(
    raw_activations: pd.DataFrame,
    alpha: float = ACTIVATION_SMOOTHING_ALPHA,
    min_holding: int = MIN_HOLDING_PERIOD,
) -> pd.DataFrame:
    """
    Apply exponential decay smoothing and minimum holding period
    to raw activation scores.
 
    This prevents rapid oscillation between strategy activations.
    """
    smoothed = raw_activations.copy()
 
    for col in smoothed.columns:
        series = smoothed[col].copy()
 
        # Exponential moving average smoothing
        ema = series.ewm(alpha=alpha, adjust=False).mean()
 
        # Minimum holding period: once activation changes significantly,
        # hold for at least min_holding days before allowing another change
        held = ema.copy()
        last_change_idx = 0
 
        for i in range(1, len(held)):
            change = abs(held.iloc[i] - held.iloc[i - 1])
            if change > 0.1 and (i - last_change_idx) < min_holding:
                # Too soon — revert to previous value
                held.iloc[i] = held.iloc[i - 1]
            elif change > 0.1:
                last_change_idx = i
 
        smoothed[col] = held
 
    # Clip to bounds
    smoothed = smoothed.clip(lower=MIN_ACTIVATION, upper=MAX_ACTIVATION)
 
    return smoothed
 
 
# ---------------------------------------------------------------------------
# Feature Assembly
# ---------------------------------------------------------------------------
 
def assemble_allocator_features(
    state_features: pd.DataFrame,
    strategy_health: pd.DataFrame,
    strategy_confidences: Dict[str, pd.Series],
    portfolio_state: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Assemble the complete feature set for the meta-allocator.
 
    Inputs (from build spec Section 6C):
        - State features: all Layer 0 observable + latent + transition
        - Strategy health: rolling Sharpe, DD, hit rate, turnover, stability
        - Strategy confidences: each strategy's self-reported confidence
        - Portfolio state: current drawdown, turnover, days since rebalance
 
    All features are aligned to a common date index.
    """
    parts = [state_features]
 
    if not strategy_health.empty:
        parts.append(strategy_health)
 
    # Add strategy confidences as features
    conf_df = pd.DataFrame(strategy_confidences)
    if not conf_df.empty:
        conf_df.columns = [f"conf_{c}" for c in conf_df.columns]
        parts.append(conf_df)
 
    if portfolio_state is not None and not portfolio_state.empty:
        parts.append(portfolio_state)
 
    combined = pd.concat(parts, axis=1)
 
    # Remove duplicate columns if any
    combined = combined.loc[:, ~combined.columns.duplicated()]
 
    return combined
 
 
# ---------------------------------------------------------------------------
# Meta-Allocator
# ---------------------------------------------------------------------------
 
class MetaAllocator:
    """
    LightGBM-based meta-allocator (Phase 1).
 
    Two-stage hierarchical decision:
        Stage 1: Strategy activation (should each strategy matter?)
        Stage 2: Capital sizing (how much does each get?)
 
    The allocator learns from historical strategy returns which
    state conditions favor which strategies, and sizes capital
    accordingly.
    """
 
    def __init__(
        self,
        strategy_names: Optional[List[str]] = None,
        train_window: int = TRAIN_WINDOW,
        retrain_every: int = RETRAIN_EVERY,
        purge_gap: int = PURGE_GAP,
        n_ensemble: int = N_ENSEMBLE,
        lgbm_params: Optional[Dict[str, Any]] = None,
        importance_threshold: float = IMPORTANCE_THRESHOLD,
    ):
        self.strategy_names = strategy_names or STRATEGY_NAMES
        self.train_window = train_window
        self.retrain_every = retrain_every
        self.purge_gap = purge_gap
        self.n_ensemble = n_ensemble
        self.lgbm_params = lgbm_params or LGBM_PARAMS.copy()
        self.importance_threshold = importance_threshold
 
        # Trained models (one per strategy, ensemble of n_ensemble)
        self.activation_models: Dict[str, List] = {}
        self.sizing_models: List = []
 
        # Feature importance tracking
        self.feature_importances: Dict[str, pd.Series] = {}
        self.pruned_features: List[str] = []
 
        # Training history
        self.training_windows: List[Dict[str, Any]] = []
 
    def train_walk_forward(
        self,
        features: pd.DataFrame,
        strategy_returns: Dict[str, pd.Series],
        benchmark_returns: pd.Series,
        strategy_turnovers: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """
        Train the meta-allocator using walk-forward validation.
 
        For each window:
            1. Train activation models on [t - train_window : t - purge_gap]
            2. Predict activations for [t : t + retrain_every]
            3. Compute blended portfolio returns
            4. Move window forward
 
        Args:
            features: Combined state + health + confidence features
            strategy_returns: Daily returns per strategy
            benchmark_returns: SPY daily returns
            strategy_turnovers: Turnover per strategy (optional)
 
        Returns:
            DataFrame with columns: date, activation per strategy,
            blended_weights, portfolio_return
        """
        import lightgbm as lgb
 
        # ---- Prepare target variables ----
        # For each strategy, the target is the forward reward
        # (how well would allocating to this strategy have performed?)
        logger.info("Meta-allocator: Preparing walk-forward training...")
 
        # Align all data
        common_idx = features.dropna(how="all").index
        for name, ret in strategy_returns.items():
            common_idx = common_idx.intersection(ret.dropna().index)
        common_idx = common_idx.intersection(benchmark_returns.dropna().index)
        common_idx = common_idx.sort_values()
 
        if len(common_idx) < self.train_window + self.purge_gap + 63:
            raise ValueError(
                f"Insufficient data for walk-forward: {len(common_idx)} days, "
                f"need at least {self.train_window + self.purge_gap + 63}"
            )
 
        logger.info(f"  Common index: {len(common_idx)} days, "
                     f"{common_idx[0].date()} → {common_idx[-1].date()}")
 
        # ---- Compute forward rewards for each strategy ----
        forward_window = 42  # evaluate strategy performance over ~2 months
        strategy_rewards = {}
 
        for name in self.strategy_names:
            if name not in strategy_returns:
                continue
 
            ret = strategy_returns[name].reindex(common_idx)
            bench = benchmark_returns.reindex(common_idx)
 
            rewards = pd.Series(np.nan, index=common_idx)
            turnover_avg = 0.20  # default
 
            if strategy_turnovers and name in strategy_turnovers:
                turnover_avg = strategy_turnovers[name].reindex(common_idx).mean()
                if pd.isna(turnover_avg):
                    turnover_avg = 0.20
 
            for i in range(len(common_idx) - forward_window):
                fwd_ret = ret.iloc[i:i + forward_window]
                fwd_bench = bench.iloc[i:i + forward_window]
 
                reward = compute_tail_aware_reward(
                    fwd_ret, fwd_bench, turnover_avg, window=forward_window
                )
                rewards.iloc[i] = reward
 
            strategy_rewards[name] = rewards
            valid = rewards.notna().sum()
            logger.info(f"  Strategy {name}: {valid} valid reward samples, "
                         f"mean={rewards.mean():.3f}")
 
        # ---- Walk-forward training ----
        all_results = []
        feature_cols = features.columns.tolist()
 
        # Start after enough data for training window
        start_idx = self.train_window
        end_idx = len(common_idx) - forward_window
 
        n_windows = 0
 
        for window_start in range(start_idx, end_idx, self.retrain_every):
            train_end = window_start - self.purge_gap
            train_start = max(0, window_start - self.train_window)
 
            test_start = window_start
            test_end = min(window_start + self.retrain_every, end_idx)
 
            if train_end <= train_start or test_end <= test_start:
                continue
 
            train_dates = common_idx[train_start:train_end]
            test_dates = common_idx[test_start:test_end]
 
            if len(train_dates) < 252 or len(test_dates) < 5:
                continue
 
            # ---- Train activation models for each strategy ----
            window_models = {}
            window_importances = {}
 
            X_train = features.loc[train_dates].copy()
            X_test = features.loc[test_dates].copy()
 
            # Drop columns that are all NaN in training data
            valid_cols = [c for c in feature_cols if X_train[c].notna().sum() > len(X_train) * 0.3]
            X_train = X_train[valid_cols]
            X_test = X_test[valid_cols]
 
            # Fill remaining NaN with column median
            train_medians = X_train.median()
            X_train = X_train.fillna(train_medians)
            X_test = X_test.fillna(train_medians)
 
            activations = pd.DataFrame(index=test_dates)
 
            for name in self.strategy_names:
                if name not in strategy_rewards:
                    activations[name] = 0.5  # default
                    continue
 
                y_train = strategy_rewards[name].loc[train_dates]
 
                # Drop rows with NaN target
                valid_mask = y_train.notna()
                X_tr = X_train.loc[valid_mask]
                y_tr = y_train.loc[valid_mask]
 
                if len(y_tr) < 100:
                    activations[name] = 0.5
                    continue
 
                # Train ensemble
                ensemble = []
                importances = []
 
                for seed in range(self.n_ensemble):
                    params = self.lgbm_params.copy()
                    params["random_state"] = seed * 42
 
                    model = lgb.LGBMRegressor(**params)
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_tr.tail(252), y_tr.tail(252))],
                        callbacks=[lgb.log_evaluation(period=0)],
                    )
                    ensemble.append(model)
                    importances.append(
                        pd.Series(model.feature_importances_, index=X_tr.columns)
                    )
 
                window_models[name] = ensemble
                window_importances[name] = pd.concat(importances, axis=1).mean(axis=1)
 
                # Predict on test set
                preds = np.mean([m.predict(X_test) for m in ensemble], axis=0)
 
                # Convert reward predictions to activation scores [0, 1]
                # Higher predicted reward → higher activation
                # Use sigmoid-like mapping
                act = self._reward_to_activation(preds, y_tr)
                activations[name] = act
 
            # Store models from the last window
            self.activation_models = window_models
            self.feature_importances = window_importances
 
            # ---- Stage 2: Capital Sizing ----
            # Use softmax-like scaling to convert activations to capital weights.
            # This amplifies differences — a strategy with 0.7 activation gets
            # meaningfully more capital than one with 0.4.
            temperature = 2.0  # higher = more differentiation
            scaled = np.exp(activations * temperature)
            capital_weights = scaled.div(scaled.sum(axis=1), axis=0)

            # Apply allocation bounds to prevent over-concentration
            # Min 10% per strategy, max 65% per strategy
            capital_weights = capital_weights.clip(lower=0.10, upper=0.65)
            # Renormalize after clipping
            capital_weights = capital_weights.div(capital_weights.sum(axis=1), axis=0)
 
            # Record results for this window
            for date in test_dates:
                if date not in activations.index:
                    continue
 
                result = {
                    "date": date,
                }
 
                # Activation scores
                for name in self.strategy_names:
                    result[f"activation_{name}"] = float(activations.loc[date].get(name, 0.5))
                    result[f"capital_weight_{name}"] = float(capital_weights.loc[date].get(name, 1/3))
 
                all_results.append(result)
 
            n_windows += 1
            window_info = {
                "window": n_windows,
                "train": f"{train_dates[0].date()} → {train_dates[-1].date()}",
                "test": f"{test_dates[0].date()} → {test_dates[-1].date()}",
                "train_size": len(train_dates),
                "test_size": len(test_dates),
            }
            self.training_windows.append(window_info)
 
            logger.info(
                f"  Window {n_windows}: train {window_info['train']} ({len(train_dates)}d), "
                f"test {window_info['test']} ({len(test_dates)}d)"
            )
 
        if not all_results:
            raise RuntimeError("No walk-forward windows completed — check data length")
 
        results_df = pd.DataFrame(all_results).set_index("date")
 
        # ---- Apply activation smoothing ----
        act_cols = [c for c in results_df.columns if c.startswith("activation_")]
        raw_activations = results_df[act_cols].copy()
        raw_activations.columns = [c.replace("activation_", "") for c in act_cols]
 
        smoothed = smooth_activations(raw_activations)
 
        for name in self.strategy_names:
            if name in smoothed.columns:
                results_df[f"activation_{name}"] = smoothed[name]
 
        # Recompute capital weights after smoothing
        act_smoothed = results_df[[f"activation_{n}" for n in self.strategy_names]].copy()
        act_smoothed.columns = self.strategy_names
        act_sum = act_smoothed.sum(axis=1)
        cap_weights = act_smoothed.div(act_sum.replace(0, 1), axis=0)
 
        for name in self.strategy_names:
            results_df[f"capital_weight_{name}"] = cap_weights[name]
 
        # ---- Prune low-importance features ----
        self._prune_features()
 
        logger.info(
            f"Meta-allocator training complete: {n_windows} windows, "
            f"{len(results_df)} out-of-sample days"
        )
 
        return results_df
 
    def compute_blended_portfolio(
        self,
        allocator_results: pd.DataFrame,
        strategy_returns: Dict[str, pd.Series],
    ) -> pd.Series:
        """
        Compute the blended portfolio daily returns using allocator capital weights.
        """
        blended = pd.Series(0.0, index=allocator_results.index)
 
        # Forward-fill capital weights for days between signals
        for name in self.strategy_names:
            weight_col = f"capital_weight_{name}"
            if weight_col not in allocator_results.columns:
                continue
            if name not in strategy_returns:
                continue
 
            weights = allocator_results[weight_col]
            returns = strategy_returns[name].reindex(allocator_results.index)
 
            blended += weights * returns.fillna(0)
 
        return blended
 
    def _reward_to_activation(
        self,
        predicted_rewards: np.ndarray,
        training_rewards: pd.Series,
    ) -> np.ndarray:
        """
        Map predicted rewards to activation scores in [0, 1].

        Uses z-score mapping against the training distribution:
            - Predicted reward at training mean → activation 0.5
            - 2 std above mean → activation ~0.88
            - 2 std below mean → activation ~0.12

        This preserves the magnitude of predictions so that cross-strategy
        comparisons remain meaningful during capital sizing.
        """
        train_vals = training_rewards.dropna().values

        if len(train_vals) < 10:
            return np.full(len(predicted_rewards), 0.5)

        train_mean = np.mean(train_vals)
        train_std = np.std(train_vals)

        if train_std < 1e-8:
            return np.full(len(predicted_rewards), 0.5)

        # Z-score the predictions against training distribution
        z_scores = (predicted_rewards - train_mean) / train_std

        # Sigmoid mapping: z=0 → 0.5, z=2 → 0.88, z=-2 → 0.12
        activations = 1.0 / (1.0 + np.exp(-z_scores))

        return np.clip(activations, MIN_ACTIVATION, MAX_ACTIVATION)
 
    def _prune_features(self) -> None:
        """
        Identify features with importance below threshold across all models.
        These features can be removed in future training iterations.
        """
        if not self.feature_importances:
            return
 
        # Average importance across all strategy models
        all_importances = pd.concat(
            [imp for imp in self.feature_importances.values()],
            axis=1,
        ).mean(axis=1)
 
        if all_importances.empty:
            return
 
        # Normalize to percentages
        total = all_importances.sum()
        if total > 0:
            normalized = all_importances / total
        else:
            return
 
        # Identify low-importance features
        self.pruned_features = normalized[
            normalized < self.importance_threshold
        ].index.tolist()
 
        n_total = len(normalized)
        n_pruned = len(self.pruned_features)
 
        logger.info(
            f"Feature pruning: {n_pruned}/{n_total} features below "
            f"{self.importance_threshold:.0%} importance threshold"
        )
 
        # Log top 10 most important features
        top_10 = normalized.nlargest(10)
        for feat, imp in top_10.items():
            logger.info(f"    {feat}: {imp:.1%}")
 
    def predict_activations(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict strategy activations for new data using trained models.
 
        Returns DataFrame with activation scores per strategy.
        """
        if not self.activation_models:
            logger.warning("No trained models — returning equal activations")
            return pd.DataFrame(
                {name: 0.33 for name in self.strategy_names},
                index=features.index,
            )
 
        activations = pd.DataFrame(index=features.index)
 
        for name in self.strategy_names:
            if name not in self.activation_models:
                activations[name] = 0.33
                continue
 
            ensemble = self.activation_models[name]
 
            # Align features to what the model was trained on
            model_features = ensemble[0].feature_name_
            available = [f for f in model_features if f in features.columns]
            missing = [f for f in model_features if f not in features.columns]
 
            if missing:
                logger.warning(f"  {name}: {len(missing)} features missing from input")
 
            X = features.reindex(columns=model_features, fill_value=0)
            X = X.fillna(0)
 
            # Ensemble prediction
            preds = np.mean([m.predict(X) for m in ensemble], axis=0)
            activations[name] = preds
 
        # Apply smoothing and bounds
        activations = smooth_activations(activations)
 
        return activations
 
 
# ---------------------------------------------------------------------------
# Backtest Integration
# ---------------------------------------------------------------------------
 
def backtest_meta_allocator(
    snapshot: Dict[str, Any],
    state_features: pd.DataFrame,
    strategy_outputs: Dict[str, list],
    adj_close: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Run a full meta-allocator backtest.
 
    Steps:
        1. Compute daily returns per strategy
        2. Compute strategy health metrics
        3. Assemble allocator features
        4. Train walk-forward meta-allocator
        5. Compute blended portfolio
        6. Compare against equal-weight baseline
 
    Args:
        snapshot: Frozen data snapshot
        state_features: State representation DataFrame
        strategy_outputs: {strategy_name: list of StrategyOutput}
        adj_close: Adjusted close prices
 
    Returns:
        Dict with results, metrics, and comparison data
    """
    from strategy_health import (
        StrategyHealthTracker,
        compute_strategy_daily_returns,
        compute_weight_changes,
    )
 
    benchmark = adj_close["SPY"].pct_change() if "SPY" in adj_close.columns else pd.Series(0, index=adj_close.index)
 
    # ---- Step 1: Compute daily returns per strategy ----
    logger.info("Computing daily strategy returns...")
    strategy_returns = {}
    strategy_turnovers = {}
    strategy_confidences = {}
 
    for name, outputs in strategy_outputs.items():
        strategy_returns[name] = compute_strategy_daily_returns(outputs, adj_close)
 
        # Extract confidence series
        conf = pd.Series(
            {pd.Timestamp(o.strategy_metadata["date"]): o.confidence for o in outputs}
        )
        strategy_confidences[name] = conf.reindex(adj_close.index).ffill()
 
        # Compute weight changes for turnover
        changes = compute_weight_changes(outputs, adj_close.index)
        strategy_turnovers[name] = changes
 
    # ---- Step 2: Compute strategy health ----
    logger.info("Computing strategy health metrics...")
    tracker = StrategyHealthTracker()
    health = tracker.compute_all_health(
        strategy_returns,
        strategy_weight_changes=strategy_turnovers,
        strategy_turnovers=strategy_turnovers,
    )
 
    # ---- Step 3: Assemble features ----
    logger.info("Assembling allocator features...")
    features = assemble_allocator_features(
        state_features, health, strategy_confidences
    )
 
    # ---- Step 4: Train walk-forward ----
    logger.info("Training meta-allocator walk-forward...")
    allocator = MetaAllocator()
    allocator_results = allocator.train_walk_forward(
        features, strategy_returns, benchmark, strategy_turnovers
    )
 
    # ---- Step 5: Compute blended portfolio ----
    blended_returns = allocator.compute_blended_portfolio(
        allocator_results, strategy_returns
    )
 
    # ---- Step 6: Compute equal-weight baseline for comparison ----
    equal_weight_returns = pd.Series(0.0, index=allocator_results.index)
    n_strategies = len(strategy_returns)
    for name, ret in strategy_returns.items():
        equal_weight_returns += ret.reindex(allocator_results.index).fillna(0) / n_strategies
 
    # ---- Compute metrics ----
    def compute_metrics(returns: pd.Series, label: str) -> Dict[str, float]:
        r = returns.dropna()
        if len(r) < 252:
            return {"label": label, "error": "insufficient data"}
 
        n_years = len(r) / 252
        cum = (1 + r).cumprod()
        cagr = cum.iloc[-1] ** (1 / n_years) - 1
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        downside = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 1e-6
        sortino = (r.mean() * 252) / downside
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        max_dd = dd.min()
        calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0
        vol = r.std() * np.sqrt(252)
 
        return {
            "label": label,
            "sharpe": float(sharpe),
            "cagr": float(cagr),
            "max_dd": float(max_dd),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "vol": float(vol),
            "n_days": len(r),
        }
 
    meta_metrics = compute_metrics(blended_returns, "Meta-Allocator")
    equal_metrics = compute_metrics(equal_weight_returns, "Equal-Weight")
    bench_metrics = compute_metrics(
        benchmark.reindex(allocator_results.index), "SPY Benchmark"
    )
 
    # ---- Activation statistics ----
    act_stats = {}
    for name in allocator.strategy_names:
        col = f"activation_{name}"
        if col in allocator_results.columns:
            act_stats[name] = {
                "mean": float(allocator_results[col].mean()),
                "std": float(allocator_results[col].std()),
                "min": float(allocator_results[col].min()),
                "max": float(allocator_results[col].max()),
            }
 
    cap_stats = {}
    for name in allocator.strategy_names:
        col = f"capital_weight_{name}"
        if col in allocator_results.columns:
            cap_stats[name] = {
                "mean": float(allocator_results[col].mean()),
                "std": float(allocator_results[col].std()),
            }
 
    return {
        "allocator": allocator,
        "allocator_results": allocator_results,
        "blended_returns": blended_returns,
        "equal_weight_returns": equal_weight_returns,
        "meta_metrics": meta_metrics,
        "equal_metrics": equal_metrics,
        "benchmark_metrics": bench_metrics,
        "activation_stats": act_stats,
        "capital_weight_stats": cap_stats,
        "feature_importances": allocator.feature_importances,
        "training_windows": allocator.training_windows,
    }
 
 
# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
 
def print_allocator_report(results: Dict[str, Any]) -> None:
    """Print a comprehensive meta-allocator performance report."""
    print(f"\n{'='*70}")
    print(f"META-ALLOCATOR PERFORMANCE REPORT")
    print(f"{'='*70}")
 
    # ---- Performance comparison ----
    print(f"\n--- Performance Comparison ---")
    for metrics in [results["meta_metrics"], results["equal_metrics"], results["benchmark_metrics"]]:
        if "error" in metrics:
            print(f"  {metrics['label']}: {metrics['error']}")
            continue
        print(
            f"  {metrics['label']:<20s} "
            f"Sharpe={metrics['sharpe']:.2f}  "
            f"CAGR={metrics['cagr']:.1%}  "
            f"MaxDD={metrics['max_dd']:.1%}  "
            f"Sortino={metrics['sortino']:.2f}  "
            f"Calmar={metrics['calmar']:.2f}  "
            f"Vol={metrics['vol']:.1%}"
        )
 
    # ---- Activation statistics ----
    print(f"\n--- Strategy Activation (mean ± std) ---")
    for name, stats in results["activation_stats"].items():
        print(f"  {name:<20s} {stats['mean']:.3f} ± {stats['std']:.3f} "
              f"[{stats['min']:.3f} — {stats['max']:.3f}]")
 
    # ---- Capital allocation ----
    print(f"\n--- Capital Allocation (mean ± std) ---")
    for name, stats in results["capital_weight_stats"].items():
        print(f"  {name:<20s} {stats['mean']:.1%} ± {stats['std']:.1%}")
 
    # ---- Feature importance ----
    print(f"\n--- Top Features (averaged across strategies) ---")
    importances = results.get("feature_importances", {})
    if importances:
        all_imp = pd.concat(
            [imp for imp in importances.values()], axis=1
        ).mean(axis=1).sort_values(ascending=False)
 
        total = all_imp.sum()
        if total > 0:
            normalized = all_imp / total
            for feat, imp in normalized.head(15).items():
                print(f"  {feat:<45s} {imp:.1%}")
 
    # ---- Training windows ----
    windows = results.get("training_windows", [])
    print(f"\n--- Walk-Forward: {len(windows)} windows ---")
    if windows:
        print(f"  First: {windows[0]['train']}")
        print(f"  Last:  {windows[-1]['test']}")
 
    # ---- Pass/Fail gates ----
    meta = results["meta_metrics"]
    equal = results["equal_metrics"]
    if "error" not in meta and "error" not in equal:
        print(f"\n--- Step 6 Verification ---")
        sharpe_pass = meta["sharpe"] > equal["sharpe"]
        dd_pass = abs(meta["max_dd"]) < abs(equal["max_dd"])
        print(f"  Meta Sharpe > Equal Sharpe: {meta['sharpe']:.2f} vs {equal['sharpe']:.2f} → {'PASS' if sharpe_pass else 'FAIL'}")
        print(f"  Meta MaxDD < Equal MaxDD:   {meta['max_dd']:.1%} vs {equal['max_dd']:.1%} → {'PASS' if dd_pass else 'FAIL'}")
        print(f"  Overall Step 6 gate:        {'PASS' if (sharpe_pass and dd_pass) else 'FAIL'}")