"""
SPY Alpha v8 — Execution Engine (Alpaca)
==========================================

Translates daily model target allocations into broker orders.
Reads latest_prediction.json, compares to current positions,
generates delta orders, and submits via Alpaca API.

Safety Controls:
    - Max single order size (% of portfolio)
    - Max daily turnover
    - Stale signal protection
    - Position verification after execution
    - Kill switch via environment variable
    - Paper trading mode by default
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("spy_alpha_v8.execution")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAPER_BASE_URL = "https://paper-api.alpaca.markets"

# Safety limits
MAX_ORDER_PCT = 0.25          # max 25% of portfolio in a single order
MAX_DAILY_TURNOVER = 0.40     # max 40% portfolio turnover per day
MAX_SIGNAL_AGE_HOURS = 24     # reject signals older than 24 hours
MIN_ORDER_VALUE = 10.0        # minimum order value in dollars

# Assets that can't be traded on Alpaca (indices)
NON_TRADEABLE = {"^VIX", "^VIX3M", "^SKEW"}


# ---------------------------------------------------------------------------
# Alpaca Client
# ---------------------------------------------------------------------------

class AlpacaExecutor:
    """
    Executes portfolio rebalancing via Alpaca API.

    Flow:
        1. Load target weights from latest_prediction.json
        2. Get current positions from Alpaca
        3. Compute delta orders
        4. Apply safety checks
        5. Submit orders
        6. Verify execution
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: str = PAPER_BASE_URL,
    ):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self.base_url = base_url

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables."
            )

        import alpaca_trade_api as tradeapi
        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            self.base_url,
            api_version="v2",
        )

        # Verify connection
        account = self.api.get_account()
        self.account_value = float(account.portfolio_value)
        logger.info(
            f"Alpaca connected: ${self.account_value:,.2f} portfolio value, "
            f"status={account.status}"
        )

    def execute_rebalance(
        self,
        signal_path: Path = Path("signals/latest_prediction.json"),
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a full portfolio rebalance based on the latest signal.

        Args:
            signal_path: Path to the signal JSON file
            dry_run: If True, compute orders but don't submit them

        Returns:
            Execution report with orders placed and results
        """
        # ---- Check kill switch ----
        if os.environ.get("KILL_SWITCH", "").lower() == "true":
            logger.warning("KILL SWITCH ACTIVE — no orders will be placed")
            return {"status": "killed", "orders": []}

        # ---- Load target weights ----
        target_weights = self._load_signal(signal_path)
        if target_weights is None:
            return {"status": "no_signal", "orders": []}

        # ---- Get current positions ----
        current_weights = self._get_current_weights()

        # ---- Compute deltas ----
        deltas = self._compute_deltas(target_weights, current_weights)

        # ---- Apply safety checks ----
        safe_deltas = self._apply_safety_checks(deltas)

        # ---- Submit orders ----
        if dry_run:
            logger.info("DRY RUN — orders computed but not submitted")
            return {
                "status": "dry_run",
                "target_weights": target_weights,
                "current_weights": current_weights,
                "deltas": deltas,
                "safe_deltas": safe_deltas,
                "orders": [],
            }

        orders = self._submit_orders(safe_deltas)

        # ---- Verify ----
        verification = self._verify_positions(target_weights)

        return {
            "status": "executed",
            "target_weights": target_weights,
            "current_weights": current_weights,
            "deltas": deltas,
            "safe_deltas": safe_deltas,
            "orders": orders,
            "verification": verification,
        }

    def _load_signal(self, signal_path: Path) -> Optional[Dict[str, float]]:
        """Load and validate the latest signal."""
        if not signal_path.exists():
            logger.error(f"Signal file not found: {signal_path}")
            return None

        with open(signal_path) as f:
            signal = json.load(f)

        # Check signal age
        generated_at = signal.get("generated_at", "")
        if generated_at:
            try:
                signal_time = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                age = datetime.now(timezone.utc) - signal_time
                if age > timedelta(hours=MAX_SIGNAL_AGE_HOURS):
                    logger.error(
                        f"Signal is {age.total_seconds()/3600:.1f} hours old "
                        f"(max {MAX_SIGNAL_AGE_HOURS}h). Rejecting stale signal."
                    )
                    return None
            except (ValueError, TypeError):
                logger.warning("Could not parse signal timestamp")

        # Extract weights
        portfolio = signal.get("portfolio", {})
        weights = portfolio.get("weights", {})

        if not weights:
            logger.error("No weights found in signal")
            return None

        # Filter out non-tradeable assets
        tradeable = {k: v for k, v in weights.items() if k not in NON_TRADEABLE}

        # Renormalize
        total = sum(tradeable.values())
        if total > 0:
            tradeable = {k: v / total for k, v in tradeable.items()}

        logger.info(f"Target weights loaded: {len(tradeable)} assets")
        return tradeable

    def _get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights from Alpaca."""
        positions = self.api.list_positions()
        account = self.api.get_account()
        portfolio_value = float(account.portfolio_value)

        if portfolio_value <= 0:
            return {}

        weights = {}
        for pos in positions:
            market_value = float(pos.market_value)
            weights[pos.symbol] = market_value / portfolio_value

        # Cash weight
        cash = float(account.cash)
        if cash > 0:
            weights["_CASH"] = cash / portfolio_value

        logger.info(
            f"Current positions: {len(positions)} assets, "
            f"cash: ${cash:,.2f} ({cash/portfolio_value:.1%})"
        )
        return weights

    def _compute_deltas(
        self,
        target: Dict[str, float],
        current: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Compute the delta orders needed to rebalance."""
        account = self.api.get_account()
        portfolio_value = float(account.portfolio_value)

        all_assets = set(list(target.keys()) + list(current.keys()))
        all_assets.discard("_CASH")

        deltas = []
        for asset in sorted(all_assets):
            target_w = target.get(asset, 0.0)
            current_w = current.get(asset, 0.0)
            delta_w = target_w - current_w

            if abs(delta_w) < 0.005:  # less than 0.5% change — skip
                continue

            delta_value = delta_w * portfolio_value

            deltas.append({
                "symbol": asset,
                "target_weight": target_w,
                "current_weight": current_w,
                "delta_weight": delta_w,
                "delta_value": delta_value,
                "side": "buy" if delta_w > 0 else "sell",
            })

        # Sort: sells first, then buys (free up cash before buying)
        deltas.sort(key=lambda x: (0 if x["side"] == "sell" else 1, -abs(x["delta_value"])))

        logger.info(f"Computed {len(deltas)} delta orders")
        return deltas

    def _apply_safety_checks(
        self,
        deltas: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply safety limits to delta orders."""
        account = self.api.get_account()
        portfolio_value = float(account.portfolio_value)

        safe = []
        total_turnover = 0.0

        # Detect initial portfolio setup (no current positions)
        is_initial_setup = all(
            d.get("current_weight", 0) == 0 for d in deltas
        )
        effective_turnover_limit = 1.0 if is_initial_setup else MAX_DAILY_TURNOVER

        for delta in deltas:
            order_pct = abs(delta["delta_value"]) / portfolio_value

            # Skip orders below minimum value
            if abs(delta["delta_value"]) < MIN_ORDER_VALUE:
                logger.debug(f"  Skipping {delta['symbol']}: below ${MIN_ORDER_VALUE} minimum")
                continue

            # Cap single order size
            if order_pct > MAX_ORDER_PCT:
                logger.warning(
                    f"  Capping {delta['symbol']} order: "
                    f"{order_pct:.1%} -> {MAX_ORDER_PCT:.1%}"
                )
                scale = MAX_ORDER_PCT / order_pct
                delta["delta_value"] *= scale
                delta["delta_weight"] *= scale
                delta["capped"] = True

            # Check total turnover
            total_turnover += abs(delta["delta_weight"])
            if total_turnover > effective_turnover_limit:
                logger.warning(
                    f"  Daily turnover limit reached ({total_turnover:.1%}). "
                    f"Remaining orders skipped."
                )
                break

            safe.append(delta)

        logger.info(
            f"Safety checks: {len(safe)}/{len(deltas)} orders passed, "
            f"turnover: {total_turnover:.1%}"
        )
        return safe

    def _submit_orders(
        self,
        deltas: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Submit orders to Alpaca."""
        results = []

        for delta in deltas:
            symbol = delta["symbol"]
            side = delta["side"]
            notional = abs(delta["delta_value"])

            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    notional=round(notional, 2),
                    side=side,
                    type="market",
                    time_in_force="day",
                )

                results.append({
                    "symbol": symbol,
                    "side": side,
                    "notional": notional,
                    "order_id": order.id,
                    "status": order.status,
                })

                logger.info(
                    f"  Order submitted: {side.upper()} ${notional:,.2f} of {symbol} "
                    f"(id: {order.id})"
                )

            except Exception as e:
                logger.error(f"  Order failed for {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "side": side,
                    "notional": notional,
                    "error": str(e),
                })

        return results

    def _verify_positions(
        self,
        target_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """Verify positions after execution."""
        import time
        time.sleep(2)  # wait for orders to fill

        current = self._get_current_weights()

        discrepancies = {}
        for asset, target_w in target_weights.items():
            actual_w = current.get(asset, 0.0)
            diff = abs(target_w - actual_w)
            if diff > 0.02:  # more than 2% off target
                discrepancies[asset] = {
                    "target": target_w,
                    "actual": actual_w,
                    "diff": diff,
                }

        return {
            "n_discrepancies": len(discrepancies),
            "discrepancies": discrepancies,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        account = self.api.get_account()
        positions = self.api.list_positions()

        holdings = []
        for pos in positions:
            holdings.append({
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "market_value": float(pos.market_value),
                "weight": float(pos.market_value) / float(account.portfolio_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
            })

        holdings.sort(key=lambda x: -x["market_value"])

        return {
            "portfolio_value": float(account.portfolio_value),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "n_positions": len(positions),
            "holdings": holdings,
        }


# ---------------------------------------------------------------------------
# CLI Integration
# ---------------------------------------------------------------------------

def print_execution_report(report: Dict[str, Any]) -> None:
    """Print execution report."""
    print(f"\n{'='*60}")
    print(f"EXECUTION REPORT — {report.get('status', 'unknown').upper()}")
    print(f"{'='*60}")

    if report["status"] == "killed":
        print("  Kill switch is active. No orders placed.")
        return

    if report["status"] == "no_signal":
        print("  No valid signal found.")
        return

    # Target vs current
    target = report.get("target_weights", {})
    current = report.get("current_weights", {})

    print(f"\n--- Target Allocation ---")
    for asset, w in sorted(target.items(), key=lambda x: -x[1]):
        curr = current.get(asset, 0)
        delta = w - curr
        direction = "+" if delta > 0 else ""
        print(f"  {asset:<8s} {w:>7.1%}  (current: {curr:>6.1%}, delta: {direction}{delta:.1%})")

    # Orders
    orders = report.get("orders", [])
    if orders:
        print(f"\n--- Orders ({len(orders)}) ---")
        for o in orders:
            if "error" in o:
                print(f"  FAILED: {o['side'].upper()} ${o['notional']:,.2f} {o['symbol']} — {o['error']}")
            else:
                print(f"  {o['status']}: {o['side'].upper()} ${o['notional']:,.2f} {o['symbol']}")

    # Verification
    verification = report.get("verification", {})
    if verification:
        n_disc = verification.get("n_discrepancies", 0)
        print(f"\n--- Verification ---")
        print(f"  Discrepancies: {n_disc}")
        for asset, disc in verification.get("discrepancies", {}).items():
            print(f"    {asset}: target={disc['target']:.1%}, actual={disc['actual']:.1%}")