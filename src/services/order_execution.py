"""
Order execution service.
Monitors pending orders and executes them when price conditions are met.
"""

from typing import List, Tuple
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import PendingOrder, PortfolioDatabase
from services.portfolio_service import PortfolioService


class OrderExecutionService:
    """Service for monitoring and executing pending orders"""

    def __init__(self, service: PortfolioService):
        """Initialize order execution service"""
        self.service = service
        self.db = service.db

    def check_and_execute_orders(self) -> List[Tuple[int, str, bool]]:
        """
        Check all pending orders and execute those that meet conditions.

        Returns:
            List of (order_id, message, success) tuples for executed orders
        """
        executed_orders = []

        # Get all pending orders
        pending_orders = self.db.get_all_pending_orders()

        if not pending_orders:
            return executed_orders

        # Get current prices for all tickers
        tickers = list(set(order.ticker for order in pending_orders))
        current_prices = self.service.get_live_prices(tickers)

        # Check each order
        for order in pending_orders:
            current_price = current_prices.get(order.ticker, 0)

            if current_price == 0:
                continue  # Skip if price unavailable

            should_execute = False
            reason = ""

            # Check execution conditions
            if order.order_type == 'BUY':
                if order.limit_price is None:
                    # Market order - execute immediately
                    should_execute = True
                    reason = "Market order"
                elif current_price <= order.limit_price:
                    # Limit buy - execute when price at or below limit
                    should_execute = True
                    reason = f"Price ${current_price:.2f} at or below limit ${order.limit_price:.2f}"

            elif order.order_type == 'SELL':
                if order.limit_price is None:
                    # Market order - execute immediately
                    should_execute = True
                    reason = "Market order"
                elif current_price >= order.limit_price:
                    # Limit sell - execute when price at or above limit
                    should_execute = True
                    reason = f"Price ${current_price:.2f} at or above limit ${order.limit_price:.2f}"

            if should_execute:
                success, message = self._execute_order(order, current_price)
                executed_orders.append((order.id, f"{reason}. {message}", success))

        return executed_orders

    def _execute_order(self, order: PendingOrder, execution_price: float) -> Tuple[bool, str]:
        """
        Execute a specific order.

        Args:
            order: The order to execute
            execution_price: Price at which to execute

        Returns:
            (success, message) tuple
        """
        try:
            if order.order_type == 'BUY':
                # Execute buy
                success, msg = self.service.buy_stock(
                    ticker=order.ticker,
                    quantity=order.quantity,
                    price=execution_price,
                    date=datetime.now().strftime("%Y-%m-%d"),
                    notes=f"Auto-executed limit order #{order.id}"
                )

                if success:
                    # Mark order as filled
                    self.db.delete_order(order.id)
                    return True, f"Bought {order.quantity:.2f} shares of {order.ticker} at ${execution_price:.2f}"
                else:
                    return False, msg

            elif order.order_type == 'SELL':
                # Execute sell
                success, msg = self.service.sell_stock(
                    ticker=order.ticker,
                    quantity=order.quantity,
                    price=execution_price,
                    date=datetime.now().strftime("%Y-%m-%d"),
                    notes=f"Auto-executed limit order #{order.id}"
                )

                if success:
                    # Mark order as filled
                    self.db.delete_order(order.id)
                    return True, f"Sold {order.quantity:.2f} shares of {order.ticker} at ${execution_price:.2f}"
                else:
                    return False, msg

        except Exception as e:
            return False, f"Error executing order: {str(e)}"

        return False, "Unknown order type"

    def check_single_order(self, order_id: int) -> Tuple[bool, str]:
        """
        Check and execute a single order if conditions are met.

        Args:
            order_id: ID of the order to check

        Returns:
            (executed, message) tuple
        """
        # Get the order
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pending_orders WHERE id = ? AND order_status = 'PENDING'", (order_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return False, "Order not found or already executed"

        order = PendingOrder(
            id=row['id'],
            ticker=row['ticker'],
            order_type=row['order_type'],
            quantity=row['quantity'],
            limit_price=row['limit_price'],
            order_status=row['order_status'],
            created_at=row['created_at']
        )

        # Get current price
        current_prices = self.service.get_live_prices([order.ticker])
        current_price = current_prices.get(order.ticker, 0)

        if current_price == 0:
            return False, "Unable to fetch current price"

        # Check if should execute
        should_execute = False

        if order.order_type == 'BUY':
            if order.limit_price is None or current_price <= order.limit_price:
                should_execute = True
        elif order.order_type == 'SELL':
            if order.limit_price is None or current_price >= order.limit_price:
                should_execute = True

        if should_execute:
            success, msg = self._execute_order(order, current_price)
            return success, msg
        else:
            if order.order_type == 'BUY':
                return False, f"Current price ${current_price:.2f} is above limit ${order.limit_price:.2f}"
            else:
                return False, f"Current price ${current_price:.2f} is below limit ${order.limit_price:.2f}"


def get_order_status_message(order: PendingOrder, current_price: float) -> str:
    """
    Get a status message for a pending order.

    Args:
        order: The pending order
        current_price: Current market price

    Returns:
        Status message string
    """
    if order.limit_price is None:
        return "PENDING: Market order - will execute on next check"

    if order.order_type == 'BUY':
        if current_price <= order.limit_price:
            return f"READY: Ready to execute! Price ${current_price:.2f} ≤ Limit ${order.limit_price:.2f}"
        else:
            diff = current_price - order.limit_price
            return f"PENDING: Waiting... Price ${current_price:.2f} is ${diff:.2f} above limit"

    elif order.order_type == 'SELL':
        if current_price >= order.limit_price:
            return f"READY: Ready to execute! Price ${current_price:.2f} ≥ Limit ${order.limit_price:.2f}"
        else:
            diff = order.limit_price - current_price
            return f"PENDING: Waiting... Price ${current_price:.2f} is ${diff:.2f} below limit"

    return "Unknown status"
