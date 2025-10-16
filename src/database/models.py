"""
Database models for portfolio management.
Handles stocks, transactions, and portfolio positions.
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Stock:
    """Stock entity"""
    ticker: str
    company_name: str
    sector: Optional[str] = None
    id: Optional[int] = None


@dataclass
class Transaction:
    """Transaction entity"""
    ticker: str
    transaction_type: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    transaction_date: str
    notes: Optional[str] = None
    id: Optional[int] = None


@dataclass
class Position:
    """Current position entity"""
    ticker: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class PendingOrder:
    """Pending order entity"""
    ticker: str
    order_type: str  # 'BUY' or 'SELL'
    quantity: float
    limit_price: Optional[float]
    order_status: str  # 'PENDING', 'FILLED', 'CANCELLED'
    created_at: str
    id: Optional[int] = None


class PortfolioDatabase:
    """Database manager for portfolio operations"""

    def __init__(self, db_path: str = "data/portfolio.db"):
        """Initialize database connection"""
        self.db_path = db_path
        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        """Create database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Stocks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT UNIQUE NOT NULL,
                company_name TEXT NOT NULL,
                sector TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                transaction_type TEXT NOT NULL CHECK(transaction_type IN ('BUY', 'SELL')),
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                transaction_date TEXT NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker) REFERENCES stocks(ticker)
            )
        """)

        # Pending orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                order_type TEXT NOT NULL CHECK(order_type IN ('BUY', 'SELL')),
                quantity REAL NOT NULL,
                limit_price REAL,
                order_status TEXT NOT NULL CHECK(order_status IN ('PENDING', 'FILLED', 'CANCELLED')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker) REFERENCES stocks(ticker)
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_ticker
            ON transactions(ticker)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_date
            ON transactions(transaction_date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_orders_ticker
            ON pending_orders(ticker)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_orders_status
            ON pending_orders(order_status)
        """)

        conn.commit()
        conn.close()

    # ============= STOCK CRUD OPERATIONS =============

    def add_stock(self, ticker: str, company_name: str, sector: Optional[str] = None) -> bool:
        """Add a new stock"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO stocks (ticker, company_name, sector) VALUES (?, ?, ?)",
                (ticker.upper(), company_name, sector)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_stock(self, ticker: str) -> Optional[Stock]:
        """Get stock by ticker"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM stocks WHERE ticker = ?", (ticker.upper(),))
        row = cursor.fetchone()
        conn.close()

        if row:
            return Stock(
                id=row['id'],
                ticker=row['ticker'],
                company_name=row['company_name'],
                sector=row['sector']
            )
        return None

    def get_all_stocks(self) -> List[Stock]:
        """Get all stocks"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM stocks ORDER BY ticker")
        rows = cursor.fetchall()
        conn.close()

        return [Stock(
            id=row['id'],
            ticker=row['ticker'],
            company_name=row['company_name'],
            sector=row['sector']
        ) for row in rows]

    def update_stock(self, ticker: str, company_name: Optional[str] = None,
                     sector: Optional[str] = None) -> bool:
        """Update stock information"""
        conn = self._get_connection()
        cursor = conn.cursor()

        updates = []
        params = []

        if company_name:
            updates.append("company_name = ?")
            params.append(company_name)
        if sector:
            updates.append("sector = ?")
            params.append(sector)

        if not updates:
            return False

        params.append(ticker.upper())
        query = f"UPDATE stocks SET {', '.join(updates)} WHERE ticker = ?"

        cursor.execute(query, params)
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()

        return rows_affected > 0

    def delete_stock(self, ticker: str) -> bool:
        """Delete a stock (and all its transactions)"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Delete transactions first (foreign key constraint)
        cursor.execute("DELETE FROM transactions WHERE ticker = ?", (ticker.upper(),))
        cursor.execute("DELETE FROM stocks WHERE ticker = ?", (ticker.upper(),))

        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()

        return rows_affected > 0

    # ============= TRANSACTION CRUD OPERATIONS =============

    def add_transaction(self, ticker: str, transaction_type: str, quantity: float,
                       price: float, transaction_date: Optional[str] = None,
                       notes: Optional[str] = None) -> Optional[int]:
        """Add a new transaction"""
        if transaction_date is None:
            transaction_date = datetime.now().strftime("%Y-%m-%d")

        # Ensure stock exists
        stock = self.get_stock(ticker)
        if not stock:
            return None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO transactions
                   (ticker, transaction_type, quantity, price, transaction_date, notes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticker.upper(), transaction_type.upper(), quantity, price,
                 transaction_date, notes)
            )
            transaction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return transaction_id
        except Exception as e:
            print(f"Error adding transaction: {e}")
            return None

    def get_transaction(self, transaction_id: int) -> Optional[Transaction]:
        """Get transaction by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transactions WHERE id = ?", (transaction_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return Transaction(
                id=row['id'],
                ticker=row['ticker'],
                transaction_type=row['transaction_type'],
                quantity=row['quantity'],
                price=row['price'],
                transaction_date=row['transaction_date'],
                notes=row['notes']
            )
        return None

    def get_transactions_by_ticker(self, ticker: str) -> List[Transaction]:
        """Get all transactions for a ticker"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM transactions WHERE ticker = ? ORDER BY transaction_date DESC",
            (ticker.upper(),)
        )
        rows = cursor.fetchall()
        conn.close()

        return [Transaction(
            id=row['id'],
            ticker=row['ticker'],
            transaction_type=row['transaction_type'],
            quantity=row['quantity'],
            price=row['price'],
            transaction_date=row['transaction_date'],
            notes=row['notes']
        ) for row in rows]

    def get_all_transactions(self, limit: Optional[int] = None) -> List[Transaction]:
        """Get all transactions"""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM transactions ORDER BY transaction_date DESC"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        return [Transaction(
            id=row['id'],
            ticker=row['ticker'],
            transaction_type=row['transaction_type'],
            quantity=row['quantity'],
            price=row['price'],
            transaction_date=row['transaction_date'],
            notes=row['notes']
        ) for row in rows]

    def update_transaction(self, transaction_id: int, quantity: Optional[float] = None,
                          price: Optional[float] = None, notes: Optional[str] = None) -> bool:
        """Update transaction"""
        conn = self._get_connection()
        cursor = conn.cursor()

        updates = []
        params = []

        if quantity is not None:
            updates.append("quantity = ?")
            params.append(quantity)
        if price is not None:
            updates.append("price = ?")
            params.append(price)
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)

        if not updates:
            return False

        params.append(transaction_id)
        query = f"UPDATE transactions SET {', '.join(updates)} WHERE id = ?"

        cursor.execute(query, params)
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()

        return rows_affected > 0

    def delete_transaction(self, transaction_id: int) -> bool:
        """Delete a transaction"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        return rows_affected > 0

    # ============= POSITION CALCULATIONS =============

    def get_current_positions(self, current_prices: Dict[str, float]) -> List[Position]:
        """Calculate current positions from transactions"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get net position for each ticker
        cursor.execute("""
            SELECT
                ticker,
                SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE -quantity END) as net_quantity,
                SUM(CASE WHEN transaction_type = 'BUY' THEN quantity * price ELSE 0 END) as total_cost,
                SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE 0 END) as total_bought
            FROM transactions
            GROUP BY ticker
            HAVING net_quantity > 0
        """)

        rows = cursor.fetchall()
        conn.close()

        positions = []
        for row in rows:
            ticker = row['ticker']
            net_quantity = row['net_quantity']
            total_cost = row['total_cost']
            total_bought = row['total_bought']

            # Calculate average cost
            avg_cost = total_cost / total_bought if total_bought > 0 else 0

            # Get current price
            current_price = current_prices.get(ticker, 0)

            # Calculate values
            market_value = net_quantity * current_price
            cost_basis = net_quantity * avg_cost
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

            positions.append(Position(
                ticker=ticker,
                quantity=net_quantity,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct
            ))

        return positions

    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get portfolio summary statistics"""
        positions = self.get_current_positions(current_prices)

        total_market_value = sum(p.market_value for p in positions)
        total_cost_basis = sum(p.quantity * p.avg_cost for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_unrealized_pnl_pct = (total_unrealized_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0

        return {
            'total_market_value': total_market_value,
            'total_cost_basis': total_cost_basis,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_unrealized_pnl_pct': total_unrealized_pnl_pct,
            'num_positions': len(positions),
            'positions': positions
        }

    # ============= PENDING ORDERS OPERATIONS =============

    def add_pending_order(self, ticker: str, order_type: str, quantity: float,
                         limit_price: Optional[float] = None) -> Optional[int]:
        """Add a pending order"""
        stock = self.get_stock(ticker)
        if not stock:
            return None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO pending_orders
                   (ticker, order_type, quantity, limit_price, order_status)
                   VALUES (?, ?, ?, ?, 'PENDING')""",
                (ticker.upper(), order_type.upper(), quantity, limit_price)
            )
            order_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return order_id
        except Exception as e:
            print(f"Error adding order: {e}")
            return None

    def get_pending_orders_by_ticker(self, ticker: str) -> List[PendingOrder]:
        """Get all pending orders for a ticker"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM pending_orders
               WHERE ticker = ? AND order_status = 'PENDING'
               ORDER BY created_at DESC""",
            (ticker.upper(),)
        )
        rows = cursor.fetchall()
        conn.close()

        return [PendingOrder(
            id=row['id'],
            ticker=row['ticker'],
            order_type=row['order_type'],
            quantity=row['quantity'],
            limit_price=row['limit_price'],
            order_status=row['order_status'],
            created_at=row['created_at']
        ) for row in rows]

    def get_all_pending_orders(self) -> List[PendingOrder]:
        """Get all pending orders"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM pending_orders
               WHERE order_status = 'PENDING'
               ORDER BY created_at DESC"""
        )
        rows = cursor.fetchall()
        conn.close()

        return [PendingOrder(
            id=row['id'],
            ticker=row['ticker'],
            order_type=row['order_type'],
            quantity=row['quantity'],
            limit_price=row['limit_price'],
            order_status=row['order_status'],
            created_at=row['created_at']
        ) for row in rows]

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending order"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE pending_orders SET order_status = 'CANCELLED' WHERE id = ?",
            (order_id,)
        )
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        return rows_affected > 0

    def delete_order(self, order_id: int) -> bool:
        """Delete an order"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM pending_orders WHERE id = ?", (order_id,))
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        return rows_affected > 0
