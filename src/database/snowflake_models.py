# src/database/snowflake_models.py
"""
Snowflake-based database models for portfolio management.
Drop-in replacement for SQLite models with cloud data warehouse capabilities.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json
import logging

from .snowflake_connector import get_snowflake_connector, SnowflakeConnector
from .models import Stock, Transaction, Position, PendingOrder

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Market data entity"""
    ticker: str
    date: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    id: Optional[int] = None


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot entity"""
    snapshot_date: str
    total_value: float
    total_cost_basis: float
    total_pnl: float
    total_pnl_pct: float
    num_positions: int
    id: Optional[int] = None


class SnowflakePortfolioDatabase:
    """Snowflake-based portfolio database operations"""
    
    def __init__(self, connector: Optional[SnowflakeConnector] = None):
        """
        Initialize Snowflake portfolio database.
        
        Args:
            connector: SnowflakeConnector instance. If None, uses singleton.
        """
        self.connector = connector or get_snowflake_connector()
        self._ensure_schema_exists()
    
    def _ensure_schema_exists(self):
        """Ensure database schema exists"""
        try:
            # Check if stocks table exists
            result = self.connector.execute_query(
                "SHOW TABLES LIKE 'stocks'"
            )
            if not result:
                logger.info("Creating Snowflake schema...")
                self.connector.create_database_schema()
        except Exception as e:
            logger.error(f"Schema check error: {str(e)}")
    
    # ============= STOCK CRUD OPERATIONS =============
    
    def add_stock(self, ticker: str, company_name: str, 
                  sector: Optional[str] = None,
                  industry: Optional[str] = None,
                  market_cap: Optional[float] = None) -> bool:
        """Add a new stock"""
        try:
            query = """
                INSERT INTO stocks (ticker, company_name, sector, industry, market_cap)
                VALUES (%s, %s, %s, %s, %s)
            """
            self.connector.execute_query(
                query,
                (ticker.upper(), company_name, sector, industry, market_cap),
                fetch=False
            )
            return True
        except Exception as e:
            logger.error(f"Error adding stock {ticker}: {str(e)}")
            return False
    
    def get_stock(self, ticker: str) -> Optional[Stock]:
        """Get stock by ticker"""
        query = "SELECT * FROM stocks WHERE ticker = %s"
        results = self.connector.execute_query(query, (ticker.upper(),))
        
        if results and len(results) > 0:
            row = results[0]
            return Stock(
                id=row['ID'],
                ticker=row['TICKER'],
                company_name=row['COMPANY_NAME'],
                sector=row.get('SECTOR')
            )
        return None
    
    def get_all_stocks(self) -> List[Stock]:
        """Get all stocks"""
        query = "SELECT * FROM stocks ORDER BY ticker"
        results = self.connector.execute_query(query)
        
        return [Stock(
            id=row['ID'],
            ticker=row['TICKER'],
            company_name=row['COMPANY_NAME'],
            sector=row.get('SECTOR')
        ) for row in results]
    
    def update_stock(self, ticker: str, company_name: Optional[str] = None,
                     sector: Optional[str] = None) -> bool:
        """Update stock information"""
        updates = []
        params = []
        
        if company_name:
            updates.append("company_name = %s")
            params.append(company_name)
        if sector:
            updates.append("sector = %s")
            params.append(sector)
        
        if not updates:
            return False
        
        updates.append("updated_at = CURRENT_TIMESTAMP()")
        params.append(ticker.upper())
        
        query = f"UPDATE stocks SET {', '.join(updates)} WHERE ticker = %s"
        
        try:
            self.connector.execute_query(query, tuple(params), fetch=False)
            return True
        except Exception as e:
            logger.error(f"Error updating stock {ticker}: {str(e)}")
            return False
    
    def delete_stock(self, ticker: str) -> bool:
        """Delete a stock and all related data"""
        try:
            # Delete in order due to foreign key constraints
            self.connector.execute_query(
                "DELETE FROM transactions WHERE ticker = %s",
                (ticker.upper(),),
                fetch=False
            )
            self.connector.execute_query(
                "DELETE FROM pending_orders WHERE ticker = %s",
                (ticker.upper(),),
                fetch=False
            )
            self.connector.execute_query(
                "DELETE FROM market_data WHERE ticker = %s",
                (ticker.upper(),),
                fetch=False
            )
            self.connector.execute_query(
                "DELETE FROM stocks WHERE ticker = %s",
                (ticker.upper(),),
                fetch=False
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting stock {ticker}: {str(e)}")
            return False
    
    # ============= TRANSACTION CRUD OPERATIONS =============
    
    def add_transaction(self, ticker: str, transaction_type: str, quantity: float,
                       price: float, transaction_date: Optional[str] = None,
                       notes: Optional[str] = None) -> Optional[int]:
        """Add a new transaction"""
        if transaction_date is None:
            transaction_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            query = """
                INSERT INTO transactions
                (ticker, transaction_type, quantity, price, transaction_date, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.connector.execute_query(
                query,
                (ticker.upper(), transaction_type.upper(), quantity, price,
                 transaction_date, notes),
                fetch=False
            )
            
            # Get the inserted ID
            result = self.connector.execute_query(
                "SELECT MAX(id) as last_id FROM transactions WHERE ticker = %s",
                (ticker.upper(),)
            )
            
            if result:
                return result[0]['LAST_ID']
            return None
            
        except Exception as e:
            logger.error(f"Error adding transaction: {str(e)}")
            return None
    
    def get_transactions_by_ticker(self, ticker: str) -> List[Transaction]:
        """Get all transactions for a ticker"""
        query = """
            SELECT * FROM transactions 
            WHERE ticker = %s 
            ORDER BY transaction_date DESC
        """
        results = self.connector.execute_query(query, (ticker.upper(),))
        
        return [Transaction(
            id=row['ID'],
            ticker=row['TICKER'],
            transaction_type=row['TRANSACTION_TYPE'],
            quantity=float(row['QUANTITY']),
            price=float(row['PRICE']),
            transaction_date=str(row['TRANSACTION_DATE']),
            notes=row.get('NOTES')
        ) for row in results]
    
    def get_all_transactions(self, limit: Optional[int] = None) -> List[Transaction]:
        """Get all transactions"""
        query = "SELECT * FROM transactions ORDER BY transaction_date DESC"
        if limit:
            query += f" LIMIT {limit}"
        
        results = self.connector.execute_query(query)
        
        return [Transaction(
            id=row['ID'],
            ticker=row['TICKER'],
            transaction_type=row['TRANSACTION_TYPE'],
            quantity=float(row['QUANTITY']),
            price=float(row['PRICE']),
            transaction_date=str(row['TRANSACTION_DATE']),
            notes=row.get('NOTES')
        ) for row in results]
    
    def delete_transaction(self, transaction_id: int) -> bool:
        """Delete a transaction"""
        try:
            self.connector.execute_query(
                "DELETE FROM transactions WHERE id = %s",
                (transaction_id,),
                fetch=False
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting transaction: {str(e)}")
            return False
    
    # ============= POSITION CALCULATIONS =============
    
    def get_current_positions(self, current_prices: Dict[str, float]) -> List[Position]:
        """Calculate current positions from transactions"""
        query = """
            SELECT
                ticker,
                SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE -quantity END) as net_quantity,
                SUM(CASE WHEN transaction_type = 'BUY' THEN quantity * price ELSE 0 END) as total_cost,
                SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE 0 END) as total_bought
            FROM transactions
            GROUP BY ticker
            HAVING net_quantity > 0
        """
        
        results = self.connector.execute_query(query)
        
        positions = []
        for row in results:
            ticker = row['TICKER']
            net_quantity = float(row['NET_QUANTITY'])
            total_cost = float(row['TOTAL_COST'])
            total_bought = float(row['TOTAL_BOUGHT'])
            
            avg_cost = total_cost / total_bought if total_bought > 0 else 0
            current_price = current_prices.get(ticker, 0)
            
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
        try:
            query = """
                INSERT INTO pending_orders
                (ticker, order_type, quantity, limit_price, order_status)
                VALUES (%s, %s, %s, %s, 'PENDING')
            """
            self.connector.execute_query(
                query,
                (ticker.upper(), order_type.upper(), quantity, limit_price),
                fetch=False
            )
            
            # Get the inserted ID
            result = self.connector.execute_query(
                "SELECT MAX(id) as last_id FROM pending_orders WHERE ticker = %s",
                (ticker.upper(),)
            )
            
            if result:
                return result[0]['LAST_ID']
            return None
            
        except Exception as e:
            logger.error(f"Error adding order: {str(e)}")
            return None
    
    def get_pending_orders_by_ticker(self, ticker: str) -> List[PendingOrder]:
        """Get all pending orders for a ticker"""
        query = """
            SELECT * FROM pending_orders
            WHERE ticker = %s AND order_status = 'PENDING'
            ORDER BY created_at DESC
        """
        results = self.connector.execute_query(query, (ticker.upper(),))
        
        return [PendingOrder(
            id=row['ID'],
            ticker=row['TICKER'],
            order_type=row['ORDER_TYPE'],
            quantity=float(row['QUANTITY']),
            limit_price=float(row['LIMIT_PRICE']) if row['LIMIT_PRICE'] else None,
            order_status=row['ORDER_STATUS'],
            created_at=str(row['CREATED_AT'])
        ) for row in results]
    
    def get_all_pending_orders(self) -> List[PendingOrder]:
        """Get all pending orders"""
        query = """
            SELECT * FROM pending_orders
            WHERE order_status = 'PENDING'
            ORDER BY created_at DESC
        """
        results = self.connector.execute_query(query)
        
        return [PendingOrder(
            id=row['ID'],
            ticker=row['TICKER'],
            order_type=row['ORDER_TYPE'],
            quantity=float(row['QUANTITY']),
            limit_price=float(row['LIMIT_PRICE']) if row['LIMIT_PRICE'] else None,
            order_status=row['ORDER_STATUS'],
            created_at=str(row['CREATED_AT'])
        ) for row in results]
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending order"""
        try:
            self.connector.execute_query(
                """UPDATE pending_orders 
                   SET order_status = 'CANCELLED', updated_at = CURRENT_TIMESTAMP()
                   WHERE id = %s""",
                (order_id,),
                fetch=False
            )
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    def delete_order(self, order_id: int) -> bool:
        """Delete an order"""
        try:
            self.connector.execute_query(
                "DELETE FROM pending_orders WHERE id = %s",
                (order_id,),
                fetch=False
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting order: {str(e)}")
            return False
    
    # ============= MARKET DATA OPERATIONS =============
    
    def add_market_data_batch(self, market_data: List[MarketData]) -> int:
        """Add batch market data"""
        query = """
            INSERT INTO market_data 
            (ticker, date, open_price, high_price, low_price, close_price, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        data = [
            (md.ticker, md.date, md.open_price, md.high_price,
             md.low_price, md.close_price, md.volume)
            for md in market_data
        ]
        
        return self.connector.execute_many(query, data)
    
    def get_market_data(self, ticker: str, start_date: str, 
                       end_date: str) -> List[MarketData]:
        """Get market data for date range"""
        query = """
            SELECT * FROM market_data
            WHERE ticker = %s AND date BETWEEN %s AND %s
            ORDER BY date
        """
        results = self.connector.execute_query(
            query,
            (ticker.upper(), start_date, end_date)
        )
        
        return [MarketData(
            id=row['ID'],
            ticker=row['TICKER'],
            date=str(row['DATE']),
            open_price=float(row['OPEN_PRICE']),
            high_price=float(row['HIGH_PRICE']),
            low_price=float(row['LOW_PRICE']),
            close_price=float(row['CLOSE_PRICE']),
            volume=int(row['VOLUME'])
        ) for row in results]
    
    # ============= PORTFOLIO SNAPSHOT OPERATIONS =============
    
    def save_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> bool:
        """Save a portfolio snapshot"""
        try:
            query = """
                INSERT INTO portfolio_snapshots
                (snapshot_date, total_value, total_cost_basis, total_pnl, 
                 total_pnl_pct, num_positions)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.connector.execute_query(
                query,
                (snapshot.snapshot_date, snapshot.total_value, 
                 snapshot.total_cost_basis, snapshot.total_pnl,
                 snapshot.total_pnl_pct, snapshot.num_positions),
                fetch=False
            )
            return True
        except Exception as e:
            logger.error(f"Error saving snapshot: {str(e)}")
            return False
    
    def get_portfolio_snapshots(self, days: int = 30) -> List[PortfolioSnapshot]:
        """Get portfolio snapshots for last N days"""
        query = """
            SELECT * FROM portfolio_snapshots
            WHERE snapshot_date >= DATEADD(day, -%s, CURRENT_DATE())
            ORDER BY snapshot_date DESC
        """
        results = self.connector.execute_query(query, (days,))
        
        return [PortfolioSnapshot(
            id=row['ID'],
            snapshot_date=str(row['SNAPSHOT_DATE']),
            total_value=float(row['TOTAL_VALUE']),
            total_cost_basis=float(row['TOTAL_COST_BASIS']),
            total_pnl=float(row['TOTAL_PNL']),
            total_pnl_pct=float(row['TOTAL_PNL_PCT']),
            num_positions=int(row['NUM_POSITIONS'])
        ) for row in results]
    
    # ============= AI INSIGHTS OPERATIONS =============
    
    def save_ai_insight(self, insight_type: str, portfolio_snapshot: Dict,
                       analysis_result: Dict, agent_name: str) -> bool:
        """Save AI agent analysis result"""
        try:
            query = """
                INSERT INTO ai_insights
                (insight_type, portfolio_snapshot, analysis_result, agent_name)
                VALUES (%s, PARSE_JSON(%s), PARSE_JSON(%s), %s)
            """
            self.connector.execute_query(
                query,
                (insight_type, json.dumps(portfolio_snapshot),
                 json.dumps(analysis_result), agent_name),
                fetch=False
            )
            return True
        except Exception as e:
            logger.error(f"Error saving AI insight: {str(e)}")
            return False
    
    def get_ai_insights(self, insight_type: Optional[str] = None,
                       limit: int = 10) -> List[Dict]:
        """Get AI insights"""
        if insight_type:
            query = """
                SELECT * FROM ai_insights
                WHERE insight_type = %s
                ORDER BY created_at DESC
                LIMIT %s
            """
            results = self.connector.execute_query(query, (insight_type, limit))
        else:
            query = """
                SELECT * FROM ai_insights
                ORDER BY created_at DESC
                LIMIT %s
            """
            results = self.connector.execute_query(query, (limit,))
        
        return results