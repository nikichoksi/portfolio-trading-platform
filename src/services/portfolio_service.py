"""
Portfolio management service.
Provides business logic for portfolio operations with live price updates.
"""

import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import PortfolioDatabase, Stock, Transaction, Position


class PortfolioService:
    """Service layer for portfolio management"""

    def __init__(self, db_path: str = "data/portfolio.db"):
        """Initialize portfolio service"""
        self.db = PortfolioDatabase(db_path)

    # ============= STOCK OPERATIONS =============

    def add_stock(self, ticker: str, company_name: Optional[str] = None,
                  sector: Optional[str] = None) -> Tuple[bool, str]:
        """
        Add a new stock to the portfolio.
        Auto-fetches company info if not provided.
        """
        ticker = ticker.upper()

        # Fetch stock info from yfinance if company_name not provided
        if not company_name:
            try:
                stock_info = yf.Ticker(ticker)
                info = stock_info.info
                company_name = info.get('longName', ticker)
                if not sector:
                    sector = info.get('sector', 'Unknown')
            except Exception as e:
                return False, f"Unable to fetch stock info: {str(e)}"

        success = self.db.add_stock(ticker, company_name, sector)
        if success:
            return True, f"Added {ticker} - {company_name}"
        else:
            return False, f"Stock {ticker} already exists"

    def get_stock(self, ticker: str) -> Optional[Stock]:
        """Get stock details"""
        return self.db.get_stock(ticker)

    def list_stocks(self) -> List[Stock]:
        """List all stocks"""
        return self.db.get_all_stocks()

    def update_stock(self, ticker: str, company_name: Optional[str] = None,
                    sector: Optional[str] = None) -> Tuple[bool, str]:
        """Update stock information"""
        success = self.db.update_stock(ticker, company_name, sector)
        if success:
            return True, f"Updated {ticker}"
        else:
            return False, f"Stock {ticker} not found"

    def delete_stock(self, ticker: str) -> Tuple[bool, str]:
        """Delete a stock"""
        success = self.db.delete_stock(ticker)
        if success:
            return True, f"Deleted {ticker} and all its transactions"
        else:
            return False, f"Stock {ticker} not found"

    # ============= TRANSACTION OPERATIONS =============

    def buy_stock(self, ticker: str, quantity: float, price: float,
                  date: Optional[str] = None, notes: Optional[str] = None) -> Tuple[bool, str]:
        """Execute a buy transaction"""
        # Ensure stock exists
        stock = self.db.get_stock(ticker)
        if not stock:
            # Try to add the stock automatically
            success, msg = self.add_stock(ticker)
            if not success:
                return False, msg

        transaction_id = self.db.add_transaction(
            ticker=ticker,
            transaction_type='BUY',
            quantity=quantity,
            price=price,
            transaction_date=date,
            notes=notes
        )

        if transaction_id:
            return True, f"Bought {quantity} shares of {ticker} at ${price:.2f}"
        else:
            return False, "Failed to record transaction"

    def sell_stock(self, ticker: str, quantity: float, price: float,
                   date: Optional[str] = None, notes: Optional[str] = None) -> Tuple[bool, str]:
        """Execute a sell transaction"""
        # Check if user has enough shares
        current_prices = self.get_live_prices([ticker])
        positions = self.db.get_current_positions(current_prices)

        position = next((p for p in positions if p.ticker == ticker), None)
        if not position or position.quantity < quantity:
            available = position.quantity if position else 0
            return False, f"Insufficient shares. You have {available} shares of {ticker}"

        transaction_id = self.db.add_transaction(
            ticker=ticker,
            transaction_type='SELL',
            quantity=quantity,
            price=price,
            transaction_date=date,
            notes=notes
        )

        if transaction_id:
            return True, f"Sold {quantity} shares of {ticker} at ${price:.2f}"
        else:
            return False, "Failed to record transaction"

    def get_transactions(self, ticker: Optional[str] = None,
                        limit: Optional[int] = None) -> List[Transaction]:
        """Get transaction history"""
        if ticker:
            return self.db.get_transactions_by_ticker(ticker)
        else:
            return self.db.get_all_transactions(limit)

    def delete_transaction(self, transaction_id: int) -> Tuple[bool, str]:
        """Delete a transaction"""
        success = self.db.delete_transaction(transaction_id)
        if success:
            return True, "Transaction deleted"
        else:
            return False, "Transaction not found"

    # ============= PRICE OPERATIONS =============

    def get_live_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch live prices for tickers"""
        prices = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                # Try to get current price from various sources
                info = stock.info
                price = info.get('currentPrice') or info.get('regularMarketPrice') or 0

                # Fallback to recent close
                if price == 0:
                    hist = stock.history(period='1d')
                    if not hist.empty:
                        price = hist['Close'].iloc[-1]

                prices[ticker] = float(price)
            except Exception as e:
                print(f"Error fetching price for {ticker}: {e}")
                prices[ticker] = 0

        return prices

    def get_price_history(self, ticker: str, period: str = "1mo") -> pd.DataFrame:
        """Get price history for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            print(f"Error fetching history for {ticker}: {e}")
            return pd.DataFrame()

    def get_stock_info(self, ticker: str) -> Dict:
        """Get detailed stock information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
            }
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            return {}

    # ============= PORTFOLIO OPERATIONS =============

    def get_positions(self) -> List[Position]:
        """Get current portfolio positions with live prices"""
        # Get all tickers with positions
        transactions = self.db.get_all_transactions()
        tickers = list(set(t.ticker for t in transactions))

        if not tickers:
            return []

        # Fetch live prices
        live_prices = self.get_live_prices(tickers)

        # Calculate positions
        positions = self.db.get_current_positions(live_prices)
        return positions

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary with live prices"""
        transactions = self.db.get_all_transactions()
        tickers = list(set(t.ticker for t in transactions))

        if not tickers:
            return {
                'total_market_value': 0,
                'total_cost_basis': 0,
                'total_unrealized_pnl': 0,
                'total_unrealized_pnl_pct': 0,
                'num_positions': 0,
                'positions': []
            }

        live_prices = self.get_live_prices(tickers)
        return self.db.get_portfolio_summary(live_prices)

    def get_portfolio_allocation(self) -> Dict[str, float]:
        """Get portfolio allocation as percentages"""
        positions = self.get_positions()
        total_value = sum(p.market_value for p in positions)

        if total_value == 0:
            return {}

        allocation = {}
        for position in positions:
            allocation[position.ticker] = (position.market_value / total_value) * 100

        return allocation

    def get_sector_allocation(self) -> Dict[str, float]:
        """Get sector allocation"""
        positions = self.get_positions()
        total_value = sum(p.market_value for p in positions)

        if total_value == 0:
            return {}

        # Get sector info for each ticker
        sector_values = {}
        for position in positions:
            stock = self.db.get_stock(position.ticker)
            sector = stock.sector if stock and stock.sector else 'Unknown'

            if sector not in sector_values:
                sector_values[sector] = 0
            sector_values[sector] += position.market_value

        # Convert to percentages
        sector_allocation = {}
        for sector, value in sector_values.items():
            sector_allocation[sector] = (value / total_value) * 100

        return sector_allocation

    def get_performance_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        summary = self.get_portfolio_summary()

        total_return = summary['total_unrealized_pnl']
        total_return_pct = summary['total_unrealized_pnl_pct']
        total_invested = summary['total_cost_basis']
        current_value = summary['total_market_value']

        # Calculate top gainers and losers
        positions = summary['positions']
        if positions:
            top_gainer = max(positions, key=lambda p: p.unrealized_pnl_pct)
            top_loser = min(positions, key=lambda p: p.unrealized_pnl_pct)
        else:
            top_gainer = None
            top_loser = None

        return {
            'total_invested': total_invested,
            'current_value': current_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_positions': summary['num_positions'],
            'top_gainer': top_gainer,
            'top_loser': top_loser
        }
