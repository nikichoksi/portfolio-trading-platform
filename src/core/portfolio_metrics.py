"""
Portfolio metrics calculation module.
Calculates key risk-return metrics for investment portfolios.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PortfolioMetrics:
    """Data class to hold portfolio metrics"""
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    var_95: float
    cvar_95: float
    holdings: Dict[str, float]
    total_value: float
    period_days: int


class PortfolioAnalyzer:
    """Analyzes portfolio risk and return characteristics"""

    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize portfolio analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation (default 4%)
        """
        self.risk_free_rate = risk_free_rate
        self.market_ticker = "^GSPC"  # S&P 500 as market proxy

    def parse_portfolio(self, portfolio_str: str) -> Dict[str, float]:
        """
        Parse portfolio from natural language string.

        Args:
            portfolio_str: String like "40% AAPL, 30% MSFT, 30% GOOGL"

        Returns:
            Dictionary of {ticker: weight}
        """
        holdings = {}
        # Simple parsing - can be enhanced with NLP
        parts = portfolio_str.replace("%", "").split(",")

        for part in parts:
            part = part.strip()
            tokens = part.split()

            if len(tokens) >= 2:
                try:
                    weight = float(tokens[0]) / 100.0
                    ticker = tokens[1].upper()
                    holdings[ticker] = weight
                except ValueError:
                    # Try reverse order: ticker then percentage
                    try:
                        ticker = tokens[0].upper()
                        weight = float(tokens[1]) / 100.0
                        holdings[ticker] = weight
                    except ValueError:
                        continue

        # Normalize weights to sum to 1
        total = sum(holdings.values())
        if total > 0:
            holdings = {k: v/total for k, v in holdings.items()}

        return holdings

    def fetch_price_data(
        self,
        tickers: List[str],
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Fetch historical price data for given tickers.

        Args:
            tickers: List of ticker symbols
            period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y)

        Returns:
            DataFrame with adjusted close prices
        """
        data = yf.download(tickers, period=period, progress=False)

        if len(tickers) == 1:
            prices = data['Adj Close'].to_frame()
            prices.columns = tickers
        else:
            prices = data['Adj Close']

        return prices.dropna()

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from prices"""
        return prices.pct_change().dropna()

    def calculate_portfolio_returns(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.Series:
        """
        Calculate portfolio returns given asset returns and weights.

        Args:
            returns: DataFrame of asset returns
            weights: Dictionary of asset weights

        Returns:
            Series of portfolio returns
        """
        # Align weights with returns columns
        weight_vector = np.array([weights.get(col, 0) for col in returns.columns])
        portfolio_returns = returns.dot(weight_vector)
        return portfolio_returns

    def calculate_annual_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        cumulative_return = (1 + returns).prod() - 1
        n_days = len(returns)
        annual_return = (1 + cumulative_return) ** (252 / n_days) - 1
        return annual_return

    def calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(252)

    def calculate_sharpe_ratio(
        self,
        annual_return: float,
        volatility: float
    ) -> float:
        """Calculate Sharpe ratio"""
        if volatility == 0:
            return 0.0
        return (annual_return - self.risk_free_rate) / volatility

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def calculate_beta(
        self,
        portfolio_returns: pd.Series,
        period: str = "1y"
    ) -> float:
        """
        Calculate portfolio beta relative to market.

        Args:
            portfolio_returns: Series of portfolio returns
            period: Period for market data

        Returns:
            Beta coefficient
        """
        try:
            market_data = yf.download(
                self.market_ticker,
                period=period,
                progress=False
            )
            market_returns = market_data['Adj Close'].pct_change().dropna()

            # Align dates
            common_dates = portfolio_returns.index.intersection(market_returns.index)
            if len(common_dates) < 20:  # Need minimum data points
                return 1.0

            port_aligned = portfolio_returns.loc[common_dates]
            market_aligned = market_returns.loc[common_dates]

            covariance = np.cov(port_aligned, market_aligned)[0, 1]
            market_variance = np.var(market_aligned)

            if market_variance == 0:
                return 1.0

            beta = covariance / market_variance
            return beta

        except Exception as e:
            print(f"Error calculating beta: {e}")
            return 1.0

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Series of returns
            confidence: Confidence level (default 95%)

        Returns:
            VaR value (positive number representing potential loss)
        """
        return -np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            returns: Series of returns
            confidence: Confidence level (default 95%)

        Returns:
            CVaR value
        """
        var = -np.percentile(returns, (1 - confidence) * 100)
        cvar = -returns[returns <= -var].mean()
        return cvar

    def analyze_portfolio(
        self,
        holdings: Dict[str, float],
        period: str = "1y"
    ) -> PortfolioMetrics:
        """
        Perform comprehensive portfolio analysis.

        Args:
            holdings: Dictionary of {ticker: weight}
            period: Analysis period

        Returns:
            PortfolioMetrics object with all calculated metrics
        """
        tickers = list(holdings.keys())

        # Fetch data
        prices = self.fetch_price_data(tickers, period)
        returns = self.calculate_returns(prices)

        # Calculate portfolio returns
        portfolio_returns = self.calculate_portfolio_returns(returns, holdings)

        # Calculate metrics
        annual_return = self.calculate_annual_return(portfolio_returns)
        volatility = self.calculate_volatility(portfolio_returns)
        sharpe = self.calculate_sharpe_ratio(annual_return, volatility)
        max_dd = self.calculate_max_drawdown(portfolio_returns)
        beta = self.calculate_beta(portfolio_returns, period)
        var_95 = self.calculate_var(portfolio_returns)
        cvar_95 = self.calculate_cvar(portfolio_returns)

        return PortfolioMetrics(
            annual_return=annual_return,
            annual_volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            beta=beta,
            var_95=var_95,
            cvar_95=cvar_95,
            holdings=holdings,
            total_value=1.0,  # Normalized to 1
            period_days=len(portfolio_returns)
        )

    def get_asset_correlations(
        self,
        tickers: List[str],
        period: str = "1y"
    ) -> pd.DataFrame:
        """Get correlation matrix of assets"""
        prices = self.fetch_price_data(tickers, period)
        returns = self.calculate_returns(prices)
        return returns.corr()
