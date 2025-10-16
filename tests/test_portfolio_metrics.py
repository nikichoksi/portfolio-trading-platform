"""
Unit tests for portfolio metrics calculations.
"""

import pytest
import numpy as np
import pandas as pd
from src.core.portfolio_metrics import PortfolioAnalyzer, PortfolioMetrics


class TestPortfolioAnalyzer:
    """Test suite for PortfolioAnalyzer class"""

    def test_initialization(self):
        """Test analyzer initialization"""
        analyzer = PortfolioAnalyzer(risk_free_rate=0.03)
        assert analyzer.risk_free_rate == 0.03
        assert analyzer.market_ticker == "^GSPC"

    def test_parse_portfolio_basic(self):
        """Test basic portfolio parsing"""
        analyzer = PortfolioAnalyzer()

        portfolio_str = "40% AAPL, 30% MSFT, 30% GOOGL"
        holdings = analyzer.parse_portfolio(portfolio_str)

        assert len(holdings) == 3
        assert holdings["AAPL"] == pytest.approx(0.4, rel=0.01)
        assert holdings["MSFT"] == pytest.approx(0.3, rel=0.01)
        assert holdings["GOOGL"] == pytest.approx(0.3, rel=0.01)
        assert sum(holdings.values()) == pytest.approx(1.0)

    def test_parse_portfolio_reverse_order(self):
        """Test parsing with ticker before percentage"""
        analyzer = PortfolioAnalyzer()

        portfolio_str = "AAPL 40%, MSFT 30%, GOOGL 30%"
        holdings = analyzer.parse_portfolio(portfolio_str)

        assert holdings["AAPL"] == pytest.approx(0.4, rel=0.01)

    def test_parse_portfolio_normalization(self):
        """Test that weights are normalized to sum to 1"""
        analyzer = PortfolioAnalyzer()

        # Weights don't sum to 100%
        portfolio_str = "50% AAPL, 30% MSFT, 30% GOOGL"
        holdings = analyzer.parse_portfolio(portfolio_str)

        assert sum(holdings.values()) == pytest.approx(1.0)

    def test_calculate_returns(self, sample_prices):
        """Test returns calculation"""
        analyzer = PortfolioAnalyzer()
        returns = analyzer.calculate_returns(sample_prices)

        assert len(returns) == len(sample_prices) - 1
        assert returns.index[0] == sample_prices.index[1]
        assert not returns.isnull().any().any()

    def test_calculate_portfolio_returns(self, sample_returns, sample_portfolio):
        """Test portfolio returns calculation"""
        analyzer = PortfolioAnalyzer()
        portfolio_returns = analyzer.calculate_portfolio_returns(
            sample_returns, sample_portfolio
        )

        assert len(portfolio_returns) == len(sample_returns)
        assert isinstance(portfolio_returns, pd.Series)

        # Manual calculation check
        expected = (
            sample_returns['AAPL'] * 0.4 +
            sample_returns['MSFT'] * 0.3 +
            sample_returns['GOOGL'] * 0.3
        )
        pd.testing.assert_series_equal(portfolio_returns, expected)

    def test_calculate_annual_return(self, sample_returns):
        """Test annual return calculation"""
        analyzer = PortfolioAnalyzer()
        annual_return = analyzer.calculate_annual_return(sample_returns['AAPL'])

        assert isinstance(annual_return, float)
        assert -1 < annual_return < 5  # Reasonable range

    def test_calculate_volatility(self, sample_returns):
        """Test volatility calculation"""
        analyzer = PortfolioAnalyzer()
        volatility = analyzer.calculate_volatility(sample_returns['AAPL'])

        assert isinstance(volatility, float)
        assert volatility > 0
        assert volatility < 2  # Reasonable upper bound

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        analyzer = PortfolioAnalyzer(risk_free_rate=0.04)

        # Positive Sharpe
        sharpe = analyzer.calculate_sharpe_ratio(0.15, 0.20)
        assert sharpe == pytest.approx((0.15 - 0.04) / 0.20)

        # Zero volatility edge case
        sharpe_zero = analyzer.calculate_sharpe_ratio(0.10, 0.0)
        assert sharpe_zero == 0.0

    def test_calculate_max_drawdown(self, sample_returns):
        """Test maximum drawdown calculation"""
        analyzer = PortfolioAnalyzer()
        max_dd = analyzer.calculate_max_drawdown(sample_returns['AAPL'])

        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative
        assert max_dd > -1  # Shouldn't lose everything in sample data

    def test_calculate_var(self, sample_returns):
        """Test Value at Risk calculation"""
        analyzer = PortfolioAnalyzer()
        var_95 = analyzer.calculate_var(sample_returns['AAPL'], confidence=0.95)

        assert isinstance(var_95, float)
        assert var_95 > 0  # VaR is reported as positive loss

    def test_calculate_cvar(self, sample_returns):
        """Test Conditional VaR calculation"""
        analyzer = PortfolioAnalyzer()
        cvar_95 = analyzer.calculate_cvar(sample_returns['AAPL'], confidence=0.95)

        assert isinstance(cvar_95, float)
        assert cvar_95 > 0

        # CVaR should typically be >= VaR
        var_95 = analyzer.calculate_var(sample_returns['AAPL'], confidence=0.95)
        assert cvar_95 >= var_95

    def test_get_asset_correlations(self, mock_yfinance_data):
        """Test correlation matrix calculation"""
        analyzer = PortfolioAnalyzer()
        tickers = ['AAPL', 'MSFT', 'GOOGL']

        corr_matrix = analyzer.get_asset_correlations(tickers, period="1y")

        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (3, 3)

        # Diagonal should be 1.0
        assert all(corr_matrix.iloc[i, i] == pytest.approx(1.0)
                   for i in range(3))

        # Matrix should be symmetric
        assert corr_matrix.equals(corr_matrix.T)


class TestPortfolioMetrics:
    """Test suite for PortfolioMetrics dataclass"""

    def test_portfolio_metrics_creation(self):
        """Test creating PortfolioMetrics instance"""
        metrics = PortfolioMetrics(
            annual_return=0.15,
            annual_volatility=0.20,
            sharpe_ratio=0.55,
            max_drawdown=-0.15,
            beta=1.1,
            var_95=0.03,
            cvar_95=0.04,
            holdings={"AAPL": 0.5, "MSFT": 0.5},
            total_value=1.0,
            period_days=252
        )

        assert metrics.annual_return == 0.15
        assert metrics.sharpe_ratio == 0.55
        assert len(metrics.holdings) == 2
        assert metrics.period_days == 252


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_portfolio_string(self):
        """Test parsing empty portfolio string"""
        analyzer = PortfolioAnalyzer()
        holdings = analyzer.parse_portfolio("")
        assert len(holdings) == 0

    def test_invalid_portfolio_string(self):
        """Test parsing invalid portfolio string"""
        analyzer = PortfolioAnalyzer()
        holdings = analyzer.parse_portfolio("not a valid portfolio")
        assert len(holdings) == 0

    def test_single_asset_portfolio(self):
        """Test portfolio with single asset"""
        analyzer = PortfolioAnalyzer()
        holdings = analyzer.parse_portfolio("100% AAPL")

        assert len(holdings) == 1
        assert holdings["AAPL"] == pytest.approx(1.0)

    def test_negative_returns(self):
        """Test metrics with negative returns"""
        analyzer = PortfolioAnalyzer()

        # Create negative returns
        negative_returns = pd.Series([-0.01] * 252)
        annual_return = analyzer.calculate_annual_return(negative_returns)

        assert annual_return < 0

    def test_zero_volatility(self):
        """Test Sharpe ratio with zero volatility"""
        analyzer = PortfolioAnalyzer()
        sharpe = analyzer.calculate_sharpe_ratio(0.05, 0.0)

        assert sharpe == 0.0


class TestIntegration:
    """Integration tests for full portfolio analysis"""

    def test_full_analysis_workflow(self, mock_yfinance_data, sample_portfolio):
        """Test complete analysis workflow"""
        analyzer = PortfolioAnalyzer()

        # This would normally call yfinance, but we've mocked it
        metrics = analyzer.analyze_portfolio(sample_portfolio, period="1y")

        # Verify all metrics are present
        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.annual_return is not None
        assert metrics.annual_volatility > 0
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown <= 0
        assert metrics.beta > 0
        assert metrics.var_95 > 0
        assert metrics.cvar_95 > 0
        assert len(metrics.holdings) == 3
        assert metrics.period_days > 0
