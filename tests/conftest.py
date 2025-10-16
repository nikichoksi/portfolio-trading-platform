"""
Pytest configuration and fixtures for portfolio tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_portfolio():
    """Sample portfolio holdings"""
    return {
        "AAPL": 0.4,
        "MSFT": 0.3,
        "GOOGL": 0.3
    }


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')

    # Generate correlated price data
    returns = np.random.randn(252, 3) * 0.02
    prices = pd.DataFrame(
        100 * np.exp(returns.cumsum(axis=0)),
        index=dates,
        columns=['AAPL', 'MSFT', 'GOOGL']
    )

    return prices


@pytest.fixture
def sample_returns(sample_prices):
    """Generate sample returns from prices"""
    return sample_prices.pct_change().dropna()


@pytest.fixture
def mock_yfinance_data(monkeypatch, sample_prices):
    """Mock yfinance download function"""
    def mock_download(tickers, period="1y", progress=False):
        # Return a dict-like structure that mimics yfinance
        data = pd.DataFrame({'Adj Close': sample_prices})
        return data

    import yfinance as yf
    monkeypatch.setattr(yf, "download", mock_download)
    return sample_prices
