"""
Sector exposure analysis for portfolios.
Analyzes portfolio composition by industry sector.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict


class SectorAnalyzer:
    """Analyzes portfolio sector exposure and concentration"""

    # Common sector mappings for major tickers
    SECTOR_MAPPING = {
        # Technology
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'GOOG': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
        'AMD': 'Technology', 'INTC': 'Technology', 'ORCL': 'Technology',
        'CRM': 'Technology', 'ADBE': 'Technology', 'CSCO': 'Technology',

        # Consumer Discretionary
        'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
        'NKE': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
        'SBUX': 'Consumer Discretionary', 'DIS': 'Consumer Discretionary',

        # Healthcare
        'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
        'ABBV': 'Healthcare', 'TMO': 'Healthcare', 'LLY': 'Healthcare',

        # Financials
        'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
        'GS': 'Financials', 'MS': 'Financials', 'BLK': 'Financials',
        'V': 'Financials', 'MA': 'Financials',

        # Consumer Staples
        'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
        'PEP': 'Consumer Staples', 'WMT': 'Consumer Staples',
        'COST': 'Consumer Staples',

        # Energy
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',

        # Utilities
        'NEE': 'Utilities', 'DUK': 'Utilities',

        # Real Estate
        'AMT': 'Real Estate', 'PLD': 'Real Estate',

        # Materials
        'LIN': 'Materials', 'APD': 'Materials',

        # Industrials
        'BA': 'Industrials', 'HON': 'Industrials', 'UPS': 'Industrials',

        # Communication Services
        'T': 'Communication Services', 'VZ': 'Communication Services',
        'NFLX': 'Communication Services', 'CMCSA': 'Communication Services',
    }

    def __init__(self):
        """Initialize sector analyzer"""
        pass

    def get_ticker_sector(self, ticker: str) -> Optional[str]:
        """
        Get sector for a ticker symbol.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Sector name or None if not found
        """
        # First check our manual mapping
        if ticker in self.SECTOR_MAPPING:
            return self.SECTOR_MAPPING[ticker]

        # Try to fetch from yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Try different sector fields
            sector = info.get('sector') or info.get('sectorKey')
            return sector if sector else 'Other'

        except Exception:
            return 'Other'

    def analyze_sector_exposure(
        self,
        holdings: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate sector exposure based on portfolio holdings.

        Args:
            holdings: Dictionary of {ticker: weight}

        Returns:
            Dictionary of {sector: total_weight}
        """
        sector_exposure = defaultdict(float)

        for ticker, weight in holdings.items():
            sector = self.get_ticker_sector(ticker)
            sector_exposure[sector] += weight

        return dict(sector_exposure)

    def calculate_sector_concentration(
        self,
        sector_exposure: Dict[str, float]
    ) -> float:
        """
        Calculate Herfindahl-Hirschman Index (HHI) for sector concentration.

        Args:
            sector_exposure: Dictionary of {sector: weight}

        Returns:
            HHI value (0-1, lower is more diversified)
        """
        hhi = sum(weight ** 2 for weight in sector_exposure.values())
        return hhi

    def get_sector_risk_profile(
        self,
        sector_exposure: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Assess risk level of each sector.

        Args:
            sector_exposure: Dictionary of {sector: weight}

        Returns:
            Dictionary of {sector: risk_level}
        """
        # Typical risk levels by sector
        sector_risk_levels = {
            'Technology': 'High',
            'Consumer Discretionary': 'High',
            'Energy': 'High',
            'Financials': 'Medium',
            'Healthcare': 'Medium',
            'Industrials': 'Medium',
            'Communication Services': 'Medium',
            'Consumer Staples': 'Low',
            'Utilities': 'Low',
            'Real Estate': 'Medium',
            'Materials': 'Medium',
            'Other': 'Unknown'
        }

        return {
            sector: sector_risk_levels.get(sector, 'Unknown')
            for sector in sector_exposure.keys()
        }

    def get_sector_summary(
        self,
        holdings: Dict[str, float]
    ) -> Dict:
        """
        Get comprehensive sector analysis summary.

        Args:
            holdings: Dictionary of {ticker: weight}

        Returns:
            Dictionary with sector exposure, concentration, and risk info
        """
        sector_exposure = self.analyze_sector_exposure(holdings)
        hhi = self.calculate_sector_concentration(sector_exposure)
        risk_profile = self.get_sector_risk_profile(sector_exposure)

        # Categorize concentration level
        if hhi < 0.15:
            concentration_level = "Well Diversified"
        elif hhi < 0.25:
            concentration_level = "Moderately Diversified"
        else:
            concentration_level = "Concentrated"

        # Count sectors
        num_sectors = len(sector_exposure)

        return {
            'sector_exposure': sector_exposure,
            'concentration_hhi': hhi,
            'concentration_level': concentration_level,
            'num_sectors': num_sectors,
            'risk_profile': risk_profile
        }

    def format_sector_analysis(
        self,
        sector_summary: Dict
    ) -> str:
        """
        Format sector analysis into readable string.

        Args:
            sector_summary: Output from get_sector_summary

        Returns:
            Formatted string report
        """
        exposure = sector_summary['sector_exposure']
        risk_profile = sector_summary['risk_profile']

        # Sort by weight
        sorted_sectors = sorted(
            exposure.items(),
            key=lambda x: x[1],
            reverse=True
        )

        sector_lines = []
        for sector, weight in sorted_sectors:
            risk = risk_profile.get(sector, 'Unknown')
            sector_lines.append(
                f"  - {sector}: {weight:.1%} (Risk: {risk})"
            )

        return f"""
Sector Analysis:
===============

Sector Exposure:
{chr(10).join(sector_lines)}

Diversification:
- Number of Sectors: {sector_summary['num_sectors']}
- Concentration Level: {sector_summary['concentration_level']}
- HHI Score: {sector_summary['concentration_hhi']:.3f}

Notes:
- HHI < 0.15: Well diversified
- HHI 0.15-0.25: Moderately diversified
- HHI > 0.25: Concentrated
"""
