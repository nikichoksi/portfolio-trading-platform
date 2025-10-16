"""
Portfolio analytics and risk metrics calculation.
Calculates volatility, beta, Sharpe ratio, max drawdown, and diversification metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PortfolioMetrics:
    """Portfolio risk and return metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    beta: float
    max_drawdown: float
    diversification_score: float
    sector_concentration: Dict[str, float]
    top_holdings: List[Tuple[str, float]]
    risk_level: str  # 'Low', 'Medium', 'High', 'Very High'


class PortfolioAnalyzer:
    """Analyzes portfolio risk and return characteristics"""

    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize portfolio analyzer.

        Args:
            risk_free_rate: Annual risk-free rate (default 4%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_portfolio_metrics(
        self,
        positions: List,
        price_history: Dict[str, pd.DataFrame],
        sectors: Dict[str, str]
    ) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio metrics.

        Args:
            positions: List of Position objects
            price_history: Dict mapping ticker to price DataFrame
            sectors: Dict mapping ticker to sector

        Returns:
            PortfolioMetrics object with all calculated metrics
        """
        if not positions:
            return self._empty_metrics()

        # Calculate portfolio weights
        total_value = sum(p.market_value for p in positions)
        weights = {p.ticker: p.market_value / total_value for p in positions}

        # Calculate returns for each position
        returns_data = {}
        for ticker, df in price_history.items():
            if ticker in weights and not df.empty:
                returns_data[ticker] = df['Close'].pct_change().dropna()

        if not returns_data:
            return self._empty_metrics()

        # Align all return series to common dates
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        if returns_df.empty or len(returns_df) < 2:
            return self._empty_metrics()

        # Calculate portfolio returns
        portfolio_returns = sum(
            returns_df[ticker] * weights[ticker]
            for ticker in returns_df.columns if ticker in weights
        )

        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Calculate beta (vs market proxy - SPY if available)
        beta = self._calculate_beta(portfolio_returns, returns_df)

        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)

        # Calculate diversification metrics
        diversification_score = self._calculate_diversification_score(weights, sectors)
        sector_concentration = self._calculate_sector_concentration(positions, sectors)

        # Top holdings
        top_holdings = sorted(
            [(p.ticker, p.market_value / total_value * 100) for p in positions],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Determine risk level
        risk_level = self._determine_risk_level(volatility, beta, diversification_score)

        return PortfolioMetrics(
            total_return=total_return * 100,
            annualized_return=annualized_return * 100,
            volatility=volatility * 100,
            sharpe_ratio=sharpe_ratio,
            beta=beta,
            max_drawdown=max_drawdown * 100,
            diversification_score=diversification_score,
            sector_concentration=sector_concentration,
            top_holdings=top_holdings,
            risk_level=risk_level
        )

    def _calculate_beta(self, portfolio_returns: pd.Series, returns_df: pd.DataFrame) -> float:
        """Calculate portfolio beta relative to market"""
        # Use equal-weighted portfolio as market proxy if SPY not available
        market_returns = returns_df.mean(axis=1)

        if len(portfolio_returns) < 2 or len(market_returns) < 2:
            return 1.0

        covariance = portfolio_returns.cov(market_returns)
        market_variance = market_returns.var()

        if market_variance == 0:
            return 1.0

        return covariance / market_variance

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_diversification_score(
        self,
        weights: Dict[str, float],
        sectors: Dict[str, str]
    ) -> float:
        """
        Calculate diversification score (0-100).
        Higher score = better diversification.
        """
        # Number of holdings factor (0-40 points)
        num_holdings = len(weights)
        holdings_score = min(num_holdings * 5, 40)

        # Concentration factor (0-30 points)
        # Penalize if any single holding > 30%
        max_weight = max(weights.values())
        concentration_score = 30 * (1 - min(max_weight / 0.3, 1))

        # Sector diversity factor (0-30 points)
        sector_weights = {}
        for ticker, weight in weights.items():
            sector = sectors.get(ticker, 'Other')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        num_sectors = len(sector_weights)
        sector_score = min(num_sectors * 6, 30)

        return holdings_score + concentration_score + sector_score

    def _calculate_sector_concentration(
        self,
        positions: List,
        sectors: Dict[str, str]
    ) -> Dict[str, float]:
        """Calculate percentage allocation by sector"""
        total_value = sum(p.market_value for p in positions)
        sector_values = {}

        for position in positions:
            sector = sectors.get(position.ticker, 'Other')
            sector_values[sector] = sector_values.get(sector, 0) + position.market_value

        return {
            sector: (value / total_value * 100)
            for sector, value in sorted(
                sector_values.items(),
                key=lambda x: x[1],
                reverse=True
            )
        }

    def _determine_risk_level(
        self,
        volatility: float,
        beta: float,
        diversification_score: float
    ) -> str:
        """Determine overall portfolio risk level"""
        # Risk factors (weighted scoring)
        vol_score = min(volatility / 0.30 * 40, 40)  # 30% vol = max score
        beta_score = min(abs(beta - 1) * 30, 30)      # Higher deviation = higher risk
        div_score = (100 - diversification_score) * 0.3  # Lower diversification = higher risk

        total_risk_score = vol_score + beta_score + div_score

        if total_risk_score < 30:
            return "Low"
        elif total_risk_score < 50:
            return "Medium"
        elif total_risk_score < 70:
            return "High"
        else:
            return "Very High"

    def _empty_metrics(self) -> PortfolioMetrics:
        """Return empty metrics for portfolios with no data"""
        return PortfolioMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            beta=1.0,
            max_drawdown=0.0,
            diversification_score=0.0,
            sector_concentration={},
            top_holdings=[],
            risk_level="Unknown"
        )


def generate_portfolio_narrative(metrics: PortfolioMetrics, positions: List) -> str:
    """
    Generate a narrative explanation of portfolio characteristics.

    Args:
        metrics: PortfolioMetrics object
        positions: List of Position objects

    Returns:
        Narrative text describing the portfolio
    """
    if not positions:
        return "Your portfolio is currently empty."

    narrative_parts = []

    # Overview
    num_positions = len(positions)
    narrative_parts.append(
        f"Your portfolio consists of {num_positions} position{'s' if num_positions > 1 else ''} "
        f"with a total return of {metrics.total_return:+.2f}%."
    )

    # Risk assessment
    risk_adjectives = {
        "Low": "conservative",
        "Medium": "moderately risky",
        "High": "aggressive",
        "Very High": "very aggressive"
    }
    risk_desc = risk_adjectives.get(metrics.risk_level, "undefined")

    narrative_parts.append(
        f"\n\n**Risk Profile:** Your portfolio shows a {risk_desc} risk profile with "
        f"{metrics.volatility:.1f}% annualized volatility and a beta of {metrics.beta:.2f}. "
    )

    if metrics.beta > 1.2:
        narrative_parts.append("This indicates higher volatility than the market.")
    elif metrics.beta < 0.8:
        narrative_parts.append("This indicates lower volatility than the market.")
    else:
        narrative_parts.append("This indicates similar volatility to the market.")

    # Diversification
    narrative_parts.append(
        f"\n\n**Diversification:** Your diversification score is {metrics.diversification_score:.0f}/100. "
    )

    if metrics.diversification_score >= 70:
        narrative_parts.append("Your portfolio is well-diversified across multiple holdings and sectors.")
    elif metrics.diversification_score >= 40:
        narrative_parts.append("Your portfolio shows moderate diversification but could be improved.")
    else:
        narrative_parts.append("WARNING: Your portfolio lacks diversification and may be exposed to concentrated risk.")

    # Sector concentration
    if metrics.sector_concentration:
        top_sector = list(metrics.sector_concentration.items())[0]
        narrative_parts.append(
            f"\n\n**Sector Allocation:** Your largest sector exposure is {top_sector[0]} "
            f"at {top_sector[1]:.1f}% of your portfolio. "
        )

        if top_sector[1] > 50:
            narrative_parts.append("WARNING: This represents significant sector concentration risk.")

    # Performance metrics
    if metrics.sharpe_ratio > 1.5:
        perf_assessment = "excellent risk-adjusted returns"
    elif metrics.sharpe_ratio > 0.5:
        perf_assessment = "good risk-adjusted returns"
    elif metrics.sharpe_ratio > 0:
        perf_assessment = "moderate risk-adjusted returns"
    else:
        perf_assessment = "poor risk-adjusted returns"

    narrative_parts.append(
        f"\n\n**Performance:** With a Sharpe ratio of {metrics.sharpe_ratio:.2f}, "
        f"your portfolio is delivering {perf_assessment}. "
        f"Your maximum drawdown of {metrics.max_drawdown:.1f}% indicates "
    )

    if abs(metrics.max_drawdown) > 30:
        narrative_parts.append("significant downside risk during market downturns.")
    elif abs(metrics.max_drawdown) > 15:
        narrative_parts.append("moderate downside risk during market downturns.")
    else:
        narrative_parts.append("relatively controlled downside risk.")

    # Top holdings
    if metrics.top_holdings:
        narrative_parts.append(
            f"\n\n**Top Holdings:** Your largest positions are: " +
            ", ".join([f"{ticker} ({weight:.1f}%)" for ticker, weight in metrics.top_holdings[:3]])
        )

    return "".join(narrative_parts)


def identify_portfolio_strengths_weaknesses(metrics: PortfolioMetrics) -> Dict[str, List[str]]:
    """
    Identify key strengths and weaknesses in portfolio allocation.

    Returns:
        Dict with 'strengths' and 'weaknesses' lists
    """
    strengths = []
    weaknesses = []

    # Diversification
    if metrics.diversification_score >= 70:
        strengths.append("Well-diversified portfolio reduces unsystematic risk")
    elif metrics.diversification_score < 40:
        weaknesses.append("Poor diversification - concentrated holdings increase risk")

    # Sharpe Ratio
    if metrics.sharpe_ratio > 1.0:
        strengths.append(f"Strong risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")
    elif metrics.sharpe_ratio < 0.5:
        weaknesses.append(f"Weak risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")

    # Volatility
    if metrics.volatility < 15:
        strengths.append(f"Low volatility ({metrics.volatility:.1f}%) provides stability")
    elif metrics.volatility > 30:
        weaknesses.append(f"High volatility ({metrics.volatility:.1f}%) increases uncertainty")

    # Drawdown
    if abs(metrics.max_drawdown) < 15:
        strengths.append(f"Limited downside risk (max drawdown: {metrics.max_drawdown:.1f}%)")
    elif abs(metrics.max_drawdown) > 30:
        weaknesses.append(f"Significant drawdown risk ({metrics.max_drawdown:.1f}%)")

    # Sector concentration
    if metrics.sector_concentration:
        top_sector_weight = list(metrics.sector_concentration.values())[0]
        if top_sector_weight > 60:
            weaknesses.append(f"Over-concentrated in one sector ({top_sector_weight:.0f}%)")

    # Beta
    if 0.8 <= metrics.beta <= 1.2:
        strengths.append(f"Market-aligned beta ({metrics.beta:.2f}) balances risk/return")
    elif metrics.beta > 1.5:
        weaknesses.append(f"High beta ({metrics.beta:.2f}) amplifies market movements")

    # Default messages if none found
    if not strengths:
        strengths.append("Portfolio requires further optimization")
    if not weaknesses:
        weaknesses.append("No major weaknesses identified")

    return {
        "strengths": strengths,
        "weaknesses": weaknesses
    }
