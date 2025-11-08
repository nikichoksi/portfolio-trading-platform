"""
Scenario Simulator Agent using LangChain.
Tests portfolio performance under hypothetical market conditions.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from datetime import datetime, timedelta

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.portfolio_analytics import PortfolioAnalyzer

# Load environment variables
load_dotenv()


class ScenarioSimulator:
    """Utility class for scenario simulation logic"""

    # Historical crisis scenarios
    HISTORICAL_SCENARIOS = {
        "2008_financial_crisis": {
            "name": "2008 Financial Crisis",
            "market_drop": -37.0,  # S&P 500 dropped 37% in 2008
            "duration_days": 365,
            "sector_impacts": {
                "Financials": -55.0,
                "Real Estate": -48.0,
                "Consumer Discretionary": -37.0,
                "Industrials": -35.0,
                "Technology": -42.0,
                "Healthcare": -22.0,
                "Consumer Staples": -14.0,
                "Utilities": -27.0,
                "Energy": -35.0,
                "Materials": -40.0,
            }
        },
        "2020_covid_crash": {
            "name": "COVID-19 Market Crash (Feb-Mar 2020)",
            "market_drop": -34.0,  # S&P 500 dropped 34% peak-to-trough
            "duration_days": 33,
            "sector_impacts": {
                "Energy": -50.0,
                "Financials": -45.0,
                "Industrials": -42.0,
                "Consumer Discretionary": -38.0,
                "Real Estate": -35.0,
                "Materials": -33.0,
                "Technology": -30.0,
                "Communication Services": -25.0,
                "Utilities": -20.0,
                "Healthcare": -18.0,
                "Consumer Staples": -15.0,
            }
        },
        "2000_dotcom_bubble": {
            "name": "Dot-com Bubble Burst (2000-2002)",
            "market_drop": -49.0,  # NASDAQ dropped 78%, S&P 49%
            "duration_days": 900,
            "sector_impacts": {
                "Technology": -78.0,
                "Communication Services": -65.0,
                "Consumer Discretionary": -45.0,
                "Financials": -42.0,
                "Industrials": -38.0,
                "Materials": -35.0,
                "Energy": -30.0,
                "Healthcare": -25.0,
                "Utilities": -20.0,
                "Consumer Staples": -15.0,
            }
        }
    }

    @classmethod
    def simulate_market_crash(cls, holdings: Dict[str, float], crash_percent: float, sector_map: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Simulate a market-wide crash scenario.

        Args:
            holdings: Dictionary of {ticker: weight}
            crash_percent: Market drop percentage (negative number)
            sector_map: Optional mapping of {ticker: sector}

        Returns:
            Simulation results dictionary
        """
        results = {
            "scenario": f"Market Crash ({crash_percent:.0f}%)",
            "initial_value": 100.0,
            "final_value": 0.0,
            "absolute_loss": 0.0,
            "percent_loss": 0.0,
            "holdings_impact": []
        }

        # Simulate impact on each holding
        for ticker, weight in holdings.items():
            # Base crash plus some randomness (±10%)
            individual_impact = crash_percent * (1 + np.random.uniform(-0.1, 0.1))

            holding_initial = 100.0 * weight
            holding_final = holding_initial * (1 + individual_impact / 100)
            holding_loss = holding_final - holding_initial

            results["holdings_impact"].append({
                "ticker": ticker,
                "weight": weight * 100,
                "impact": individual_impact,
                "initial_value": holding_initial,
                "final_value": holding_final,
                "loss": holding_loss
            })

            results["final_value"] += holding_final

        results["absolute_loss"] = results["final_value"] - results["initial_value"]
        results["percent_loss"] = (results["final_value"] / results["initial_value"] - 1) * 100

        return results

    @classmethod
    def simulate_sector_decline(cls, holdings: Dict[str, float], target_sector: str, decline_percent: float, sector_map: Dict[str, str]) -> Dict[str, Any]:
        """
        Simulate a specific sector decline.

        Args:
            holdings: Dictionary of {ticker: weight}
            target_sector: Sector to simulate decline in
            decline_percent: Decline percentage (negative)
            sector_map: Mapping of {ticker: sector}

        Returns:
            Simulation results
        """
        results = {
            "scenario": f"{target_sector} Sector Decline ({decline_percent:.0f}%)",
            "initial_value": 100.0,
            "final_value": 0.0,
            "absolute_loss": 0.0,
            "percent_loss": 0.0,
            "holdings_impact": [],
            "exposed_holdings": 0,
            "sector_exposure": 0.0
        }

        for ticker, weight in holdings.items():
            sector = sector_map.get(ticker, "Unknown")
            holding_initial = 100.0 * weight

            # Apply decline only to holdings in target sector
            if sector == target_sector:
                individual_impact = decline_percent * (1 + np.random.uniform(-0.05, 0.05))
                results["exposed_holdings"] += 1
                results["sector_exposure"] += weight * 100
            else:
                # Other sectors have minor sympathetic moves
                individual_impact = decline_percent * 0.15  # 15% correlation effect

            holding_final = holding_initial * (1 + individual_impact / 100)
            holding_loss = holding_final - holding_initial

            results["holdings_impact"].append({
                "ticker": ticker,
                "sector": sector,
                "weight": weight * 100,
                "impact": individual_impact,
                "initial_value": holding_initial,
                "final_value": holding_final,
                "loss": holding_loss,
                "in_target_sector": sector == target_sector
            })

            results["final_value"] += holding_final

        results["absolute_loss"] = results["final_value"] - results["initial_value"]
        results["percent_loss"] = (results["final_value"] / results["initial_value"] - 1) * 100

        return results

    @classmethod
    def simulate_historical_crisis(cls, holdings: Dict[str, float], scenario_name: str, sector_map: Dict[str, str]) -> Dict[str, Any]:
        """
        Simulate a historical crisis scenario.

        Args:
            holdings: Dictionary of {ticker: weight}
            scenario_name: Key from HISTORICAL_SCENARIOS
            sector_map: Mapping of {ticker: sector}

        Returns:
            Simulation results
        """
        if scenario_name not in cls.HISTORICAL_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = cls.HISTORICAL_SCENARIOS[scenario_name]

        results = {
            "scenario": scenario["name"],
            "market_drop": scenario["market_drop"],
            "duration": scenario["duration_days"],
            "initial_value": 100.0,
            "final_value": 0.0,
            "absolute_loss": 0.0,
            "percent_loss": 0.0,
            "holdings_impact": []
        }

        for ticker, weight in holdings.items():
            sector = sector_map.get(ticker, "Unknown")
            holding_initial = 100.0 * weight

            # Get sector-specific impact from historical data
            sector_impact = scenario["sector_impacts"].get(sector, scenario["market_drop"])

            # Add some randomness (±5%) to represent individual stock variation
            individual_impact = sector_impact * (1 + np.random.uniform(-0.05, 0.05))

            holding_final = holding_initial * (1 + individual_impact / 100)
            holding_loss = holding_final - holding_initial

            results["holdings_impact"].append({
                "ticker": ticker,
                "sector": sector,
                "weight": weight * 100,
                "impact": individual_impact,
                "initial_value": holding_initial,
                "final_value": holding_final,
                "loss": holding_loss
            })

            results["final_value"] += holding_final

        results["absolute_loss"] = results["final_value"] - results["initial_value"]
        results["percent_loss"] = (results["final_value"] / results["initial_value"] - 1) * 100

        return results

    @classmethod
    def monte_carlo_simulation(cls, holdings: Dict[str, float], metrics: Any, num_simulations: int = 1000, time_horizon_days: int = 252) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for probabilistic outcomes.

        Args:
            holdings: Dictionary of {ticker: weight}
            metrics: Portfolio metrics with volatility info
            num_simulations: Number of simulation paths
            time_horizon_days: Time horizon in trading days

        Returns:
            Monte Carlo results with percentile outcomes
        """
        # Use portfolio volatility (annualized) and convert to daily
        annual_vol = metrics.volatility / 100
        daily_vol = annual_vol / np.sqrt(252)

        # Assume expected return (could be enhanced with historical returns)
        annual_return = metrics.annualized_return / 100 if hasattr(metrics, 'annualized_return') else 0.10
        daily_return = annual_return / 252

        # Run simulations
        final_values = []
        for _ in range(num_simulations):
            value = 100.0
            for _ in range(time_horizon_days):
                daily_change = np.random.normal(daily_return, daily_vol)
                value *= (1 + daily_change)
            final_values.append(value)

        final_values = np.array(final_values)

        # Calculate statistics
        results = {
            "scenario": f"Monte Carlo Simulation ({num_simulations} paths, {time_horizon_days} days)",
            "initial_value": 100.0,
            "simulations": num_simulations,
            "time_horizon_days": time_horizon_days,
            "median_outcome": float(np.percentile(final_values, 50)),
            "mean_outcome": float(np.mean(final_values)),
            "percentiles": {
                "5th": float(np.percentile(final_values, 5)),
                "10th": float(np.percentile(final_values, 10)),
                "25th": float(np.percentile(final_values, 25)),
                "50th": float(np.percentile(final_values, 50)),
                "75th": float(np.percentile(final_values, 75)),
                "90th": float(np.percentile(final_values, 90)),
                "95th": float(np.percentile(final_values, 95)),
            },
            "prob_loss": float(np.sum(final_values < 100.0) / num_simulations * 100),
            "prob_double": float(np.sum(final_values >= 200.0) / num_simulations * 100),
            "best_case": float(np.max(final_values)),
            "worst_case": float(np.min(final_values)),
        }

        return results


class ScenarioSimulatorAgent:
    """
    Scenario Simulator Agent that performs what-if analysis and stress testing.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.1,
        use_openai: bool = False,
        service=None
    ):
        """Initialize the Scenario Simulator Agent"""
        self.analyzer = PortfolioAnalyzer()
        self.simulator = ScenarioSimulator()
        self.service = service
        self.use_openai = use_openai

        # Initialize LLM
        if use_openai:
            self.llm = ChatOpenAI(
                model=model if "gpt" in model else "gpt-4-turbo-preview",
                temperature=temperature
            )
        else:
            self.llm = ChatAnthropic(
                model=model,
                temperature=temperature
            )

        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def _create_tools(self) -> List[Tool]:
        """Create tools for scenario simulation"""

        def simulate_market_crash_tool(portfolio_and_crash: str) -> str:
            """
            Simulate market crash scenario.
            Input format: "portfolio: 40% AAPL, 30% MSFT | crash: -20"
            """
            try:
                parts = portfolio_and_crash.split("|")
                if len(parts) != 2:
                    return "Invalid format. Use: 'portfolio: X% TICKER... | crash: -20'"

                portfolio_str = parts[0].replace("portfolio:", "").strip()
                crash_str = parts[1].replace("crash:", "").strip()

                holdings = self.analyzer.parse_portfolio(portfolio_str)
                if not holdings:
                    return "Could not parse portfolio."

                crash_percent = float(crash_str)
                if crash_percent > 0:
                    crash_percent = -crash_percent  # Ensure negative

                results = self.simulator.simulate_market_crash(holdings, crash_percent)

                output = f"""
Market Crash Simulation: {crash_percent:.0f}% Market Drop
================================================

Initial Portfolio Value: ${results['initial_value']:.2f}
Final Portfolio Value: ${results['final_value']:.2f}
Total Loss: ${results['absolute_loss']:.2f} ({results['percent_loss']:.1f}%)

Impact by Holding:
"""
                for holding in sorted(results['holdings_impact'], key=lambda x: x['loss']):
                    output += f"\n{holding['ticker']} ({holding['weight']:.1f}% of portfolio):\n"
                    output += f"  Impact: {holding['impact']:.1f}%\n"
                    output += f"  Value change: ${holding['initial_value']:.2f} → ${holding['final_value']:.2f}\n"
                    output += f"  Loss: ${holding['loss']:.2f}\n"

                return output

            except Exception as e:
                return f"Error simulating crash: {str(e)}"

        def simulate_sector_decline_tool(portfolio_sector_decline: str) -> str:
            """
            Simulate sector-specific decline.
            Input: "portfolio: 40% AAPL, 30% MSFT | sector: Technology | decline: -30"
            """
            try:
                parts = portfolio_sector_decline.split("|")
                if len(parts) != 3:
                    return "Invalid format. Use: 'portfolio: X% TICKER | sector: Technology | decline: -30'"

                portfolio_str = parts[0].replace("portfolio:", "").strip()
                sector = parts[1].replace("sector:", "").strip()
                decline_str = parts[2].replace("decline:", "").strip()

                holdings = self.analyzer.parse_portfolio(portfolio_str)
                if not holdings:
                    return "Could not parse portfolio."

                decline_percent = float(decline_str)
                if decline_percent > 0:
                    decline_percent = -decline_percent

                # Get sector mapping (simplified - in production would use real sector data)
                sector_map = {}
                for ticker in holdings.keys():
                    # Simple heuristic - would use actual sector lookup
                    if ticker in ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN"]:
                        sector_map[ticker] = "Technology"
                    elif ticker in ["JPM", "BAC", "GS", "WFC"]:
                        sector_map[ticker] = "Financials"
                    elif ticker in ["XOM", "CVX"]:
                        sector_map[ticker] = "Energy"
                    else:
                        sector_map[ticker] = "Other"

                results = self.simulator.simulate_sector_decline(holdings, sector, decline_percent, sector_map)

                output = f"""
Sector Decline Simulation: {sector} {decline_percent:.0f}%
=================================================

Portfolio Exposure to {sector}: {results['sector_exposure']:.1f}%
Holdings in {sector}: {results['exposed_holdings']}

Initial Portfolio Value: ${results['initial_value']:.2f}
Final Portfolio Value: ${results['final_value']:.2f}
Total Impact: ${results['absolute_loss']:.2f} ({results['percent_loss']:.1f}%)

Impact by Holding:
"""
                for holding in sorted(results['holdings_impact'], key=lambda x: x['in_target_sector'], reverse=True):
                    marker = " [EXPOSED]" if holding['in_target_sector'] else ""
                    output += f"\n{holding['ticker']} - {holding['sector']}{marker} ({holding['weight']:.1f}%):\n"
                    output += f"  Impact: {holding['impact']:.1f}%\n"
                    output += f"  Loss: ${holding['loss']:.2f}\n"

                return output

            except Exception as e:
                return f"Error simulating sector decline: {str(e)}"

        def simulate_historical_crisis_tool(portfolio_and_crisis: str) -> str:
            """
            Simulate historical crisis scenario.
            Input: "portfolio: 40% AAPL, 30% MSFT | crisis: 2008_financial_crisis"
            Available crises: 2008_financial_crisis, 2020_covid_crash, 2000_dotcom_bubble
            """
            try:
                parts = portfolio_and_crisis.split("|")
                if len(parts) != 2:
                    return "Invalid format. Use: 'portfolio: X% TICKER | crisis: 2008_financial_crisis'"

                portfolio_str = parts[0].replace("portfolio:", "").strip()
                crisis_name = parts[1].replace("crisis:", "").strip()

                holdings = self.analyzer.parse_portfolio(portfolio_str)
                if not holdings:
                    return "Could not parse portfolio."

                # Simple sector mapping
                sector_map = {}
                for ticker in holdings.keys():
                    if ticker in ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]:
                        sector_map[ticker] = "Technology"
                    elif ticker in ["JPM", "BAC", "GS"]:
                        sector_map[ticker] = "Financials"
                    else:
                        sector_map[ticker] = "Consumer Discretionary"

                results = self.simulator.simulate_historical_crisis(holdings, crisis_name, sector_map)

                output = f"""
Historical Crisis Simulation: {results['scenario']}
==================================================

Market Drop: {results['market_drop']:.1f}%
Crisis Duration: {results['duration']} days

Initial Portfolio Value: ${results['initial_value']:.2f}
Final Portfolio Value: ${results['final_value']:.2f}
Total Loss: ${results['absolute_loss']:.2f} ({results['percent_loss']:.1f}%)

Impact by Holding:
"""
                for holding in sorted(results['holdings_impact'], key=lambda x: x['loss']):
                    output += f"\n{holding['ticker']} - {holding['sector']} ({holding['weight']:.1f}%):\n"
                    output += f"  Historical Impact: {holding['impact']:.1f}%\n"
                    output += f"  Value: ${holding['initial_value']:.2f} → ${holding['final_value']:.2f}\n"
                    output += f"  Loss: ${holding['loss']:.2f}\n"

                return output

            except Exception as e:
                return f"Error simulating historical crisis: {str(e)}"

        tools = [
            Tool(
                name="simulate_market_crash",
                func=simulate_market_crash_tool,
                description=(
                    "Simulates a market-wide crash scenario. "
                    "Input: 'portfolio: 40% AAPL, 30% MSFT | crash: -20'. "
                    "Returns impact on portfolio value and individual holdings."
                )
            ),
            Tool(
                name="simulate_sector_decline",
                func=simulate_sector_decline_tool,
                description=(
                    "Simulates a sector-specific decline. "
                    "Input: 'portfolio: X% TICKER | sector: Technology | decline: -30'. "
                    "Shows exposure and impact on sector holdings."
                )
            ),
            Tool(
                name="simulate_historical_crisis",
                func=simulate_historical_crisis_tool,
                description=(
                    "Simulates historical crisis scenarios (2008_financial_crisis, 2020_covid_crash, 2000_dotcom_bubble). "
                    "Input: 'portfolio: X% TICKER | crisis: 2008_financial_crisis'. "
                    "Uses actual historical sector impacts."
                )
            )
        ]

        return tools

    def _create_agent(self):
        """Create agent with system prompt"""

        system_prompt = """You are a Scenario Simulator Agent, specializing in portfolio stress testing and what-if analysis.

Your role is to:
1. Test portfolios under various market conditions (crashes, recessions, sector declines)
2. Run historical crisis simulations (2008 crisis, COVID crash, dot-com bubble)
3. Perform Monte Carlo simulations for probabilistic outcomes
4. Calculate Value-at-Risk (VaR) and Conditional VaR (CVaR)
5. Help investors understand downside risks

Available scenarios:
- Market crash: Simulate market-wide drops (-10%, -20%, -30%, etc.)
- Sector decline: Test specific sector weaknesses (Tech down 20%, Financials down 30%)
- Historical crises: Replay 2008 financial crisis, 2020 COVID crash, 2000 dot-com bubble

When analyzing scenarios:
- Be specific about assumptions (e.g., "This assumes all correlations hold")
- Explain what the numbers mean in practical terms
- Highlight which holdings are most vulnerable
- Provide context from historical events
- Note that past performance doesn't guarantee future results

Communication style:
- Clear about the "what-if" nature of simulations
- Specific with numbers and impacts
- Educational about historical context
- Balanced - not fear-mongering, but realistic about risks

Remember: Scenarios are hypothetical. Real market behavior is complex and unpredictable."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return agent

    def analyze(self, query: str, chat_history: Optional[List] = None) -> str:
        """Analyze scenario simulation query"""
        try:
            response = self.agent_executor.invoke({
                "input": query,
                "chat_history": chat_history or []
            })
            return self._extract_output(response["output"])
        except Exception as e:
            return f"Error during simulation: {str(e)}"

    def _extract_output(self, output: Any) -> str:
        """Extract string output from agent response"""
        if isinstance(output, list):
            if len(output) > 0 and isinstance(output[0], dict):
                if 'text' in output[0]:
                    return output[0]['text']
            return ' '.join(str(item) for item in output)
        if isinstance(output, str):
            return output
        return str(output)


def main():
    """Example usage"""
    print("Scenario Simulator Agent")
    print("=" * 50)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set API key in .env file")
        return

    agent = ScenarioSimulatorAgent()

    # Example: Market crash simulation
    print("\n[Market Crash Simulation]\n")
    response = agent.analyze(
        "What happens to my portfolio if the market drops 25%? Portfolio: 40% AAPL, 30% MSFT, 30% GOOGL"
    )
    print(response)


if __name__ == "__main__":
    main()
