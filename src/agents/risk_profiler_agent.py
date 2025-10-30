"""
Risk Profiler Agent using LangChain.
Evaluates if portfolio aligns with investor's risk profile.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from enum import Enum

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.portfolio_analytics import PortfolioAnalyzer, PortfolioMetrics

# Load environment variables
load_dotenv()


class RiskProfile(Enum):
    """Risk profile categories"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class RiskProfiler:
    """Utility class for risk profiling logic"""

    # Risk profile benchmarks
    RISK_BENCHMARKS = {
        RiskProfile.CONSERVATIVE: {
            "max_volatility": 12.0,  # 12% annual volatility
            "max_beta": 0.7,
            "min_diversification": 80,
            "max_sector_concentration": 25.0,  # Max 25% in any sector
            "max_position_size": 15.0,  # Max 15% in single stock
            "target_sharpe": 1.5,
        },
        RiskProfile.MODERATE: {
            "max_volatility": 20.0,  # 20% annual volatility
            "max_beta": 1.1,
            "min_diversification": 60,
            "max_sector_concentration": 40.0,
            "max_position_size": 25.0,
            "target_sharpe": 1.2,
        },
        RiskProfile.AGGRESSIVE: {
            "max_volatility": 35.0,  # 35% annual volatility
            "max_beta": 1.5,
            "min_diversification": 40,
            "max_sector_concentration": 60.0,
            "max_position_size": 40.0,
            "target_sharpe": 1.0,
        }
    }

    @classmethod
    def evaluate_profile_fit(cls, metrics: PortfolioMetrics, target_profile: RiskProfile) -> Dict[str, Any]:
        """
        Evaluate how well a portfolio matches a target risk profile.

        Args:
            metrics: Portfolio metrics
            target_profile: Target risk profile

        Returns:
            Dictionary with fit score and mismatches
        """
        benchmarks = cls.RISK_BENCHMARKS[target_profile]

        mismatches = []
        fit_scores = []

        # Check volatility
        if metrics.volatility > benchmarks["max_volatility"]:
            mismatches.append({
                "metric": "Volatility",
                "current": f"{metrics.volatility:.1f}%",
                "target": f"<{benchmarks['max_volatility']:.1f}%",
                "severity": "high" if metrics.volatility > benchmarks["max_volatility"] * 1.5 else "medium"
            })
            fit_scores.append(0)
        else:
            fit_scores.append(100)

        # Check beta
        if metrics.beta > benchmarks["max_beta"]:
            mismatches.append({
                "metric": "Beta (Market Sensitivity)",
                "current": f"{metrics.beta:.2f}",
                "target": f"<{benchmarks['max_beta']:.2f}",
                "severity": "high" if metrics.beta > benchmarks["max_beta"] * 1.3 else "medium"
            })
            fit_scores.append(0)
        else:
            fit_scores.append(100)

        # Check diversification
        if metrics.diversification_score < benchmarks["min_diversification"]:
            mismatches.append({
                "metric": "Diversification",
                "current": f"{metrics.diversification_score:.0f}/100",
                "target": f">{benchmarks['min_diversification']}/100",
                "severity": "high" if metrics.diversification_score < benchmarks["min_diversification"] * 0.7 else "medium"
            })
            fit_scores.append(0)
        else:
            fit_scores.append(100)

        # Check sector concentration
        if metrics.sector_concentration:
            max_sector_pct = max(metrics.sector_concentration.values())
            if max_sector_pct > benchmarks["max_sector_concentration"]:
                top_sector = max(metrics.sector_concentration.items(), key=lambda x: x[1])
                mismatches.append({
                    "metric": "Sector Concentration",
                    "current": f"{top_sector[0]}: {top_sector[1]:.1f}%",
                    "target": f"<{benchmarks['max_sector_concentration']:.1f}% per sector",
                    "severity": "high" if max_sector_pct > benchmarks["max_sector_concentration"] * 1.5 else "medium"
                })
                fit_scores.append(0)
            else:
                fit_scores.append(100)

        # Check top holdings concentration
        if metrics.top_holdings:
            top_holding_pct = metrics.top_holdings[0][1] if metrics.top_holdings else 0
            if top_holding_pct > benchmarks["max_position_size"]:
                mismatches.append({
                    "metric": "Position Size",
                    "current": f"{metrics.top_holdings[0][0]}: {top_holding_pct:.1f}%",
                    "target": f"<{benchmarks['max_position_size']:.1f}% per stock",
                    "severity": "high" if top_holding_pct > benchmarks["max_position_size"] * 1.5 else "medium"
                })
                fit_scores.append(0)
            else:
                fit_scores.append(100)

        overall_fit = sum(fit_scores) / len(fit_scores) if fit_scores else 0

        return {
            "fit_score": overall_fit,
            "mismatches": mismatches,
            "profile": target_profile.value,
            "recommendation": cls._get_recommendation(overall_fit, target_profile)
        }

    @classmethod
    def _get_recommendation(cls, fit_score: float, profile: RiskProfile) -> str:
        """Generate recommendation based on fit score"""
        if fit_score >= 80:
            return f"Portfolio is well-aligned with {profile.value} risk profile."
        elif fit_score >= 60:
            return f"Portfolio is mostly aligned with {profile.value} profile, with minor adjustments needed."
        elif fit_score >= 40:
            return f"Portfolio has moderate misalignment with {profile.value} profile. Consider rebalancing."
        else:
            return f"Portfolio does not match {profile.value} profile. Significant changes recommended."

    @classmethod
    def suggest_profile(cls, metrics: PortfolioMetrics) -> Tuple[RiskProfile, str]:
        """
        Suggest the most appropriate risk profile for a portfolio.

        Args:
            metrics: Portfolio metrics

        Returns:
            Tuple of (suggested_profile, reasoning)
        """
        # Score each profile
        scores = {}
        for profile in RiskProfile:
            eval_result = cls.evaluate_profile_fit(metrics, profile)
            scores[profile] = eval_result["fit_score"]

        best_profile = max(scores, key=scores.get)
        best_score = scores[best_profile]

        reasoning = f"Based on current volatility ({metrics.volatility:.1f}%), beta ({metrics.beta:.2f}), "
        reasoning += f"and diversification ({metrics.diversification_score:.0f}/100), "
        reasoning += f"this portfolio best matches a {best_profile.value} investor profile (fit score: {best_score:.0f}/100)."

        return best_profile, reasoning


class RiskProfilerAgent:
    """
    Risk Profiler Agent that assesses portfolio-investor risk alignment.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.1,
        use_openai: bool = False,
        service=None
    ):
        """
        Initialize the Risk Profiler Agent.

        Args:
            model: Model name (Claude or GPT)
            temperature: Model temperature (0-1)
            use_openai: If True, use OpenAI instead of Anthropic
            service: Optional PortfolioService for direct integration
        """
        self.analyzer = PortfolioAnalyzer()
        self.profiler = RiskProfiler()
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

        # Create tools
        self.tools = self._create_tools()

        # Create agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent to use"""

        def evaluate_risk_fit_tool(portfolio_and_profile: str) -> str:
            """
            Evaluate if portfolio matches investor risk profile.
            Input format: "portfolio: 40% AAPL, 30% MSFT, 30% GOOGL | profile: conservative"
            """
            try:
                parts = portfolio_and_profile.split("|")
                if len(parts) != 2:
                    return "Invalid format. Use: 'portfolio: X% TICKER... | profile: conservative/moderate/aggressive'"

                portfolio_str = parts[0].replace("portfolio:", "").strip()
                profile_str = parts[1].replace("profile:", "").strip().lower()

                # Parse profile
                try:
                    target_profile = RiskProfile(profile_str)
                except ValueError:
                    return f"Invalid profile: {profile_str}. Use: conservative, moderate, or aggressive"

                # Analyze portfolio
                holdings = self.analyzer.parse_portfolio(portfolio_str)
                if not holdings:
                    return "Could not parse portfolio."

                metrics = self.analyzer.analyze_portfolio(holdings)

                # Evaluate fit
                result = self.profiler.evaluate_profile_fit(metrics, target_profile)

                output = f"""
Risk Profile Evaluation for {profile_str.upper()} investor:
==========================================================

Overall Fit Score: {result['fit_score']:.0f}/100

{result['recommendation']}

Current Portfolio Metrics:
- Volatility: {metrics.volatility:.1f}%
- Beta: {metrics.beta:.2f}
- Diversification: {metrics.diversification_score:.0f}/100
"""

                if result['mismatches']:
                    output += "\n\nMismatches Identified:\n"
                    for mismatch in result['mismatches']:
                        output += f"\n{mismatch['metric']} ({mismatch['severity'].upper()}):\n"
                        output += f"  Current: {mismatch['current']}\n"
                        output += f"  Target:  {mismatch['target']}\n"
                else:
                    output += "\n\nNo significant mismatches found. Portfolio is well-aligned!"

                return output

            except Exception as e:
                return f"Error evaluating risk fit: {str(e)}"

        def suggest_best_profile_tool(portfolio_str: str) -> str:
            """
            Suggest the most appropriate risk profile for a portfolio.
            Input: portfolio description like "40% AAPL, 30% MSFT, 30% GOOGL"
            """
            try:
                holdings = self.analyzer.parse_portfolio(portfolio_str)
                if not holdings:
                    return "Could not parse portfolio."

                metrics = self.analyzer.analyze_portfolio(holdings)

                suggested_profile, reasoning = self.profiler.suggest_profile(metrics)

                # Get detailed evaluation for suggested profile
                evaluation = self.profiler.evaluate_profile_fit(metrics, suggested_profile)

                output = f"""
Suggested Risk Profile: {suggested_profile.value.upper()}
=====================================================

{reasoning}

Why this profile fits:
- Current volatility: {metrics.volatility:.1f}% (benchmark: {self.profiler.RISK_BENCHMARKS[suggested_profile]['max_volatility']:.1f}%)
- Current beta: {metrics.beta:.2f} (benchmark: {self.profiler.RISK_BENCHMARKS[suggested_profile]['max_beta']:.2f})
- Diversification: {metrics.diversification_score:.0f}/100 (benchmark: >{self.profiler.RISK_BENCHMARKS[suggested_profile]['min_diversification']})

Alternative profiles fit scores:
"""
                # Show how other profiles would fit
                for profile in RiskProfile:
                    eval_result = self.profiler.evaluate_profile_fit(metrics, profile)
                    output += f"- {profile.value.capitalize()}: {eval_result['fit_score']:.0f}/100\n"

                return output

            except Exception as e:
                return f"Error suggesting profile: {str(e)}"

        def get_profile_benchmarks_tool(profile_name: str) -> str:
            """
            Get risk benchmarks for a specific profile.
            Input: profile name (conservative, moderate, or aggressive)
            """
            try:
                profile = RiskProfile(profile_name.lower())
                benchmarks = self.profiler.RISK_BENCHMARKS[profile]

                return f"""
Risk Benchmarks for {profile.value.upper()} Investor:
===================================================

Volatility Limits:
- Maximum Annual Volatility: {benchmarks['max_volatility']:.1f}%

Market Sensitivity:
- Maximum Beta: {benchmarks['max_beta']:.2f}

Diversification Requirements:
- Minimum Diversification Score: {benchmarks['min_diversification']}/100
- Maximum Sector Concentration: {benchmarks['max_sector_concentration']:.1f}%
- Maximum Single Position: {benchmarks['max_position_size']:.1f}%

Performance Target:
- Target Sharpe Ratio: >{benchmarks['target_sharpe']:.2f}

These benchmarks help ensure portfolios match investor risk tolerance and goals.
"""
            except ValueError:
                return f"Invalid profile: {profile_name}. Use: conservative, moderate, or aggressive"
            except Exception as e:
                return f"Error getting benchmarks: {str(e)}"

        tools = [
            Tool(
                name="evaluate_risk_fit",
                func=evaluate_risk_fit_tool,
                description=(
                    "Evaluates if a portfolio matches an investor's risk profile (conservative/moderate/aggressive). "
                    "Input format: 'portfolio: 40% AAPL, 30% MSFT | profile: conservative'. "
                    "Returns fit score, mismatches, and recommendations."
                )
            ),
            Tool(
                name="suggest_best_profile",
                func=suggest_best_profile_tool,
                description=(
                    "Suggests the most appropriate risk profile for a given portfolio. "
                    "Input: portfolio description like '40% AAPL, 30% MSFT, 30% GOOGL'. "
                    "Returns suggested profile with reasoning and fit scores for all profiles."
                )
            ),
            Tool(
                name="get_profile_benchmarks",
                func=get_profile_benchmarks_tool,
                description=(
                    "Gets risk benchmarks for a specific investor profile. "
                    "Input: profile name (conservative, moderate, or aggressive). "
                    "Returns detailed benchmarks for volatility, beta, diversification, etc."
                )
            )
        ]

        return tools

    def _create_agent(self):
        """Create the LangChain agent with system prompt"""

        system_prompt = """You are a Risk Profiler Agent, an expert in investor risk tolerance assessment and portfolio alignment.

Your role is to:
1. Evaluate if portfolios match investor risk profiles (conservative/moderate/aggressive)
2. Identify specific mismatches between portfolio risk and investor tolerance
3. Suggest appropriate risk profiles based on portfolio characteristics
4. Recommend adjustments to align portfolio risk with investor goals

Risk Profile Definitions:
- CONSERVATIVE: Low volatility (<12%), defensive positioning, high diversification
- MODERATE: Balanced risk-return, moderate volatility (<20%), market-like exposure
- AGGRESSIVE: Growth-focused, higher volatility acceptable (<35%), concentrated positions OK

When assessing portfolios:
- Use evaluate_risk_fit to check alignment with a stated profile
- Use suggest_best_profile to recommend the most appropriate profile
- Use get_profile_benchmarks to explain what each profile entails
- Be specific about which metrics are misaligned and by how much
- Provide actionable recommendations to improve alignment

Communication style:
- Empathetic and educational - risk tolerance is personal
- Clear about trade-offs (lower risk = lower expected return)
- Specific with numbers and thresholds
- Avoid jargon - explain terms like beta, volatility simply

Remember: You assess risk alignment, not suitability. You help investors understand if their portfolio matches their stated risk tolerance."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return agent

    def analyze(self, query: str, chat_history: Optional[List] = None) -> str:
        """
        Analyze risk profile alignment based on query.

        Args:
            query: User's question about risk profile
            chat_history: Optional conversation history

        Returns:
            Agent's response with analysis
        """
        try:
            response = self.agent_executor.invoke({
                "input": query,
                "chat_history": chat_history or []
            })
            return self._extract_output(response["output"])
        except Exception as e:
            return f"Error during analysis: {str(e)}"

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
    """Example usage of the Risk Profiler Agent"""
    print("Risk Profiler Agent")
    print("=" * 50)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env file")
        return

    agent = RiskProfilerAgent()

    # Example 1: Check if portfolio fits conservative profile
    print("\n[Example 1: Evaluate conservative fit]\n")
    response = agent.analyze(
        "Is this portfolio suitable for a conservative investor? Portfolio: 40% AAPL, 30% TSLA, 30% NVDA"
    )
    print(response)

    # Example 2: Suggest best profile
    print("\n\n[Example 2: Suggest best profile]\n")
    response = agent.analyze(
        "What risk profile best matches this portfolio: 30% VOO, 30% BND, 20% VTI, 20% VXUS"
    )
    print(response)


if __name__ == "__main__":
    main()
