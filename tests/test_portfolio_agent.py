"""
Tests for Portfolio Insight Agent LangChain implementation.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.agents.portfolio_agent import PortfolioInsightAgent
from src.core.portfolio_metrics import PortfolioMetrics


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")


@pytest.fixture
def mock_portfolio_metrics():
    """Mock PortfolioMetrics for testing"""
    return PortfolioMetrics(
        annual_return=0.15,
        annual_volatility=0.20,
        sharpe_ratio=0.55,
        max_drawdown=-0.15,
        beta=1.1,
        var_95=0.03,
        cvar_95=0.04,
        holdings={"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3},
        total_value=1.0,
        period_days=252
    )


class TestPortfolioInsightAgent:
    """Test suite for PortfolioInsightAgent"""

    @patch('src.agents.portfolio_agent.ChatAnthropic')
    def test_agent_initialization_anthropic(self, mock_chat, mock_env_vars):
        """Test agent initialization with Anthropic"""
        agent = PortfolioInsightAgent(use_openai=False)

        assert agent.analyzer is not None
        assert agent.use_openai is False
        assert len(agent.tools) == 2
        mock_chat.assert_called_once()

    @patch('src.agents.portfolio_agent.ChatOpenAI')
    def test_agent_initialization_openai(self, mock_chat, mock_env_vars):
        """Test agent initialization with OpenAI"""
        agent = PortfolioInsightAgent(use_openai=True)

        assert agent.analyzer is not None
        assert agent.use_openai is True
        assert len(agent.tools) == 2
        mock_chat.assert_called_once()

    @patch('src.agents.portfolio_agent.ChatAnthropic')
    def test_tools_creation(self, mock_chat, mock_env_vars):
        """Test that tools are created correctly"""
        agent = PortfolioInsightAgent()

        assert len(agent.tools) == 2

        tool_names = [tool.name for tool in agent.tools]
        assert "analyze_portfolio" in tool_names
        assert "get_correlations" in tool_names

    @patch('src.agents.portfolio_agent.ChatAnthropic')
    def test_format_metrics(self, mock_chat, mock_env_vars, mock_portfolio_metrics):
        """Test metrics formatting"""
        agent = PortfolioInsightAgent()

        formatted = agent._format_metrics(mock_portfolio_metrics)

        assert isinstance(formatted, str)
        assert "AAPL" in formatted
        assert "15.00%" in formatted  # Annual return
        assert "20.00%" in formatted  # Volatility
        assert "0.550" in formatted   # Sharpe ratio
        assert "1.100" in formatted   # Beta

    @patch('src.agents.portfolio_agent.ChatAnthropic')
    @patch('src.agents.portfolio_agent.PortfolioAnalyzer')
    def test_analyze_portfolio_tool(self, mock_analyzer_class, mock_chat, mock_env_vars, mock_portfolio_metrics):
        """Test analyze_portfolio tool execution"""
        # Setup mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.parse_portfolio.return_value = {"AAPL": 0.5, "MSFT": 0.5}
        mock_analyzer.analyze_portfolio.return_value = mock_portfolio_metrics
        mock_analyzer_class.return_value = mock_analyzer

        agent = PortfolioInsightAgent()
        agent.analyzer = mock_analyzer

        # Get the analyze_portfolio tool
        analyze_tool = next(t for t in agent.tools if t.name == "analyze_portfolio")

        # Execute the tool
        result = analyze_tool.func("50% AAPL, 50% MSFT")

        assert isinstance(result, str)
        assert "Portfolio Analysis Results" in result
        mock_analyzer.parse_portfolio.assert_called_once()
        mock_analyzer.analyze_portfolio.assert_called_once()

    @patch('src.agents.portfolio_agent.ChatAnthropic')
    @patch('src.agents.portfolio_agent.PortfolioAnalyzer')
    def test_analyze_portfolio_tool_parsing_error(self, mock_analyzer_class, mock_chat, mock_env_vars):
        """Test analyze_portfolio tool with invalid input"""
        mock_analyzer = Mock()
        mock_analyzer.parse_portfolio.return_value = {}
        mock_analyzer_class.return_value = mock_analyzer

        agent = PortfolioInsightAgent()
        agent.analyzer = mock_analyzer

        analyze_tool = next(t for t in agent.tools if t.name == "analyze_portfolio")
        result = analyze_tool.func("invalid portfolio")

        assert "Could not parse portfolio" in result

    @patch('src.agents.portfolio_agent.ChatAnthropic')
    @patch('src.agents.portfolio_agent.PortfolioAnalyzer')
    def test_get_correlations_tool(self, mock_analyzer_class, mock_chat, mock_env_vars):
        """Test get_correlations tool execution"""
        import pandas as pd

        mock_analyzer = Mock()
        mock_corr = pd.DataFrame({
            'AAPL': [1.0, 0.8],
            'MSFT': [0.8, 1.0]
        }, index=['AAPL', 'MSFT'])
        mock_analyzer.get_asset_correlations.return_value = mock_corr
        mock_analyzer_class.return_value = mock_analyzer

        agent = PortfolioInsightAgent()
        agent.analyzer = mock_analyzer

        corr_tool = next(t for t in agent.tools if t.name == "get_correlations")
        result = corr_tool.func("AAPL,MSFT")

        assert isinstance(result, str)
        assert "AAPL" in result
        assert "MSFT" in result
        mock_analyzer.get_asset_correlations.assert_called_once()

    @patch('src.agents.portfolio_agent.ChatAnthropic')
    def test_agent_executor_exists(self, mock_chat, mock_env_vars):
        """Test that agent executor is created"""
        agent = PortfolioInsightAgent()

        assert agent.agent_executor is not None
        assert hasattr(agent.agent_executor, 'invoke')

    @patch('src.agents.portfolio_agent.ChatAnthropic')
    @patch.object(PortfolioInsightAgent, 'analyze')
    def test_chat_method(self, mock_analyze, mock_chat, mock_env_vars):
        """Test chat method for conversation"""
        mock_analyze.return_value = "This is a test response"

        agent = PortfolioInsightAgent()
        response, history = agent.chat("Test message", chat_history=[])

        assert response == "This is a test response"
        assert len(history) == 2  # User message + AI response
        mock_analyze.assert_called_once()

    @patch('src.agents.portfolio_agent.ChatAnthropic')
    def test_agent_error_handling(self, mock_chat, mock_env_vars):
        """Test error handling in agent"""
        agent = PortfolioInsightAgent()

        # Mock the executor to raise an error
        agent.agent_executor.invoke = Mock(side_effect=Exception("Test error"))

        result = agent.analyze("Test query")

        assert "Error during analysis" in result
        assert "Test error" in result


class TestAgentTools:
    """Test individual agent tools"""

    @patch('src.agents.portfolio_agent.ChatAnthropic')
    @patch('src.agents.portfolio_agent.PortfolioAnalyzer')
    def test_tool_descriptions(self, mock_analyzer_class, mock_chat, mock_env_vars):
        """Test that tools have proper descriptions"""
        agent = PortfolioInsightAgent()

        for tool in agent.tools:
            assert tool.name is not None
            assert tool.description is not None
            assert len(tool.description) > 20  # Meaningful description

    @patch('src.agents.portfolio_agent.ChatAnthropic')
    def test_tool_error_handling(self, mock_chat, mock_env_vars):
        """Test error handling within tools"""
        agent = PortfolioInsightAgent()

        # Force an error in the analyzer
        agent.analyzer.parse_portfolio = Mock(side_effect=Exception("Parse error"))

        analyze_tool = next(t for t in agent.tools if t.name == "analyze_portfolio")
        result = analyze_tool.func("50% AAPL, 50% MSFT")

        assert "Error analyzing portfolio" in result


class TestSystemPrompt:
    """Test agent system prompt configuration"""

    @patch('src.agents.portfolio_agent.ChatAnthropic')
    def test_system_prompt_content(self, mock_chat, mock_env_vars):
        """Test that system prompt guides agent behavior"""
        agent = PortfolioInsightAgent()

        # The agent should be created with a proper prompt
        assert agent.agent is not None

        # Verify agent was created with tools
        assert len(agent.tools) > 0


class TestIntegration:
    """Integration tests for the agent"""

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="No API key available for integration test"
    )
    @patch('src.agents.portfolio_agent.PortfolioAnalyzer')
    def test_full_agent_workflow(self, mock_analyzer_class, mock_portfolio_metrics):
        """Test complete agent workflow with real LLM (if API key available)"""
        mock_analyzer = Mock()
        mock_analyzer.parse_portfolio.return_value = {"AAPL": 0.5, "MSFT": 0.5}
        mock_analyzer.analyze_portfolio.return_value = mock_portfolio_metrics
        mock_analyzer_class.return_value = mock_analyzer

        agent = PortfolioInsightAgent()
        agent.analyzer = mock_analyzer

        # This would make a real API call
        response = agent.analyze("Analyze portfolio: 50% AAPL, 50% MSFT")

        assert isinstance(response, str)
        assert len(response) > 0
