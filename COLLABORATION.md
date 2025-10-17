# Team Collaboration Guide

## Team Members

- **Niki Choksi** (nikichoksi) - Portfolio Insight Agent ✅
- **Swara Joshi** (Swara-Joshi) - Risk Profiler Agent
- **Princy** (princyy23) - Scenario Simulator Agent

## Branching Strategy

```
main (production code)
  │
  ├── development (integration)
  │     │
  │     ├── swara (Swara's work)
  │     │
  │     └── princy (Princy's work)
```

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/nikichoksi/portfolio-trading-platform.git
cd portfolio-trading-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup .env file
cp .env.example .env
# Add your API keys
```

### 2. Create Your Branch

**Swara:**
```bash
git checkout development
git pull origin development
git checkout -b swara
git push -u origin swara
```

**Princy:**
```bash
git checkout development
git pull origin development
git checkout -b princy
git push -u origin princy
```

## Daily Workflow

### Start Working

```bash
# 1. Go to your branch
git checkout swara  # or princy

# 2. Get latest changes
git pull origin development

# 3. Code!
```

### Save Your Work

```bash
# 1. See changes
git status

# 2. Add files
git add -A

# 3. Commit
git commit -m "feat: Added risk assessment function"

# 4. Push
git push origin swara  # or princy
```

### Commit Message Tips

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Adding tests

Examples:
- `feat: Add Monte Carlo simulation`
- `fix: Correct volatility calculation`
- `docs: Update API documentation`

## Merging to Development

### Create Pull Request

1. Push your branch: `git push origin swara`
2. Go to: https://github.com/nikichoksi/portfolio-trading-platform
3. Click "Pull requests" → "New pull request"
4. Select: `development` ← `swara` (or `princy`)
5. Write description
6. Request review from teammates
7. Wait for approval
8. Merge!

## Avoiding Conflicts

### Work in Your Own Files

**Swara's Files:**
```
src/agents/risk_profiler_agent.py
src/utils/risk_profiling.py
tests/test_risk_profiler.py
```

**Princy's Files:**
```
src/agents/scenario_simulator_agent.py
src/utils/scenario_simulation.py
tests/test_scenario_simulator.py
```

### Shared Files (Coordinate!)

- `src/app_trading.py` - Talk before editing
- `requirements.txt` - Announce when adding packages

### If You Get Conflicts

```bash
git checkout development
git pull origin development
git checkout swara  # your branch
git merge development

# Fix conflicts in files
git add [fixed-files]
git commit -m "merge: Resolve conflicts"
git push origin swara
```

## Communication

- **Before editing shared files**: Message group chat
- **When stuck**: Ask for help!
- **Before merging**: Give teammates heads up

## Testing

```bash
# Test your agent
python src/agents/risk_profiler_agent.py

# Run all tests
pytest

# Run the app
streamlit run src/app_trading.py
```

## Quick Commands

```bash
# Check current branch
git branch

# See changes
git status

# Pull latest
git pull origin development

# Switch branch
git checkout swara

# Undo last commit (keep changes)
git reset --soft HEAD~1
```

## Project Structure

```
src/
  ├── agents/
  │   ├── portfolio_agent.py          # Niki ✅
  │   ├── risk_profiler_agent.py      # Swara
  │   └── scenario_simulator_agent.py # Princy
  ├── utils/
  │   ├── portfolio_analytics.py      # Shared
  │   ├── risk_profiling.py           # Swara
  │   └── scenario_simulation.py      # Princy
  └── app_trading.py                  # Shared - coordinate!
```

## What Each Agent Does

### Risk Profiler Agent (Swara)
- Assess portfolio risk level
- Compare to investor risk tolerance
- Suggest adjustments

### Scenario Simulator Agent (Princy)
- Simulate market crashes
- Run Monte Carlo simulations
- Calculate stress test results

## Need Help?

Ask in the group chat!

Happy coding! 🚀
