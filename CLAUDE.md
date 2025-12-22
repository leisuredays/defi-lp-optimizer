# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DefiLab Uniswap V3 Simulator is a React application for simulating and backtesting Uniswap V3 liquidity provider strategies. The app allows users to analyze concentrated liquidity positions, compare strategies, and backtest performance using historical data from The Graph protocol.

Live site: https://defi-lab.xyz/uniswapv3simulator

## Development Commands

### Initial Setup

1. **Configure The Graph API Key**
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env and add your Graph API key
   # Get your API key from: https://thegraph.com/studio/
   ```

   Required environment variables:
   - `REACT_APP_GRAPH_API_KEY`: Your API key from The Graph Studio

2. **Install Dependencies**
   ```bash
   npm install
   ```

### Start Development Server
```bash
npm start
# Note: Uses --openssl-legacy-provider flag for compatibility
```

### Build for Production
```bash
npm build
```

### Run Tests
```bash
npm test
```

## Python Backend Environment

**CRITICAL**: This project uses conda for Python dependency management. All Python commands MUST be executed within the conda environment.

### Conda Environment Setup

- **Environment Name**: `uniswap-v3-backend`
- **Python Version**: 3.9
- **Configuration File**: `backend/environment.yml`

### First-time Setup

```bash
cd backend
./setup_env.sh
```

### Running Python Commands in Claude Code

**MANDATORY PATTERN**: Always use this exact pattern when executing ANY Python-related command:

```bash
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend && <your-command>
```

**Examples:**

```bash
# Install a package
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend && pip install package-name

# Run a Python script
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend && python script.py

# Start FastAPI server
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend && uvicorn app.main:app --reload

# Run tests
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend && pytest

# Check installed packages
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend && pip list
```

### Important Rules

1. **NEVER** run `pip install` or `python` commands without activating the conda environment first
2. **NEVER** use bare `pip` or `python` commands - always prepend with the conda activation
3. The `pip.conf` file enforces virtual environment usage and will block installations outside conda
4. When working on backend tasks, ALWAYS change directory to `backend/` and activate the environment

### Environment Management

```bash
# List all conda environments
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda env list

# Update environment after environment.yml changes
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend && conda env update -f environment.yml --prune

# Check current environment
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda info --envs
```

### Key Python Dependencies

- **FastAPI** (0.104.1): Web API framework
- **PyTorch** (2.5.1): Deep learning framework with CUDA 12.4 support (conda-managed)
- **stable-baselines3** (2.1.0): Reinforcement learning library
- **NumPy** (1.24.3): Numerical computing (conda-managed)
- **Pandas** (2.0.3): Data analysis (conda-managed)

For full dependency list, see `backend/environment.yml` and `backend/requirements.txt`

### Machine Learning Model Training

**CRITICAL**: All model training MUST use GPU acceleration for acceptable performance.

#### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (Currently: NVIDIA GeForce RTX 4070 SUPER, 12GB VRAM)
- **CUDA Version**: 12.4 (via PyTorch)
- **Driver**: NVIDIA driver 581.29+ (CUDA 13.0 compatible)

#### GPU Verification

Before training, verify GPU is available:

```bash
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend && python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU device:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA available: True
GPU device: NVIDIA GeForce RTX 4070 SUPER
```

#### Training Commands

```bash
# PPO model training (MUST use GPU)
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend && python scripts/train_ppo_single_pool.py

# Data collection
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend && python scripts/collect_single_pool_data.py
```

#### Performance Expectations

- **CPU Training**: ~1,000-1,500 steps/sec (~60-90 minutes for 5M steps) - **NOT RECOMMENDED**
- **GPU Training**: ~10,000-20,000 steps/sec (~5-10 minutes for 5M steps) - **REQUIRED**

The training configuration (`config/training_config.yaml`) uses `device: "auto"` which automatically selects GPU when available.

#### Troubleshooting GPU Issues

If CUDA is not available:

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA support
source /home/zekiya/miniconda3/etc/profile.d/conda.sh && conda activate uniswap-v3-backend && conda uninstall -y pytorch torchvision torchaudio && conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## Architecture

### State Management (Redux Toolkit)

The application uses Redux Toolkit with the following store slices (see `src/store.js`):

- **pool**: Pool data, token info, liquidity data, and daily statistics from The Graph
- **protocol**: Selected protocol (Ethereum, Polygon, Arbitrum, Optimism, Celo)
- **investment**: Investment amount configuration
- **strategyRanges**: Min/max price ranges for liquidity strategies (S1, S2, v2/unbounded)
- **strategies**: Strategy definitions and visual properties
- **window**: Window dimensions for responsive layouts
- **tokenRatios**: Token composition ratios for strategies

### Core Modules

#### The Graph API Integration (`src/api/thegraph/`)

GraphQL queries for fetching Uniswap V3 data using The Graph's decentralized network:
- `uniPools.js`: Pool data, TVL, prices, and pool search
- `uniPoolDayDatas.js`: Daily historical price and volume data
- `uniPoolHourDatas.js`: Hourly candle data for backtesting
- `uniTicks.js`: Liquidity distribution across price ticks
- `helpers.js`: Protocol URLs, API key configuration, and subgraph ID mapping

**Subgraph IDs** (from Uniswap official documentation):
- Ethereum Mainnet: `5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV`
- Optimism: `Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj`
- Arbitrum: `FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM`
- Polygon: `3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm`
- Celo: `ESdrTJ3twMwWVoQ1hUE2u7PugEHX3QkenudD6aXCkDQ4`

**API Authentication**: The app uses The Graph's decentralized network with API keys from The Graph Studio. If no API key is provided, it falls back to deprecated hosted service endpoints (may stop working).

#### Strategy Calculations (`src/helpers/uniswap/`)

Core mathematical models for Uniswap V3:

- **strategies.js**: Position value calculations
  - `strategyV3()`: Concentrated liquidity position value across price ranges
  - `hodl5050()`, `hodlToken1()`, `hodlToken2()`: HODL strategy comparisons
  - `V2Unbounded()`: Uniswap V2 style unbounded liquidity
  - `V3ImpLossData()`: Impermanent loss calculations
  - `genTokenRatios()`: Token composition at given prices

- **liquidity.js**: Liquidity math
  - `roundToNearestTick()`: Snap prices to valid tick boundaries
  - `calcLiquidity0()`, `calcLiquidity1()`: Liquidity from token amounts
  - `tokensFromLiquidity()`: Token amounts from liquidity at current price
  - `activeLiquidityForCandle()`: Determine if position is in range for a time period

- **backtest.js**: Historical performance simulation
  - `calcFees()`: Estimate fees earned based on hourly data, active liquidity, and fee growth
  - `pivotFeeData()`: Aggregate hourly fee data into daily summaries
  - `backtestIndicators()`: Calculate ROI, APR, confidence metrics

#### Layout Components (`src/layout/`)

Main UI sections that compose the dashboard:
- `StrategyOverview.js`: Strategy comparison chart and controls
- `StrategyBacktest.js`: Historical backtest results and metrics
- `PoolPriceLiquidity.js`: Price chart and liquidity distribution
- `PoolOverview.js`: Pool information and token details
- `SideBar.js`: Pool search, protocol selector, investment inputs
- `NavBar.js`: Top navigation with protocol branding

### Key Patterns

#### Pool Data Flow

1. User searches for pool via `PoolSearch` component
2. `fetchPoolData()` thunk (in `src/store/pool.js`) fetches:
   - Tick liquidity data
   - Daily price history
   - Calculates statistics (std dev, mean)
3. Pool data propagates to all components via Redux selectors
4. Strategy ranges auto-initialize based on pool statistics

#### Strategy Range Updates

When user adjusts min/max price ranges:
1. Input value validated and rounded to nearest valid tick (`roundToNearestTick`)
2. Redux action updates `strategyRanges` slice
3. Charts re-render using `strategyV3()` with new ranges
4. Token ratios and liquidity multipliers recalculate

#### Backtest Calculation

Backtest flow (triggered when pool/strategy changes):
1. Fetch hourly pool data from The Graph
2. For each hour, calculate:
   - Whether position is in range (using tick boundaries)
   - Fee growth per hour based on global fee growth
   - Tokens held at that price point
3. Aggregate to daily data
4. Calculate final metrics (APR, total return, confidence score)

Note: Arbitrum backtests are disabled due to data accuracy issues (see `UniswapSimulator.js:110`)

#### Base Token Toggle

Users can flip the price axis between token0/token1:
- Updates both `pool.baseToken` and `pool.quoteToken`
- Inverts all price calculations
- Updates strategy range inputs
- Preserves liquidity position value

## Important Codebase Conventions

### Price and Decimal Handling

- Prices from The Graph include decimal adjustments
- `price0N` and `price1N` calculations account for token decimal differences
- Always use `Math.pow(10, decimals)` for token amount conversions
- Tick prices: `Math.pow(1.0001, tickIdx)`

### Fee Tier and Tick Spacing

- Fee tiers (500, 3000, 10000 bps) determine valid tick spacing
- Tick spacing = feeTier / 50
- All price inputs must snap to valid ticks for on-chain accuracy

### Liquidity Math

Uniswap V3 uses concentrated liquidity formulas:
- Virtual liquidity: `L = sqrt(x * y)` at current price
- Token amounts depend on whether price is in range
- See `strategyV3()` for full implementation of position value across price ranges

### Data Fetching

- All The Graph queries use AbortController for cleanup
- Protocol ID determines which subgraph endpoint to use
- Pool data includes nested `poolDayData` for historical context

## Testing Notes

- Single test file exists: `src/App.test.js`
- Uses @testing-library/react
- Run with `npm test`
