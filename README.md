# DefiLab Uniswap V3 Simulator

Code for site built at https://defi-lab.xyz/uniswapv3simulator

## Setup

### 1. Get The Graph API Key

The app now uses The Graph's decentralized network (the hosted service has been deprecated).

1. Go to [The Graph Studio](https://thegraph.com/studio/)
2. Create an account or sign in
3. Navigate to "API Keys" section
4. Create a new API key

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
# REACT_APP_GRAPH_API_KEY=your_api_key_here
```

### 3. Set Up Python Backend (Required)

This project uses Python for backend services with **conda** for dependency management.

#### Quick Setup (Recommended)

```bash
cd backend
./setup_env.sh
```

This script will automatically:
- Check if conda is installed
- Create the conda environment from `environment.yml`
- Install all dependencies including PyTorch, FastAPI, and ML libraries

#### Manual Setup

```bash
cd backend

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate uniswap-v3-backend
```

#### Update Existing Environment

```bash
cd backend
conda activate uniswap-v3-backend
conda env update -f environment.yml --prune
```

### 4. Install and Run Frontend

```bash
npm install
npm start
```

**Note**: The `npm install` command will check if the Python virtual environment is set up and display a warning if it's missing.

## Migration Notes

This project has been migrated from The Graph's deprecated hosted service to the decentralized network. The following changes were made:

- **API Endpoints**: Updated to use `https://gateway.thegraph.com/api/[key]/subgraphs/id/[id]`
- **Subgraph IDs**: Using official Uniswap V3 subgraph IDs from [Uniswap documentation](https://docs.uniswap.org/api/subgraph/overview)
- **Environment Variables**: API key now required in `.env` file
- **Fallback**: Temporarily falls back to hosted service if no API key is provided (will stop working when hosted service is fully shutdown)

## Development

Currently work in progress. We are working on code refactor and will push commits as updates are made. Once all code is published, we'll update this README with more documentation about the code and a contribution guide.

