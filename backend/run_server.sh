#!/bin/bash
# Backend Server Startup Script

cd /home/zekiya/liquidity/uniswap-v3-simulator/backend

source /home/zekiya/miniconda3/etc/profile.d/conda.sh
conda activate uniswap-v3-backend

echo "Starting FastAPI backend server..."
echo "Server will be available at http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
