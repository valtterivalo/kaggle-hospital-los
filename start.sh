#!/bin/bash

# healthcare ai predictor startup script
# starts both backend and frontend servers

# check dependencies
if ! command -v uv &> /dev/null; then
    echo "error: uv not found. install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command -v pnpm &> /dev/null; then
    echo "error: pnpm not found. install with: npm install -g pnpm"
    exit 1
fi

echo "starting healthcare ai predictor..."

# install backend dependencies
echo "installing backend dependencies..."
cd backend
uv sync --quiet
cd ..

# install frontend dependencies  
echo "installing frontend dependencies..."
cd frontend
pnpm install --silent
cd ..

# start backend server in background
echo "starting backend server on port 8000..."
cd backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# wait for backend to start
sleep 3

# start frontend server in background
echo "starting frontend server on port 3000..."
cd frontend
pnpm dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "healthcare ai predictor running:"
echo "  frontend: http://localhost:3000"
echo "  backend:  http://localhost:8000"
echo "  api docs: http://localhost:8000/docs"
echo ""
echo "press ctrl+c to stop both servers"

# trap ctrl+c and kill both processes
trap 'echo "stopping servers..."; kill $BACKEND_PID $FRONTEND_PID; exit 0' INT

# wait for both processes
wait $BACKEND_PID $FRONTEND_PID