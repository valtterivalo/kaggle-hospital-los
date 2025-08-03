# Healthcare AI Predictor - Development Makefile

.PHONY: help setup data train dev-backend dev-frontend dev notebook export-model test clean

# Default target
help:
	@echo "Healthcare AI Predictor - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        Install all dependencies"
	@echo "  make data         Generate synthetic dataset"
	@echo "  make train        Train the ML model"
	@echo ""
	@echo "Development:"
	@echo "  make dev-backend  Start FastAPI server (port 8000)"
	@echo "  make dev-frontend Start Next.js server (port 3000)"
	@echo "  make dev          Start both servers in background"
	@echo ""
	@echo "ML Experimentation:"
	@echo "  make notebook     Start Jupyter Lab for model experiments"
	@echo "  make export-model Export improved model from notebook"
	@echo ""
	@echo "Utilities:"
	@echo "  make test         Run tests (when implemented)"
	@echo "  make clean        Clean generated files"
	@echo "  make kill-dev     Kill development servers"

# Setup commands
setup:
	@echo "Installing backend dependencies..."
	uv sync
	@echo "Installing frontend dependencies..."
	cd frontend && pnpm install
	@echo "✅ Setup complete!"

data:
	@echo "Generating synthetic healthcare dataset..."
	uv run python scripts/generate_synthetic_data.py
	@echo "Validating dataset..."
	uv run python scripts/download_data.py
	@echo "✅ Dataset ready!"

train:
	@echo "Training XGBoost model..."
	uv run python scripts/simple_train.py
	@echo "✅ Model training complete!"

# Development servers
dev-backend:
	@echo "Starting FastAPI server on http://localhost:8000"
	cd backend && PYTHONPATH=$(PWD)/backend uv run uvicorn app.main:app --reload --port 8000

dev-frontend:
	@echo "Starting Next.js server on http://localhost:3000"
	cd frontend && pnpm dev

dev:
	@echo "Starting both development servers..."
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo "API Docs: http://localhost:8000/docs"
	cd backend && PYTHONPATH=$(PWD)/backend uv run uvicorn app.main:app --reload --port 8000 &
	cd frontend && pnpm dev &
	@echo "✅ Both servers started in background"
	@echo "Use 'make kill-dev' to stop servers"

# Utilities
test:
	@echo "Running tests (placeholder - tests not yet implemented)"
	# uv run pytest backend/tests/
	# cd frontend && pnpm test

clean:
	@echo "Cleaning generated files..."
	rm -rf data/models/*.joblib
	rm -rf data/processed/*
	rm -rf backend/__pycache__
	rm -rf frontend/.next
	@echo "✅ Clean complete!"

kill-dev:
	@echo "Stopping development servers..."
	-pkill -f "uvicorn.*app.main"
	-pkill -f "next dev"
	-lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	-lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	@echo "✅ Development servers stopped"

# ML Experimentation
notebook:
	@echo "Starting Jupyter Lab for ML experimentation..."
	@echo "Notebook URL will open automatically"
	@echo "Open notebooks/01_model_experimentation.ipynb to get started"
	uv run jupyter lab notebooks/

export-model:
	@echo "Checking for improved model from notebook..."
	@if [ -f "data/models/improved_model.joblib" ]; then \
		echo "✅ Found improved model, copying to production..."; \
		cp data/models/improved_model.joblib data/models/simple_model.joblib; \
		echo "✅ Production model updated!"; \
	else \
		echo "❌ No improved model found. Run notebook experiments first."; \
		echo "Use 'make notebook' to start model experimentation."; \
	fi

# Full workflow
full-setup: setup data train
	@echo "🎉 Full setup complete! Ready for development."
	@echo "Run 'make dev' to start both servers"