"""FastAPI application entry point for Healthcare AI Predictor."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, predictions, data

app = FastAPI(
    title="Healthcare AI Predictor",
    description="Real-time length-of-stay prediction with model interpretability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(data.router, prefix="/api/data", tags=["data"])


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint returning basic API information."""
    return {"message": "Healthcare AI Predictor API", "version": "1.0.0"}