"""Health check API endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Response model for health checks."""
    
    status: str
    message: str


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(status="healthy", message="Healthcare AI Predictor API is running")


@router.get("/detailed", response_model=dict)
async def detailed_health_check() -> dict[str, str | bool]:
    """Detailed health check with service status."""
    # TODO: Add actual service checks (database, model loading, etc.)
    return {
        "status": "healthy",
        "api": True,
        "model_loaded": True,  # Will be updated when ML service is implemented
        "data_available": True,  # Will be updated when data service is implemented
    }