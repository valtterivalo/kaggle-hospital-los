"""Data exploration and querying API endpoints."""

from fastapi import APIRouter
from app.models.prediction import DataSummaryResponse

router = APIRouter()


@router.get("/summary", response_model=DataSummaryResponse)
async def get_data_summary() -> DataSummaryResponse:
    """Get summary statistics of the healthcare dataset."""
    # TODO: Implement actual data summary with data service
    mock_summary = DataSummaryResponse(
        total_records=100000,
        average_los=4.8,
        feature_stats={
            "age": {"mean": 45.2, "std": 18.5, "min": 0, "max": 120},
            "length_of_stay": {"mean": 4.8, "std": 3.2, "min": 1, "max": 30},
            "billing_amount": {"mean": 15000.0, "std": 8500.0, "min": 500, "max": 75000}
        }
    )
    return mock_summary


@router.get("/explore/{feature}")
async def explore_feature(feature: str) -> dict:
    """Explore a specific feature in the dataset."""
    # TODO: Implement actual feature exploration
    return {
        "feature": feature,
        "type": "categorical" if feature in ["gender", "medical_condition", "admission_type"] else "numerical",
        "unique_values": 10 if feature == "medical_condition" else None,
        "null_percentage": 2.5,
        "message": f"Feature exploration for {feature} - implementation pending"
    }