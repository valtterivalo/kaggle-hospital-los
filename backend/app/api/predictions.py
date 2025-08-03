"""Prediction API endpoints for healthcare length-of-stay predictions."""

from fastapi import APIRouter, HTTPException
from app.models.prediction import (
    PatientInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from app.services.ml_service import MLService

router = APIRouter()

# Initialize ML service
ml_service = MLService()


@router.post("/single", response_model=PredictionResponse)
async def predict_single(patient: PatientInput) -> PredictionResponse:
    """Generate a single length-of-stay prediction with explanation."""
    try:
        # Convert Pydantic model to dict
        patient_dict = patient.model_dump()
        
        # Get prediction from ML service
        result = ml_service.predict_single(patient_dict)
        
        return PredictionResponse(
            predicted_los=result["predicted_los"],
            confidence_interval=result["confidence_interval"],
            shap_values=result["shap_values"],
            explanation=result["explanation"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Generate batch length-of-stay predictions."""
    if len(request.patients) > 1000:
        raise HTTPException(
            status_code=400, 
            detail="Batch size too large. Maximum 1000 patients per request."
        )
    
    try:
        # Convert to list of dicts
        patients_data = [patient.model_dump() for patient in request.patients]
        
        # Get batch predictions
        results = ml_service.predict_batch(patients_data)
        
        # Convert to response format
        predictions = []
        for result in results:
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            
            predictions.append(PredictionResponse(
                predicted_los=result["predicted_los"],
                confidence_interval=result["confidence_interval"],
                shap_values=result["shap_values"],
                explanation=result["explanation"]
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/model-info")
async def get_model_info():
    """Get information about the current model."""
    return ml_service.get_model_info()