"""Pydantic models for prediction requests and responses."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    """Input model for clinical patient data used in predictions."""
    
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., pattern="^(M|F)$", description="Patient gender (M/F)")
    facility: str = Field(..., pattern="^[A-E]$", description="Healthcare facility (A-E)")
    readmissions: int = Field(..., ge=0, le=5, description="Previous readmission count")
    medical_condition: str = Field(..., description="Primary medical condition")
    creatinine: float = Field(..., ge=0.2, le=5.0, description="Serum creatinine level (mg/dL)")
    glucose: int = Field(..., ge=50, le=300, description="Blood glucose level (mg/dL)")
    hematocrit: float = Field(..., ge=4.0, le=25.0, description="Hematocrit level (g/dL)")
    bun: int = Field(..., ge=5, le=100, description="Blood urea nitrogen (mg/dL)")
    bmi: float = Field(..., ge=15.0, le=50.0, description="Body mass index (kg/m²)")
    pulse: int = Field(..., ge=40, le=150, description="Heart rate (beats per minute)")
    respiration: int = Field(..., ge=8, le=30, description="Respiratory rate (breaths per minute)")
    sodium: int = Field(..., ge=125, le=155, description="Serum sodium level (mmol/L)")
    neutrophils: float = Field(..., ge=0.1, le=50.0, description="Neutrophil count (K/μL)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 65,
                "gender": "F",
                "facility": "A",
                "readmissions": 1,
                "medical_condition": "Pneumonia",
                "creatinine": 1.1,
                "glucose": 120,
                "hematocrit": 12.0,
                "bun": 18,
                "bmi": 28.5,
                "pulse": 80,
                "respiration": 18,
                "sodium": 140,
                "neutrophils": 8.5
            }
        }


class PredictionResponse(BaseModel):
    """Response model for length-of-stay predictions."""
    
    predicted_los: float = Field(..., description="Predicted length of stay in days")
    confidence_interval: List[float] = Field(..., description="95% confidence interval [lower, upper]")
    shap_values: Dict[str, float] = Field(..., description="SHAP feature importance values")
    explanation: str = Field(..., description="Human-readable explanation of the prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_los": 4.2,
                "confidence_interval": [3.1, 5.3],
                "shap_values": {
                    "age": 0.5,
                    "medical_condition": 1.2,
                    "admission_type": -0.3
                },
                "explanation": "Patient likely to stay 4.2 days based on age and medical condition"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    patients: List[PatientInput] = Field(..., description="List of patients for batch prediction")
    
    
class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")


class DataSummaryResponse(BaseModel):
    """Response model for dataset summary statistics."""
    
    total_records: int = Field(..., description="Total number of records")
    average_los: float = Field(..., description="Average length of stay")
    feature_stats: Dict[str, Dict[str, float]] = Field(..., description="Feature statistics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_records": 100000,
                "average_los": 4.8,
                "feature_stats": {
                    "age": {"mean": 45.2, "std": 18.5, "min": 0, "max": 120},
                    "los": {"mean": 4.8, "std": 3.2, "min": 1, "max": 30}
                }
            }
        }