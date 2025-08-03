"""Pydantic models for prediction requests and responses."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    """Input model for complete clinical patient data matching Kaggle dataset schema."""
    
    # Demographics
    gender: str = Field(..., pattern="^(M|F)$", description="Patient gender (M/F)")
    facid: int = Field(..., ge=1, le=5, description="Facility ID (1-5)")
    rcount: int = Field(..., ge=0, le=5, description="Readmissions within 180 days")
    
    # Clinical Conditions (binary flags)
    dialysisrenalendstage: int = Field(..., ge=0, le=1, description="End-stage renal disease flag")
    asthma: int = Field(..., ge=0, le=1, description="Asthma flag")
    irondef: int = Field(..., ge=0, le=1, description="Iron deficiency flag")
    pneum: int = Field(..., ge=0, le=1, description="Pneumonia flag")
    substancedependence: int = Field(..., ge=0, le=1, description="Substance dependence flag")
    psychologicaldisordermajor: int = Field(..., ge=0, le=1, description="Major psychological disorder flag")
    depress: int = Field(..., ge=0, le=1, description="Depression flag")
    psychother: int = Field(..., ge=0, le=1, description="Other psychological disorder flag")
    fibrosisandother: int = Field(..., ge=0, le=1, description="Fibrosis and other conditions flag")
    malnutrition: int = Field(..., ge=0, le=1, description="Malnutrition flag")
    hemo: int = Field(..., ge=0, le=1, description="Blood disorder flag")
    secondarydiagnosisnonicd9: int = Field(..., ge=0, le=10, description="Non-ICD9 secondary diagnoses count")
    
    # Laboratory Values (continuous)
    hematocrit: float = Field(..., ge=4.0, le=25.0, description="Hematocrit level (g/dL)")
    neutrophils: float = Field(..., ge=0.1, le=50.0, description="Neutrophil count (cells/μL)")
    sodium: int = Field(..., ge=125, le=155, description="Sodium level (mmol/L)")
    glucose: float = Field(..., ge=2.0, le=20.0, description="Glucose level (mmol/L)")
    bloodureanitro: int = Field(..., ge=5, le=100, description="Blood urea nitrogen (mg/dL)")
    creatinine: float = Field(..., ge=0.2, le=5.0, description="Creatinine level (mg/dL)")
    
    # Vital Signs (continuous)
    bmi: float = Field(..., ge=15.0, le=50.0, description="Body mass index (kg/m²)")
    pulse: int = Field(..., ge=40, le=150, description="Heart rate (beats/min)")
    respiration: int = Field(..., ge=8, le=30, description="Respiratory rate (breaths/min)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "F",
                "facid": 1,
                "rcount": 1,
                "dialysisrenalendstage": 0,
                "asthma": 0,
                "irondef": 0,
                "pneum": 1,
                "substancedependence": 0,
                "psychologicaldisordermajor": 0,
                "depress": 0,
                "psychother": 0,
                "fibrosisandother": 0,
                "malnutrition": 0,
                "hemo": 0,
                "secondarydiagnosisnonicd9": 2,
                "hematocrit": 12.0,
                "neutrophils": 8.5,
                "sodium": 140,
                "glucose": 6.7,
                "bloodureanitro": 18,
                "creatinine": 1.1,
                "bmi": 28.5,
                "pulse": 80,
                "respiration": 18
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