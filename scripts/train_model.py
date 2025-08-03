"""Script to train the healthcare length-of-stay prediction model."""

import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.data_service import DataService
from app.services.ml_service import MLService
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Train the healthcare prediction model."""
    try:
        logger.info("🏥 Starting healthcare ML model training...")
        
        # Initialize services
        data_service = DataService()
        ml_service = MLService(data_service)
        
        # Validate data quality first
        logger.info("🔍 Validating data quality...")
        quality_report = data_service.validate_data_quality()
        
        if quality_report["issues"]:
            logger.warning(f"Data quality issues found: {quality_report['issues']}")
        
        if quality_report["warnings"]:
            logger.info(f"Data quality warnings: {quality_report['warnings']}")
        
        # Train model
        logger.info("🚀 Training XGBoost model...")
        training_results = ml_service.train_model(hyperparameter_tuning=False)  # Skip tuning for faster demo
        
        # Display results
        logger.info("📊 Training Results:")
        logger.info(f"  - Train R²: {training_results['train_r2']:.3f}")
        logger.info(f"  - Test R²: {training_results['test_r2']:.3f}")
        logger.info(f"  - Test RMSE: {training_results['test_metrics']['rmse']:.2f} days")
        logger.info(f"  - Test MAE: {training_results['test_metrics']['mae']:.2f} days")
        logger.info(f"  - Within 1 day accuracy: {training_results['test_metrics']['within_1_day']:.1f}%")
        
        # Show top features
        feature_importance = training_results["feature_importance"]
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        logger.info("🎯 Top 5 Most Important Features:")
        for feature, importance in top_features:
            logger.info(f"  - {feature}: {importance:.3f}")
        
        # Test prediction
        logger.info("🧪 Testing single prediction...")
        test_patient = {
            "age": 65,
            "gender": "Male",
            "admission_type": "Emergency",
            "medical_condition": "Pneumonia",
            "blood_type": "O+",
            "billing_amount": 18000
        }
        
        prediction_result = ml_service.predict_single(test_patient)
        logger.info(f"Test prediction result: {prediction_result['predicted_los']:.1f} days")
        logger.info(f"Explanation: {prediction_result['explanation']}")
        
        logger.info("✅ Model training completed successfully!")
        return training_results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()