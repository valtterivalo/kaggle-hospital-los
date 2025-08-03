"""Machine learning service for healthcare length-of-stay prediction."""

import joblib
import shap
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from app.config import settings
from app.services.data_service import DataService

logger = logging.getLogger(__name__)


class MLService:
    """Service for machine learning model training, prediction, and explanation."""
    
    def __init__(self, data_service: Optional[DataService] = None):
        """Initialize MLService with optional data service dependency."""
        self.data_service = data_service  # Only needed for training, not prediction
        self.model: Optional[XGBRegressor] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: Optional[List[str]] = None
        self.scaler: Optional[StandardScaler] = None
        self.model_metadata: Dict[str, Any] = {}
        
        # Use relative paths from backend directory to data/models
        model_dir = Path("../data/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load our optimized clinical model first
        self.clinical_model_path = model_dir / "clinical_optimized_model.joblib"
        self.model_path = model_dir / f"{settings.model_name}_{settings.model_version}.joblib"
        self.explainer_path = model_dir / f"{settings.model_name}_explainer_{settings.model_version}.joblib"
        self.metadata_path = model_dir / f"{settings.model_name}_metadata_{settings.model_version}.joblib"
        
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for model training."""
        if self.data_service is None:
            raise ValueError("DataService required for training. Initialize with DataService for training operations.")
        
        logger.info("Preparing training data...")
        
        # Get ML-ready data from data service
        ml_data, feature_names = self.data_service.prepare_ml_data()
        
        # Split features and target
        X = ml_data.select(feature_names).to_numpy()
        y = ml_data.select("length_of_stay").to_numpy().ravel()
        
        logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names
    
    def train_model(self, hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """Train XGBoost model with optional hyperparameter tuning."""
        logger.info("Starting model training...")
        
        # Prepare data
        X, y, feature_names = self.prepare_training_data()
        self.feature_names = feature_names
        
        # Split data for training/testing
        ml_data, _ = self.data_service.prepare_ml_data()
        train_data, test_data = self.data_service.split_data(ml_data, test_size=0.2, temporal_split=False)
        
        # Prepare train/test splits
        X_train = train_data.select(feature_names).to_numpy()
        y_train = train_data.select("length_of_stay").to_numpy().ravel()
        X_test = test_data.select(feature_names).to_numpy()
        y_test = test_data.select("length_of_stay").to_numpy().ravel()
        
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            self.model = self._hyperparameter_tuning(X_train, y_train)
        else:
            # Use default parameters optimized for healthcare data
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1
            )
            
        # Train the model
        logger.info("Training final model...")
        self.model.fit(X_train, y_train)
        
        # Create SHAP explainer
        logger.info("Creating SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate comprehensive metrics
        training_results = {
            "timestamp": datetime.now().isoformat(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": len(feature_names),
            "feature_names": feature_names,
            "train_metrics": self._calculate_metrics(y_train, y_pred_train),
            "test_metrics": self._calculate_metrics(y_test, y_pred_test),
            "train_r2": train_score,
            "test_r2": test_score,
            "feature_importance": dict(zip(feature_names, self.model.feature_importances_)),
        }
        
        # Store metadata
        self.model_metadata = training_results
        
        # Save model and components
        self.save_model()
        
        logger.info(f"Model training completed. Test R²: {test_score:.3f}")
        return training_results
    
    def _hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> XGBRegressor:
        """Perform hyperparameter tuning using GridSearchCV."""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
        }
        
        base_model = XGBRegressor(random_state=42, n_jobs=-1)
        
        # Use 3-fold CV due to potentially smaller dataset
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=3, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        return {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100,
            "within_1_day": np.mean(np.abs(y_true - y_pred) <= 1) * 100,
            "within_2_days": np.mean(np.abs(y_true - y_pred) <= 2) * 100,
        }
    
    def predict_single(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate single prediction with SHAP explanation."""
        if self.model is None or self.explainer is None:
            self.load_model()
        
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train the model first.")
        
        # Convert patient data to model input format
        X_input = self._prepare_prediction_input(patient_data)
        
        # Make prediction
        prediction = self.model.predict(X_input)[0]
        
        # Calculate SHAP values for explanation
        shap_values = self.explainer.shap_values(X_input)[0]
        
        # Create feature importance dictionary
        shap_dict = dict(zip(self.feature_names, shap_values))
        
        # Generate human-readable explanation
        explanation = self._generate_explanation(patient_data, shap_dict, prediction)
        
        # Calculate confidence interval (rough estimate)
        confidence_interval = self._estimate_confidence_interval(X_input, prediction)
        
        return {
            "predicted_los": float(prediction),
            "confidence_interval": confidence_interval,
            "shap_values": {k: float(v) for k, v in shap_dict.items()},
            "explanation": explanation,
            "model_version": settings.model_version,
            "prediction_timestamp": datetime.now().isoformat()
        }
    
    def predict_batch(self, patients_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate batch predictions with explanations."""
        predictions = []
        for patient_data in patients_data:
            try:
                prediction = self.predict_single(patient_data)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting for patient {patient_data}: {e}")
                predictions.append({
                    "error": str(e),
                    "patient_data": patient_data
                })
        
        return predictions
    
    def _prepare_prediction_input(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """Convert patient data dictionary to model input format."""
        if self.feature_names is None:
            raise ValueError("Feature names not available. Please train or load model first.")
        
        # Check if this is the clinical model (has clinical features)
        is_clinical_model = any('creatinine' in feature or 'glucose' in feature for feature in self.feature_names)
        
        if is_clinical_model:
            return self._prepare_clinical_input(patient_data)
        else:
            return self._prepare_legacy_input(patient_data)
    
    def _prepare_clinical_input(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """Prepare input for clinical model with lab values."""
        input_row = {}
        
        # Clinical lab values with realistic defaults
        clinical_defaults = {
            'rcount': patient_data.get('readmissions', 0),
            'dialysisrenalendstage': 0,
            'asthma': 1 if patient_data.get('medical_condition') == 'Asthma' else 0,
            'irondef': 0,
            'pneum': 1 if patient_data.get('medical_condition') == 'Pneumonia' else 0,
            'substancedependence': 0,
            'psychologicaldisordermajor': 0,
            'depress': 0,
            'psychother': 0,
            'fibrosisandother': 0,
            'malnutrition': 0,
            'hemo': 0,
            'hematocrit': patient_data.get('hematocrit', 12.0),
            'neutrophils': patient_data.get('neutrophils', 7.0),
            'sodium': patient_data.get('sodium', 140.0),
            'glucose': patient_data.get('glucose', 100.0),
            'bloodureanitro': patient_data.get('bun', 15.0),
            'creatinine': patient_data.get('creatinine', 1.0),
            'bmi': patient_data.get('bmi', 25.0),
            'pulse': patient_data.get('pulse', 75),
            'respiration': patient_data.get('respiration', 16.0),
            'secondarydiagnosisnonicd9': patient_data.get('secondary_diagnoses', 2),
        }
        
        # Add clinical defaults
        for feature, value in clinical_defaults.items():
            if feature in self.feature_names:
                input_row[feature] = value
        
        # Engineered features
        if 'creatinine_bun_ratio' in self.feature_names:
            input_row['creatinine_bun_ratio'] = input_row['creatinine'] / (input_row['bloodureanitro'] + 0.01)
        
        if 'bmi_category' in self.feature_names:
            bmi = input_row['bmi']
            if bmi < 18.5:
                input_row['bmi_category'] = 0
            elif bmi < 25:
                input_row['bmi_category'] = 1
            elif bmi < 30:
                input_row['bmi_category'] = 2
            else:
                input_row['bmi_category'] = 3
        
        if 'anemia_flag' in self.feature_names:
            input_row['anemia_flag'] = 1 if input_row['hematocrit'] < 36 else 0
        
        if 'hyperglycemia_flag' in self.feature_names:
            input_row['hyperglycemia_flag'] = 1 if input_row['glucose'] > 140 else 0
        
        if 'kidney_dysfunction' in self.feature_names:
            input_row['kidney_dysfunction'] = 1 if input_row['creatinine'] > 1.2 else 0
        
        if 'readmission_risk' in self.feature_names:
            input_row['readmission_risk'] = 1 if input_row['rcount'] >= 2 else 0
        
        if 'diabetes_kidney' in self.feature_names:
            input_row['diabetes_kidney'] = 1 if (input_row['glucose'] > 140 and input_row['creatinine'] > 1.2) else 0
        
        if 'psych_complexity' in self.feature_names:
            input_row['psych_complexity'] = (input_row['psychologicaldisordermajor'] + 
                                           input_row['depress'] + input_row['psychother'])
        
        if 'comorbidity_count' in self.feature_names:
            comorbidity_fields = ['dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 
                                'substancedependence', 'psychologicaldisordermajor', 
                                'depress', 'psychother', 'fibrosisandother', 'malnutrition']
            input_row['comorbidity_count'] = sum(input_row.get(field, 0) for field in comorbidity_fields)
        
        if 'high_comorbidity' in self.feature_names:
            input_row['high_comorbidity'] = 1 if input_row.get('comorbidity_count', 0) >= 3 else 0
        
        if 'tachycardia' in self.feature_names:
            input_row['tachycardia'] = 1 if input_row['pulse'] > 100 else 0
        
        if 'tachypnea' in self.feature_names:
            input_row['tachypnea'] = 1 if input_row['respiration'] > 20 else 0
        
        # Gender encoding
        if 'is_male' in self.feature_names:
            input_row['is_male'] = 1 if patient_data.get('gender', 'Male').upper() == 'M' else 0
        
        # Facility encoding (default to facility A)
        facility_id = patient_data.get('facility', 'A')
        for facility in ['B', 'C', 'D', 'E']:
            facility_col = f'facility_{facility}'
            if facility_col in self.feature_names:
                input_row[facility_col] = 1 if facility_id == facility else 0
        
        # Set any missing features to 0
        for feature in self.feature_names:
            if feature not in input_row:
                input_row[feature] = 0
        
        # Create input array in correct order
        input_array = np.array([[input_row.get(feature, 0) for feature in self.feature_names]])
        return input_array
    
    def _prepare_legacy_input(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """Prepare input for legacy model format."""
        input_row = {}
        
        # Handle numerical features
        numerical_mapping = {
            "age": patient_data.get("age", 50),
            "billing_amount": patient_data.get("billing_amount", 15000),
            "room_number": patient_data.get("room_number", 200),
            "admission_weekday": patient_data.get("admission_weekday", 1),
            "admission_month": patient_data.get("admission_month", 6),
        }
        
        for feature, value in numerical_mapping.items():
            if feature in self.feature_names:
                input_row[feature] = value
        
        # Handle categorical features
        categorical_values = {
            "gender": patient_data.get("gender", "Male").upper(),
            "admission_type": patient_data.get("admission_type", "Emergency").title(),
            "medical_condition": patient_data.get("medical_condition", "Pneumonia").title(),
            "blood_type": patient_data.get("blood_type", "O+"),
            "age_group": "adult",
            "billing_tier": "medium"
        }
        
        # Calculate derived features
        age = numerical_mapping["age"]
        if age < 18:
            categorical_values["age_group"] = "pediatric"
        elif age >= 65:
            categorical_values["age_group"] = "senior"
        
        # Set all categorical dummy variables to 0 first
        for feature in self.feature_names:
            if feature not in input_row:
                input_row[feature] = 0
        
        # Set appropriate dummy variables to 1
        for category, value in categorical_values.items():
            dummy_col = f"{category}_{value}"
            if dummy_col in input_row:
                input_row[dummy_col] = 1
        
        input_array = np.array([[input_row.get(feature, 0) for feature in self.feature_names]])
        return input_array
    
    def _generate_explanation(self, patient_data: Dict[str, Any], 
                            shap_values: Dict[str, float], prediction: float) -> str:
        """Generate human-readable explanation of prediction."""
        # Find most important features
        important_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        age = patient_data.get("age", "unknown")
        gender = patient_data.get("gender", "unknown")
        condition = patient_data.get("medical_condition", "unknown condition")
        admission_type = patient_data.get("admission_type", "unknown")
        
        explanation = f"Based on a {age}-year-old {gender.lower()} with {condition} "
        explanation += f"({admission_type.lower()} admission), predicted stay is {prediction:.1f} days. "
        
        # Add feature importance insights
        if important_features:
            explanation += "Key factors: "
            for feature, impact in important_features:
                if abs(impact) > 0.1:  # Only mention significant impacts
                    direction = "increases" if impact > 0 else "decreases"
                    explanation += f"{feature.replace('_', ' ')} {direction} stay duration; "
        
        return explanation.rstrip("; ")
    
    def _estimate_confidence_interval(self, X_input: np.ndarray, prediction: float, 
                                    confidence: float = 0.95) -> List[float]:
        """Estimate confidence interval for prediction (simplified approach)."""
        # This is a simplified confidence interval estimation
        # In production, you might want to use quantile regression or ensemble methods
        
        # Use model's feature importance and training error to estimate uncertainty
        if hasattr(self.model_metadata, 'test_metrics'):
            rmse = self.model_metadata.get('test_metrics', {}).get('rmse', 1.0)
        else:
            rmse = 1.0  # Default uncertainty
        
        # Simple interval based on RMSE
        margin = 1.96 * rmse  # 95% confidence interval approximation
        
        lower_bound = max(1, prediction - margin)  # At least 1 day
        upper_bound = prediction + margin
        
        return [float(lower_bound), float(upper_bound)]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get model feature importance scores."""
        if self.model is None:
            self.load_model()
        
        if self.model is None or self.feature_names is None:
            return {}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save_model(self) -> None:
        """Save trained model, explainer, and metadata."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        logger.info(f"Saving model to {self.model_path}")
        joblib.dump(self.model, self.model_path)
        
        if self.explainer is not None:
            logger.info(f"Saving explainer to {self.explainer_path}")
            joblib.dump(self.explainer, self.explainer_path)
        
        if self.model_metadata:
            logger.info(f"Saving metadata to {self.metadata_path}")
            joblib.dump(self.model_metadata, self.metadata_path)
        
        logger.info("Model artifacts saved successfully")
    
    def load_model(self) -> bool:
        """Load trained model, explainer, and metadata."""
        try:
            # First try to load our optimized clinical model
            if self.clinical_model_path.exists():
                logger.info(f"Loading optimized clinical model from {self.clinical_model_path}")
                model_data = joblib.load(self.clinical_model_path)
                
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                self.model_metadata = model_data['performance']
                
                # Create SHAP explainer for the clinical model
                self.explainer = shap.TreeExplainer(self.model)
                
                logger.info("Optimized clinical model loaded successfully")
                return True
                
            # Fallback to old model format
            elif self.model_path.exists():
                logger.info(f"Loading model from {self.model_path}")
                self.model = joblib.load(self.model_path)
                
                if self.explainer_path.exists():
                    logger.info(f"Loading explainer from {self.explainer_path}")
                    self.explainer = joblib.load(self.explainer_path)
                
                if self.metadata_path.exists():
                    logger.info(f"Loading metadata from {self.metadata_path}")
                    self.model_metadata = joblib.load(self.metadata_path)
                    self.feature_names = self.model_metadata.get("feature_names", [])
                
                logger.info("Model loaded successfully")
                return True
            else:
                logger.warning(f"No model files found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.model is None:
            self.load_model()
        
        info = {
            "model_loaded": self.model is not None,
            "explainer_loaded": self.explainer is not None,
            "model_version": settings.model_version,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
        }
        
        if self.model_metadata:
            info.update({
                "training_timestamp": self.model_metadata.get("timestamp"),
                "test_r2": self.model_metadata.get("test_r2"),
                "test_rmse": self.model_metadata.get("test_metrics", {}).get("rmse"),
                "train_samples": self.model_metadata.get("train_samples"),
                "test_samples": self.model_metadata.get("test_samples"),
            })
        
        return info