"""Simple training script that bypasses the complex data service."""

import polars as pl
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path


def main():
    """Simple training approach."""
    print("🏥 Simple healthcare ML model training...")
    
    # Load data directly
    data_path = Path("data/raw/hospital_los.csv")
    df = pl.read_csv(data_path)
    
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Basic cleaning
    clean_df = df.filter(
        (pl.col("age") >= 0) & 
        (pl.col("age") <= 120) &
        (pl.col("length_of_stay") > 0) &
        (pl.col("length_of_stay") <= 30)  # Cap at 30 days
    )
    
    print(f"After cleaning: {len(clean_df)} records")
    
    # Simple feature engineering
    features_df = clean_df.with_columns([
        # Age groups
        pl.when(pl.col("age") < 18).then(0)
        .when(pl.col("age") < 65).then(1)
        .otherwise(2).alias("age_group"),
        
        # Gender encoding
        pl.when(pl.col("gender") == "Male").then(1).otherwise(0).alias("gender_male"),
        
        # Admission type encoding
        pl.when(pl.col("admission_type") == "Emergency").then(1).otherwise(0).alias("admission_emergency"),
        pl.when(pl.col("admission_type") == "Elective").then(1).otherwise(0).alias("admission_elective"),
        
        # Medical condition - encode top conditions
        pl.when(pl.col("medical_condition") == "Pneumonia").then(1).otherwise(0).alias("condition_pneumonia"),
        pl.when(pl.col("medical_condition") == "Diabetes").then(1).otherwise(0).alias("condition_diabetes"),
        pl.when(pl.col("medical_condition") == "Hypertension").then(1).otherwise(0).alias("condition_hypertension"),
    ])
    
    # Select features for training
    feature_cols = [
        "age", "billing_amount", "room_number", "age_group",
        "gender_male", "admission_emergency", "admission_elective",
        "condition_pneumonia", "condition_diabetes", "condition_hypertension"
    ]
    
    # Prepare training data
    ml_df = features_df.select(feature_cols + ["length_of_stay"]).drop_nulls()
    
    print(f"ML-ready data: {len(ml_df)} records, {len(feature_cols)} features")
    
    # Convert to arrays
    X = ml_df.select(feature_cols).to_numpy()
    y = ml_df.select("length_of_stay").to_numpy().ravel()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("🚀 Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"📊 Results:")
    print(f"  - Train R²: {train_r2:.3f}")
    print(f"  - Test R²: {test_r2:.3f}")
    print(f"  - Test RMSE: {test_rmse:.2f} days")
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("🎯 Top 5 Features:")
    for feature, imp in top_features:
        print(f"  - {feature}: {imp:.3f}")
    
    # Save model
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "simple_model.joblib"
    joblib.dump({
        'model': model,
        'feature_names': feature_cols,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse
    }, model_path)
    
    print(f"✅ Model saved to {model_path}")
    
    # Test prediction
    print("🧪 Testing prediction...")
    test_input = np.array([[65, 18000, 205, 2, 1, 1, 0, 1, 0, 0]])  # Sample patient
    prediction = model.predict(test_input)[0]
    print(f"Sample prediction: {prediction:.1f} days")
    
    return model, feature_cols


if __name__ == "__main__":
    main()