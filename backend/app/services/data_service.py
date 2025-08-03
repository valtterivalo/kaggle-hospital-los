"""Data processing service using Polars for healthcare dataset operations."""

import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class DataService:
    """Service for healthcare data processing and feature engineering."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize DataService with data path."""
        self.data_path = data_path or Path("../data/kaggle-data/LengthOfStay.csv")
        self.processed_data_path = settings.processed_data_dir / "processed_data.parquet"
        self._raw_data: Optional[pl.LazyFrame] = None
        self._processed_data: Optional[pl.DataFrame] = None
        
        # Ensure directories exist
        settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self) -> pl.LazyFrame:
        """Load raw healthcare data with lazy evaluation for efficiency."""
        if self._raw_data is None:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
            logger.info(f"Loading raw data from {self.data_path}")
            self._raw_data = pl.scan_csv(self.data_path)
        
        return self._raw_data
    
    def get_data_summary(self) -> Dict[str, any]:
        """Get comprehensive summary statistics of the dataset."""
        df = self.load_raw_data().collect()
        
        summary = {
            "total_records": len(df),
            "columns": df.columns,
            "missing_values": df.null_count().to_dict(as_series=False),
            "data_types": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        }
        
        # Numerical statistics  
        numerical_cols = df.select(pl.col(pl.Int64, pl.Float64)).columns
        if numerical_cols:
            numerical_stats = df.select(numerical_cols).describe().to_dict(as_series=False)
            summary["numerical_stats"] = numerical_stats
        
        # Categorical statistics
        categorical_cols = ["gender", "admission_type", "medical_condition", "blood_type"]
        categorical_stats = {}
        for col in categorical_cols:
            if col in df.columns:
                value_counts = df[col].value_counts().to_dict(as_series=False)
                categorical_stats[col] = {
                    "unique_values": len(value_counts[col]),
                    "top_values": dict(zip(value_counts[col][:5], value_counts["count"][:5]))
                }
        summary["categorical_stats"] = categorical_stats
        
        return summary
    
    def validate_data_quality(self) -> Dict[str, any]:
        """Validate data quality and identify potential issues."""
        df = self.load_raw_data()
        
        quality_report = {
            "total_records": df.select(pl.count()).collect().item(),
            "issues": [],
            "warnings": []
        }
        
        # Check for logical inconsistencies
        collected_df = df.collect()
        
        # Age validation
        invalid_ages = collected_df.filter(
            (pl.col("age") < 0) | (pl.col("age") > 120)
        ).select(pl.count()).item()
        if invalid_ages > 0:
            quality_report["issues"].append(f"Found {invalid_ages} records with invalid ages")
        
        # Length of stay validation
        invalid_los = collected_df.filter(
            pl.col("length_of_stay") <= 0
        ).select(pl.count()).item()
        if invalid_los > 0:
            quality_report["issues"].append(f"Found {invalid_los} records with invalid length_of_stay")
        
        # Date consistency
        date_issues = collected_df.filter(
            pl.col("discharge_date") < pl.col("date_of_admission")
        ).select(pl.count()).item()
        if date_issues > 0:
            quality_report["issues"].append(f"Found {date_issues} records with discharge before admission")
        
        # Missing values check
        null_counts = collected_df.null_count()
        for col in null_counts.columns:
            null_count = null_counts[col].item()
            if null_count > 0:
                percentage = (null_count / len(collected_df)) * 100
                if percentage > 10:
                    quality_report["issues"].append(f"Column '{col}' has {percentage:.1f}% missing values")
                elif percentage > 5:
                    quality_report["warnings"].append(f"Column '{col}' has {percentage:.1f}% missing values")
        
        return quality_report
    
    def clean_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Clean and standardize the healthcare data."""
        return (
            df
            # Remove invalid records
            .filter(
                (pl.col("age") >= 0) & 
                (pl.col("age") <= 120) &
                (pl.col("length_of_stay") > 0) &
                (pl.col("length_of_stay") <= 365)  # Cap at 1 year
            )
            # Standardize categorical values
            .with_columns([
                pl.col("gender").str.to_uppercase(),
                pl.col("admission_type").str.to_titlecase(),
                pl.col("medical_condition").str.to_titlecase(),
            ])
            # Handle missing values for core columns
            .filter(
                pl.col("age").is_not_null() &
                pl.col("gender").is_not_null() &
                pl.col("medical_condition").is_not_null() &
                pl.col("length_of_stay").is_not_null()
            )
        )
    
    def engineer_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Engineer features for better model performance."""
        return (
            df
            .with_columns([
                # Age groups for non-linear relationships
                pl.when(pl.col("age") < 18).then("pediatric")
                .when(pl.col("age") < 65).then("adult")
                .otherwise("senior").alias("age_group"),
                
                # Billing amount categories (using cut instead of qcut for lazy frames)
                pl.when(pl.col("billing_amount") < 10000).then("low")
                .when(pl.col("billing_amount") < 20000).then("medium")
                .otherwise("high").alias("billing_tier"),
                
                # Simple derived features (skip complex date parsing for now)
                (pl.col("age") / 10).round(0).alias("age_decade"),
            ])
        )
    
    def prepare_ml_data(self, target_col: str = "length_of_stay") -> Tuple[pl.DataFrame, List[str]]:
        """Prepare data for machine learning with proper encoding."""
        df = self.load_raw_data()
        
        # Clean and engineer features
        processed_df = self.engineer_features(self.clean_data(df)).collect()
        
        # Define feature columns
        numerical_features = [
            "age", "billing_amount", "room_number", "age_decade"
        ]
        
        categorical_features = [
            "gender", "admission_type", "medical_condition", 
            "blood_type", "age_group", "billing_tier"
        ]
        
        # One-hot encode categorical variables
        encoded_df = processed_df
        dummy_columns = []
        
        for col in categorical_features:
            if col in processed_df.columns:
                # Create dummy variables for this column
                dummies = processed_df.select(pl.col(col)).to_dummies(separator="_")
                dummy_columns.extend(dummies.columns)
                encoded_df = encoded_df.hconcat(dummies)
        
        # Select final features - only use columns that actually exist
        available_numerical = [col for col in numerical_features if col in encoded_df.columns]
        available_features = available_numerical + dummy_columns
        
        # Final dataset with features and target
        ml_ready_df = encoded_df.select(available_features + [target_col]).drop_nulls()
        
        return ml_ready_df, available_features
    
    def split_data(self, df: pl.DataFrame, test_size: float = 0.2, 
                   temporal_split: bool = False) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Split data into train/test sets with temporal awareness for healthcare data."""
        if temporal_split and "date_of_admission" in df.columns:
            # Temporal split to avoid data leakage (using string date for now)
            sorted_df = df.sort("date_of_admission")
            split_idx = int(len(sorted_df) * (1 - test_size))
            
            train_df = sorted_df[:split_idx]
            test_df = sorted_df[split_idx:]
        else:
            # Random split (default for simplicity)
            shuffled_df = df.sample(fraction=1.0, seed=42)
            split_idx = int(len(shuffled_df) * (1 - test_size))
            
            train_df = shuffled_df[:split_idx]
            test_df = shuffled_df[split_idx:]
        
        return train_df, test_df
    
    def get_feature_info(self, feature_name: str) -> Dict[str, any]:
        """Get detailed information about a specific feature."""
        df = self.load_raw_data().collect()
        
        if feature_name not in df.columns:
            return {"error": f"Feature '{feature_name}' not found"}
        
        feature_data = df.select(feature_name)
        feature_info = {
            "name": feature_name,
            "type": str(feature_data.dtypes[0]),
            "non_null_count": feature_data.select(pl.col(feature_name).count()).item(),
            "null_count": feature_data.select(pl.col(feature_name).null_count()).item(),
        }
        
        # Type-specific statistics
        if feature_data.dtypes[0] in [pl.Int64, pl.Float64]:
            stats = feature_data.describe()
            feature_info.update({
                "min": stats["min"].item(),
                "max": stats["max"].item(),
                "mean": stats["mean"].item(),
                "std": stats["std"].item(),
                "median": stats["50%"].item(),
            })
        else:
            # Categorical feature
            value_counts = feature_data[feature_name].value_counts()
            feature_info.update({
                "unique_values": len(value_counts),
                "top_values": value_counts.to_dict(as_series=False),
            })
        
        return feature_info
    
    def cache_processed_data(self, df: pl.DataFrame) -> None:
        """Cache processed data for faster access."""
        logger.info(f"Caching processed data to {self.processed_data_path}")
        df.write_parquet(self.processed_data_path)
        self._processed_data = df
    
    def load_processed_data(self) -> Optional[pl.DataFrame]:
        """Load cached processed data if available."""
        if self._processed_data is not None:
            return self._processed_data
        
        if self.processed_data_path.exists():
            logger.info(f"Loading cached processed data from {self.processed_data_path}")
            self._processed_data = pl.read_parquet(self.processed_data_path)
            return self._processed_data
        
        return None