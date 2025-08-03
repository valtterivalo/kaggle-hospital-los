"""Debug data processing pipeline."""

import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.data_service import DataService
import polars as pl

def main():
    """Debug data processing step by step."""
    data_service = DataService()
    
    print("1. Loading raw data...")
    raw_df = data_service.load_raw_data()
    print(f"Raw data columns: {raw_df.columns}")
    
    print("\n2. Testing clean_data...")
    cleaned_df = data_service.clean_data(raw_df)
    print("Clean data step passed")
    
    print("\n3. Testing engineer_features...")
    try:
        engineered_df = data_service.engineer_features(cleaned_df)
        print("Feature engineering step passed")
        print(f"Engineered columns: {engineered_df.columns}")
        
        print("\n4. Collecting data...")
        collected_df = engineered_df.collect()
        print(f"Final columns: {collected_df.columns}")
        print(f"Data shape: {collected_df.shape}")
        
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        
        # Let's try a minimal version
        print("\n5. Trying minimal feature engineering...")
        minimal_df = (
            cleaned_df
            .with_columns([
                pl.when(pl.col("age") < 18).then("pediatric")
                .when(pl.col("age") < 65).then("adult")  
                .otherwise("senior").alias("age_group")
            ])
        )
        
        print("Minimal engineering columns:", minimal_df.columns)
        collected_minimal = minimal_df.collect()
        print(f"Minimal data shape: {collected_minimal.shape}")
        print(f"Age group values: {collected_minimal['age_group'].unique()}")


if __name__ == "__main__":
    main()