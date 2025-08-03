"""Script to download and validate healthcare dataset.

Since we can't automate Kaggle downloads without API credentials,
this script provides instructions and validation for manual download.
"""

import polars as pl
from pathlib import Path


def validate_downloaded_data(data_path: Path) -> bool:
    """Validate the downloaded healthcare dataset."""
    expected_columns = [
        "patient_id", "age", "gender", "blood_type", "medical_condition",
        "date_of_admission", "doctor", "hospital", "insurance_provider",
        "billing_amount", "room_number", "admission_type", "discharge_date",
        "medication", "test_results", "length_of_stay"
    ]
    
    if not data_path.exists():
        print(f"❌ Dataset not found at {data_path}")
        return False
    
    try:
        df = pl.read_csv(data_path)
        
        # Check columns
        if set(df.columns) != set(expected_columns):
            print("❌ Column mismatch detected")
            print(f"Expected: {sorted(expected_columns)}")
            print(f"Found: {sorted(df.columns)}")
            return False
        
        # Check record count (relaxed for development)
        if len(df) < 1000:
            print(f"❌ Insufficient records: {len(df)} (expected >1,000)")
            return False
        elif len(df) < 90000:
            print(f"⚠️  Using smaller dataset: {len(df)} records (production expects >90,000)")
        
        print("✅ Dataset validation passed")
        print(f"Records: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        return True
        
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False


def display_download_instructions():
    """Display instructions for manual dataset download."""
    print("""
📋 Healthcare Dataset Download Instructions
==========================================

1. Visit Kaggle dataset page:
   https://www.kaggle.com/datasets/aayushchou/hospital-length-of-stay-dataset-microsoft

2. Download the CSV file (hospital_los.csv or similar)

3. Place the file in: data/raw/hospital_los.csv

4. Run this script again to validate the download

Alternative: If you have Kaggle API set up:
   pip install kaggle
   kaggle datasets download -d aayushchou/hospital-length-of-stay-dataset-microsoft

Note: This project uses synthetic data for demo purposes only.
    """)


def main():
    """Main function to handle data download and validation."""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for common dataset filenames
    possible_files = [
        data_dir / "hospital_los.csv",
        data_dir / "hospital_length_of_stay.csv",
        data_dir / "healthcare_dataset.csv"
    ]
    
    dataset_file = None
    for file_path in possible_files:
        if file_path.exists():
            dataset_file = file_path
            break
    
    if dataset_file is None:
        print("📥 Dataset not found. Download required.")
        display_download_instructions()
        return
    
    print(f"🔍 Validating dataset: {dataset_file}")
    if validate_downloaded_data(dataset_file):
        print(f"🎉 Dataset ready for use: {dataset_file}")
    else:
        print("❌ Dataset validation failed. Please check the file.")


if __name__ == "__main__":
    main()