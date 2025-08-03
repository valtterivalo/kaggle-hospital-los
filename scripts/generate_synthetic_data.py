"""Generate synthetic healthcare dataset for development and demo purposes.

This creates a dataset that matches the expected schema from the Kaggle
Hospital Length of Stay dataset for development purposes.
"""

import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random


def generate_synthetic_healthcare_data(n_records: int = 10000) -> pl.DataFrame:
    """Generate synthetic healthcare data matching the expected schema."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define categorical options
    medical_conditions = [
        "Pneumonia", "Diabetes", "Hypertension", "Asthma", "Heart Disease",
        "Arthritis", "Cancer", "Obesity", "Depression", "Stroke",
        "Kidney Disease", "COPD", "Migraine", "Epilepsy", "Anemia"
    ]
    
    blood_types = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    
    admission_types = ["Emergency", "Elective", "Urgent"]
    
    genders = ["Male", "Female"]
    
    insurance_providers = [
        "Aetna", "Blue Cross", "Cigna", "Humana", "UnitedHealth",
        "Medicare", "Medicaid", "Kaiser", "Anthem", "Molina"
    ]
    
    hospitals = [
        "General Hospital", "Medical Center", "Regional Hospital",
        "University Hospital", "Community Hospital", "City Hospital"
    ]
    
    medications = [
        "Aspirin", "Lisinopril", "Metformin", "Atorvastatin", "Amlodipine",
        "Omeprazole", "Metoprolol", "Losartan", "Albuterol", "Gabapentin"
    ]
    
    test_results = ["Normal", "Abnormal", "Inconclusive"]
    
    doctors = [f"Dr. {name}" for name in [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
        "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez"
    ]]
    
    # Generate data
    data = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(n_records):
        # Patient demographics
        age = max(0, int(np.random.normal(50, 20)))
        age = min(age, 120)  # Cap at 120
        gender = random.choice(genders)
        blood_type = random.choice(blood_types)
        
        # Medical information
        medical_condition = random.choice(medical_conditions)
        admission_type = random.choice(admission_types)
        
        # Hospital information
        hospital = random.choice(hospitals)
        doctor = random.choice(doctors)
        room_number = random.randint(100, 999)
        
        # Insurance and billing
        insurance_provider = random.choice(insurance_providers)
        base_billing = np.random.lognormal(9, 0.8)  # Log-normal distribution for billing
        billing_amount = round(base_billing, 2)
        
        # Treatment information
        medication = random.choice(medications)
        test_results_val = random.choice(test_results)
        
        # Dates and length of stay
        admission_date = base_date + timedelta(days=random.randint(0, 365))
        
        # Length of stay based on medical condition and age (realistic distribution)
        if medical_condition in ["Cancer", "Heart Disease", "Stroke"]:
            base_los = np.random.gamma(3, 2)  # Longer stays for serious conditions
        elif medical_condition in ["Pneumonia", "Diabetes"]:
            base_los = np.random.gamma(2, 1.5)
        else:
            base_los = np.random.gamma(1.5, 1)
        
        # Age adjustment
        if age > 70:
            base_los *= 1.3
        elif age < 18:
            base_los *= 0.8
        
        # Admission type adjustment
        if admission_type == "Emergency":
            base_los *= 1.2
        elif admission_type == "Elective":
            base_los *= 0.9
        
        length_of_stay = max(1, int(base_los))  # At least 1 day
        length_of_stay = min(length_of_stay, 30)  # Cap at 30 days
        
        discharge_date = admission_date + timedelta(days=length_of_stay)
        
        data.append({
            "patient_id": f"P{i+1:06d}",
            "age": age,
            "gender": gender,
            "blood_type": blood_type,
            "medical_condition": medical_condition,
            "date_of_admission": admission_date.strftime("%Y-%m-%d"),
            "doctor": doctor,
            "hospital": hospital,
            "insurance_provider": insurance_provider,
            "billing_amount": billing_amount,
            "room_number": room_number,
            "admission_type": admission_type,
            "discharge_date": discharge_date.strftime("%Y-%m-%d"),
            "medication": medication,
            "test_results": test_results_val,
            "length_of_stay": length_of_stay
        })
    
    return pl.DataFrame(data)


def main():
    """Generate and save synthetic healthcare dataset."""
    print("🏥 Generating synthetic healthcare dataset...")
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    df = generate_synthetic_healthcare_data(n_records=10000)
    
    # Save to CSV
    output_path = data_dir / "hospital_los.csv"
    df.write_csv(output_path)
    
    print(f"✅ Synthetic dataset generated: {output_path}")
    print(f"📊 Records: {len(df):,}")
    print(f"📊 Columns: {len(df.columns)}")
    
    # Display basic statistics
    print("\n📈 Basic Statistics:")
    print(f"Average LOS: {df['length_of_stay'].mean():.1f} days")
    print(f"LOS Range: {df['length_of_stay'].min()}-{df['length_of_stay'].max()} days")
    print(f"Age Range: {df['age'].min()}-{df['age'].max()} years")
    
    # Validate the generated data
    print("\n🔍 Validating generated data...")
    from download_data import validate_downloaded_data
    if validate_downloaded_data(output_path):
        print("🎉 Synthetic dataset is ready for development!")
    
    return output_path


if __name__ == "__main__":
    main()