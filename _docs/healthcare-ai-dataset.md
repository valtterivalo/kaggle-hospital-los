# healthcare ai predictor - dataset analysis & requirements

**Status**: ✅ IMPLEMENTED (Synthetic) + 🆕 REAL DATASET ACQUIRED  
**Current**: Synthetic data (10K records) - Test R² = 0.424  
**Next Session**: Transition to real Kaggle dataset (100K records)

## implementation history & learnings

### phase 1: synthetic dataset (completed)
- **source**: `/scripts/generate_synthetic_data.py` (custom implementation)
- **size**: 10,000 synthetic patient records (~1.3MB CSV)
- **why we used**: Faster prototyping, no access issues, guaranteed schema
- **performance achieved**: Test R² = 0.424, RMSE = 2.099 days
- **limitation discovered**: Synthetic data caps model performance potential

### phase 2: real kaggle dataset (ready for next session)
- **source**: kaggle - "hospital length of stay dataset"
- **size**: 100,000 real patient records (~10MB CSV)
- **location**: `/data/kaggle-data/LengthOfStay.csv`
- **expectation**: Should achieve significantly better model performance

#### real kaggle dataset schema (28 features) - CORRECTED
**Source**: Kaggle Hospital Length of Stay Dataset Documentation

```
eid                         : encounter id (integer)
vdate                       : visit date (string)
rcount                      : readmission count (string: "0", "1", "2", "3", "4", "5+")
gender                      : M/F (string)
dialysisrenalendstage      : renal disease flag (integer 0/1)
asthma                     : asthma flag (integer 0/1)
irondef                    : iron deficiency flag (integer 0/1)
pneum                      : pneumonia flag (integer 0/1)
substancedependence        : substance dependence flag (integer 0/1)
psychologicaldisordermajor : major psychological disorder flag (integer 0/1)
depress                    : depression flag (integer 0/1)
psychother                 : other psychological disorder flag (integer 0/1)
fibrosisandother          : fibrosis flag (integer 0/1)
malnutrition              : malnutrition flag (integer 0/1)
hemo                      : BLOOD DISORDER FLAG (integer 0/1) - NOT lab value!
hematocrit                : hematocrit value (float, g/dL)
neutrophils               : neutrophil count (float, cells/microL)
sodium                    : sodium level (float, mmol/L)
glucose                   : glucose level (float, mmol/L)
bloodureanitro            : blood urea nitrogen (float, mg/dL)
creatinine                : creatinine level (float, mg/dL)
bmi                       : body mass index (float, kg/m²)
pulse                     : pulse rate (float, beats/min)
respiration               : respiration rate (float, breaths/min)
secondarydiagnosisnonicd9 : non-ICD9 secondary diagnosis flag (integer)
discharged                : discharge date (string)
facid                     : facility id (string: A-E)
lengthofstay              : target variable (integer, days)
lengthofstay              : target variable (integer, days)
```

#### synthetic dataset schema (for comparison)
```
patient_id, age, gender, blood_type, medical_condition, 
date_of_admission, doctor, hospital, insurance_provider, 
billing_amount, room_number, admission_type, discharge_date, 
medication, test_results, length_of_stay (16 features)
```

#### data characteristics
- **target distribution**: right-skewed (1-30 days, median ~4 days)
- **missing values**: <5% across all features
- **categorical features**: 8 categorical variables requiring encoding
- **numerical features**: 3+ continuous variables for modeling
- **temporal features**: admission/discharge dates for feature engineering

## alternative datasets (backup options)

### option 1: mimic-iv demo dataset
- **pros**: real clinical terminology, research credibility
- **cons**: more complex preprocessing, larger size
- **use case**: if synthetic data feels too artificial

### option 2: finnish thl open datasets
- **pros**: local relevance, official health statistics
- **cons**: aggregated data, limited individual predictions
- **use case**: for contextual analysis and background research

### option 3: kaggle healthcare competitions
- **specific**: "predict hospital readmissions", "er wait times"
- **pros**: well-cleaned, competition-ready datasets
- **cons**: may be overused in portfolios

## data preprocessing pipeline

### initial data exploration (polars-based eda)
```python
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# load and inspect
df = pl.read_csv("hospital_los.csv")
print(df.describe())
print(df.null_count())

# target analysis
df.select("length_of_stay").to_pandas().hist(bins=30)

# categorical analysis
df.select("medical_condition", "length_of_stay").group_by("medical_condition").agg(
    pl.col("length_of_stay").mean().alias("avg_los"),
    pl.col("length_of_stay").count().alias("count")
)
```

### data quality assessment
```python
# identify data quality issues
def assess_data_quality(df: pl.LazyFrame) -> dict:
    return {
        "total_records": df.select(pl.count()).collect().item(),
        "missing_values": df.null_count().collect(),
        "duplicates": df.filter(pl.col("patient_id").is_duplicated()).select(pl.count()).collect().item(),
        "outliers": df.filter(pl.col("length_of_stay") > 30).select(pl.count()).collect().item(),
        "date_range": df.select([pl.col("date_of_admission").min(), pl.col("date_of_admission").max()]).collect()
    }
```

### feature engineering strategy
```python
def engineer_features(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df
        .with_columns([
            # age groups for non-linear relationships
            pl.when(pl.col("age") < 18).then("pediatric")
            .when(pl.col("age") < 65).then("adult")
            .otherwise("senior").alias("age_group"),
            
            # los derived from dates
            (pl.col("discharge_date") - pl.col("date_of_admission")).dt.days().alias("calculated_los"),
            
            # day of week admission patterns
            pl.col("date_of_admission").dt.weekday().alias("admission_weekday"),
            
            # medical condition severity proxy
            pl.col("billing_amount").qcut(3, labels=["low", "medium", "high"]).alias("billing_tier"),
            
            # doctor specialization proxy
            pl.col("doctor").value_counts().over("doctor").alias("doctor_caseload")
        ])
        .filter(pl.col("calculated_los") == pl.col("length_of_stay"))  # data consistency check
    )
```

## modeling approach

### baseline models for comparison
1. **linear regression**: simple interpretable baseline
2. **random forest**: ensemble method with built-in feature importance
3. **xgboost**: primary model choice for production

### target model: xgboost configuration
```python
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

# hyperparameter search space
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# model with healthcare-specific considerations
model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=10
)
```

### evaluation metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model_performance(y_true, y_pred):
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'within_1_day': np.mean(np.abs(y_true - y_pred) <= 1) * 100
    }
```

## data validation and testing

### data integrity checks
```python
def validate_healthcare_data(df: pl.LazyFrame) -> bool:
    checks = [
        # logical constraints
        df.filter(pl.col("age") < 0).select(pl.count()).collect().item() == 0,
        df.filter(pl.col("length_of_stay") <= 0).select(pl.count()).collect().item() == 0,
        df.filter(pl.col("discharge_date") < pl.col("date_of_admission")).select(pl.count()).collect().item() == 0,
        
        # referential integrity
        df.select(pl.col("patient_id").is_unique().all()).collect().item(),
        
        # business rules
        df.filter(pl.col("length_of_stay") > 365).select(pl.count()).collect().item() < df.select(pl.count()).collect().item() * 0.01
    ]
    return all(checks)
```

### test data preparation
```python
from sklearn.model_selection import train_test_split

def create_temporal_split(df: pl.DataFrame, test_size: float = 0.2):
    """Create temporal split for healthcare data to avoid data leakage."""
    sorted_df = df.sort("date_of_admission")
    split_idx = int(len(sorted_df) * (1 - test_size))
    
    train_df = sorted_df[:split_idx]
    test_df = sorted_df[split_idx:]
    
    return train_df, test_df
```

## performance benchmarks & learnings

### synthetic data performance (baseline)
- **baseline rmse**: 2.081 days 
- **achieved r²**: 0.424 (limited by synthetic data)
- **cross-validation r²**: 0.434 ± 0.019 (stable)
- **overfitting control**: 0.065 gap (excellent)

### real clinical dataset performance (achieved)
- **final rmse**: 0.397 days (~10 hour prediction accuracy)
- **final r²**: 0.971 (97.1% variance explained)
- **overfitting control**: 0.003 gap (excellent)
- **performance journey**: 0.424 → 0.960 (baseline) → 0.969 (features) → 0.971 (optimized)
- **key factors**: readmission history, comorbidity count, kidney function (creatinine)

### inference performance requirements
- **prediction latency**: <500ms for single prediction
- **batch prediction**: <5 seconds for 1000 predictions
- **model loading time**: <2 seconds on startup
- **memory footprint**: <100mb for model artifacts

## data privacy and compliance

### synthetic data advantages
- **no patient privacy concerns**: fully synthetic dataset
- **gdpr compliance**: no real personal data processing
- **research ethics**: no irb approval required
- **public sharing**: can be included in public repositories

### production considerations
```python
# data anonymization patterns for real deployment
def anonymize_patient_data(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df
        .with_columns([
            pl.col("patient_id").str.slice(0, 8).alias("patient_id_hash"),  # truncate ids
            pl.col("doctor").hash().alias("doctor_id"),  # hash names
            pl.col("age").map_elements(lambda x: (x // 5) * 5).alias("age_group_5yr")  # age binning
        ])
        .drop(["patient_id", "doctor", "age"])
    )
```

## dataset acquisition instructions

### download process
```bash
# manual download from kaggle
1. visit: https://www.kaggle.com/datasets/aayushchou/hospital-length-of-stay-dataset-microsoft
2. download hospital_los.csv
3. place in data/raw/ directory

# automated download (requires kaggle api)
pip install kaggle
kaggle datasets download -d aayushchou/hospital-length-of-stay-dataset-microsoft
unzip hospital-length-of-stay-dataset-microsoft.zip -d data/raw/
```

### data validation after download
```python
def validate_downloaded_data():
    expected_columns = [
        "patient_id", "age", "gender", "blood_type", "medical_condition",
        "date_of_admission", "doctor", "hospital", "insurance_provider",
        "billing_amount", "room_number", "admission_type", "discharge_date",
        "medication", "test_results", "length_of_stay"
    ]
    
    df = pl.read_csv("data/raw/hospital_los.csv")
    assert set(df.columns) == set(expected_columns), "column mismatch"
    assert len(df) > 90000, "insufficient records"
    print("data validation passed ✓")
```

## exploratory data analysis checklist

### univariate analysis
- [ ] target variable distribution and statistics
- [ ] categorical feature distributions
- [ ] numerical feature distributions and outliers
- [ ] missing value patterns and percentages

### bivariate analysis
- [ ] correlation matrix for numerical features
- [ ] categorical vs target relationships
- [ ] temporal patterns in admissions and los
- [ ] hospital and doctor-specific patterns

### multivariate analysis
- [ ] feature interactions exploration
- [ ] dimensionality reduction visualization (pca/tsne)
- [ ] clustering analysis for patient segments
- [ ] time series patterns in healthcare utilization

### healthcare-specific analysis
- [ ] seasonal patterns in medical conditions
- [ ] age and gender interactions with los
- [ ] insurance provider impact on treatment duration
- [ ] emergency vs elective admission patterns

## 🚀 next session: real dataset integration plan

### immediate steps
1. **exploratory data analysis** of kaggle dataset
   ```python
   df = pl.read_csv("data/kaggle-data/LengthOfStay.csv")
   df.describe()  # understand lab value ranges
   df.null_count()  # check data quality
   ```

2. **compare dataset characteristics**
   - target distribution (synthetic vs real)
   - feature correlation patterns
   - missing value patterns
   - clinical value ranges validation

3. **feature engineering for clinical data**
   ```python
   # lab value derived features
   - creatinine_to_bun_ratio = creatinine / bloodureanitro
   - anemia_flag = hemo < 12.0  # clinical threshold
   - diabetes_proxy = glucose > 140  # fasting glucose threshold
   - kidney_function = 1 / creatinine  # inverse relationship
   ```

4. **model performance comparison**
   - retrain XGBoost on real data
   - compare performance metrics
   - analyze SHAP feature importance differences

### expected improvements
- **better clinical relevance**: lab values should create stronger predictive signals
- **higher r²**: expecting >0.6 with richer feature set
- **lower rmse**: targeting <1.5 days with clinical indicators
- **more interpretable**: lab values have known clinical associations

### technical challenges to address
- **different schema**: need new preprocessing pipeline
- **larger dataset**: 100K vs 10K records (10x more data)
- **lab value normalization**: clinical ranges vary significantly
- **feature selection**: 28 features vs 16 (avoid overfitting)

### session success criteria
- [ ] complete EDA of real dataset
- [ ] achieve test R² >0.6 (vs current 0.424)
- [ ] RMSE <1.5 days (vs current 2.099)
- [ ] deploy improved model to production API
- [ ] update frontend with new feature explanations

tags:: #dataset #healthcare #eda #data-quality #modeling #real-data #clinical-features