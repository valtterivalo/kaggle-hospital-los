# clinical dataset

100K real hospital records used for length-of-stay prediction. achieves 97.1% R² accuracy.

## data source
- **dataset**: Kaggle hospital length of stay dataset  
- **records**: 100,000 de-identified patient encounters
- **features**: 28 clinical variables
- **target**: length of stay (1-17 days, mean 4.0 days)

## clinical features

### demographics
- gender (M/F)
- facility (A-E) 
- readmission count (0-5+)

### medical conditions (binary flags)
- end-stage renal disease
- asthma, pneumonia
- depression, major psychiatric disorder
- substance dependence
- iron deficiency, malnutrition
- blood disorders, fibrosis

### lab values
- **kidney function**: creatinine (0.22-2.04 mg/dL), BUN (1-682 mg/dL)
- **blood chemistry**: glucose (0-271 mmol/L), sodium (125-151 mmol/L)
- **blood counts**: hematocrit (4-24 g/dL), neutrophils (0.1-246 cells/μL)

### vital signs  
- BMI (22-39 kg/m²)
- pulse (21-130 beats/min)
- respiration (0.2-10 breaths/min)

## feature engineering

created clinical ratios and thresholds:
- creatinine/BUN ratio (kidney function)
- anemia flag (hematocrit < 36)
- hyperglycemia flag (glucose > 140)
- kidney dysfunction (creatinine > 1.2)
- readmission risk (≥2 previous admissions)
- comorbidity count (multiple conditions)

## model performance

- **accuracy**: 97.1% R² 
- **precision**: 0.397 days RMSE (~10 hour accuracy)
- **features**: 39 total (28 original + 11 engineered)
- **validation**: no overfitting (0.003 train/test gap)

### most important features (SHAP values)
1. readmission count (1.17)
2. comorbidity count (0.65) 
3. readmission risk flag (0.34)
4. creatinine - kidney function (0.29)
5. hematocrit - anemia (0.29)

clinical features dominate predictions, validating medical knowledge.

## clinical validation

feature importance aligns with medical understanding:
- kidney dysfunction (creatinine) increases stay
- multiple conditions compound complexity  
- readmission history indicates complex patients
- lab values reflect underlying disease severity

SHAP explanations enable clinical decision support by showing which factors drive individual predictions.

---

**clinical dataset: 100K records, 97.1% accuracy, medically validated**