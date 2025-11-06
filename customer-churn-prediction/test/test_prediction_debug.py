#!/usr/bin/env python3
"""Debug prediction issue"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.predict import ChurnPredictor
import pandas as pd

# Create predictor
p = ChurnPredictor('xgboost')

# Create test data
test_data = {
    'tenure': 12,
    'MonthlyCharges': 65.5,
    'gender': 'Male',
    'Contract': 'Month-to-month',
    'InternetService': 'Fiber optic'
}

# Convert to DataFrame
df = pd.DataFrame([test_data])

# Apply feature engineering
df_eng = p._engineer_features(df)
print(f"After engineering: {len(df_eng.columns)} columns")
print(f"Columns: {list(df_eng.columns)}")

# Convert object to category
for col in df_eng.columns:
    if df_eng[col].dtype == 'object':
        df_eng[col] = df_eng[col].astype('category')

print(f"\nFeature names expected: {len(p.feature_names)}")
print(f"Missing: {set(p.feature_names) - set(df_eng.columns)}")

# Add missing features
for feature in set(p.feature_names) - set(df_eng.columns):
    if feature in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                   'PaperlessBilling', 'PaymentMethod']:
        df_eng[feature] = pd.Categorical(['No'] * len(df_eng))
    elif feature == 'SeniorCitizen':
        df_eng[feature] = 0
    else:
        df_eng[feature] = 0

print(f"\nAfter adding missing: {len(df_eng.columns)} columns")
print(f"Still missing: {set(p.feature_names) - set(df_eng.columns)}")

# Try to reorder
try:
    df_final = df_eng[p.feature_names]
    print(f"\nFinal: {len(df_final.columns)} columns")
    print("Success!")
except Exception as e:
    print(f"\nError: {e}")
