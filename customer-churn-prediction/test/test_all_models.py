#!/usr/bin/env python3
"""Test all three models"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.predict import ChurnPredictor

test_data = {
    'tenure': 12,
    'MonthlyCharges': 65.5,
    'gender': 'Male',
    'Contract': 'Month-to-month',
    'InternetService': 'Fiber optic'
}

models = ['xgboost', 'random_forest', 'logistic_regression']

print("Testing all models with sample data:")
print(f"Input: {test_data}\n")

for model_name in models:
    try:
        predictor = ChurnPredictor(model_name)
        result = predictor.predict_single(test_data)
        print(f"{model_name:20s}: {result['prediction_label']:10s} ({result['churn_probability']:.1%})")
    except Exception as e:
        print(f"{model_name:20s}: ERROR - {str(e)[:50]}")
