#!/usr/bin/env python3
"""Quick test to verify models load correctly"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.predict import ChurnPredictor

def test_model_loading():
    """Test if models load correctly"""
    model_names = ['xgboost', 'random_forest', 'logistic_regression']
    
    for model_name in model_names:
        try:
            print(f"\nTesting {model_name}...")
            predictor = ChurnPredictor(model_name=model_name)
            print(f"  ✅ Loaded successfully")
            print(f"  Type: {type(predictor.model).__name__}")
            print(f"  Has predict: {hasattr(predictor.model, 'predict')}")
            print(f"  Has predict_proba: {hasattr(predictor.model, 'predict_proba')}")
            
            # Test a simple prediction
            test_data = {
                'tenure': 12,
                'MonthlyCharges': 65.5,
                'TotalCharges': 786.0,
            }
            result = predictor.predict_single(test_data)
            print(f"  Test prediction: {result['prediction_label']} ({result['churn_probability']:.2%})")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)[:100]}")

if __name__ == "__main__":
    test_model_loading()
