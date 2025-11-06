#!/usr/bin/env python3
"""Test script to check ChurnPredictor initialization."""

import sys
import os
sys.path.append('.')

try:
    from src.models.predict import ChurnPredictor
    print("ChurnPredictor import successful")

    predictor = ChurnPredictor('models/xgboost.pkl')
    print("ChurnPredictor initialized successfully")
    print(f"Model type: {type(predictor.model)}")
    print(f"Label encoders loaded: {len(predictor.label_encoders) if hasattr(predictor, 'label_encoders') else 'None'}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
