#!/usr/bin/env python3
"""Test script to verify the app initialization."""

import sys
import os
sys.path.append('.')

try:
    from app.app import ChurnPredictionApp
    print('Testing ChurnPredictionApp initialization...')

    app = ChurnPredictionApp()
    print(f'Models loaded: {len(app.models)}')
    print(f'Predictor initialized: {app.predictor is not None}')

    if app.predictor:
        print('Predictor type:', type(app.predictor))
        print('Label encoders loaded:', hasattr(app.predictor, 'label_encoders') and app.predictor.label_encoders is not None)

    print('SUCCESS: App initialized without errors')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
