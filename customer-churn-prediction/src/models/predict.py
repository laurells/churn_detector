"""
Model prediction module for customer churn prediction.
"""

import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any, Union
import os


class ChurnPredictor:
    """Class for making predictions with trained churn models."""

    def __init__(self, model_path: str):
        """Initialize predictor with a trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        self.model_path = model_path

        # Load label encoders for consistent encoding
        encoders_path = model_path.replace('.pkl', '_encoders.pkl').replace('xgboost.pkl', 'label_encoders.pkl')
        if os.path.exists(encoders_path):
            self.label_encoders = joblib.load(encoders_path)
        else:
            # Try alternative path
            encoders_path = os.path.join(os.path.dirname(model_path), 'label_encoders.pkl')
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
            else:
                raise FileNotFoundError(f"Label encoders not found. Expected at: {encoders_path}")

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same encoding used during training."""
        df_encoded = df.copy()

        # Apply label encoding to categorical columns
        for col, encoder in self.label_encoders.items():
            if col in df_encoded.columns:
                # Handle unseen categories gracefully
                try:
                    df_encoded[col] = encoder.transform(df_encoded[col])
                except ValueError as e:
                    # For prediction, use the most frequent category (mode)
                    mode_value = df_encoded[col].mode().iloc[0] if not df_encoded[col].mode().empty else encoder.classes_[0]
                    df_encoded[col] = encoder.transform([mode_value])[0]

        return df_encoded

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make churn predictions."""
        # Ensure X is encoded
        if not self._is_encoded(X):
            X = self.encode_features(X)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make churn probability predictions."""
        # Ensure X is encoded
        if not self._is_encoded(X):
            X = self.encode_features(X)
        return self.model.predict_proba(X)

    def _is_encoded(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame contains encoded (numeric) categorical values."""
        categorical_like_columns = []

        # Identify columns that should be categorical based on encoders
        for col in self.label_encoders.keys():
            if col in df.columns:
                categorical_like_columns.append(col)

        # Check if any categorical columns contain strings
        for col in categorical_like_columns:
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                return False

        return True

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single customer."""
        # Convert dict to DataFrame
        df = pd.DataFrame([customer_data])

        # Get prediction and probability
        prediction = self.predict(df)[0]
        proba = self.predict_proba(df)[0]

        return {
            'churn_prediction': int(prediction),
            'churn_probability': float(proba[1]),
            'no_churn_probability': float(proba[0])
        }

    def batch_predict(self, customers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions for multiple customers."""
        results = []

        for customer_data in customers_data:
            result = self.predict_single(customer_data)
            result['customer_id'] = customer_data.get('customer_id', 'unknown')
            results.append(result)

        return results

    def save_predictions(self, predictions: List[Dict[str, Any]], output_path: str) -> None:
        """Save predictions to a file."""
        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
