"""
Custom transformers for the churn prediction pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical columns to numeric for models that require numeric input"""
    
    def __init__(self):
        self.encoders_ = {}
        self.categorical_columns_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn encodings from training data"""
        self.categorical_columns_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.categorical_columns_:
            encoder = LabelEncoder()
            # Handle both object and category dtypes
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                encoder.fit(X[col].astype(str))
            else:
                encoder.fit(X[col])
            self.encoders_[col] = encoder
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned encodings"""
        X = X.copy()
        
        for col in self.categorical_columns_:
            if col not in X.columns:
                continue
            
            encoder = self.encoders_[col]
            
            # Convert to string if categorical dtype
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                col_values = X[col].astype(str)
            else:
                col_values = X[col]
            
            # Handle unseen categories
            unseen_mask = ~col_values.isin(encoder.classes_)
            if unseen_mask.any():
                # Replace unseen values with the most common class
                col_values[unseen_mask] = encoder.classes_[0]
            
            X[col] = encoder.transform(col_values)
        
        return X


class DataCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer for data cleaning to prevent data leakage"""

    def __init__(self):
        self.numeric_columns_ = None
        self.medians_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """Learn medians from training data only"""
        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns
        self.medians_ = X[self.numeric_columns_].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning using training statistics"""
        X = X.copy()

        # Only process columns that exist in the input and are still numeric
        for col in self.numeric_columns_:
            if col not in X.columns:
                continue
            
            # Get the column
            col_data = X[col]
            
            # Skip categorical columns (even if they have numeric dtype)
            if isinstance(col_data.dtype, pd.CategoricalDtype):
                continue
                
            # Check if column is still numeric (not categorical or object)
            if not pd.api.types.is_numeric_dtype(col_data):
                continue
                
            # Replace infinite values with NaN
            col_data = col_data.replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values with training medians
            if col in self.medians_ and col_data.isna().any():
                col_data = col_data.fillna(self.medians_[col])
            
            # Assign back to dataframe
            X[col] = col_data

        return X
