"""
Data preprocessing module for customer churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from config.settings import config


class DataPreprocessor:
    """Enhanced data preprocessor with business-driven feature engineering."""

    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []

    def load_data(self) -> pd.DataFrame:
        """Load and initial data cleaning"""
        df = pd.read_csv(config.data_url)
        # Handle TotalCharges missing values
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create business-driven features"""
        # Tenure buckets
        df['tenure_bucket'] = pd.cut(df['tenure'], bins=config.tenure_buckets,
                                   labels=['0-6m', '6-12m', '12-24m', '24m+'])

        # Services count
        df['services_count'] = df[config.services_columns].apply(
            lambda x: (x == 'Yes').sum() if x.name in config.services_columns else 0, axis=1
        )

        # Monthly to total ratio
        df['monthly_to_total_ratio'] = df['TotalCharges'] / np.maximum(1, df['tenure'] * df['MonthlyCharges'])

        # Business flags
        df['internet_no_tech_support'] = ((df['InternetService'] != 'No') &
                                        (df['TechSupport'] == 'No')).astype(int)

        # Additional business flags
        df['phone_no_security'] = ((df['PhoneService'] == 'Yes') &
                                 (df['OnlineSecurity'] == 'No')).astype(int)

        df['premium_no_backup'] = ((df['MonthlyCharges'] > df['MonthlyCharges'].median()) &
                                 (df['OnlineBackup'] == 'No')).astype(int)

        df['long_tenure_low_value'] = ((df['tenure'] > 24) &
                                     (df['MonthlyCharges'] < df['MonthlyCharges'].median())).astype(int)

        # Expected tenure assumption: industry average + customer behavior
        # Using 36 months as expected tenure for CLV calculation
        df['expected_tenure'] = 36  # Document this assumption

        # CLV calculation
        df['clv'] = df['MonthlyCharges'] * df['expected_tenure']

        return df

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables with consistent encoding"""
        df_encoded = df.copy()

        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = categorical_columns.drop('customerID', errors='ignore')

        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                # Fit on full data to ensure consistency
                self.label_encoders[col].fit(df[col])

            df_encoded[col] = self.label_encoders[col].transform(df[col])

        return df_encoded

    def prepare_splits(self, df: pd.DataFrame) -> dict:
        """Create train/validation/test splits with stratification"""
        X = df.drop(columns=[config.target_column, 'customerID'], errors='ignore')
        y = df[config.target_column]

        # First split: test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=config.model_settings['test_size'], stratify=y, random_state=config.model_settings['random_state']
        )

        # Second split: train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=config.model_settings['val_size'], stratify=y_temp, random_state=config.model_settings['random_state']
        )

        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_names': list(X.columns)
        }

    def run_pipeline(self):
        """Execute full data preparation pipeline"""
        print("Loading data...")
        df = self.load_data()

        print("Engineering features...")
        df = self.engineer_features(df)

        print("Encoding features...")
        df_encoded = self.encode_features(df)

        print("Creating data splits...")
        splits = self.prepare_splits(df_encoded)

        # Save processed data
        config.processed_data_dir.mkdir(parents=True, exist_ok=True)
        for name, data in splits.items():
            if name.startswith('X_'):
                data.to_parquet(config.processed_data_dir / f"{name}.parquet")
            else:
                data.to_parquet(config.processed_data_dir / f"{name}.parquet")

        # Save encoders with consistent naming
        joblib.dump(self.label_encoders, config.models_dir / "label_encoders.pkl")
        joblib.dump(splits['feature_names'], config.models_dir / "feature_names.pkl")

        print("Data preparation completed!")
        return splits, df  # Return original df for CLV analysis
