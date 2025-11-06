"""
Data preprocessing module for customer churn prediction.
Fixed version addressing split ratios and documentation.
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
        self.training_medians = {}  # Store medians for prediction

    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial data cleaning.
        
        Approach for TotalCharges missing values:
        - Convert to numeric (coerces invalid strings to NaN)
        - Fill NaN with 0 (assumes new customers with no charges yet)
        - Rationale: Missing TotalCharges typically occurs for new customers
          with tenure=0 or very short tenure where billing hasn't occurred.
        """
        df = pd.read_csv(config.data_url)
        
        # Handle TotalCharges missing values
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"TotalCharges missing values filled: {df['TotalCharges'].isna().sum()} remaining NaNs")
        
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create business-driven features for churn prediction.
        
        Features created:
        1. tenure_bucket: Categorical tenure grouping
        2. services_count: Number of active services
        3. monthly_to_total_ratio: Spending consistency indicator
        4. Business flags: Risk indicators based on service combinations
        5. CLV: Customer Lifetime Value
        """
        # Store medians for consistent prediction
        self.training_medians['MonthlyCharges'] = df['MonthlyCharges'].median()
        
        # 1. Tenure buckets (early customers are higher risk)
        df['tenure_bucket'] = pd.cut(
            df['tenure'], 
            bins=config.tenure_buckets,
            labels=['0-6m', '6-12m', '12-24m', '24m+']
        )

        # 2. Services count (more services = higher engagement)
        df['services_count'] = df[config.services_columns].apply(
            lambda row: (row == 'Yes').sum(), axis=1
        )

        # 3. Monthly to total ratio (spending consistency)
        # Ratio close to 1.0 = consistent spending, >1 = recent increase, <1 = decrease
        df['monthly_to_total_ratio'] = df['TotalCharges'] / np.maximum(
            1, df['tenure'] * df['MonthlyCharges']
        )

        # 4. Business risk flags
        # Customers with internet but no tech support (higher support costs)
        df['internet_no_tech_support'] = (
            (df['InternetService'] != 'No') & 
            (df['TechSupport'] == 'No')
        ).astype(int)

        # Phone service without online security (potential upsell opportunity)
        df['phone_no_security'] = (
            (df['PhoneService'] == 'Yes') & 
            (df['OnlineSecurity'] == 'No')
        ).astype(int)

        # Premium customer without backup (potential dissatisfaction)
        df['premium_no_backup'] = (
            (df['MonthlyCharges'] > self.training_medians['MonthlyCharges']) & 
            (df['OnlineBackup'] == 'No')
        ).astype(int)

        # Long tenure but low value (churn risk indicator)
        df['long_tenure_low_value'] = (
            (df['tenure'] > 24) & 
            (df['MonthlyCharges'] < self.training_medians['MonthlyCharges'])
        ).astype(int)

        # 5. Customer Lifetime Value (CLV)
        # Expected Tenure Assumption: 36 months
        # Rationale: Industry average for telco customers based on:
        # - Typical contract lengths (12-24 months)
        # - Customer renewal patterns
        # - Balance between optimistic and conservative estimates
        df['expected_tenure'] = 36
        df['clv'] = df['MonthlyCharges'] * df['expected_tenure']
        
        print(f"Features engineered. New feature count: {len(df.columns)}")
        print(f"Median MonthlyCharges (stored): ${self.training_medians['MonthlyCharges']:.2f}")
        
        return df

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder.
        
        Important: LabelEncoder sorts categories ALPHABETICALLY
        Examples:
        - Gender: Female=0, Male=1
        - PaymentMethod: Bank transfer=0, Credit card=1, Electronic check=2, Mailed check=3
        - MultipleLines: No=0, No phone service=1, Yes=2
        
        These encoders are saved and must be used consistently in prediction!
        """
        df_encoded = df.copy()

        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        categorical_columns = categorical_columns.drop(['customerID', config.target_column], errors='ignore')

        print("\nEncoding categorical features:")
        for col in categorical_columns:
            # Convert column to string type to handle mixed data types
            str_series = df[col].astype(str)
            
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(str_series)
                
                # Print encoding mapping for verification
                unique_values = sorted(str_series.unique())
                encoded_values = self.label_encoders[col].transform(unique_values)
                print(f"  {col}: {dict(zip(unique_values, encoded_values))}")

            # Transform using the fitted encoder
            df_encoded[col] = self.label_encoders[col].transform(str_series)

        return df_encoded

    def prepare_splits(self, df: pd.DataFrame) -> dict:
        """
        Create train/validation/test splits with stratification.
        
        Split Strategy: 60% train / 20% validation / 20% test
        - 60% training: Sufficient data for model learning
        - 20% validation: Hyperparameter tuning without test set leakage
        - 20% test: Final evaluation on unseen data
        
        Stratification ensures balanced churn distribution across all splits.
        """
        X = df.drop(columns=[config.target_column, 'customerID'], errors='ignore')
        y = df[config.target_column]

        # Calculate split sizes for 60/20/20
        # First split: separate 20% for test
        test_size = 0.20
        
        # Second split: from remaining 80%, take 25% for validation
        # (25% of 80% = 20% of total)
        val_size_from_temp = 0.25
        
        # First split: 80% temp (train+val) / 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            stratify=y, 
            random_state=config.model_settings['random_state']
        )

        # Second split: 75% train / 25% val (of the temp set)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size_from_temp,
            stratify=y_temp, 
            random_state=config.model_settings['random_state']
        )
        
        print("\nData split sizes:")
        print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        print(f"\nChurn distribution:")
        print(f"  Train: {y_train.value_counts(normalize=True).to_dict()}")
        print(f"  Val:   {y_val.value_counts(normalize=True).to_dict()}")
        print(f"  Test:  {y_test.value_counts(normalize=True).to_dict()}")

        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_names': list(X.columns)
        }

    def run_pipeline(self):
        """Execute full data preparation pipeline"""
        print("="*60)
        print("DATA PREPARATION PIPELINE")
        print("="*60)
        
        print("\n[1/5] Loading data...")
        df = self.load_data()

        print("\n[2/5] Engineering features...")
        df = self.engineer_features(df)

        print("\n[3/5] Encoding features...")
        df_encoded = self.encode_features(df)

        print("\n[4/5] Creating data splits...")
        splits = self.prepare_splits(df_encoded)

        print("\n[5/5] Saving processed data...")
        # Save processed data
        config.processed_data_dir.mkdir(parents=True, exist_ok=True)
        for name, data in splits.items():
            if name.startswith('X_'):
                data.to_parquet(config.processed_data_dir / f"{name}.parquet")
            else:
                pd.DataFrame(data).to_parquet(config.processed_data_dir / f"{name}.parquet")

        # Save encoders and medians with consistent naming
        config.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.label_encoders, config.models_dir / "label_encoders.pkl")
        joblib.dump(splits['feature_names'], config.models_dir / "feature_names.pkl")
        joblib.dump(self.training_medians, config.models_dir / "training_medians.pkl")

        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nProcessed files saved to: {config.processed_data_dir}")
        print(f"Model artifacts saved to: {config.models_dir}")
        
        return splits, df_encoded  # Return encoded df for CLV analysis


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    splits, df_encoded = preprocessor.run_pipeline()