"""
Configuration settings for the churn prediction project.
"""

import os
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration class for the project."""

    def __init__(self):
        # Project paths
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.external_data_dir = self.data_dir / "external"
        self.models_dir = self.project_root / "models"
        self.notebooks_dir = self.project_root / "notebooks"

        # Data configuration
        self.data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        self.target_column = "Churn"

        # Feature engineering
        self.tenure_buckets = [0, 6, 12, 24, 72]  # 0-6m, 6-12m, 12-24m, 24m+
        self.services_columns = ['PhoneService', 'MultipleLines', 'InternetService',
                               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                               'TechSupport', 'StreamingTV', 'StreamingMovies']

        # Model settings - 60/20/20 split
        self.model_settings = {
            'test_size': 0.2,           # 20% for test
            'val_size': 0.25,          # 25% of remaining (20% of total) -> 60/20/20 split
            'random_state': 42,
            'cv_folds': 5,
            'model_types': ['random_forest', 'logistic_regression', 'svm', 'xgboost']
        }

        # Feature engineering settings
        self.feature_settings = {
            'categorical_encoders': ['label', 'onehot'],
            'scaling_method': 'standard',
            'handle_missing': 'mean',
            'create_interactions': True,
            'polynomial_features_degree': 2,
            'tenure_buckets': self.tenure_buckets,
            'services_columns': self.services_columns,
            'flag_columns': [
                'internet_no_tech_support',
                'phone_no_security',
                'premium_no_backup',
                'long_tenure_low_value'
            ]
        }

        # CLV analysis settings
        self.clv_settings = {
            'average_lifespan_months': 36,
            'discount_rate': 0.1,
            'cohort_period': 'M',
            'n_segments': 4
        }

        # App settings
        self.app_settings = {
            'debug': os.getenv('DEBUG', 'False').lower() == 'true',
            'host': os.getenv('HOST', 'localhost'),
            'port': int(os.getenv('PORT', '8501')),
            'theme': 'light'
        }

        # Data settings
        self.data_settings = {
            'chunk_size': 10000,
            'date_format': '%Y-%m-%d',
            'encoding': 'utf-8'
        }

    def get_path(self, path_type: str) -> Path:
        """Get path based on type."""
        paths = {
            'data': self.data_dir,
            'raw': self.raw_data_dir,
            'processed': self.processed_data_dir,
            'external': self.external_data_dir,
            'models': self.models_dir,
            'notebooks': self.notebooks_dir
        }
        return paths.get(path_type, self.project_root)

    def create_directories(self):
        """Create all necessary directories."""
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.external_data_dir,
            self.models_dir,
            self.notebooks_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'project_root': str(self.project_root),
            'model_settings': self.model_settings,
            'feature_settings': self.feature_settings,
            'clv_settings': self.clv_settings,
            'app_settings': self.app_settings,
            'data_settings': self.data_settings
        }


# Global config instance
config = Config()
