"""
Utility functions for the churn prediction project.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def ensure_directory_exists(directory_path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data_safely(file_path: str, **kwargs) -> pd.DataFrame:
    """Safely load data with error handling."""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, **kwargs)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path, **kwargs)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def save_data_safely(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """Safely save data with error handling."""
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        if file_path.endswith('.csv'):
            df.to_csv(file_path, **kwargs)
        elif file_path.endswith('.xlsx'):
            df.to_excel(file_path, **kwargs)
        elif file_path.endswith('.json'):
            df.to_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {str(e)}")
        raise


def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """Calculate memory usage of DataFrame."""
    memory_usage = df.memory_usage(deep=True).sum()
    memory_mb = memory_usage / (1024 ** 2)

    return {
        'memory_bytes': memory_usage,
        'memory_mb': f"{memory_mb:.2f} MB",
        'shape': str(df.shape)
    }


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get information about a file."""
    path = Path(file_path)
    return {
        'name': path.name,
        'size': path.stat().st_size,
        'modified': path.stat().st_mtime,
        'exists': path.exists()
    }


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample customer data for testing."""
    np.random.seed(42)

    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'tenure': np.random.randint(1, 120, n_samples),
        'monthly_charges': np.random.uniform(20, 200, n_samples),
        'total_charges': np.random.uniform(100, 10000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }

    return pd.DataFrame(data)
