"""
Robust model prediction module for customer churn prediction.
Compatible with the preprocessing pipeline and trained models.
"""

import pickle
import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class ChurnPredictor:
    """Robust class for making predictions with trained churn models.
    Handles feature engineering consistent with the preprocessing pipeline.
    """

    def __init__(self, model_name: str = "xgboost", models_dir: Optional[Union[str, Path]] = None):
        """Initialize predictor with a trained model.
        
        Args:
            model_name: Name of the model to load ('xgboost', 'random_forest', 'logistic_regression')
            models_dir: Directory containing model artifacts
        """
        self.logger = self._setup_logging()
        
        # Set paths based on project structure
        if models_dir is None:
            # Look for models in the new location
            project_root = Path(__file__).parent.parent.parent  # Go up to customer-churn-prediction
            self.models_dir = project_root / "models"
            if not self.models_dir.exists():
                # Fallback to old location for backward compatibility
                self.models_dir = project_root / "models"
        else:
            self.models_dir = Path(models_dir)
        
        self.model_name = model_name
        self.model_path = self.models_dir / f"{model_name}.pkl"
        
        if not self.model_path.exists():
            available_models = [f.stem for f in self.models_dir.glob("*.pkl") if f.stem != "label_encoders"]
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                f"Available models: {available_models}"
            )

        # Load model and related artifacts
        self.model = self._load_model(self.model_path)
        self.label_encoders = self._load_label_encoders()
        self.feature_names = self._load_feature_names()
        self.training_medians = self._load_training_medians()
        self.preprocessing_config = self._load_preprocessing_config()
        
        self.logger.info(f"Successfully loaded {self.model_name} model")
        self.logger.info(f"Model type: {type(self.model).__name__}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _load_model(self, model_path: Path) -> BaseEstimator:
        """Safely load model file with error handling and module remapping.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model object
            
        Raises:
            RuntimeError: If the model cannot be loaded or is invalid
        """
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Handle module path remapping
                if isinstance(module, str):
                    if module == 'models.train':
                        # DataCleaner was moved to transformers module
                        if name == 'DataCleaner':
                            module = 'src.models.transformers'
                        else:
                            module = 'src.models.train'
                    elif module == 'models': 
                        module = 'src.models'
                    elif module.startswith('models.'): 
                        module = 'src.' + module
                return super().find_class(module, name)

        model = None
        try:
            # Try with joblib first (preferred for scikit-learn and xgboost)
            try:
                model = joblib.load(model_path)
                self.logger.info(f"Joblib loaded model. Type: {type(model).__name__}")
            except Exception as joblib_error:
                self.logger.info(f"Joblib loading failed ({joblib_error}), trying custom unpickler...")
                with open(model_path, 'rb') as f:
                    unpickler = CustomUnpickler(f)
                    model = unpickler.load()
                    self.logger.info(f"Custom unpickler loaded model. Type: {type(model).__name__}")
            
            # --- STRICT VALIDATION --- #
            if isinstance(model, np.ndarray):
                raise ValueError("Loaded model is a numpy array, not a valid model object.")
            if not (hasattr(model, 'predict') and hasattr(model, 'predict_proba')):
                raise TypeError("Loaded object is not a valid model (missing predict/predict_proba methods).")
            
            self.logger.info("✓ Model validation successful.")
            return model

        except Exception as e:
            file_info = f"File size: {model_path.stat().st_size} bytes."
            error_msg = (
                f"FATAL: Failed to load and validate model from {model_path}.\n"
                f"File info: {file_info}\n"
                f"Error: {str(e)}\n\n"
                "This is a critical error, likely due to a corrupted or incorrectly saved model file. "
                "The application cannot proceed.\n\n"
                "TO FIX THIS:\n"
                "1. DELETE the problematic model file: {model_path}\n"
                "2. RETRAIN the model to generate a correct .pkl file. Ensure joblib.dump() is used.\n"
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _load_label_encoders(self) -> Dict[str, Any]:
        """Load label encoders if they exist."""
        encoder_path = self.models_dir / "label_encoders.pkl"
        
        if encoder_path.exists():
            try:
                encoders = joblib.load(encoder_path)
                self.logger.info(f"Loaded {len(encoders)} label encoders")
                return encoders
            except Exception as e:
                self.logger.warning(f"Failed to load encoders from {encoder_path}: {str(e)}")
        
        self.logger.info("No label encoders found - assuming features are already encoded")
        return {}

    def _load_feature_names(self) -> Optional[List[str]]:
        """Load feature names used during training."""
        feature_path = self.models_dir / "feature_names.pkl"
        
        if feature_path.exists():
            try:
                feature_names = joblib.load(feature_path)
                self.logger.info(f"Loaded {len(feature_names)} feature names")
                return feature_names
            except Exception as e:
                self.logger.warning(f"Failed to load feature names: {str(e)}")
        
        return None

    def _load_training_medians(self) -> Dict[str, float]:
        """Load training medians for consistent feature engineering."""
        medians_path = self.models_dir / "training_medians.pkl"
        
        if medians_path.exists():
            try:
                medians = joblib.load(medians_path)
                self.logger.info(f"Loaded training medians: {list(medians.keys())}")
                return medians
            except Exception as e:
                self.logger.warning(f"Failed to load training medians: {str(e)}")
        
        self.logger.warning("No training medians found - using input data medians")
        return {}

    def _load_preprocessing_config(self) -> Dict[str, Any]:
        """Load preprocessing configuration for feature engineering."""
        try:
            from config.settings import config as app_config
            return {
                'tenure_buckets': getattr(app_config, 'tenure_buckets', [0, 6, 12, 24, 72]),
                'services_columns': getattr(app_config, 'services_columns', [
                    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
                ]),
                'target_column': getattr(app_config, 'target_column', 'Churn')
            }
        except ImportError:
            self.logger.warning("Could not import config, using default values")
            return {
                'tenure_buckets': [0, 6, 12, 24, 72],
                'services_columns': [
                    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
                ],
                'target_column': 'Churn'
            }

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replicate the feature engineering from the preprocessing pipeline."""
        df_engineered = df.copy()
        
        # Handle TotalCharges like in preprocessing
        if 'TotalCharges' in df_engineered.columns:
            df_engineered['TotalCharges'] = pd.to_numeric(
                df_engineered['TotalCharges'], errors='coerce'
            ).fillna(0)
        else:
            df_engineered['TotalCharges'] = 0
        
        # Tenure buckets (matching preprocessing exactly)
        tenure_buckets = self.preprocessing_config['tenure_buckets']
        if 'tenure' in df_engineered.columns:
            df_engineered['tenure_bucket'] = pd.cut(
                df_engineered['tenure'], 
                bins=tenure_buckets,
                labels=['0-6m', '6-12m', '12-24m', '24m+'],
                include_lowest=True
            )
        
        # Services count
        services_columns = self.preprocessing_config['services_columns']
        available_services_cols = [col for col in services_columns if col in df_engineered.columns]
        if available_services_cols:
            df_engineered['services_count'] = df_engineered[available_services_cols].apply(
                lambda x: (x == 'Yes').sum(), axis=1
            )
        else:
            df_engineered['services_count'] = 0
        
        # Monthly to total ratio
        if all(col in df_engineered.columns for col in ['TotalCharges', 'tenure', 'MonthlyCharges']):
            df_engineered['monthly_to_total_ratio'] = df_engineered['TotalCharges'] / np.maximum(
                1, df_engineered['tenure'] * df_engineered['MonthlyCharges']
            )
        else:
            df_engineered['monthly_to_total_ratio'] = 0
        
        # Business flags - use training medians for consistency
        monthly_median = self.training_medians.get('MonthlyCharges', 
                         df_engineered['MonthlyCharges'].median() if 'MonthlyCharges' in df_engineered.columns else 0)
        
        # Internet no tech support
        if all(col in df_engineered.columns for col in ['InternetService', 'TechSupport']):
            df_engineered['internet_no_tech_support'] = (
                (df_engineered['InternetService'] != 'No') & 
                (df_engineered['TechSupport'] == 'No')
            ).astype(int)
        else:
            df_engineered['internet_no_tech_support'] = 0
        
        # Phone no security
        if all(col in df_engineered.columns for col in ['PhoneService', 'OnlineSecurity']):
            df_engineered['phone_no_security'] = (
                (df_engineered['PhoneService'] == 'Yes') & 
                (df_engineered['OnlineSecurity'] == 'No')
            ).astype(int)
        else:
            df_engineered['phone_no_security'] = 0
        
        # Premium no backup (using training median)
        if all(col in df_engineered.columns for col in ['MonthlyCharges', 'OnlineBackup']):
            df_engineered['premium_no_backup'] = (
                (df_engineered['MonthlyCharges'] > monthly_median) & 
                (df_engineered['OnlineBackup'] == 'No')
            ).astype(int)
        else:
            df_engineered['premium_no_backup'] = 0
        
        # Long tenure low value (using training median)
        if all(col in df_engineered.columns for col in ['tenure', 'MonthlyCharges']):
            df_engineered['long_tenure_low_value'] = (
                (df_engineered['tenure'] > 24) & 
                (df_engineered['MonthlyCharges'] < monthly_median)
            ).astype(int)
        else:
            df_engineered['long_tenure_low_value'] = 0
        
        # CLV calculation
        df_engineered['expected_tenure'] = 36
        if 'MonthlyCharges' in df_engineered.columns:
            df_engineered['clv'] = df_engineered['MonthlyCharges'] * df_engineered['expected_tenure']
        else:
            df_engineered['clv'] = 0
        
        self.logger.debug(f"Engineered features. Total columns: {len(df_engineered.columns)}")
        return df_engineered

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same encoding used during training."""
        if not self.label_encoders:
            return df

        df_encoded = df.copy()
        
        for col, encoder in self.label_encoders.items():
            if col in df_encoded.columns:
                # Handle unseen categories
                unique_values = df_encoded[col].unique()
                unseen_categories = [val for val in unique_values if val not in encoder.classes_]
                
                if unseen_categories:
                    self.logger.warning(f"Found {len(unseen_categories)} unseen categories in column '{col}': {unseen_categories}")
                    # Use the first training category for unseen values
                    default_value = encoder.classes_[0]
                    mask = df_encoded[col].isin(unseen_categories)
                    df_encoded.loc[mask, col] = default_value
                
                try:
                    df_encoded[col] = encoder.transform(df_encoded[col])
                except Exception as e:
                    self.logger.error(f"Failed to encode column '{col}': {str(e)}")
                    raise
        
        return df_encoded

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps including feature engineering and encoding."""
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Apply feature engineering (matching training pipeline)
        X_engineered = self._engineer_features(X)
        
        # Encode categorical features
        if self.label_encoders:
            # Use external label encoders if available
            X_encoded = self._encode_features(X_engineered)
        else:
            # No external encoders - use simple label encoding for categorical columns
            # This is needed for models like Logistic Regression and Random Forest
            # XGBoost can handle category dtype natively
            X_encoded = X_engineered.copy()
            
            # Encode categorical columns to numeric
            from sklearn.preprocessing import LabelEncoder
            for col in X_encoded.columns:
                if X_encoded[col].dtype == 'object' or isinstance(X_encoded[col].dtype, pd.CategoricalDtype):
                    # Simple label encoding
                    le = LabelEncoder()
                    # Convert to string first to handle both object and category dtypes
                    col_str = X_encoded[col].astype(str)
                    X_encoded[col] = le.fit_transform(col_str)
        
        # Remove target column and customerID if present
        target_column = self.preprocessing_config['target_column']
        columns_to_drop = [target_column, 'customerID']
        X_clean = X_encoded.drop(columns=[col for col in columns_to_drop if col in X_encoded.columns], errors='ignore')
        
        # Reorder features to match training
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X_clean.columns)
            if missing_features:
                self.logger.warning(f"Adding {len(missing_features)} missing features with default values")
                for feature in missing_features:
                    # Add missing features with default numeric values (0)
                    # Since all categorical features are now encoded to numeric
                    X_clean[feature] = 0
            
            # Reorder columns to match training
            X_clean = X_clean[self.feature_names]
        
        return X_clean

    def predict(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """Make churn predictions."""
        try:
            X_processed = self._prepare_features(X)
            
            # Log model type and available methods for debugging
            model_type = type(self.model).__name__
            self.logger.info(f"Model type: {model_type}")
            self.logger.info(f"Available methods: {[m for m in dir(self.model) if not m.startswith('_')]}")
            
            # Handle different model types
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(X_processed)
            elif hasattr(self.model, 'predict_proba'):
                # If only predict_proba is available, use it with a threshold
                proba = self.model.predict_proba(X_processed)
                predictions = (proba[:, 1] > 0.5).astype(int)
                self.logger.info("Used predict_proba with threshold=0.5 for predictions")
            else:
                raise AttributeError(
                    f"Model of type {model_type} does not have required prediction methods. "
                    f"Available methods: {[m for m in dir(self.model) if not m.startswith('_')]}"
                )
            
            self.logger.info(f"Successfully made predictions for {len(predictions)} samples")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            self.logger.error(f"Input data shape: {X_processed.shape if hasattr(X_processed, 'shape') else 'N/A'}")
            self.logger.error(f"Input data columns: {getattr(X_processed, 'columns', 'N/A')}")
            raise

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """Make churn probability predictions."""
        try:
            X_processed = self._prepare_features(X)
            
            # Handle pipeline vs regular model
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_processed)
            else:
                raise AttributeError("Loaded model does not have predict_proba method")
            
            self.logger.info(f"Made probability predictions for {len(probabilities)} samples")
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Probability prediction failed: {str(e)}")
            raise

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single customer."""
        try:
            # Validate required fields
            required_fields = ['tenure', 'MonthlyCharges']
            missing_fields = [field for field in required_fields if field not in customer_data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Handle TotalCharges like in preprocessing
            if 'TotalCharges' not in customer_data:
                customer_data['TotalCharges'] = 0
            else:
                try:
                    customer_data['TotalCharges'] = float(customer_data['TotalCharges'])
                except (ValueError, TypeError):
                    customer_data['TotalCharges'] = 0
            
            # Convert to DataFrame and predict
            df = pd.DataFrame([customer_data])
            prediction = self.predict(df)[0]
            proba = self.predict_proba(df)[0]
            
            result = {
                'churn_prediction': int(prediction),
                'churn_probability': float(proba[1]),
                'no_churn_probability': float(proba[0]),
                'customer_id': customer_data.get('customerID', customer_data.get('customer_id', 'unknown')),
                'model_used': self.model_name,
                'prediction_label': 'Churn' if prediction == 1 else 'No Churn'
            }
            
            self.logger.info(f"Prediction for {result['customer_id']}: {result['prediction_label']} ({result['churn_probability']:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Single prediction failed: {str(e)}")
            raise

    def batch_predict(self, customers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions for multiple customers."""
        if not customers_data:
            self.logger.warning("Empty customer data provided")
            return []

        try:
            # Convert to DataFrame for efficient processing
            df = pd.DataFrame(customers_data)
            
            # Handle TotalCharges
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            else:
                df['TotalCharges'] = 0
            
            # Get predictions
            predictions = self.predict(df)
            probabilities = self.predict_proba(df)
            
            # Compile results
            results = []
            for i, customer_data in enumerate(customers_data):
                result = {
                    'churn_prediction': int(predictions[i]),
                    'churn_probability': float(probabilities[i][1]),
                    'no_churn_probability': float(probabilities[i][0]),
                    'customer_id': customer_data.get('customerID', customer_data.get('customer_id', f'customer_{i}')),
                    'model_used': self.model_name,
                    'prediction_label': 'Churn' if predictions[i] == 1 else 'No Churn'
                }
                results.append(result)
            
            churn_count = sum(1 for r in results if r['churn_prediction'] == 1)
            self.logger.info(f"Batch prediction: {len(results)} customers, {churn_count} churn predictions")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'is_pipeline': hasattr(self.model, 'named_steps'),
            'feature_count': len(self.feature_names) if self.feature_names else 'Unknown',
            'encoders_loaded': len(self.label_encoders),
            'training_medians_loaded': bool(self.training_medians)
        }

    def save_predictions(self, predictions: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
        """Save predictions to file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(predictions)
            
            if output_path.suffix.lower() == '.csv':
                df.to_csv(output_path, index=False)
            elif output_path.suffix.lower() == '.parquet':
                df.to_parquet(output_path, index=False)
            else:
                output_path = output_path.with_suffix('.csv')
                df.to_csv(output_path, index=False)
            
            self.logger.info(f"Predictions saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save predictions: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Example of using the predictor
    try:
        predictor = ChurnPredictor(model_name="xgboost")
        
        print("Model info:", predictor.get_model_info())
        
        # Example customer data
        customer_data = {
            'customerID': 'test-001',
            'tenure': 15,
            'MonthlyCharges': 70.0,
            'TotalCharges': 1050.0,
            'InternetService': 'Fiber optic',
            'TechSupport': 'No',
            'PhoneService': 'Yes',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'Contract': 'Month-to-month'
        }
        
        result = predictor.predict_single(customer_data)
        print("Prediction result:", result)
        
    except Exception as e:
        print(f"Error: {e}")