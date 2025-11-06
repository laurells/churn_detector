"""
Model interpretability module for customer churn prediction.

Implements:
- SHAP TreeExplainer for RF/XGB (global + local)
- Coefficient analysis for Logistic Regression (no SHAP)
- Efficient sampling for performance
- Visualization exports
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from config.settings import config
from scipy import sparse as sp


class ModelInterpreter:
    """
    Model interpreter for understanding feature importance and predictions.
    
    Features:
    - SHAP TreeExplainer for tree-based models (RF, XGB)
    - Standardized coefficient analysis for Logistic Regression
    - Global feature importance plots
    - Local explanations for individual predictions
    - Efficient sampling for large datasets
    """

    def __init__(self, models: Dict[str, Any], X_train: pd.DataFrame, X_test: pd.DataFrame = None):
        """
        Initialize interpreter with trained models.
        
        Args:
            models: Dictionary of trained model pipelines
            X_train: Training features (for computing statistics)
            X_test: Test features (optional, for global explanations)
        """
        self.models = models
        self.X_train = X_train
        self.X_test = X_test if X_test is not None else X_train
        self.feature_names = list(X_train.columns)
        
        # Store explainers for reuse
        self.explainers = {}
        self.shap_values_cache = {}
        self.global_importance = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Interpreter initialized with {len(models)} models")
        self.logger.info(f"Features: {len(self.feature_names)}")

    def _extract_classifier(self, model_pipeline, allow_numpy: bool = False) -> Any:
        """Extract the classifier from model object.
        
        Args:
            model_pipeline: The model or pipeline to extract classifier from
            allow_numpy: If True, allows returning numpy arrays. Set to False when using with SHAP.
            
        Returns:
            The extracted classifier or model
            
        Raises:
            ValueError: If the model is a numpy array and allow_numpy is False
        """
        self.logger.info(f"Extracting classifier from: {type(model_pipeline).__name__}")
        
        # If it's a numpy array and we don't allow numpy, raise an error
        if isinstance(model_pipeline, np.ndarray):
            if not allow_numpy:
                error_msg = (
                    "Numpy array model detected. This typically means the model was saved incorrectly.\n"
                    "Possible causes and solutions:\n"
                    "1. The model was saved using numpy.save() instead of joblib or pickle\n"
                    "   - Fix: Use joblib.dump(model, 'model.joblib') to save the model\n"
                    "2. Only the model's internal arrays were saved, not the model object\n"
                    "   - Fix: Save the entire model object, not just its parameters\n"
                    "3. The model was loaded incorrectly\n"
                    "   - Fix: Use joblib.load() to load the model, not numpy.load()"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            self.logger.warning("Using numpy array as model - this may cause issues with some operations")
            return model_pipeline
        
        # If it's a pipeline with named steps, get the classifier
        if hasattr(model_pipeline, 'named_steps'):
            classifier = model_pipeline.named_steps.get('classifier', model_pipeline)
            self.logger.info(f"Extracted from pipeline: {type(classifier).__name__}")
            return classifier
            
        # If it has a 'model' attribute (like ChurnPredictor)
        if hasattr(model_pipeline, 'model'):
            model = model_pipeline.model
            self.logger.info(f"Extracted from model attribute: {type(model).__name__}")
            return model
            
        # If it's a scikit-learn model or similar
        if hasattr(model_pipeline, 'predict'):
            self.logger.info(f"Using model directly: {type(model_pipeline).__name__}")
            return model_pipeline
            
        self.logger.warning(f"Could not determine how to extract classifier from {type(model_pipeline).__name__}")
        return model_pipeline
            
        self.logger.warning(f"Could not determine how to extract classifier from {type(model_pipeline).__name__}")
        return model_pipeline

    def _preprocess_data(self, model_pipeline, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data through pipeline up to (but not including) classifier.
        This ensures SHAP sees the same data the model sees.
        """
        if hasattr(model_pipeline, 'named_steps'):
            # Apply all preprocessing steps except final classifier
            X_processed = X.copy()
            output_feature_names = None
            for step_name, transformer in model_pipeline.named_steps.items():
                if step_name != 'classifier':
                    # Only apply steps that expose a transform method (skip SMOTE, etc.)
                    if hasattr(transformer, 'transform'):
                        X_processed = transformer.transform(X_processed)
                        # Capture transformed feature names if available
                        try:
                            if hasattr(transformer, 'get_feature_names_out'):
                                output_feature_names = list(transformer.get_feature_names_out())
                        except Exception:
                            pass
                    else:
                        self.logger.debug(f"Skipping step without transform: {step_name} ({type(transformer).__name__})")
            # Ensure dense array
            if sp.issparse(X_processed):
                X_processed = X_processed.toarray()
            # If we have valid feature names matching shape, return DataFrame for plotting-friendly usage
            if output_feature_names is not None:
                try:
                    if isinstance(X_processed, np.ndarray) and X_processed.ndim == 2 and X_processed.shape[1] == len(output_feature_names):
                        return pd.DataFrame(X_processed, columns=output_feature_names)
                except Exception:
                    pass
            return X_processed
        
        # If model has a preprocess method, use it
        if hasattr(model_pipeline, 'preprocess'):
            return model_pipeline.preprocess(X)
            
        # If model has a transform method, use it
        if hasattr(model_pipeline, 'transform'):
            return model_pipeline.transform(X)
            
        # Otherwise return as is (assume already preprocessed)
        return X.values if isinstance(X, pd.DataFrame) else X

    def _to_2d_numeric_array(self, X: Any) -> np.ndarray:
        """Ensure input is a dense 2D numeric numpy array.
        Handles pandas, scipy.sparse, and object arrays of arrays.
        """
        # Sparse
        if sp.issparse(X):
            X = X.toarray()
        # Pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        # Pandas Series where each element is an array/list
        if isinstance(X, pd.Series):
            values = X.to_numpy()
            if len(values) > 0 and isinstance(values[0], (np.ndarray, list)):
                X = np.vstack([np.asarray(v) for v in values])
            else:
                X = values.reshape(-1, 1)
        # Numpy array
        X = np.asarray(X, dtype=object)
        # Object array of arrays (1D): stack rows
        if X.ndim == 1 and X.dtype == object and len(X) > 0 and isinstance(X[0], (np.ndarray, list)):
            X = np.vstack([np.asarray(v) for v in X])
        # If still object dtype, try column-wise extraction
        if X.dtype == object:
            try:
                X = np.asarray(X, dtype=float)
            except Exception:
                cols = []
                X_2d = X if X.ndim == 2 else X.reshape(-1, 1)
                for j in range(X_2d.shape[1]):
                    col = X_2d[:, j]
                    if isinstance(col[0], (np.ndarray, list)):
                        col_arr = np.vstack([np.asarray(v).reshape(-1) for v in col])
                    else:
                        col_arr = np.asarray(col).reshape(-1, 1)
                    cols.append(col_arr)
                X = np.concatenate(cols, axis=1)
        # Final cast to float if possible
        try:
            X = np.asarray(X, dtype=float)
        except Exception:
            pass
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _sample_data_for_shap(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Sample data for SHAP computation to improve performance.
        
        Args:
            n_samples: Number of samples to use
            
        Returns:
            Sampled DataFrame from test set
        """
        if len(self.X_test) <= n_samples:
            self.logger.info(f"Using all {len(self.X_test)} test instances for SHAP")
            return self.X_test
        else:
            self.logger.info(f"Sampling {n_samples} instances for SHAP (performance optimization)")
            return self.X_test.sample(n=n_samples, random_state=42)

    def compute_logistic_regression_importance(self, model_name: str) -> pd.DataFrame:
        """
        Compute feature importance for Logistic Regression using standardized coefficients.
        
        Formula: importance = |coefficient × std_dev_of_feature|
        
        This is:
        - Faster than SHAP KernelExplainer
        - More interpretable for linear models
        - Industry-standard approach
        
        Args:
            model_name: Name of the logistic regression model
            
        Returns:
            DataFrame with features and their importance scores
        """
        self.logger.info(f"Computing coefficient-based importance for {model_name}...")
        
        model_pipeline = self.models[model_name]
        
        # Try to get the classifier, but don't fail if we can't
        try:
            classifier = self._extract_classifier(model_pipeline)
            
            # Get coefficients
            if hasattr(classifier, 'coef_'):
                coef = classifier.coef_[0]  # For binary classification
            else:
                self.logger.warning(f"Model {model_name} does not have coefficients, using zeros")
                coef = np.zeros(len(self.feature_names))
        except Exception as e:
            self.logger.warning(f"Could not extract classifier for {model_name}: {str(e)}. Using zeros for importance.")
            coef = np.zeros(len(self.feature_names))
        
        # Compute standard deviations of features
        try:
            X_processed = self._preprocess_data(model_pipeline, self.X_train)
            
            if isinstance(X_processed, pd.DataFrame):
                std_devs = X_processed.std().values
            else:
                std_devs = np.std(X_processed, axis=0)
                
            # Ensure we have the right number of features
            if len(std_devs) != len(coef):
                self.logger.warning(f"Mismatch in number of features: expected {len(coef)}, got {len(std_devs)}. Using ones for std_dev.")
                std_devs = np.ones(len(coef))
                
        except Exception as e:
            self.logger.warning(f"Could not compute feature standard deviations: {str(e)}. Using ones for std_dev.")
            std_devs = np.ones(len(coef))
        
        # Standardized importance: |coef × std_dev|
        importance = np.abs(coef * std_devs)
        
        # Ensure we have the right number of features
        if len(importance) != len(self.feature_names):
            self.logger.warning(f"Mismatch in number of features: expected {len(self.feature_names)}, got {len(importance)}. Truncating or padding.")
            importance = np.pad(importance, (0, max(0, len(self.feature_names) - len(importance))), 
                              'constant', constant_values=0)
            importance = importance[:len(self.feature_names)]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'coefficient': coef[:len(self.feature_names)],
            'std_dev': std_devs[:len(self.feature_names)]
        }).sort_values('importance', ascending=False)
        
        self.global_importance[model_name] = importance_df
        
        self.logger.info(f"✓ Top 5 features for {model_name}:")
        self.logger.info("\n" + importance_df.head(5)[['feature', 'importance']].to_string(index=False))
        
        return importance_df

    def compute_shap_explanations(self, model_name: str, X_sample: Optional[np.ndarray] = None, n_samples: int = 100) -> Tuple[Optional[Any], Optional[np.ndarray]]:
        """Compute SHAP values for a model.
    
            Args:
            model_name: Name of the model in self.models
            X_sample: Optional pre-sampled data for SHAP (must be preprocessed)
            n_samples: Number of samples to use for SHAP (if X_sample not provided)
        
        Returns:
            Tuple of (explainer, shap_values) or (None, None) if SHAP cannot be used
        """
        model = self.models[model_name]
    
        try:
            # First check if this is a numpy array model
            if isinstance(model, np.ndarray):
                self.logger.warning(f"Skipping SHAP for {model_name} - model is a numpy array")
                return None, None
            
            # Extract the classifier from the pipeline if needed
            try:
                # Don't allow numpy arrays for SHAP
                classifier = self._extract_classifier(model, allow_numpy=False)
            except ValueError as ve:
                self.logger.warning(f"Cannot use SHAP for {model_name}: {str(ve)}")
                return None, None
        
            # Double-check that the extracted classifier is not a numpy array
            if isinstance(classifier, np.ndarray):
                self.logger.warning(f"Skipping SHAP for {model_name} - extracted classifier is a numpy array")
                return None, None
        
            # Sample data if not provided
            if X_sample is None:
                X_sample = self._sample_data_for_shap(n_samples)
        
            # Preprocess the data if needed and convert to dense 2D numeric array
            try:
                X_sample_processed = self._preprocess_data(model, X_sample)
                # Robust conversion to dense 2D numeric array
                X_tree_input = self._to_2d_numeric_array(X_sample_processed)
            except Exception as preprocess_error:
                self.logger.error(f"Error preprocessing data for {model_name}: {str(preprocess_error)}")
                return None, None
        
            # Handle different model types
            try:
                self.logger.info(f"Trying TreeExplainer for {model_name}")
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_tree_input)
                try:
                    self.logger.debug(f"TreeExplainer SHAP shapes: X={X_tree_input.shape}, shap={np.array(shap_values).shape}")
                except Exception:
                    pass
            
                # If it's a list (multi-class), take the first class
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                self.logger.info(f"✓ Successfully computed SHAP values for {model_name} using TreeExplainer")
            
                # Compute global importance (mean absolute SHAP)
                if isinstance(shap_values, np.ndarray):
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names[:len(mean_abs_shap)],
                        'importance': mean_abs_shap
                    }).sort_values('importance', ascending=False)
                
                    self.global_importance[model_name] = importance_df
                
                    self.logger.info("SHAP values computed successfully.")
                    self.logger.info("Top 5 features by SHAP importance:")
                    self.logger.info("\n" + importance_df.head(5).to_string(index=False))
                else:
                    self.logger.warning("Could not compute feature importance from SHAP values")
            
                # Store for reuse
                self.explainers[model_name] = explainer
                self.shap_values_cache[model_name] = {
                    'shap_values': shap_values,
                    'X_sample': X_tree_input,
                    'X_sample_original': X_sample
                }
            
                return explainer, shap_values
            
            except Exception as e:
                self.logger.warning(f"TreeExplainer failed for {model_name}: {str(e)}")
                self.logger.info("Falling back to KernelExplainer...")
            
                # Fall back to KernelExplainer for non-tree models
                try:
                    def predict_proba_wrapper(X):
                        X_input = X
                        if isinstance(X_input, np.ndarray) and hasattr(classifier, 'feature_names_in_'):
                            try:
                                if X_input.shape[1] == len(classifier.feature_names_in_):
                                    X_input = pd.DataFrame(X_input, columns=list(classifier.feature_names_in_))
                            except Exception:
                                pass
                        if hasattr(classifier, 'predict_proba'):
                            return classifier.predict_proba(X_input)
                        else:
                            # If no predict_proba, use decision_function or predict
                            if hasattr(classifier, 'decision_function'):
                                return classifier.decision_function(X_input)
                            return classifier.predict(X_input)
                
                    # Use a smaller sample for KernelExplainer as it's slower
                    kernel_sample = X_tree_input[:min(50, len(X_tree_input))]
                    explainer = shap.KernelExplainer(predict_proba_wrapper, kernel_sample)
                    shap_values = explainer.shap_values(kernel_sample, l1_reg="num_features(10)")
                    try:
                        self.logger.debug(f"KernelExplainer SHAP shapes: X={kernel_sample.shape}, shap={np.array(shap_values).shape}")
                    except Exception:
                        pass
                
                    if isinstance(shap_values, list):
                        shap_values = np.array(shap_values)
                
                    self.logger.info(f"✓ Successfully computed SHAP values for {model_name} using KernelExplainer")
                
                    # Compute global importance (mean absolute SHAP)
                    if isinstance(shap_values, np.ndarray):
                        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                        importance_df = pd.DataFrame({
                            'feature': self.feature_names[:len(mean_abs_shap)],
                            'importance': mean_abs_shap
                        }).sort_values('importance', ascending=False)
                    
                        self.global_importance[model_name] = importance_df
                    
                        self.logger.info("SHAP values computed successfully.")
                        self.logger.info("Top 5 features by SHAP importance:")
                        self.logger.info("\n" + importance_df.head(5).to_string(index=False))
                    else:
                        self.logger.warning("Could not compute feature importance from SHAP values")
                
                    # Store for reuse
                    self.explainers[model_name] = explainer
                    self.shap_values_cache[model_name] = {
                        'shap_values': shap_values,
                    'X_sample': kernel_sample,
                        'X_sample_original': X_sample[:len(kernel_sample)]
                    }
                
                    return explainer, shap_values
                
                except Exception as ke:
                    self.logger.error(f"KernelExplainer also failed for {model_name}: {str(ke)}")
                    return None, None
                
        except Exception as e:
            self.logger.error(f"Error computing SHAP values for {model_name}: {str(e)}", exc_info=True)
            return None, None

    def get_global_importance(self, compute_shap: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Compute global feature importance for all models.
        
        - Logistic Regression: Standardized coefficients
        - RF/XGB: SHAP TreeExplainer with sampling or default feature importance
        
        Args:
            compute_shap: Whether to attempt SHAP computation (can be slow, default: True)
            
        Returns:
            Dictionary mapping model names to importance DataFrames
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("COMPUTING GLOBAL FEATURE IMPORTANCE")
        self.logger.info("="*70)
        
        for name, model in self.models.items():
            try:
                if name in self.global_importance and compute_shap:
                    self.logger.info(f"Using cached importance for {name}")
                    continue
                    
                if name == 'logistic_regression':
                    self.logger.info(f"\nComputing importance for {name} using coefficients...")
                    self.compute_logistic_regression_importance(name)
                elif name in ['random_forest', 'xgboost']:
                    if compute_shap:
                        self.logger.info(f"\nAttempting to compute SHAP importance for {name}...")
                        explainer, shap_values = self.compute_shap_explanations(name)
                        
                        # If SHAP succeeded, compute global importance from SHAP values
                        if explainer is not None and shap_values is not None:
                            self.logger.info(f"Computing global importance from SHAP values for {name}...")
                            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                            importance_df = pd.DataFrame({
                                'feature': self.feature_names[:len(mean_abs_shap)],
                                'importance': mean_abs_shap
                            }).sort_values('importance', ascending=False)
                            self.global_importance[name] = importance_df
                            self.logger.info(f"✓ Top 5 features for {name}:")
                            self.logger.info("\n" + importance_df.head(5)[['feature', 'importance']].to_string(index=False))
                        else:
                            # If SHAP failed, fall back to default feature importance
                            self.logger.warning(f"SHAP failed for {name}, falling back to default feature importance")
                            self._compute_default_feature_importance(name)
                    else:
                        self._compute_default_feature_importance(name)
                else:
                    self.logger.warning(f"No specific importance computation method for model: {name}")
                    
                    # Try to compute default feature importance if available
                    try:
                        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                            self._compute_default_feature_importance(name)
                        else:
                            # If we can't compute importance, create a dummy importance with zeros
                            self.logger.warning(f"Could not compute importance for {name}, using dummy importance")
                            self.global_importance[name] = pd.DataFrame({
                                'feature': self.feature_names,
                                'importance': np.zeros(len(self.feature_names))
                            })
                    except Exception as e:
                        self.logger.error(f"Error computing default feature importance for {name}: {str(e)}")
                        # Create a dummy importance with zeros
                        self.global_importance[name] = pd.DataFrame({
                            'feature': self.feature_names,
                            'importance': np.zeros(len(self.feature_names))
                        })
            except Exception as e:
                self.logger.error(f"Error computing importance for {name}: {str(e)}")
                # Create a dummy importance with zeros
                self.global_importance[name] = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': np.zeros(len(self.feature_names))
                })
        
        return self.global_importance
        
    def _compute_default_feature_importance(self, model_name: str) -> None:
        """Compute default feature importance for a model.
        
        This is a fallback when SHAP is not available. Handles various model types.
        """
        try:
            model = self.models[model_name]
            
            # Check if model is a numpy array
            if isinstance(model, np.ndarray):
                self.logger.warning(f"Model {model_name} is a numpy array, cannot compute feature importance")
                self.global_importance[model_name] = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': np.zeros(len(self.feature_names))
                })
                return
            
            try:
                classifier = self._extract_classifier(model, allow_numpy=True)
            except Exception as e:
                self.logger.warning(f"Could not extract classifier for {model_name}: {str(e)}")
                self.global_importance[model_name] = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': np.zeros(len(self.feature_names))
                })
                return
            
            # Check if extracted classifier is a numpy array
            if isinstance(classifier, np.ndarray):
                self.logger.warning(f"Extracted classifier for {model_name} is a numpy array, cannot compute feature importance")
                self.global_importance[model_name] = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': np.zeros(len(self.feature_names))
                })
                return
            
            # Try different ways to get feature importance
            if hasattr(classifier, 'feature_importances_'):
                # Tree-based models (Random Forest, XGBoost, etc.)
                importance = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                # Linear models (Logistic Regression, etc.)
                # Take absolute value of coefficients for importance
                if len(classifier.coef_.shape) > 1:
                    # For multi-class, take max across classes
                    importance = np.max(np.abs(classifier.coef_), axis=0)
                else:
                    importance = np.abs(classifier.coef_)
            elif hasattr(classifier, 'feature_importances'):
                # Some models might have a method instead of attribute
                importance = classifier.feature_importances()
            else:
                raise AttributeError("No feature importance method found")
            
            # Ensure we have the right number of features
            if len(importance) != len(self.feature_names):
                self.logger.warning(
                    f"Feature importance length ({len(importance)}) doesn't match "
                    f"number of features ({len(self.feature_names)}). Truncating/padding."
                )
                importance = np.pad(
                    importance, 
                    (0, max(0, len(self.feature_names) - len(importance))),
                    'constant', 
                    constant_values=0
                )
                importance = importance[:len(self.feature_names)]
            
            # Create and store the importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.global_importance[model_name] = importance_df
            
            self.logger.info(f"✓ Computed default feature importance for {model_name}")
            if not importance_df.empty:
                self.logger.info("Top 5 features by default importance:")
                self.logger.info("\n" + importance_df.head(5).to_string(index=False))
            
        except Exception as e:
            self.logger.error(f"Error computing default feature importance for {model_name}: {str(e)}")
            # Create a dummy importance with zeros as fallback
            self.global_importance[model_name] = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.zeros(len(self.feature_names))
            })
            self.logger.info(f"✓ Created dummy feature importance for {model_name}")

    def get_local_explanation(self, model_name: str, X_instance: pd.DataFrame) -> Dict[str, Any]:
        """
        Get local explanation for a single customer prediction.
        
        For tree models: Returns SHAP values
        For Logistic Regression: Returns contribution of each feature
        
        Args:
            model_name: Name of the model
            X_instance: Single customer data (1 row DataFrame)
            
        Returns:
            Dictionary with explanation data
        """
        self.logger.info(f"Computing local explanation for {model_name}...")
        
        model_pipeline = self.models[model_name]
        
        # Get prediction
        prediction = model_pipeline.predict(X_instance)[0]
        prediction_proba = model_pipeline.predict_proba(X_instance)[0, 1]
        
        explanation = {
            'model_name': model_name,
            'prediction': int(prediction),
            'prediction_proba': float(prediction_proba),
            'features': {}
        }
        
        if model_name == 'logistic_regression':
            # Coefficient-based explanation
            classifier = self._extract_classifier(model_pipeline)
            X_processed = self._preprocess_data(model_pipeline, X_instance)
            if isinstance(X_processed, np.ndarray) and hasattr(self._extract_classifier(model_pipeline), 'feature_names_in_'):
                clf = self._extract_classifier(model_pipeline)
                try:
                    if X_processed.shape[1] == len(clf.feature_names_in_):
                        X_processed = pd.DataFrame(X_processed, columns=list(clf.feature_names_in_))
                except Exception:
                    pass
            # Ensure numeric array for SHAP
            X_tree_input = self._to_2d_numeric_array(X_processed)
            
            if isinstance(X_processed, pd.DataFrame):
                X_values = X_processed.values[0]
            else:
                X_values = X_processed[0]
            
            coef = classifier.coef_[0]
            contributions = X_values * coef
            
            for i, feature in enumerate(self.feature_names):
                explanation['features'][feature] = {
                    'value': float(X_instance[feature].values[0]),
                    'contribution': float(contributions[i]),
                    'coefficient': float(coef[i])
                }
        
        elif model_name in ['random_forest', 'xgboost']:
            # SHAP-based explanation
            if model_name not in self.explainers:
                # Initialize explainer if not already done
                self.compute_shap_explanations(model_name, n_samples=100)
            
            explainer = self.explainers[model_name]
            X_processed = self._preprocess_data(model_pipeline, X_instance)
            # Ensure numeric array for SHAP
            X_tree_input = self._to_2d_numeric_array(X_processed)
            
            # Compute SHAP values for this instance
            shap_values = explainer.shap_values(X_tree_input)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            if len(shap_values.shape) == 2:
                shap_values = shap_values[0]
            
            for i, feature in enumerate(self.feature_names):
                explanation['features'][feature] = {
                    'value': float(X_instance[feature].values[0]),
                    'shap_value': float(shap_values[i])
                }
        
        return explanation

    def plot_global_importance(self, model_name: str, top_n: int = 20, 
                               save_path: Path = None) -> plt.Figure:
        """
        Plot global feature importance for a model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to show
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if model_name not in self.global_importance:
            self.logger.warning(f"Global importance not computed for {model_name}")
            return None
        
        importance_df = self.global_importance[model_name].head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Horizontal bar plot
        y_pos = np.arange(len(importance_df))
        bars = ax.barh(y_pos, importance_df['importance'].values, alpha=0.8)
        
        # Color bars by importance
        colors = plt.cm.RdYlGn_r(importance_df['importance'].values / importance_df['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} Features - {model_name.replace("_", " ").title()}', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path / f'{model_name}_global_importance.png', 
                       dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved global importance plot to {save_path}")
        
        return fig

    def plot_shap_summary(self, model_name: str, save_path: Path = None) -> plt.Figure:
        """
        Create SHAP summary plot (beeswarm plot) for tree-based models.
        
        Args:
            model_name: Name of the model (must be 'random_forest' or 'xgboost')
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if model_name not in ['random_forest', 'xgboost']:
            self.logger.warning(f"SHAP summary plot only for tree models, not {model_name}")
            return None
        
        if model_name not in self.shap_values_cache:
            self.logger.warning(f"SHAP values not computed for {model_name}")
            return None
        
        cache = self.shap_values_cache[model_name]
        shap_values = cache['shap_values']
        X_sample = cache['X_sample']
        
        # Create SHAP summary plot
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, 
                         show=False, max_display=20)
        plt.title(f'SHAP Summary - {model_name.replace("_", " ").title()}', 
                 fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / f'{model_name}_shap_summary.png', 
                       dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved SHAP summary plot to {save_path}")
        
        return fig

    def plot_shap_waterfall(self, model_name: str, instance_idx: int = 0, 
                           save_path: Path = None) -> plt.Figure:
        """
        Create SHAP waterfall plot for a single prediction (local explanation).
        
        Args:
            model_name: Name of the model
            instance_idx: Index of instance in test set
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if model_name not in ['random_forest', 'xgboost']:
            self.logger.warning(f"SHAP waterfall only for tree models, not {model_name}")
            return None
        
        if model_name not in self.explainers:
            self.logger.warning(f"SHAP explainer not initialized for {model_name}")
            return None
        
        # Get single instance
        X_instance = self.X_test.iloc[[instance_idx]]
        X_processed = self._preprocess_data(self.models[model_name], X_instance)
        
        # Get SHAP explanation
        explainer = self.explainers[model_name]
        shap_values = explainer.shap_values(X_processed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create waterfall plot
        fig = plt.figure(figsize=(10, 8))
        
        # Create explanation object for waterfall
        if isinstance(X_processed, pd.DataFrame):
            X_display = X_processed.iloc[0]
        else:
            X_display = pd.Series(X_processed[0], index=self.feature_names)
        
        shap_explanation = shap.Explanation(
            values=shap_values[0] if len(shap_values.shape) == 2 else shap_values,
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) 
                       else explainer.expected_value[1],
            data=X_display.values,
            feature_names=self.feature_names
        )
        
        shap.waterfall_plot(shap_explanation, show=False, max_display=15)
        plt.title(f'SHAP Explanation - Customer {instance_idx}', 
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / f'{model_name}_waterfall_customer_{instance_idx}.png', 
                       dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved SHAP waterfall plot to {save_path}")
        
        return fig

    def compare_model_importances(self, top_n: int = 15, save_path: Path = None) -> plt.Figure:
        """
        Compare feature importance across all models.
        
        Args:
            top_n: Number of top features to compare
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not self.global_importance:
            self.logger.warning("No global importance computed yet")
            return None
        
        # Get union of top features across all models
        all_features = set()
        for importance_df in self.global_importance.values():
            all_features.update(importance_df.head(top_n)['feature'].tolist())
        
        # Create comparison DataFrame
        comparison_data = []
        for feature in all_features:
            row = {'feature': feature}
            for model_name, importance_df in self.global_importance.items():
                importance_value = importance_df[importance_df['feature'] == feature]['importance'].values
                row[model_name] = importance_value[0] if len(importance_value) > 0 else 0
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by average importance
        model_cols = [col for col in comparison_df.columns if col != 'feature']
        comparison_df['avg_importance'] = comparison_df[model_cols].mean(axis=1)
        comparison_df = comparison_df.sort_values('avg_importance', ascending=False).head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(comparison_df))
        width = 0.25
        
        for i, model_name in enumerate(model_cols):
            offset = (i - len(model_cols)/2 + 0.5) * width
            ax.bar(x + offset, comparison_df[model_name], width, 
                  label=model_name.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel('Features', fontsize=11, fontweight='bold')
        ax.set_ylabel('Importance', fontsize=11, fontweight='bold')
        ax.set_title(f'Feature Importance Comparison Across Models (Top {top_n})', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['feature'], rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path / 'model_importance_comparison.png', 
                       dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved comparison plot to {save_path}")
        
        return fig

    def generate_all_plots(self, save_dir: Path = None) -> Dict[str, plt.Figure]:
        """
        Generate all interpretability plots.
        
        Args:
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of figure objects
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("\n" + "="*70)
        self.logger.info("GENERATING INTERPRETABILITY PLOTS")
        self.logger.info("="*70)
        
        figures = {}
        
        # Global importance plots for each model
        for model_name in self.models.keys():
            if model_name in self.global_importance:
                fig = self.plot_global_importance(model_name, save_path=save_dir)
                figures[f'{model_name}_importance'] = fig
        
        # SHAP summary plots for tree models
        for model_name in ['random_forest', 'xgboost']:
            if model_name in self.models and model_name in self.shap_values_cache:
                fig = self.plot_shap_summary(model_name, save_path=save_dir)
                figures[f'{model_name}_shap_summary'] = fig
        
        # Model comparison
        fig = self.compare_model_importances(save_path=save_dir)
        figures['comparison'] = fig
        
        self.logger.info(f"\n✓ Generated {len(figures)} interpretability plots")
        
        return figures

    def save_importance_tables(self, save_dir: Path) -> None:
        """
        Save feature importance tables as CSV files.
        
        Args:
            save_dir: Directory to save CSV files
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, importance_df in self.global_importance.items():
            csv_path = save_dir / f'{model_name}_feature_importance.csv'
            importance_df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved {model_name} importance table to {csv_path}")