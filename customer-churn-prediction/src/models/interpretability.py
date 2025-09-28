"""
Model interpretability module for customer churn prediction.
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance


class ModelInterpreter:
    """Model interpreter for understanding feature importance and predictions."""

    def __init__(self, models, feature_names):
        self.models = models
        self.feature_names = feature_names
        self.explainers = {}

    def compute_logistic_regression_importance(self, model, X_train):
        """Compute feature importance for logistic regression"""
        # Standardize coefficients by feature standard deviations
        coef = model.coef_[0]
        std_devs = X_train.std()
        importance = np.abs(coef * std_devs)
        return dict(zip(self.feature_names, importance))

    def compute_shap_explanations(self, model, X_sample, model_name):
        """Compute SHAP explanations for tree-based models"""
        if model_name in ['random_forest', 'xgboost']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            self.explainers[model_name] = explainer
            return shap_values
        return None

    def get_global_importance(self, X_train):
        """Get global feature importance for all models"""
        global_importance = {}

        for name, model in self.models.items():
            if name == 'logistic_regression':
                global_importance[name] = self.compute_logistic_regression_importance(model, X_train)
            else:
                # Sample data for SHAP (for performance)
                X_sample = X_train.sample(min(200, len(X_train)), random_state=42)
                shap_values = self.compute_shap_explanations(model, X_sample, name)
                if shap_values is not None:
                    if isinstance(shap_values, list):  # For binary classification
                        shap_values = shap_values[1]
                    global_importance[name] = np.mean(np.abs(shap_values), axis=0)

        return global_importance

    def get_local_explanation(self, model_name, X_instance):
        """Get local explanation for a single prediction"""
        if model_name in self.explainers:
            explainer = self.explainers[model_name]
            shap_values = explainer.shap_values(X_instance)
            return shap_values
        return None
