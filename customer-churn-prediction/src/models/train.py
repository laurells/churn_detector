"""
Model training module for customer churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
from config.settings import config


class ModelTrainer:
    """Enhanced model trainer with hyperparameter tuning and comprehensive evaluation."""

    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.metrics = {}

    def load_data(self):
        """Load processed data splits"""
        self.X_train = pd.read_parquet(config.processed_data_dir / "X_train.parquet")
        self.X_val = pd.read_parquet(config.processed_data_dir / "X_val.parquet")
        self.X_test = pd.read_parquet(config.processed_data_dir / "X_test.parquet")
        self.y_train = pd.read_parquet(config.processed_data_dir / "y_train.parquet").squeeze()
        self.y_val = pd.read_parquet(config.processed_data_dir / "y_val.parquet").squeeze()
        self.y_test = pd.read_parquet(config.processed_data_dir / "y_test.parquet").squeeze()

    def train_logistic_regression(self):
        """Train and tune Logistic Regression with class imbalance handling"""
        print("Training Logistic Regression...")

        # Calculate class weights for imbalance handling
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))

        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'class_weight': [class_weight_dict, None]  # Try both balanced and unbalanced
        }

        lr = LogisticRegression(random_state=config.model_settings['random_state'], max_iter=1000)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        self.models['logistic_regression'] = grid_search.best_estimator_
        self.best_params['logistic_regression'] = grid_search.best_params_

        return grid_search.best_estimator_

    def train_random_forest(self):
        """Train and tune Random Forest with class imbalance handling"""
        print("Training Random Forest...")

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]  # Handle class imbalance
        }

        rf = RandomForestClassifier(random_state=config.model_settings['random_state'], n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        self.models['random_forest'] = grid_search.best_estimator_
        self.best_params['random_forest'] = grid_search.best_params_

        return grid_search.best_estimator_

    def train_xgboost(self):
        """Train and tune XGBoost with class imbalance handling"""
        print("Training XGBoost...")

        # Calculate scale_pos_weight for class imbalance
        neg_class_count = (self.y_train == 0).sum()
        pos_class_count = (self.y_train == 1).sum()
        scale_pos_weight = neg_class_count / pos_class_count

        param_grid = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200],
            'scale_pos_weight': [1, scale_pos_weight]  # Handle class imbalance
        }

        xgb = XGBClassifier(random_state=config.model_settings['random_state'], n_jobs=-1)
        grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        self.models['xgboost'] = grid_search.best_estimator_
        self.best_params['xgboost'] = grid_search.best_params_

        return grid_search.best_estimator_

    def evaluate_models(self):
        """Evaluate all models on test set"""
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            self.metrics[name] = {
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'auc_roc': roc_auc_score(self.y_test, y_pred_proba)
            }

    def train_final_models(self):
        """Retrain best models on combined train + validation data (industry best practice)"""
        print("Final training on combined train+validation data...")

        X_full = pd.concat([self.X_train, self.X_val])
        y_full = pd.concat([self.y_train, self.y_val])

        print(f"Combined dataset: {len(X_full)} samples")

        for name, model in self.models.items():
            # Get best parameters and retrain on full data
            best_params = self.best_params[name]

            # Remove GridSearchCV-specific parameters
            clean_params = {k: v for k, v in best_params.items() if k not in ['scoring', 'cv', 'n_jobs']}

            final_model = model.__class__(**clean_params, random_state=config.model_settings['random_state'])
            final_model.fit(X_full, y_full)
            self.models[name] = final_model
            print(f"  ✅ {name} retrained on full dataset")

    def save_models(self):
        """Save trained models and metrics"""
        config.models_dir.mkdir(exist_ok=True)

        for name, model in self.models.items():
            joblib.dump(model, config.models_dir / f"{name}.pkl")

        joblib.dump(self.metrics, config.models_dir / "model_metrics.pkl")
        joblib.dump(self.best_params, config.models_dir / "best_params.pkl")

    def run_training_pipeline(self):
        """Execute full model training pipeline"""
        self.load_data()
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.evaluate_models()
        self.train_final_models()
        self.save_models()

        print("Model training completed!")
        print("Final metrics:")
        for model_name, metrics in self.metrics.items():
            print(f"{model_name}: {metrics}")
