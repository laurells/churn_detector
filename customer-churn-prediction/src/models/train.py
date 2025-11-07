"""
Enhanced model training module for customer churn prediction.
Optimized for production with efficient hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, make_scorer,
    balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import config
from src.models.transformers import DataCleaner, CategoricalEncoder


class ModelTrainer:
    """Enhanced model trainer with efficient hyperparameter tuning"""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict] = {}
        self.metrics: Dict[str, Dict] = {}
        self.val_metrics: Dict[str, Dict] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
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

    def load_data(self) -> None:
        """Load processed data splits"""
        try:
            self.logger.info("="*70)
            self.logger.info("Loading processed data...")
            self.X_train = pd.read_parquet(config.processed_data_dir / "X_train.parquet")
            self.X_val = pd.read_parquet(config.processed_data_dir / "X_val.parquet")
            self.X_test = pd.read_parquet(config.processed_data_dir / "X_test.parquet")

            # Load and convert target variables to numeric (0/1)
            self.y_train = (pd.read_parquet(config.processed_data_dir / "y_train.parquet").squeeze() == 'Yes').astype(int)
            self.y_val = (pd.read_parquet(config.processed_data_dir / "y_val.parquet").squeeze() == 'Yes').astype(int)
            self.y_test = (pd.read_parquet(config.processed_data_dir / "y_test.parquet").squeeze() == 'Yes').astype(int)

            # Verify class distribution
            train_churn_rate = (self.y_train == 1).mean() * 100
            val_churn_rate = (self.y_val == 1).mean() * 100
            test_churn_rate = (self.y_test == 1).mean() * 100

            self.logger.info(f"Data loaded successfully:")
            self.logger.info(f"  Train: {len(self.X_train)} samples ({train_churn_rate:.1f}% churn)")
            self.logger.info(f"  Val:   {len(self.X_val)} samples ({val_churn_rate:.1f}% churn)")
            self.logger.info(f"  Test:  {len(self.X_test)} samples ({test_churn_rate:.1f}% churn)")
            self.logger.info(f"  Features: {self.X_train.shape[1]}")

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _get_sample_weights(self, y: pd.Series) -> np.ndarray:
        """Calculate sample weights for imbalanced data"""
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        return np.array([class_weights[int(label)] for label in y])

    def _create_logistic_regression_pipeline(self) -> ImbPipeline:
        """Create pipeline for Logistic Regression with preprocessing and SMOTE"""
        return ImbPipeline([
            ('cleaner', DataCleaner()),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(
                sampling_strategy='auto',
                random_state=config.model_settings['random_state']
            )),
            ('classifier', LogisticRegression(
                random_state=config.model_settings['random_state'],
                max_iter=5000,
                n_jobs=-1
            ))
        ])

    def _create_random_forest_pipeline(self) -> ImbPipeline:
        """Create pipeline for Random Forest with class weights"""
        return ImbPipeline([
            ('cleaner', DataCleaner()),
            ('classifier', RandomForestClassifier(
                random_state=config.model_settings['random_state'],
                class_weight='balanced_subsample',
                n_jobs=-1,
                oob_score=True
            ))
        ])

    def _create_xgboost_pipeline(self) -> ImbPipeline:
        """Create pipeline for XGBoost with optimized settings"""
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = int((self.y_train == 0).sum() / (self.y_train == 1).sum())

        return ImbPipeline([
            ('cleaner', DataCleaner()),
            ('classifier', XGBClassifier(
                random_state=config.model_settings['random_state'],
                n_jobs=-1,
                eval_metric='logloss',
                tree_method='hist',
                enable_categorical=True,  # Enable categorical feature support
                scale_pos_weight=scale_pos_weight,
                verbosity=1
            ))
        ])

    def _get_optimized_hyperparams(self, model_type: str) -> Dict:
        """Get optimized hyperparameters for each model type"""
        if model_type == 'logistic_regression':
            return {
                'classifier__C': [0.1, 0.5, 1, 2, 5, 10],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga'],
                'smote__sampling_strategy': [0.5, 0.75, 1.0]
            }
        elif model_type == 'random_forest':
            return {
                'classifier__n_estimators': [300, 400, 500],  # Focus on more trees
                'classifier__max_depth': [15, 20, 25, 30, None],  # Even deeper - key parameter
                'classifier__min_samples_split': [2, 4, 6],
                'classifier__min_samples_leaf': [1, 2, 3, 4, 5],  # Expanded - key parameter
                'classifier__max_features': ['sqrt', 'log2', 0.25, 0.33],
                'classifier__bootstrap': [True],
                'classifier__min_impurity_decrease': [0.0, 0.001, 0.01]  # Added for pruning
            }
        elif model_type == 'xgboost':
            return {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 4, 5, 6],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__subsample': [0.8, 0.9, 1.0],
                'classifier__colsample_bytree': [0.8, 0.9, 1.0],
                'classifier__gamma': [0, 0.1, 0.2],
                'classifier__reg_alpha': [0, 0.1, 1],
                'classifier__reg_lambda': [0.5, 1, 2]
            }
        return {}

    def _create_search_cv(self, pipeline, param_grid, model_name: str):
        """Create and configure RandomizedSearchCV"""
        scoring = {
            'recall': make_scorer(recall_score, pos_label=1),
            'precision': make_scorer(precision_score, pos_label=1),
            'f1': make_scorer(f1_score, pos_label=1),
            'roc_auc': 'roc_auc'
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=config.model_settings['random_state']
        )

        # Adjust n_iter based on model type - more iterations for complex models
        # RF has large search space due to key parameters: max_depth, min_samples_leaf
        n_iter = 75 if model_name == 'random_forest' else (50 if model_name == 'xgboost' else 30)

        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            refit='roc_auc',
            n_jobs=-1,
            verbose=2,
            random_state=config.model_settings['random_state'],
            return_train_score=True
        )

    def train_logistic_regression(self) -> ImbPipeline:
        """Train Logistic Regression with advanced tuning and SMOTE"""
        self.logger.info("\n" + "="*70)
        self.logger.info("Training Logistic Regression with SMOTE")
        self.logger.info("="*70)

        param_grid = self._get_optimized_hyperparams('logistic_regression')
        pipeline = self._create_logistic_regression_pipeline()
        search = self._create_search_cv(pipeline, param_grid, 'logistic_regression')

        self.logger.info(f"Starting randomized search with parameters...")
        search.fit(self.X_train, self.y_train)

        # Store results
        self.models['logistic_regression'] = search.best_estimator_
        self.best_params['logistic_regression'] = search.best_params_

        self.logger.info(f"Best parameters: {search.best_params_}")
        self.logger.info(f"Best CV score (AUC-ROC): {search.best_score_:.4f}")

        return search.best_estimator_

    def train_random_forest(self) -> ImbPipeline:
        """Train Random Forest with advanced tuning"""
        self.logger.info("\n" + "="*70)
        self.logger.info("Training Random Forest with Advanced Tuning")
        self.logger.info("="*70)

        param_grid = self._get_optimized_hyperparams('random_forest')
        pipeline = self._create_random_forest_pipeline()
        search = self._create_search_cv(pipeline, param_grid, 'random_forest')

        self.logger.info(f"Starting randomized search...")
        search.fit(self.X_train, self.y_train)

        self.models['random_forest'] = search.best_estimator_
        self.best_params['random_forest'] = search.best_params_

        self.logger.info(f"Best parameters: {search.best_params_}")
        self.logger.info(f"Best CV score (AUC-ROC): {search.best_score_:.4f}")

        return search.best_estimator_

    def train_xgboost(self) -> ImbPipeline:
        """Train XGBoost with advanced tuning"""
        self.logger.info("\n" + "="*70)
        self.logger.info("Training XGBoost with Advanced Tuning")
        self.logger.info("="*70)

        # Calculate scale_pos_weight for class imbalance
        neg_class_count = (self.y_train == 0).sum()
        pos_class_count = (self.y_train == 1).sum()
        scale_pos_weight = neg_class_count / pos_class_count

        self.logger.info(f"Scale pos weight: {scale_pos_weight:.2f} (neg={neg_class_count}, pos={pos_class_count})")

        param_grid = self._get_optimized_hyperparams('xgboost')
        pipeline = self._create_xgboost_pipeline()
        search = self._create_search_cv(pipeline, param_grid, 'xgboost')

        self.logger.info(f"Starting randomized search...")
        search.fit(self.X_train, self.y_train)

        self.models['xgboost'] = search.best_estimator_
        self.best_params['xgboost'] = search.best_params_

        self.logger.info(f"Best parameters: {search.best_params_}")
        self.logger.info(f"Best CV score (AUC-ROC): {search.best_score_:.4f}")

        return search.best_estimator_

    def evaluate_models(self) -> None:
        """Evaluate all trained models on validation and test sets"""
        self.logger.info("\n" + "="*70)
        self.logger.info("EVALUATING MODELS")
        self.logger.info("="*70)

        for model_name, model in self.models.items():
            self.logger.info(f"\nEvaluating {model_name}...")

            # Validation metrics
            y_val_pred = model.predict(self.X_val)
            y_val_proba = model.predict_proba(self.X_val)[:, 1]

            self.val_metrics[model_name] = {
                'precision': precision_score(self.y_val, y_val_pred),
                'recall': recall_score(self.y_val, y_val_pred),
                'f1': f1_score(self.y_val, y_val_pred),
                'auc_roc': roc_auc_score(self.y_val, y_val_proba),
                'balanced_accuracy': balanced_accuracy_score(self.y_val, y_val_pred),
                'confusion_matrix': confusion_matrix(self.y_val, y_val_pred).tolist()
            }

            # Test metrics
            y_test_pred = model.predict(self.X_test)
            y_test_proba = model.predict_proba(self.X_test)[:, 1]

            self.metrics[model_name] = {
                'precision': precision_score(self.y_test, y_test_pred),
                'recall': recall_score(self.y_test, y_test_pred),
                'f1': f1_score(self.y_test, y_test_pred),
                'auc_roc': roc_auc_score(self.y_test, y_test_proba),
                'balanced_accuracy': balanced_accuracy_score(self.y_test, y_test_pred),
                'confusion_matrix': confusion_matrix(self.y_test, y_test_pred).tolist()
            }

            self.logger.info(f"  Validation AUC-ROC: {self.val_metrics[model_name]['auc_roc']:.4f}")
            self.logger.info(f"  Test AUC-ROC: {self.metrics[model_name]['auc_roc']:.4f}")
            self.logger.info(f"  Test Recall: {self.metrics[model_name]['recall']:.4f}")
            self.logger.info(f"  Test Precision: {self.metrics[model_name]['precision']:.4f}")

    def train_final_models(self) -> None:
        """Train final models on combined train+val data for deployment"""
        self.logger.info("\nTraining final models on combined train+val data...")

        # Combine train and val
        X_combined = pd.concat([self.X_train, self.X_val], axis=0)
        y_combined = pd.concat([self.y_train, self.y_val], axis=0)

        for model_name in self.models.keys():
            self.logger.info(f"  Retraining {model_name} on {len(X_combined)} samples...")

            # Get the best pipeline and retrain on combined data
            best_pipeline = self.models[model_name]
            best_pipeline.fit(X_combined, y_combined)

            # Update stored model
            self.models[model_name] = best_pipeline

    def save_models(self) -> None:
        """Save trained models and related artifacts"""
        self.logger.info("\n" + "="*70)
        self.logger.info("SAVING MODELS AND ARTIFACTS")
        self.logger.info("="*70)

        # Ensure models directory exists
        config.models_dir.mkdir(parents=True, exist_ok=True)

        # Save each model
        for model_name, model in self.models.items():
            model_path = config.models_dir / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            self.logger.info(f"  Saved {model_name} to {model_path}")

        # Save metrics
        metrics_path = config.models_dir / "model_metrics.pkl"
        joblib.dump(self.metrics, metrics_path)
        self.logger.info(f"  Saved metrics to {metrics_path}")

        # Save validation metrics
        val_metrics_path = config.models_dir / "val_metrics.pkl"
        joblib.dump(self.val_metrics, val_metrics_path)
        self.logger.info(f"  Saved validation metrics to {val_metrics_path}")

        # Save best parameters
        params_path = config.models_dir / "best_params.pkl"
        joblib.dump(self.best_params, params_path)
        self.logger.info(f"  Saved best parameters to {params_path}")
        
        # Save feature names from training data
        feature_names_path = config.models_dir / "feature_names.pkl"
        feature_names = list(self.X_train.columns)
        joblib.dump(feature_names, feature_names_path)
        self.logger.info(f"  Saved {len(feature_names)} feature names to {feature_names_path}")

    def print_final_summary(self) -> None:
        """Print final summary of model performance"""
        self.logger.info("\n" + "="*70)
        self.logger.info("FINAL SUMMARY")
        self.logger.info("="*70)

        # Create summary table
        summary_data = []
        for model_name in self.models.keys():
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Val AUC': f"{self.val_metrics[model_name]['auc_roc']:.4f}",
                'Test AUC': f"{self.metrics[model_name]['auc_roc']:.4f}",
                'Test Recall': f"{self.metrics[model_name]['recall']:.4f}",
                'Test Precision': f"{self.metrics[model_name]['precision']:.4f}",
                'Test F1': f"{self.metrics[model_name]['f1']:.4f}"
            })

        summary_df = pd.DataFrame(summary_data)
        self.logger.info("\n" + summary_df.to_string(index=False))

        # Best model
        best_model_name = max(self.val_metrics.items(), key=lambda x: x[1]['auc_roc'])[0]
        self.logger.info(f"\nBest model (by validation AUC): {best_model_name.replace('_', ' ').title()}")
        self.logger.info(f"Validation AUC: {self.val_metrics[best_model_name]['auc_roc']:.4f}")
        self.logger.info(f"Test AUC: {self.metrics[best_model_name]['auc_roc']:.4f}")

    def run_training_pipeline(self) -> Dict[str, Any]:
        """Execute full model training pipeline"""
        try:
            # Load and prepare data
            self.logger.info("\n" + "="*70)
            self.logger.info("STARTING MODEL TRAINING PIPELINE")
            self.logger.info("="*70)

            self.load_data()

            # Train models with cross-validation
            self.logger.info("\n" + "="*70)
            self.logger.info("TRAINING MODELS")
            self.logger.info("="*70)

            self.train_logistic_regression()
            self.train_random_forest()
            self.train_xgboost()

            # Evaluate models
            self.evaluate_models()

            # Train final models on combined train+val
            self.train_final_models()

            # Save models and artifacts
            self.save_models()

            # Print final summary
            self.print_final_summary()

            # Get best model based on validation AUC-ROC
            best_model_name = max(self.val_metrics.items(), key=lambda x: x[1]['auc_roc'])[0]
            best_auc = self.val_metrics[best_model_name]['auc_roc']
            best_recall = self.val_metrics[best_model_name]['recall']

            # Check if performance meets targets
            if best_auc < 0.8:
                self.logger.warning(f"\n⚠️ Validation AUC-ROC ({best_auc:.2f}) is below target (0.8)")
            if best_recall < 0.6:
                self.logger.warning(f"⚠️ Validation Recall ({best_recall:.2f}) is below target (0.6)")

            return {
                'status': 'success',
                'best_model': best_model_name,
                'val_auc_roc': best_auc,
                'val_recall': best_recall,
                'metrics': {k: v for k, v in self.metrics.items()}
            }

        except Exception as e:
            self.logger.error(f"\n❌ Error in training pipeline: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }
