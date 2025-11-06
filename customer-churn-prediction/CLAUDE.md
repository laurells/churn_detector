# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A production-ready customer churn prediction system built with machine learning. The project combines churn prediction (using Logistic Regression, Random Forest, and XGBoost), customer lifetime value (CLV) analysis, and model interpretability (SHAP) in an interactive Streamlit web application.

**Key Features:**
- Multi-model churn prediction with hyperparameter tuning
- Customer Lifetime Value segmentation and business insights
- SHAP-based explainability for tree models, coefficient analysis for linear models
- Business-driven feature engineering (tenure buckets, service counts, risk flags)
- Professional Streamlit UI with prediction, performance, and CLV analysis tabs

## Development Commands

### Environment Setup
```bash
# Windows (activate virtual environment)
venv\Scripts\activate.bat

# Unix/Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Python 3.13+ compatibility issues
pip install -r requirements_py313.txt
```

### Pipeline Execution
```bash
# Run complete ML pipeline (data prep, CLV analysis, model training, demo predictions)
python run_pipeline.py

# Skip specific steps
python run_pipeline.py --skip-data        # Skip data preparation
python run_pipeline.py --skip-clv         # Skip CLV analysis
python run_pipeline.py --skip-models      # Skip model training
python run_pipeline.py --skip-predictions # Skip prediction demo
```

### Run the Application
```bash
# Launch Streamlit web interface (runs on http://localhost:8501)
streamlit run app/app.py
```

### Testing
```bash
# Test prediction module
python test_predictor.py

# Test Streamlit app
python test_app.py
```

## Architecture & Data Flow

### ML Pipeline Flow
1. **Data Acquisition** → Downloads Telco Customer Churn dataset from IBM GitHub
2. **Preprocessing** (`src/data/preprocessing.py`) → Feature engineering, encoding, train/val/test split (60/20/20)
3. **CLV Analysis** (`src/clv/analysis.py`) → Compute CLV, quartile segmentation, churn patterns
4. **Model Training** (`src/models/train.py`) → Train 3 models with hyperparameter tuning, save artifacts
5. **Evaluation** → Compute metrics (AUC-ROC, F1, precision, recall, confusion matrix)
6. **Interpretability** (`src/models/interpretability.py`) → SHAP for tree models, coefficient analysis for logistic regression
7. **Deployment** (`app/app.py`) → Streamlit interface for predictions and analysis

### Data Split Strategy
- **60% Training** → Model training with cross-validation
- **20% Validation** → Hyperparameter tuning and model selection
- **20% Test** → Final performance evaluation

**Important:** Models are trained on 60%, validated on 20%, and tested on 20%. The split uses stratification to preserve class distribution.

### Model Artifacts Location
All trained models and preprocessing objects are saved in `models/`:
- `xgboost.pkl`, `random_forest.pkl`, `logistic_regression.pkl` → Trained model pipelines
- `label_encoders.pkl` → LabelEncoder objects for categorical features
- `feature_names.pkl` → Ordered list of features (critical for prediction consistency)
- `training_medians.pkl` → Median values from training set (for filling missing values)
- `preprocessing_config.pkl` → Configuration for feature engineering
- `model_metrics.pkl` → Performance metrics dictionary

### Feature Engineering Pipeline
The preprocessing creates business-driven features:
1. **Tenure Buckets**: Categorical grouping (0-6m, 6-12m, 12-24m, 24m+) - early customers are higher churn risk
2. **Services Count**: Number of active services (higher = better engagement)
3. **Monthly to Total Ratio**: Spending consistency indicator
4. **Business Risk Flags**:
   - `internet_no_tech_support`: Internet without tech support (higher support cost)
   - `phone_no_security`: Phone without security (upsell opportunity)
   - `premium_no_backup`: High charges without backup (dissatisfaction risk)
   - `long_tenure_low_value`: Long tenure but low spend (churn indicator)
5. **CLV Calculation**: `MonthlyCharges × 36` (expected tenure)

**Critical:** The `ChurnPredictor` class handles all feature engineering internally using saved artifacts. Never manually engineer features when making predictions - always use `ChurnPredictor.predict_single()` or `ChurnPredictor.predict_batch()`.

### Module Responsibilities

**`src/data/preprocessing.py`** → DataPreprocessor class
- Downloads and cleans raw data
- Feature engineering (tenure buckets, service counts, risk flags)
- Label encoding for categorical variables
- Train/val/test splitting with stratification
- Saves processed data as Parquet files in `data/processed/`

**`src/models/train.py`** → ModelTrainer class
- Loads processed data from Parquet files
- Trains 3 models: Logistic Regression, Random Forest, XGBoost
- Uses GridSearchCV/RandomizedSearchCV for hyperparameter tuning
- Handles class imbalance with class weights (NOT SMOTE by default)
- Evaluates on validation and test sets
- Saves models and metrics to `models/`

**`src/models/predict.py`** → ChurnPredictor class
- Loads trained model and preprocessing artifacts
- Handles feature engineering for new data (consistent with training)
- Provides `predict_single()` for one customer, `predict_batch()` for multiple
- Returns predictions with probabilities and confidence levels

**`src/models/interpretability.py`** → ModelInterpreter class
- **Tree models (RF, XGB)**: Uses SHAP TreeExplainer for global and local explanations
- **Logistic Regression**: Uses standardized coefficients (coefficient × feature std)
- Provides `get_global_importance()` for feature importance ranking
- Provides `get_local_explanation()` for individual prediction explanations
- Efficient sampling (uses subset of data for SHAP computation)

**`src/clv/analysis.py`** → CLVAnalyzer class
- Computes CLV quartiles (Low/Medium/High/Premium)
- Analyzes churn rates by CLV segment
- Provides business insights on revenue at risk
- Used by Streamlit app for CLV Analysis tab

**`app/app.py`** → ChurnPredictionApp class
- Streamlit interface with 3 tabs: Predict, Model Performance, CLV Analysis
- Loads all models via ChurnPredictor (handles preprocessing automatically)
- Displays predictions with SHAP explanations (tree models) or coefficient contributions (logistic)
- Shows ROC curves, confusion matrices, feature importance
- CLV distribution and churn rate by segment

**`config/settings.py`** → Configuration management
- Centralized configuration (paths, model settings, feature engineering settings)
- Global `config` object imported throughout codebase
- Key settings: `data_url`, `model_settings`, `feature_settings`, `clv_settings`

## Key Implementation Details

### Prediction Consistency
The ChurnPredictor ensures consistency between training and prediction:
1. Uses saved `label_encoders.pkl` to encode categorical features
2. Uses saved `training_medians.pkl` to fill missing values
3. Uses saved `feature_names.pkl` to ensure correct feature order
4. Applies same feature engineering logic as training

**Never bypass ChurnPredictor** - always use it for predictions to ensure consistency.

### Model Interpretability Strategy
- **Tree models (Random Forest, XGBoost)**: Use SHAP TreeExplainer
  - Global: Mean absolute SHAP values across all predictions
  - Local: SHAP values for individual prediction (waterfall plot)
- **Logistic Regression**: Coefficient-based interpretation
  - Global: |coefficient × feature_std| for importance ranking
  - Local: coefficient × feature_value for contribution to prediction

SHAP is NOT used for logistic regression because coefficients provide exact linear contributions.

### Class Imbalance Handling
Default approach uses **class weights** (`class_weight='balanced'` for sklearn models, `scale_pos_weight` for XGBoost):
- Automatically adjusts loss function to penalize minority class errors more
- No data augmentation (SMOTE commented out in code but available)
- Preserves original data distribution

### Streamlit App Loading Pattern
The app uses ChurnPredictor for ALL predictions:
```python
# App initialization
predictor = ChurnPredictor(model_name="xgboost")

# Making predictions (predictor handles feature engineering internally)
result = predictor.predict_single(customer_dict)
```

**Important:** The app does NOT manually engineer features - it relies on ChurnPredictor's `_prepare_features()` method.

### Custom Unpickler for Model Loading
The codebase includes a custom unpickler to handle module path changes:
- Maps `models` → `src.models` when loading pickled models
- Required because models may have been saved with different module paths
- Used in both `ChurnPredictor` and `app.py`

## Common Development Patterns

### Adding a New Feature
1. Add feature engineering logic in `DataPreprocessor.engineer_features()`
2. Ensure same logic is in `ChurnPredictor._prepare_features()`
3. Retrain models: `python run_pipeline.py`
4. Test predictions: `python test_predictor.py`

### Adding a New Model
1. Add model definition in `ModelTrainer.get_model_params()`
2. Define hyperparameter grid in same method
3. Add model to `config.model_settings['model_types']` in `config/settings.py`
4. Retrain: `python run_pipeline.py`
5. Model will automatically appear in Streamlit app

### Debugging Prediction Issues
If predictions fail or give unexpected results:
1. Check `models/feature_names.pkl` matches training features
2. Verify `models/label_encoders.pkl` exists and has all categorical columns
3. Ensure input data has required columns (see sample in `run_pipeline.py` line 63-71)
4. Check `ChurnPredictor._prepare_features()` applies same engineering as `DataPreprocessor.engineer_features()`

### Updating Dependencies
Use `requirements.txt` for Python < 3.13, `requirements_py313.txt` for Python 3.13+.

If you encounter scikit-learn/numpy compatibility issues:
```bash
pip install --upgrade pip
pip install -r requirements_py313.txt
```

## Data File Locations

- **Raw data**: Downloaded to `data/raw/` (from `config.data_url`)
- **Processed data**: Parquet files in `data/processed/`
  - `X_train.parquet`, `X_val.parquet`, `X_test.parquet`
  - `y_train.parquet`, `y_val.parquet`, `y_test.parquet`
- **Model artifacts**: `models/` directory
- **Notebooks**: `notebooks/` (if any exploratory analysis)

## Critical Configuration Values

From `config/settings.py`:
- **Test size**: 20% (config.model_settings['test_size'])
- **Validation size**: 25% of remaining 80% = 20% of total (config.model_settings['val_size'])
- **Random state**: 42 (for reproducibility)
- **Tenure buckets**: [0, 6, 12, 24, 72] months
- **Expected lifetime**: 36 months (for CLV calculation)
- **Data URL**: IBM Telco Customer Churn dataset

## Troubleshooting

**"Model file not found"**
→ Run `python run_pipeline.py` to train and save models

**"Feature names mismatch"**
→ Retrain models after changing features: `python run_pipeline.py`

**"Label encoder KeyError"**
→ Ensure input data has all categorical columns; check `label_encoders.pkl`

**"SHAP explainer error"**
→ SHAP only works with tree models; logistic regression uses coefficient analysis

**ImportError for src modules**
→ Ensure you're running from project root: `customer-churn-prediction/`

**Streamlit app shows "Models not loaded"**
→ Train models first: `python run_pipeline.py`, then launch app

## Testing Before Commits

1. Run full pipeline: `python run_pipeline.py`
2. Test predictions: `python test_predictor.py`
3. Launch app and verify all 3 tabs load: `streamlit run app/app.py`
4. Make a test prediction in the app to verify end-to-end flow
