"""
Professional Streamlit application for Customer Churn Prediction.
Modern, clean UI with comprehensive ML pipeline integration.
"""

import sys
import os
import logging
import pickle
import importlib
from pathlib import Path
from sklearn.base import BaseEstimator

# Add the project root to Python path before any imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.predict import ChurnPredictor

# Custom unpickler to handle missing 'models' module
class CustomUnpickler(pickle.Unpickler):    
    def find_class(self, module, name):
        if module == 'models':
            # Try to import from src.models instead
            try:
                module = 'src.' + module
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                # If that fails, try to find the class in the global namespace
                try:
                    return globals()[name]
                except KeyError:
                    pass
        return super().find_class(module, name)

# Path is already set at the top of the file

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
from src.models.predict import ChurnPredictor
from src.clv.analysis import CLVAnalyzer
from src.models.interpretability import ModelInterpreter
from config.settings import config



# Professional color scheme
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#2ca02c',    # Green
    'danger': '#d62728',       # Red
    'warning': '#ff7f0e',      # Orange
    'info': '#17becf',         # Cyan
    'neutral': '#7f7f7f',      # Gray
    'background': '#ffffff',   # White
    'text': '#2c3e50'          # Dark blue-gray
}

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border-radius: 6px;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #1558a0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 6px;
        border-left: 4px solid #1f77b4;
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 6px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


class ChurnPredictionApp:
    """Professional Streamlit app with full ML pipeline integration."""

    def __init__(self):
        # Initialize all attributes first
        self.models = {}
        self.predictors = {}
        self.metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.clv_analyzer = None
        self.interpreter = None
        
        # Then set up logging and load resources
        self._setup_logging()
        self.initialize_session_state()
        self.load_all_resources()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
        if 'prediction_made' not in st.session_state:
            st.session_state.prediction_made = False
        if 'global_importance_computed' not in st.session_state:
            st.session_state.global_importance_computed = False
        if 'shap_explainers_initialized' not in st.session_state:
            st.session_state.shap_explainers_initialized = set()  # Track which models have explainers

    def load_all_resources(self):
        """Load all required models, data, and analyzers"""
        try:
            # Use ChurnPredictor for consistent preprocessing
            self.predictors = {}
            self.models = {}
            model_names = ['xgboost', 'random_forest', 'logistic_regression']
            
            # Look for models in the new location first, then fall back to the old location
            project_root = Path(__file__).parent.parent  # Go up to customer-churn-prediction
            models_dir = project_root / "models"
            if not models_dir.exists():
                models_dir = project_root / "models"  # Fallback to old location
            self.logger.info(f"Using models directory: {models_dir}")
            
            for model_name in model_names:
                try:
                    # Use ChurnPredictor for consistent preprocessing
                    predictor = ChurnPredictor(model_name=model_name, models_dir=models_dir)
                    self.predictors[model_name] = predictor
                    self.models[model_name] = predictor.model  # Extract model for interpreter
                    self.logger.info(f"Loaded {model_name} via ChurnPredictor from {models_dir}")
                except Exception as e:
                    error_msg = str(e)
                    self.logger.error(f"Failed to load {model_name}: {error_msg}")
                    st.sidebar.error(f"Could not load {model_name}")
                    # Show detailed error for corrupted models
                    if "numpy array" in error_msg.lower():
                        st.sidebar.warning(f"{model_name}.pkl is corrupted. Please retrain models.")

            # Check if any models loaded successfully
            if not self.predictors:
                st.error("""
                No models could be loaded!
                
                Your model files appear to be corrupted (saved as numpy arrays instead of model objects).
                
                To fix this, retrain the models:
                ```bash
                python run_pipeline.py --skip-data --skip-clv --skip-predictions
                ```
                
                Or run the full pipeline:
                ```bash
                python run_pipeline.py
                ```
                """)
                st.session_state.models_loaded = False
                return
            
            # Load metrics
            metrics_path = config.models_dir / "model_metrics.pkl"
            self.metrics = joblib.load(metrics_path) if metrics_path.exists() else {}

            # Load data for analysis
            self._load_customer_data()
            
            # Initialize interpreter with loaded models (lazy - don't compute SHAP yet)
            if self.models and hasattr(self, 'X_train'):
                self.interpreter = ModelInterpreter(
                    self.models, 
                    self.X_train, 
                    self.X_test if hasattr(self, 'X_test') else self.X_train
                )
                # Don't compute global importance here - it's slow! Compute on-demand instead
                # self.interpreter.get_global_importance()  # Removed - compute lazily
            else:
                self.interpreter = None

            st.session_state.models_loaded = True
            st.sidebar.success(f"Loaded {len(self.predictors)} model(s) successfully")

        except Exception as e:
            st.error(f"Error loading resources: {str(e)}")
            traceback.print_exc()
    
    def _setup_logging(self):
        """Setup logging for the app"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        return self.logger

    def _load_customer_data(self):
        """Load customer data for CLV and SHAP analysis"""
        try:
            X_train_path = config.processed_data_dir / "X_train.parquet"
            X_test_path = config.processed_data_dir / "X_test.parquet"
            y_train_path = config.processed_data_dir / "y_train.parquet"
            y_test_path = config.processed_data_dir / "y_test.parquet"
            
            if X_train_path.exists():
                self.X_train = pd.read_parquet(X_train_path)
                self.X_test = pd.read_parquet(X_test_path) if X_test_path.exists() else self.X_train
                self.y_train = pd.read_parquet(y_train_path).squeeze()
                self.y_test = pd.read_parquet(y_test_path).squeeze() if y_test_path.exists() else self.y_train
                
                # Combine for CLV analysis (need decoded labels)
                customer_data = self.X_train.copy()
                customer_data['Churn'] = self.y_train
                
                # CLV already computed during preprocessing
                if 'clv' in customer_data.columns:
                    self.clv_analyzer = CLVAnalyzer(customer_data)
                else:
                    # Fallback: compute CLV
                    customer_data['clv'] = customer_data['MonthlyCharges'] * 36
                    self.clv_analyzer = CLVAnalyzer(customer_data)
            else:
                st.warning("Training data not found. Some features may be limited.")
                self.X_train = None
                self.clv_analyzer = None
                
        except Exception as e:
            st.warning(f"Could not load customer data: {str(e)}")
            self.X_train = None
            self.clv_analyzer = None

    def _load_model(self, model_path: Path) -> BaseEstimator:
        """Safely load model file with error handling."""
        try:
            # First try with the ChurnPredictor which has our custom unpickler
            model_name = model_path.stem
            predictor = ChurnPredictor(model_name=model_name, models_dir=model_path.parent)
            return predictor.model
        except Exception as e:
            self.logger.error(f"Failed to load model using ChurnPredictor: {str(e)}")
            # Fall back to direct loading if ChurnPredictor fails
            try:
                return joblib.load(model_path)
            except Exception as e2:
                self.logger.error(f"Direct loading also failed: {str(e2)}")
                raise RuntimeError(f"All model loading attempts failed. Last error: {str(e2)}")

    def prepare_customer_data(self, input_dict: dict) -> pd.DataFrame:
        """
        Prepare customer data for prediction with proper feature engineering.
        
        Args:
            input_dict: Dictionary with customer features
            
        Returns:
            DataFrame ready for model prediction
        """
        # Create base dataframe
        df = pd.DataFrame([input_dict])
        
        # Feature engineering (matching preprocessing pipeline)
        # 1. Tenure buckets
        df['tenure_bucket'] = pd.cut(
            df['tenure'], 
            bins=[-np.inf, 6, 12, 24, np.inf],
            labels=['0-6m', '6-12m', '12-24m', '24m+']
        )
        
        # 2. Services count
        service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies']
        df['services_count'] = sum((df[col] == 'Yes').astype(int) for col in service_cols if col in df.columns)
        
        # 3. Monthly to total ratio
        df['monthly_to_total_ratio'] = df['TotalCharges'] / np.maximum(
            1, df['tenure'] * df['MonthlyCharges']
        )
        
        # 4. Business flags
        df['internet_no_tech_support'] = (
            (df['InternetService'] != 'No') & 
            (df['TechSupport'] == 'No')
        ).astype(int)
        
        df['phone_no_security'] = (
            (df['PhoneService'] == 'Yes') & 
            (df['OnlineSecurity'] == 'No')
        ).astype(int)
        
        median_charges = self.training_medians.get('MonthlyCharges', 70)
        df['premium_no_backup'] = (
            (df['MonthlyCharges'] > median_charges) & 
            (df['OnlineBackup'] == 'No')
        ).astype(int)
        
        df['long_tenure_low_value'] = (
            (df['tenure'] > 24) & 
            (df['MonthlyCharges'] < median_charges)
        ).astype(int)
        
        # 5. CLV
        df['expected_tenure'] = 36
        df['clv'] = df['MonthlyCharges'] * df['expected_tenure']
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except:
                    # Handle unseen categories
                    df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only model features in correct order
        df = df[self.feature_names]
        
        return df

    def predict_with_model(self, model_name: str, customer_df: pd.DataFrame) -> dict:
        """
        Make prediction with specified model.
        
        Args:
            model_name: Name of the model to use
            customer_df: Prepared customer data
            
        Returns:
            Dictionary with prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Get prediction
        prediction = model.predict(customer_df)[0]
        prediction_proba = model.predict_proba(customer_df)[0, 1]
        
        return {
            'model_name': model_name,
            'prediction': int(prediction),
            'prediction_label': 'Churn' if prediction == 1 else 'No Churn',
            'churn_probability': float(prediction_proba),
            'confidence_level': 'High' if prediction_proba > 0.75 or prediction_proba < 0.25 else 'Medium'
        }

    def render_prediction_tab(self):
        """Render Tab 1: Predict"""
        st.header("Customer Churn Prediction")
        
        if not self.predictors:
            st.error("Models not loaded. Please ensure model training has been completed.")
            return
        
        st.markdown("### Customer Information")
        
        # Input form with validation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tenure = st.number_input(
                "Tenure (months)", 
                min_value=0, 
                max_value=100, 
                value=12,
                help="Number of months the customer has been with the company"
            )
            
            monthly_charges = st.number_input(
                "Monthly Charges ($)", 
                min_value=0.0, 
                max_value=200.0, 
                value=65.0,
                step=5.0,
                help="Monthly subscription fee"
            )
            
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                help="Type of contract agreement"
            )
            
            senior_citizen = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"],
                help="Is the customer a senior citizen?"
            )
        
        with col2:
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
                help="Type of internet service subscribed"
            )
            
            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"],
                help="Has tech support service?"
            )
            
            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"],
                help="Has online security service?"
            )
            
            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"],
                help="Has online backup service?"
            )
        
        with col3:
            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"],
                help="Has phone service?"
            )
            
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"],
                help="Has multiple phone lines?"
            )
            
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                help="Method of payment"
            )
            
            partner = st.selectbox(
                "Has Partner",
                ["No", "Yes"],
                help="Does the customer have a partner?"
            )
        
        # Model selection
        st.markdown("### Model Selection")
        selected_model = st.selectbox(
            "Choose prediction model",
            list(self.predictors.keys()),
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Select which model to use for prediction"
        )
        
        # Predict button
        if st.button("Predict Churn Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing customer data..."):
                try:
                    # Prepare input data (ChurnPredictor handles feature engineering)
                    input_dict = {
                        'customerID': 'streamlit_customer',
                        'tenure': tenure,
                        'MonthlyCharges': monthly_charges,
                        'TotalCharges': tenure * monthly_charges,
                        'Contract': contract,
                        'InternetService': internet_service,
                        'TechSupport': tech_support,
                        'OnlineSecurity': online_security,
                        'OnlineBackup': online_backup,
                        'PhoneService': phone_service,
                        'MultipleLines': multiple_lines,
                        'PaymentMethod': payment_method,
                        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
                        'Partner': partner,
                        'gender': 'Male',  # Default
                        'Dependents': 'No',  # Default
                        'DeviceProtection': 'No',  # Default
                        'StreamingTV': 'No',  # Default
                        'StreamingMovies': 'No',  # Default
                        'PaperlessBilling': 'Yes'  # Default
                    }
                    
                    # Use ChurnPredictor (handles all preprocessing internally)
                    predictor = self.predictors[selected_model]
                    result = predictor.predict_single(input_dict)
                    
                    # Display results
                    self._display_prediction_results(result, input_dict, selected_model)
                    
                    st.session_state.prediction_made = True
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

    def _display_prediction_results(self, result: dict, input_dict: dict, model_name: str):
        """Display prediction results with visualizations"""
        
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_label = "High Risk" if result['churn_prediction'] == 1 else "Low Risk"
            risk_color = COLORS['danger'] if result['churn_prediction'] == 1 else COLORS['secondary']
            st.markdown(f"<h3 style='color: {risk_color};'>{risk_label}</h3>", unsafe_allow_html=True)
            st.caption("Churn Prediction")
        
        with col2:
            st.metric("Churn Probability", f"{result['churn_probability']:.1%}")
        
        with col3:
            confidence = "High" if result['churn_probability'] > 0.75 or result['churn_probability'] < 0.25 else "Medium"
            st.metric("Confidence", confidence)
        
        with col4:
            clv = input_dict['MonthlyCharges'] * 36
            st.metric("Estimated CLV", f"${clv:,.0f}")
        
        # Probability gauge
        st.markdown("### Risk Assessment")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result['churn_probability'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk Score", 'font': {'size': 20}},
            delta={'reference': 50, 'increasing': {'color': COLORS['danger']}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': COLORS['primary']},
                'steps': [
                    {'range': [0, 33], 'color': "#e8f5e9"},
                    {'range': [33, 66], 'color': "#fff3e0"},
                    {'range': [66, 100], 'color': "#ffebee"}
                ],
                'threshold': {
                    'line': {'color': COLORS['danger'], 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Local explanation (SHAP or feature importance)
        st.markdown("### Explanation: Key Factors Influencing Prediction")
        
        if self.interpreter:
            try:
                # Prepare customer data for interpreter
                predictor = self.predictors[model_name]
                customer_df = pd.DataFrame([input_dict])
                customer_processed = predictor._prepare_features(customer_df)
                
                # Get local explanation (with progress indicator for tree models)
                if model_name in ['random_forest', 'xgboost']:
                    # For tree models, check if explainer exists, if not initialize it
                    if model_name not in self.interpreter.explainers:
                        if model_name not in st.session_state.shap_explainers_initialized:
                            with st.spinner(f"Initializing SHAP explainer for {model_name} (first time only, ~10-30 seconds)..."):
                                # Initialize explainer with small sample for faster computation
                                self.interpreter.compute_shap_explanations(model_name, n_samples=50)
                                st.session_state.shap_explainers_initialized.add(model_name)
                        else:
                            # Explainer should exist but check anyway
                            if model_name not in self.interpreter.explainers:
                                with st.spinner(f"Re-initializing SHAP explainer for {model_name}..."):
                                    self.interpreter.compute_shap_explanations(model_name, n_samples=50)
                
                # Get local explanation
                with st.spinner("Computing explanation..."):
                    explanation = self.interpreter.get_local_explanation(model_name, customer_processed)
                
                # Extract top features
                if model_name in ['random_forest', 'xgboost']:
                    # SHAP values
                    feature_impacts = []
                    for feature, values in explanation['features'].items():
                        feature_impacts.append({
                            'Feature': feature,
                            'Value': values['value'],
                            'Impact (SHAP)': values['shap_value']
                        })
                    impact_df = pd.DataFrame(feature_impacts)
                    impact_df = impact_df.reindex(
                        impact_df['Impact (SHAP)'].abs().sort_values(ascending=False).index
                    ).head(15)
                    
                    # Create SHAP waterfall plot
                    fig = go.Figure(go.Waterfall(
                        orientation="h",
                        y=impact_df['Feature'],
                        x=impact_df['Impact (SHAP)'],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        decreasing={"marker": {"color": COLORS['secondary']}},
                        increasing={"marker": {"color": COLORS['danger']}},
                        totals={"marker": {"color": COLORS['primary']}}
                    ))
                    fig.update_layout(
                        title="Top 15 Feature Contributions (SHAP Values)",
                        xaxis_title="Impact on Prediction",
                        yaxis_title="",
                        height=500,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # Logistic Regression: coefficient contributions
                    feature_impacts = []
                    for feature, values in explanation['features'].items():
                        feature_impacts.append({
                            'Feature': feature,
                            'Value': values['value'],
                            'Contribution': values['contribution']
                        })
                    impact_df = pd.DataFrame(feature_impacts)
                    impact_df = impact_df.reindex(
                        impact_df['Contribution'].abs().sort_values(ascending=False).index
                    ).head(15)
                    
                    # Create bar plot
                    fig = go.Figure(go.Bar(
                        y=impact_df['Feature'],
                        x=impact_df['Contribution'],
                        orientation='h',
                        marker_color=[COLORS['danger'] if x > 0 else COLORS['secondary'] 
                                     for x in impact_df['Contribution']]
                    ))
                    fig.update_layout(
                        title="Top 15 Feature Contributions (Coefficient × Value)",
                        xaxis_title="Contribution to Prediction",
                        yaxis_title="",
                        height=500,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation text
                st.markdown("**Interpretation:**")
                st.markdown("""
                - **Positive values** (red) increase churn probability
                - **Negative values** (green) decrease churn probability
                - Larger absolute values indicate stronger influence on the prediction
                """)
                
            except Exception as e:
                st.warning(f"Could not generate explanation: {str(e)}")
                # Fallback: Show business insights
                self._display_business_insights(input_dict)
        else:
            st.info("Model interpreter not available.")
            self._display_business_insights(input_dict)
        
        # CLV Information
        st.markdown("### Customer Lifetime Value (CLV)")
        col1, col2 = st.columns(2)
        
        with col1:
            clv = input_dict['MonthlyCharges'] * 36
            st.metric("Estimated CLV", f"${clv:,.0f}")
            st.caption("Formula: Monthly Charges × 36 months (expected tenure)")
        
        with col2:
            if clv > 2500:
                priority = "HIGH"
                color = COLORS['danger']
            elif clv > 1500:
                priority = "MEDIUM"
                color = COLORS['warning']
            else:
                priority = "STANDARD"
                color = COLORS['info']
            
            st.markdown(f"<h3 style='color: {color};'>{priority} VALUE</h3>", unsafe_allow_html=True)
            st.caption("Customer Priority Level")
        
        # Business recommendations
        st.markdown("### Recommended Actions")
        
        if result['churn_probability'] > 0.7:
            st.error("**URGENT: High Churn Risk**")
            recommendations = [
                "Immediate personal outreach by retention specialist",
                "Offer premium retention package or loyalty discount",
                "Conduct satisfaction survey to identify pain points",
                "Review account for service issues or billing problems",
                "Consider contract upgrade incentives"
            ]
        elif result['churn_probability'] > 0.4:
            st.warning("**MODERATE: Monitor Closely**")
            recommendations = [
                "Schedule proactive check-in call",
                "Send satisfaction survey",
                "Offer loyalty program enrollment",
                "Review service usage patterns",
                "Provide educational materials on service benefits"
            ]
        else:
            st.success("**LOW RISK: Stable Customer**")
            recommendations = [
                "Standard engagement protocols",
                "Consider upselling opportunities",
                "Maintain service quality",
                "Send periodic satisfaction checks",
                "Reward loyalty with small perks"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    
    def _display_business_insights(self, input_dict: dict):
        """Display business-driven insights when SHAP is unavailable"""
        st.markdown("**Key Risk Factors Based on Customer Profile:**")
        
        risk_factors = []
        
        if input_dict['Contract'] == 'Month-to-month':
            risk_factors.append("Month-to-month contract increases churn risk")
        
        if input_dict['InternetService'] == 'Fiber optic' and input_dict['TechSupport'] == 'No':
            risk_factors.append("Fiber optic without tech support may lead to dissatisfaction")
        
        if input_dict['PaymentMethod'] == 'Electronic check':
            risk_factors.append("Electronic check payment has higher failure rates")
        
        if input_dict['tenure'] < 6:
            risk_factors.append("New customers (< 6 months) have elevated churn risk")
        
        if input_dict['MonthlyCharges'] > 80:
            risk_factors.append("High monthly charges may indicate price sensitivity")
        
        if input_dict['OnlineSecurity'] == 'No' and input_dict['OnlineBackup'] == 'No':
            risk_factors.append("Lack of security services suggests low engagement")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.markdown("- Customer profile suggests stable engagement")

    def render_model_performance_tab(self):
        """Render Tab 2: Model Performance"""
        st.header("Model Performance Analysis")
        
        if not self.metrics:
            st.error("Performance metrics not available. Please train models first.")
            return
        
        # Metrics table
        st.markdown("### Performance Metrics Summary")
        
        metrics_df = pd.DataFrame(self.metrics).T
        display_metrics = metrics_df[['precision', 'recall', 'f1', 'auc_roc', 'balanced_accuracy']].copy()
        display_metrics.columns = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Balanced Accuracy']
        display_metrics.index = [idx.replace('_', ' ').title() for idx in display_metrics.index]
        
        # Highlight best scores
        styled_df = display_metrics.style.format("{:.4f}").background_gradient(
            cmap='YlGn', subset=['AUC-ROC'], vmin=0.7, vmax=0.9
        ).background_gradient(
            cmap='YlGn', subset=['F1-Score'], vmin=0.5, vmax=0.8
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Best model indicators
        st.markdown("### Top Performers")
        col1, col2, col3 = st.columns(3)
        
        best_auc = display_metrics['AUC-ROC'].idxmax()
        best_f1 = display_metrics['F1-Score'].idxmax()
        best_precision = display_metrics['Precision'].idxmax()
        
        with col1:
            st.metric(
                "Best AUC-ROC",
                best_auc,
                f"{display_metrics.loc[best_auc, 'AUC-ROC']:.4f}"
            )
        
        with col2:
            st.metric(
                "Best F1-Score",
                best_f1,
                f"{display_metrics.loc[best_f1, 'F1-Score']:.4f}"
            )
        
        with col3:
            st.metric(
                "Best Precision",
                best_precision,
                f"{display_metrics.loc[best_precision, 'Precision']:.4f}"
            )
        
        # ROC Curves (overlay)
        st.markdown("### ROC Curves Comparison")
        
        fig = go.Figure()
        
        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        # Plot ROC curves for each model (if available)
        # Note: Need to compute ROC curves from test predictions
        st.info("ROC curves require test set predictions. Computing...")
        
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            from sklearn.metrics import roc_curve, auc
            
            for model_name, model in self.models.items():
                try:
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                    # Convert y_test to binary if it's string labels
                    y_test_binary = (self.y_test == 'Yes').astype(int) if self.y_test.dtype == 'object' else self.y_test
                    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f"{model_name.replace('_', ' ').title()} (AUC={roc_auc:.3f})",
                        line=dict(width=2)
                    ))
                except Exception as e:
                    st.warning(f"Could not compute ROC for {model_name}: {str(e)}")
        
        fig.update_layout(
            title="ROC Curves - All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix for selected model
        st.markdown("### Confusion Matrix")
        
        selected_model = st.selectbox(
            "Select model for confusion matrix",
            list(self.models.keys()),
            format_func=lambda x: x.replace('_', ' ').title(),
            key='cm_model_select'
        )
        
        if selected_model in self.metrics:
            cm = np.array(self.metrics[selected_model]['confusion_matrix'])
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted No Churn', 'Predicted Churn'],
                y=['Actual No Churn', 'Actual Churn'],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title=f"Confusion Matrix - {selected_model.replace('_', ' ').title()}",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix metrics
            tn, fp, fn, tp = cm.ravel()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("True Negatives", f"{tn:,}")
                st.caption("Correctly predicted no churn")
            
            with col2:
                st.metric("False Positives", f"{fp:,}")
                st.caption("Incorrectly predicted churn")
            
            with col3:
                st.metric("False Negatives", f"{fn:,}")
                st.caption("Missed churn cases")
            
            with col4:
                st.metric("True Positives", f"{tp:,}")
                st.caption("Correctly predicted churn")
        
        # Global feature importance
        st.markdown("### Global Feature Importance")
        
        # Lazy load global importance (compute on-demand with caching)
        if self.interpreter:
            # Check if already computed in session state
            if 'global_importance_computed' not in st.session_state:
                st.session_state.global_importance_computed = False
            
            if not st.session_state.global_importance_computed:
                with st.spinner("Computing feature importance (this may take a minute for SHAP)..."):
                    try:
                        self.interpreter.get_global_importance(compute_shap=True)
                        st.session_state.global_importance_computed = True
                        st.success("Feature importance computed successfully!")
                    except Exception as e:
                        st.error(f"Error computing feature importance: {str(e)}")
                        st.session_state.global_importance_computed = False
        
        if self.interpreter and self.interpreter.global_importance:
            importance_model = st.selectbox(
                "Select model for feature importance",
                list(self.interpreter.global_importance.keys()),
                format_func=lambda x: x.replace('_', ' ').title(),
                key='importance_model_select'
            )
            
            importance_df = self.interpreter.global_importance[importance_model].head(20)
            
            # Create horizontal bar chart
            fig = go.Figure(go.Bar(
                y=importance_df['feature'][::-1],
                x=importance_df['importance'][::-1],
                orientation='h',
                marker_color=COLORS['primary'],
                text=importance_df['importance'][::-1].round(3),
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"Top 20 Features - {importance_model.replace('_', ' ').title()}",
                xaxis_title="Importance Score",
                yaxis_title="",
                height=600,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance explanation
            if importance_model == 'logistic_regression':
                st.info("""
                **Logistic Regression Feature Importance**: 
                Calculated as |coefficient × standard deviation of feature|. 
                Represents the absolute impact each feature has on the prediction.
                """)
            else:
                st.info("""
                **Tree Model Feature Importance (SHAP)**: 
                Mean absolute SHAP values across all predictions. 
                Higher values indicate features that more strongly influence predictions.
                """)
        else:
            st.warning("Feature importance not available. Run model interpretation first.")
        
        # Model selection guidance
        st.markdown("### Model Selection Guidance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**When to use each model:**")
            st.markdown("""
            - **XGBoost**: Best overall performance, production deployment
            - **Random Forest**: Robust, handles non-linear patterns well
            - **Logistic Regression**: High interpretability, regulatory compliance
            """)
        
        with col2:
            st.markdown("**Metric priorities:**")
            st.markdown("""
            - **High Precision**: Minimize false alarms, target high-value customers
            - **High Recall**: Catch all potential churners, aggressive retention
            - **Balanced F1**: General-purpose churn prediction
            - **AUC-ROC**: Overall model discrimination ability
            """)

    def render_clv_tab(self):
        """Render Tab 3: CLV Overview"""
        st.header("Customer Lifetime Value Analysis")
        
        if self.clv_analyzer is None:
            st.error("CLV analysis not available. Customer data not loaded.")
            return
        
        try:
            # Run CLV analysis
            self.clv_analyzer.compute_clv_quartiles()
            churn_analysis = self.clv_analyzer.analyze_churn_by_clv()
            
            # Key metrics
            st.markdown("### CLV Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            customer_data = self.clv_analyzer.df
            
            with col1:
                st.metric("Total Customers", f"{len(customer_data):,}")
            
            with col2:
                avg_clv = customer_data['clv'].mean()
                st.metric("Average CLV", f"${avg_clv:,.0f}")
            
            with col3:
                total_clv = customer_data['clv'].sum()
                st.metric("Total CLV", f"${total_clv:,.0f}")
            
            with col4:
                # Handle both string ('Yes'/'No') and numeric (1/0) churn values
                if customer_data['Churn'].dtype == 'object' or customer_data['Churn'].dtype.name == 'category':
                    overall_churn = (customer_data['Churn'] == 'Yes').mean()
                else:
                    overall_churn = (customer_data['Churn'] == 1).mean()
                st.metric("Overall Churn Rate", f"{overall_churn:.1%}")
            
            # CLV Distribution
            st.markdown("### CLV Distribution")
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=customer_data['clv'],
                nbinsx=50,
                marker_color=COLORS['primary'],
                opacity=0.7,
                name='CLV Distribution'
            ))
            
            # Add quartile lines
            quartiles = customer_data['clv'].quantile([0.25, 0.5, 0.75])
            for q, label in zip(quartiles, ['Q1', 'Q2 (Median)', 'Q3']):
                fig.add_vline(
                    x=q, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"{label}: ${q:,.0f}",
                    annotation_position="top"
                )
            
            fig.update_layout(
                title="Distribution of Customer Lifetime Values",
                xaxis_title="Customer Lifetime Value ($)",
                yaxis_title="Number of Customers",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Churn Rate by CLV Quartile
            st.markdown("### Churn Rate by CLV Segment")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Bar chart
                churn_data = churn_analysis.reset_index()
                churn_data['clv_quartile'] = pd.Categorical(
                    churn_data['clv_quartile'],
                    categories=['Low', 'Medium', 'High', 'Premium'],
                    ordered=True
                )
                churn_data = churn_data.sort_values('clv_quartile')
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=churn_data['clv_quartile'],
                    y=churn_data['churn_rate_pct'],
                    marker_color=[COLORS['danger'], COLORS['warning'], COLORS['info'], COLORS['secondary']],
                    text=churn_data['churn_rate_pct'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside',
                    name='Churn Rate'
                ))
                
                # Add overall churn line
                fig.add_hline(
                    y=overall_churn * 100,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Overall: {overall_churn*100:.1f}%"
                )
                
                fig.update_layout(
                    title="Churn Rate by CLV Quartile",
                    xaxis_title="CLV Segment",
                    yaxis_title="Churn Rate (%)",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Segment Breakdown:**")
                
                for segment in ['Low', 'Medium', 'High', 'Premium']:
                    if segment in churn_analysis.index:
                        count = int(churn_analysis.loc[segment, 'total_customers'])
                        rate = churn_analysis.loc[segment, 'churn_rate_pct']
                        avg_clv = churn_analysis.loc[segment, 'avg_clv']
                        
                        st.markdown(f"**{segment}**")
                        st.markdown(f"- Customers: {count:,}")
                        st.markdown(f"- Churn: {rate:.1f}%")
                        st.markdown(f"- Avg CLV: ${avg_clv:,.0f}")
                        st.markdown("")
            
            # Customer distribution by segment
            st.markdown("### Customer Distribution by Segment")
            
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'bar'}, {'type': 'pie'}]],
                subplot_titles=('Customer Count', 'CLV Value Distribution')
            )
            
            # Bar chart: Customer count
            fig.add_trace(
                go.Bar(
                    x=churn_data['clv_quartile'],
                    y=churn_data['total_customers'],
                    marker_color=[COLORS['danger'], COLORS['warning'], COLORS['info'], COLORS['secondary']],
                    showlegend=False,
                    text=churn_data['total_customers'],
                    textposition='outside'
                ),
                row=1, col=1
            )
            
            # Pie chart: Total CLV value
            segment_clv = customer_data.groupby('clv_quartile')['clv'].sum()
            fig.add_trace(
                go.Pie(
                    labels=segment_clv.index,
                    values=segment_clv.values,
                    marker_colors=[COLORS['danger'], COLORS['warning'], COLORS['info'], COLORS['secondary']],
                    textinfo='label+percent',
                    showlegend=True
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key takeaway
            st.markdown("### Key Takeaway")
            
            # Calculate insights
            premium_churn = churn_analysis.loc['Premium', 'churn_rate_pct'] if 'Premium' in churn_analysis.index else 0
            low_churn = churn_analysis.loc['Low', 'churn_rate_pct'] if 'Low' in churn_analysis.index else 0
            premium_count = int(churn_analysis.loc['Premium', 'total_customers']) if 'Premium' in churn_analysis.index else 0
            premium_avg_clv = churn_analysis.loc['Premium', 'avg_clv'] if 'Premium' in churn_analysis.index else 0
            
            # Revenue at risk calculation
            premium_at_risk = int(premium_count * (premium_churn / 100) * premium_avg_clv)
            
            takeaway = f"""
            **Strategic Priority: Focus on Premium Segment**
            
            The analysis reveals clear segmentation in customer value and churn behavior. 
            The **Premium segment** (highest 25% CLV) represents our most valuable customers 
            with an average CLV of ${premium_avg_clv:,.0f}, yet faces a {premium_churn:.1f}% churn rate. 
            This translates to approximately ${premium_at_risk:,.0f} in annual revenue at risk.
            
            **Why prioritize Premium customers:**
            - Highest lifetime value (${premium_avg_clv:,.0f} vs company average ${customer_data['clv'].mean():,.0f})
            - Lower acquisition cost to retain vs replace
            - Disproportionate impact on revenue (25% of customers contribute significantly to total CLV)
            - Better retention ROI: Every 1% reduction in Premium churn saves ${int(premium_count * 0.01 * premium_avg_clv):,.0f}
            
            **Recommended Actions:**
            1. Deploy dedicated retention team for Premium segment
            2. Implement early warning system using predictive models
            3. Offer premium-tier loyalty programs and personalized service
            4. Conduct quarterly satisfaction reviews for high-value accounts
            5. For Medium/High segments: Focus on value migration to Premium tier
            6. For Low segment: Evaluate cost-benefit of retention investments
            """
            
            st.info(takeaway)
            
            # Additional insights table
            st.markdown("### Detailed Segment Analysis")
            
            detailed_analysis = churn_analysis.copy()
            detailed_analysis['revenue_at_risk'] = (
                detailed_analysis['total_customers'] * 
                (detailed_analysis['churn_rate_pct'] / 100) * 
                detailed_analysis['avg_clv']
            )
            
            display_cols = {
                'total_customers': 'Customers',
                'churn_rate_pct': 'Churn Rate (%)',
                'avg_clv': 'Avg CLV ($)',
                'churned_count': 'Churned',
                'revenue_at_risk': 'Revenue at Risk ($)'
            }
            
            detailed_display = detailed_analysis[list(display_cols.keys())].copy()
            detailed_display.columns = list(display_cols.values())
            detailed_display.index = detailed_display.index.map(lambda x: f"{x} Value")
            
            st.dataframe(
                detailed_display.style.format({
                    'Customers': '{:,.0f}',
                    'Churn Rate (%)': '{:.1f}',
                    'Avg CLV ($)': '${:,.0f}',
                    'Churned': '{:,.0f}',
                    'Revenue at Risk ($)': '${:,.0f}'
                }).background_gradient(cmap='RdYlGn_r', subset=['Churn Rate (%)']),
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error in CLV analysis: {str(e)}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())

    def render_sidebar(self):
        """Render sidebar with app info and settings"""
        with st.sidebar:
            st.markdown("## About")
            
            st.markdown("""
            **Customer Churn Prediction Platform**
            
            Advanced ML platform for predicting customer churn 
            and analyzing customer lifetime value.
            """)
            
            st.markdown("---")
            
            st.markdown("### System Status")
            
            models_loaded = len(self.predictors) if hasattr(self, 'predictors') else 0
            st.metric("Models Loaded", models_loaded)
            
            if hasattr(self, 'X_train') and self.X_train is not None:
                st.metric("Training Samples", f"{len(self.X_train):,}")
            
            if self.metrics:
                best_model = max(self.metrics.items(), key=lambda x: x[1]['auc_roc'])[0]
                best_auc = self.metrics[best_model]['auc_roc']
                st.metric("Best Model AUC", f"{best_auc:.4f}")
            
            st.markdown("---")
            
            st.markdown("### Model Info")
            
            if hasattr(self, 'predictors') and self.predictors:
                for model_name in self.predictors.keys():
                    st.markdown(f"✓ {model_name.replace('_', ' ').title()}")
            else:
                st.warning("No models loaded")
            
            st.markdown("---")
            
            st.markdown("### Technical Stack")
            st.markdown("""
            - **ML**: XGBoost, Random Forest, Logistic Regression
            - **Interpretability**: SHAP TreeExplainer
            - **Framework**: Streamlit, Plotly
            - **Data**: Scikit-learn, Pandas
            """)


def main():
    """Main application entry point"""
    
    # Initialize app
    app = ChurnPredictionApp()
    
    # Render sidebar
    app.render_sidebar()
    
    # Main title
    st.title("Customer Churn Prediction Platform")
    st.markdown("Predict customer churn and analyze lifetime value with advanced machine learning")
    
    # Check if models are loaded
    if not st.session_state.models_loaded:
        st.error("""
        **Models not loaded.** Please ensure:
        1. Model training has been completed
        2. Model files exist in the `models/` directory
        3. All required dependencies are installed
        
        Run `python run_pipeline.py` to train models.
        """)
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Predict Churn", 
        "Model Performance", 
        "CLV Analysis"
    ])
    
    with tab1:
        app.render_prediction_tab()
    
    with tab2:
        app.render_model_performance_tab()
    
    with tab3:
        app.render_clv_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #7f7f7f;'>"
        "Professional ML Platform | Built with Modern Data Science Practices"
        "</p>", 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()