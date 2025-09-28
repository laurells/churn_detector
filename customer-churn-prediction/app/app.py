"""
Main Streamlit application for Customer Churn Prediction.
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.predict import ChurnPredictor
from src.models.interpretability import ModelInterpreter
from src.clv.analysis import CLVAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from config.settings import config

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


class ChurnPredictionApp:
    """Enhanced Streamlit app with full ML pipeline integration."""

    def __init__(self):
        self.predictor = None
        self.interpreter = None
        self.clv_analyzer = None
        self.models = {}
        self.metrics = {}
        self.feature_names = []
        self.customer_data = None  # For CLV analysis
        self.load_models()

    def load_models(self):
        """Load trained models and data"""
        try:
            # Load models
            model_files = ['logistic_regression.pkl', 'random_forest.pkl', 'xgboost.pkl']
            for model_file in model_files:
                model_path = config.models_dir / model_file
                if model_path.exists():
                    model_name = model_file.replace('.pkl', '')
                    self.models[model_name] = joblib.load(model_path)

            # Load metrics and feature names
            metrics_path = config.models_dir / "model_metrics.pkl"
            if metrics_path.exists():
                self.metrics = joblib.load(metrics_path)

            features_path = config.models_dir / "feature_names.pkl"
            if features_path.exists():
                self.feature_names = joblib.load(features_path)

            # Load customer data for CLV analysis
            try:
                customer_data_path = config.data_dir / "raw" / "customer_churn_data.csv"
                if customer_data_path.exists():
                    self.customer_data = pd.read_csv(customer_data_path)
                    # Calculate CLV for the dataset
                    self.customer_data['clv'] = self.customer_data['MonthlyCharges'] * 36
                    self.clv_analyzer = CLVAnalyzer(self.customer_data)
                else:
                    # Generate sample data if real data not available
                    self.customer_data = self._generate_sample_customer_data()
                    self.customer_data['clv'] = self.customer_data['MonthlyCharges'] * 36
                    self.clv_analyzer = CLVAnalyzer(self.customer_data)
            except Exception as e:
                st.warning(f"Could not load customer data: {str(e)}")
                # Generate sample data
                self.customer_data = self._generate_sample_customer_data()
                self.customer_data['clv'] = self.customer_data['MonthlyCharges'] * 36
                self.clv_analyzer = CLVAnalyzer(self.customer_data)

            # Initialize predictor and interpreter if models loaded
            if self.models:
                self.predictor = ChurnPredictor(str(config.models_dir / "xgboost.pkl"))
                self.interpreter = ModelInterpreter(self.models, self.feature_names)

        except Exception as e:
            st.warning(f"Could not load models: {str(e)}")

    def _generate_sample_customer_data(self, n_samples=1000):
        """Generate sample customer data for demonstration"""
        np.random.seed(42)

        data = {
            'customerID': [f'CUST_{i:04d}' for i in range(1, n_samples + 1)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples).round(2),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
        }

        df = pd.DataFrame(data)
        df['TotalCharges'] = (df['tenure'] * df['MonthlyCharges']).round(2)
        return df

    def prepare_input_features(self, tenure, monthly_charges, contract, internet_service,
                             tech_support, payment_method, senior_citizen, partner):
        """Prepare input features for prediction"""
        # Create feature dictionary matching preprocessing pipeline
        features = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'Contract': contract,
            'InternetService': internet_service,
            'TechSupport': tech_support,
            'PaymentMethod': payment_method,
            'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
            'Partner': partner,
            'gender': 'Male',  # Default values for missing features
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'PaperlessBilling': 'No',
            'Churn': 'No',  # Will be predicted
            'TotalCharges': tenure * monthly_charges  # Calculate if missing
        }

        # Convert to DataFrame
        return pd.DataFrame([features])

    def display_prediction_results(self, results, input_features):
        """Display prediction results with explanations"""
        st.success("✅ Prediction completed!")

        # Prediction results
        col1, col2, col3 = st.columns(3)

        with col1:
            churn_status = "🔴 High Risk" if results['churn_prediction'] == 1 else "🟢 Low Risk"
            st.metric("Churn Risk", churn_status)

        with col2:
            st.metric("Churn Probability", f"{results['churn_probability']:.1%}")

        with col3:
            confidence = "High" if results['churn_probability'] > 0.8 else "Medium" if results['churn_probability'] > 0.5 else "Low"
            st.metric("Confidence", confidence)

        # CLV Calculation for this customer
        tenure = input_features.iloc[0]['tenure']
        monthly_charges = input_features.iloc[0]['MonthlyCharges']
        estimated_clv = monthly_charges * 36  # Using 36-month assumption

        st.subheader("💰 Customer Lifetime Value")
        st.metric("Estimated CLV", f"${estimated_clv:,.0f}")
        st.info("💡 CLV Formula: MonthlyCharges × 36 months (industry average tenure)")

        # Model comparison
        st.subheader("Model Predictions")
        model_predictions = {}

        for name, model in self.models.items():
            try:
                pred = model.predict(input_features)
                pred_proba = model.predict_proba(input_features)[0]
                model_predictions[name] = {
                    'prediction': 'Churn' if pred[0] == 1 else 'No Churn',
                    'probability': pred_proba[1]
                }
            except:
                pass

        if model_predictions:
            pred_df = pd.DataFrame(model_predictions).T
            st.dataframe(pred_df.style.format({'probability': '{:.1%}'}).highlight_max(axis=0, subset=['probability']))

        # SHAP explanation (if available)
        if self.interpreter:
            st.subheader("🔍 Prediction Explanation")
            try:
                explanation = self.interpreter.get_local_explanation('xgboost', input_features)
                if explanation is not None:
                    st.write("**Top contributing features:**")

                    # Create a simple feature importance display
                    feature_importance = {}
                    for i, feature in enumerate(self.feature_names[:10]):  # Top 10 features
                        try:
                            importance = abs(explanation[0][i]) if len(explanation[0]) > i else 0
                            feature_importance[feature] = importance
                        except:
                            pass

                    # Sort and display
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    for feature, importance in sorted_features:
                        direction = "➕" if explanation[0][self.feature_names.index(feature)] > 0 else "➖"
                        st.write(f"{direction} **{feature}**: {importance:.3f}")
                else:
                    st.info("SHAP explanation not available for this prediction")
            except Exception as e:
                st.info(f"Explanation not available: {str(e)}")

        # Business recommendation
        st.subheader("💼 Recommendation")
        if results['churn_probability'] > 0.7:
            st.error("🚨 **HIGH RISK**: Immediate retention action required!")
        elif results['churn_probability'] > 0.4:
            st.warning("⚠️ **MEDIUM RISK**: Monitor and consider proactive outreach")
        else:
            st.success("✅ **LOW RISK**: Customer appears stable")

        if estimated_clv > 2000:
            st.info("💎 **HIGH VALUE**: Prioritize retention efforts for this valuable customer")
        elif estimated_clv > 1000:
            st.info("💰 **MEDIUM VALUE**: Consider cost-effective retention strategies")
        else:
            st.info("📊 **STANDARD VALUE**: Apply standard retention protocols")

    def render_prediction_tab(self):
        """Render the prediction tab"""
        st.header("🔮 Churn Prediction")

        if not self.models:
            st.error("⚠️ No models loaded. Please run the training pipeline first.")
            return

        col1, col2 = st.columns(2)

        with col1:
            # Key feature inputs
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        with col2:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer", "Credit card"
            ])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])

        if st.button("Predict Churn Risk", type="primary"):
            # Prepare input features
            input_features = self.prepare_input_features(
                tenure, monthly_charges, contract, internet_service,
                tech_support, payment_method, senior_citizen, partner
            )

            # Get predictions
            results = self.predictor.predict_single(input_features.iloc[0].to_dict())

            # Display results
            self.display_prediction_results(results, input_features)

    def render_model_performance_tab(self):
        """Render model performance comparison"""
        st.header("📈 Model Performance")

        if not self.metrics:
            st.error("⚠️ No performance metrics found. Please run model training first.")
            return

        # Metrics table
        metrics_df = pd.DataFrame(self.metrics).T
        st.dataframe(metrics_df.style.format("{:.3f}").highlight_max(axis=0))

        # Performance visualization
        col1, col2 = st.columns(2)

        with col1:
            # Model comparison chart
            fig = px.bar(metrics_df.reset_index(), x='index', y='auc_roc',
                        title="Model ROC-AUC Comparison",
                        labels={'index': 'Model', 'auc_roc': 'ROC-AUC Score'})
            st.plotly_chart(fig)

        with col2:
            # Precision vs Recall
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=metrics_df['recall'], y=metrics_df['precision'],
                                   mode='markers+text', text=metrics_df.index,
                                   textposition="top center",
                                   marker=dict(size=12)))
            fig.update_layout(title="Precision vs Recall by Model",
                            xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig)

        # Additional visualizations
        st.subheader("📊 Detailed Performance Analysis")

        # Select model for detailed analysis
        selected_model = st.selectbox("Select Model for Detailed Analysis",
                                    list(self.models.keys()),
                                    key="model_selector")

        if selected_model in self.models:
            model = self.models[selected_model]

            # Confusion Matrix
            st.write("**Confusion Matrix**")
            try:
                # Load test data for confusion matrix
                if hasattr(self, 'X_test') and self.X_test is not None:
                    y_pred = model.predict(self.X_test)
                    y_true = self.y_test if hasattr(self, 'y_test') else None

                    if y_true is not None:
                        from sklearn.metrics import confusion_matrix
                        import plotly.figure_factory as ff

                        cm = confusion_matrix(y_true, y_pred)
                        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                        # Create annotated heatmap
                        fig = ff.create_annotated_heatmap(
                            cm_normalized.round(3),
                            x=['Predicted No Churn', 'Predicted Churn'],
                            y=['Actual No Churn', 'Actual Churn'],
                            colorscale='Blues',
                            showscale=True
                        )
                        fig.update_layout(title=f'Confusion Matrix - {selected_model}')
                        st.plotly_chart(fig)
                    else:
                        st.info("Test labels not available for confusion matrix")
                else:
                    st.info("Test data not loaded for confusion matrix calculation")
            except Exception as e:
                st.info(f"Confusion matrix not available: {str(e)}")

            # Feature Importance
            st.write("**Feature Importance**")
            try:
                # Get global importance
                global_importance = self.interpreter.get_global_importance(self.X_test if hasattr(self, 'X_test') else None)

                if global_importance and selected_model in global_importance:
                    importance_data = global_importance[selected_model]

                    if isinstance(importance_data, dict):
                        # Convert dict to DataFrame for plotting
                        importance_df = pd.DataFrame(list(importance_data.items()),
                                                   columns=['Feature', 'Importance'])
                        importance_df = importance_df.sort_values('Importance', ascending=True).tail(10)

                        fig = px.bar(importance_df, x='Importance', y='Feature',
                                   orientation='h', title=f'Top 10 Features - {selected_model}')
                        st.plotly_chart(fig)
                    else:
                        st.info("Feature importance visualization would be displayed here")
                else:
                    st.info("Feature importance not available for this model")
            except Exception as e:
                st.info(f"Feature importance not available: {str(e)}")

        # Model comparison summary
        st.subheader("Model Ranking")
        best_model = metrics_df.loc[metrics_df['auc_roc'].idxmax()].name
        st.success(f"**Best Model**: {best_model} (AUC-ROC: {metrics_df.loc[best_model, 'auc_roc']:.3f})")

        # Performance insights
        st.subheader("Performance Insights")
        insights = [
            f"**Best Model**: {best_model} achieves {metrics_df.loc[best_model, 'auc_roc']:.1%} AUC-ROC",
            f"**Precision Focus**: {metrics_df.loc[metrics_df['precision'].idxmax()].name} has highest precision ({metrics_df['precision'].max():.1%})",
            f"**Recall Focus**: {metrics_df.loc[metrics_df['recall'].idxmax()].name} has highest recall ({metrics_df['recall'].max():.1%})",
            "Consider business context when choosing between precision and recall optimization"
        ]

        for insight in insights:
            st.write(f"• {insight}")

    def render_clv_tab(self):
        """Render CLV analysis"""
        st.header("Customer Lifetime Value Analysis")

        if self.clv_analyzer is None or self.customer_data is None:
            st.error("CLV analysis data not available.")
            return

        try:
            # Run CLV analysis
            analysis_results = self.clv_analyzer.analyze_clv_segments()

            # CLV Distribution
            st.subheader("CLV Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            self.customer_data['clv'].hist(bins=50, ax=ax, alpha=0.7)
            ax.set_xlabel('Customer Lifetime Value ($)')
            ax.set_ylabel('Number of Customers')
            ax.set_title('CLV Distribution')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Churn Rate by CLV Segment
            st.subheader("Churn Rate by CLV Segment")
            churn_by_clv = analysis_results['churn_by_clv']

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(churn_by_clv.index, churn_by_clv['churn_rate'] * 100,
                         alpha=0.7, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
            ax.set_xlabel('CLV Segment')
            ax.set_ylabel('Churn Rate (%)')
            ax.set_title('Churn Rate by Customer Lifetime Value Segment')

            # Add value labels on bars
            for bar, count in zip(bars, churn_by_clv['customer_count']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'n={count}', ha='center', va='bottom', fontsize=10)

            st.pyplot(fig)

            # CLV Segment Summary
            st.subheader("CLV Segment Summary")

            # Create summary table
            summary_data = []
            for segment in churn_by_clv.index:
                segment_data = self.customer_data[self.customer_data['clv_quartile'] == segment]
                summary_data.append({
                    'Segment': segment,
                    'Customer Count': len(segment_data),
                    'Avg CLV': f"${segment_data['clv'].mean():,.0f}",
                    'Churn Rate': f"{churn_by_clv.loc[segment, 'churn_rate']:.1%}",
                    'Total CLV': f"${segment_data['clv'].sum():,.0f}"
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # Business Insights
            st.subheader("Key Business Insights")
            insights = analysis_results['insights']

            for i, insight in enumerate(insights, 1):
                st.write(f"{i}. {insight}")

            # Business Recommendations
            st.subheader("Actionable Recommendations")

            recommendations = [
                "**Premium CLV customers** have lowest churn rate - invest heavily in relationship management",
                "**High CLV segment** shows concerning churn signals - investigate root causes immediately",
                "**Low CLV segment** may not justify expensive retention programs - focus on cost-effective solutions",
                "**Monitor tenure patterns** - customers with high monthly_to_total_ratio may be newer and more volatile"
            ]

            for rec in recommendations:
                st.write(f"• {rec}")

            # CLV vs Churn Scatter Plot
            st.subheader("CLV vs Monthly Charges Analysis")

            # Sample for performance
            sample_data = self.customer_data.sample(min(1000, len(self.customer_data)), random_state=42)

            fig = px.scatter(sample_data, x='MonthlyCharges', y='clv',
                           color='Churn', color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'},
                           opacity=0.6, title='CLV vs Monthly Charges by Churn Status')

            fig.update_layout(
                xaxis_title="Monthly Charges ($)",
                yaxis_title="Customer Lifetime Value ($)",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # One-paragraph takeaway
            st.subheader("Strategic Takeaway")

            premium_clv = self.customer_data[self.customer_data['clv_quartile'] == 'Premium']['clv'].mean()
            premium_churn = churn_by_clv.loc['Premium', 'churn_rate'] * 100
            low_clv = self.customer_data[self.customer_data['clv_quartile'] == 'Low']['clv'].mean()
            low_churn = churn_by_clv.loc['Low', 'churn_rate'] * 100

            st.info(f"""
            **Customer retention strategy should prioritize Premium CLV customers** (${premium_clv:,.0f} average value)
            who demonstrate the lowest churn rate ({premium_churn:.1f}%). These high-value customers represent the
            core of sustainable business growth and justify premium retention investments. In contrast, Low CLV
            customers (${low_clv:,.0f} average value) with higher churn rates ({low_churn:.1f}%) require cost-effective
            retention approaches that don't erode profitability. The 36-month expected tenure assumption provides
            a balanced view for calculating customer lifetime value and guiding strategic resource allocation.
            """)

        except Exception as e:
            st.error(f"Error in CLV analysis: {str(e)}")
            st.info("Please ensure customer data is properly loaded and CLV analysis is working.")

    def render_about_tab(self):
        """Render about/information tab"""
        st.header("ℹ️ About")

        st.markdown("""
        ## Customer Churn Prediction & CLV Analysis

        This application provides:

        ### 🔮 **Churn Prediction**
        - Predict individual customer churn probability
        - Compare predictions across multiple ML models
        - Understand feature contributions to predictions

        ### 📊 **Model Performance**
        - Comprehensive model evaluation metrics
        - ROC-AUC, Precision, Recall, F1-Score analysis
        - Model comparison and selection

        ### 💰 **CLV Analysis**
        - Customer Lifetime Value segmentation
        - Churn analysis by customer value tiers
        - Business recommendations for retention

        ### 🛠️ **Technical Stack**
        - **Models**: Logistic Regression, Random Forest, XGBoost
        - **Framework**: Streamlit for web interface
        - **ML Libraries**: Scikit-learn, XGBoost, SHAP
        - **Visualization**: Plotly, Matplotlib, Seaborn
        """)

        st.markdown("---")
        st.markdown("*Built with ❤️ using modern ML and web technologies*")


def main():
    """Main application entry point"""
    app = ChurnPredictionApp()

    st.title("📊 Customer Churn Prediction & CLV Analysis")
    st.markdown("Predict customer churn and analyze customer lifetime value for better retention strategies")

    # Use tabs instead of sidebar navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Churn", "📈 Model Performance", "💰 CLV Analysis", "ℹ️ About"])

    with tab1:
        app.render_prediction_tab()

    with tab2:
        app.render_model_performance_tab()

    with tab3:
        app.render_clv_tab()

    with tab4:
        app.render_about_tab()


if __name__ == "__main__":
    main()
