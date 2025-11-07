# Customer Churn Prediction & CLV Analysis

A comprehensive machine learning application for predicting customer churn and analyzing customer lifetime value (CLV) using advanced analytics and explainable AI.

## Features

- **Churn Prediction**: Predict individual customer churn probability with ML models
- **Model Performance**: Comprehensive evaluation with ROC curves and feature importance
- **CLV Analysis**: Customer lifetime value segmentation and business insights
- **Interactive UI**: Professional Streamlit application with tabbed interface
- **Explainable AI**: SHAP explanations for model transparency

## Technical Stack

- **Models**: Logistic Regression, Random Forest, XGBoost
- **Framework**: Streamlit for web interface
- **ML Libraries**: Scikit-learn, XGBoost, SHAP for interpretability
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy with business-driven feature engineering

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation & Setup

#### Option 1: Automated Setup (Recommended)
```bash
# Unix/Linux/Mac
./setup.sh

# Windows
setup.bat
```

#### Option 2: Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Unix/Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import streamlit, pandas, numpy, sklearn, xgboost, shap; print('All packages installed!')"
```

### Run the Application
```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`

## Key Insights

- **Contract Type**: Month-to-month contracts highest churn risk
- **Tenure**: New customers (<6 months) most likely to churn
- **Monthly Charges**: Higher charges correlate with higher churn
- **CLV Segmentation**: Premium customers have lowest churn rates
- **Business Strategy**: Prioritize high-CLV customer retention


## Testing High-Risk Scenarios

The model correctly identifies high-risk customers:
- **Senior citizen** + **Month-to-month contract** + **Fiber optic internet** + **No tech support** + **Electronic check** + **High monthly charges** → **78.5% churn probability**

## Contributing

1. Create/activate virtual environment: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and demonstration purposes.

## Troubleshooting

**Common Issues:**
- **Import Errors**: Ensure virtual environment is activated
- **Model Loading**: Run training pipeline first: `python run_pipeline.py`
- **Memory Issues**: Reduce SHAP sample size in interpretability module
- **Plot Display**: Ensure Plotly is properly installed

**Virtual Environment Issues:**
- **Activation Problems**: Use `source venv/bin/activate` (Unix) or `venv\Scripts\activate.bat` (Windows)
- **Package Not Found**: Try `pip install --upgrade pip` then reinstall requirements
- **Python Version**: Ensure Python 3.8+ is installed and being used

---

*Built with ❤️ using modern ML and web technologies*
