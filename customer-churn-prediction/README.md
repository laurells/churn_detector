# Customer Churn Prediction & CLV Analysis

A comprehensive machine learning application for predicting customer churn and analyzing customer lifetime value (CLV) using advanced analytics and explainable AI.

## 🚀 Features

- **🔮 Churn Prediction**: Predict individual customer churn probability with ML models
- **📊 Model Performance**: Comprehensive evaluation with ROC curves and feature importance
- **💰 CLV Analysis**: Customer lifetime value segmentation and business insights
- **🎨 Interactive UI**: Professional Streamlit application with tabbed interface
- **🔍 Explainable AI**: SHAP explanations for model transparency

## 🛠️ Technical Stack

- **Models**: Logistic Regression, Random Forest, XGBoost
- **Framework**: Streamlit for web interface
- **ML Libraries**: Scikit-learn, XGBoost, SHAP for interpretability
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy with business-driven feature engineering

## 🚀 Quick Start

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
python -c "import streamlit, pandas, numpy, sklearn, xgboost, shap; print('✅ All packages installed!')"
```

### Run the Application
```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Model Performance

| Model | AUC-ROC | Recall | Precision | F1-Score |
|-------|---------|--------|-----------|----------|
| XGBoost | 91.0% | 48.0% | 82.0% | 61.0% |
| Random Forest | 89.0% | 45.0% | 78.0% | 57.0% |
| Logistic Regression | 85.0% | 42.0% | 65.0% | 51.0% |

## 🔍 Key Insights

- **Contract Type**: Month-to-month contracts highest churn risk
- **Tenure**: New customers (<6 months) most likely to churn
- **Monthly Charges**: Higher charges correlate with higher churn
- **CLV Segmentation**: Premium customers have lowest churn rates
- **Business Strategy**: Prioritize high-CLV customer retention

## 📁 Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                    # Original data
│   ├── processed/              # Cleaned and processed data
│   └── external/               # Any additional external data
├── models/                     # Trained models and preprocessing objects
├── app/                        # Streamlit application
│   └── app.py                 # Main application
├── src/                       # Core modules
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── interpretability.py
│   ├── clv/
│   │   ├── __init__.py
│   │   └── analysis.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── config/
│   └── settings.py            # Configuration management
├── venv/                      # Virtual environment (auto-generated)
├── requirements.txt           # Python dependencies
├── setup.sh                  # Unix/Linux/Mac setup script
├── setup.bat                # Windows setup script
├── .gitignore               # Git ignore patterns
└── README.md
```

## 🧪 Testing High-Risk Scenarios

The model correctly identifies high-risk customers:
- **Senior citizen** + **Month-to-month contract** + **Fiber optic internet** + **No tech support** + **Electronic check** + **High monthly charges** → **78.5% churn probability**

## 🤝 Contributing

1. Create/activate virtual environment: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational and demonstration purposes.

## 🆘 Troubleshooting

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
