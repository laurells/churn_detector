@echo off
REM Virtual Environment Setup Script for Windows

echo 🚀 Setting up Customer Churn Prediction Environment...
echo ==================================================

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
pip install --upgrade pip

REM Install requirements
echo 📚 Installing Python packages...
pip install -r requirements.txt

REM Verify installation
echo ✅ Verifying installation...
python -c "import streamlit, pandas, numpy, sklearn, xgboost, shap; print('All packages installed successfully!')"

echo.
echo 🎉 Setup complete!
echo To activate the environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To run the application:
echo   streamlit run app/app.py
echo.
echo To deactivate:
echo   deactivate
pause
