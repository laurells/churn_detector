#!/bin/bash
# Virtual Environment Setup Script for Unix/Linux/Mac

echo "🚀 Setting up Customer Churn Prediction Environment..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "import streamlit, pandas, numpy, sklearn, xgboost, shap; print('All packages installed successfully!')"

echo ""
echo "🎉 Setup complete!"
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application:"
echo "  streamlit run app/app.py"
echo ""
echo "To deactivate:"
echo "  deactivate"
