#!/usr/bin/env python3
"""
Main pipeline runner for Customer Churn Prediction project
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.preprocessing import DataPreprocessor
from clv.analysis import CLVAnalyzer
from models.train import ModelTrainer
from models.predict import ChurnPredictor


def run_data_preparation():
    """Run data preparation pipeline"""
    print("=== DATA PREPARATION ===")
    preprocessor = DataPreprocessor()
    splits, original_df = preprocessor.run_pipeline()
    return original_df, preprocessor

def run_clv_analysis(df):
    """Run CLV analysis"""
    print("\n=== CLV ANALYSIS ===")
    analyzer = CLVAnalyzer(df)
    # First compute quartiles
    analyzer.compute_clv_quartiles()
    # Then analyze churn by those quartiles
    churn_analysis = analyzer.analyze_churn_by_clv()
    
    print("\nChurn Analysis by CLV Quartile:")
    print("-" * 70)
    print(churn_analysis[['churn_rate', 'churned_count', 'total_customers', 'avg_clv', 'median_clv']].to_string())
    
    return analyzer

def run_model_training():
    """Run model training pipeline"""
    print("\n=== MODEL TRAINING ===")
    trainer = ModelTrainer()
    result = trainer.run_training_pipeline()
    
    if result['status'] == 'success':
        print("Model training completed successfully!")
        for model_name, metrics in result['metrics'].items():
            print(f"{model_name}: AUC = {metrics['auc_roc']:.4f}")
    else:
        print(f"Model training failed: {result['error']}")
    
    return trainer

def run_predictions():
    """Run prediction examples"""
    print("\n=== MODEL PREDICTIONS ===")
    try:
        predictor = ChurnPredictor(model_name="xgboost")
        
        # Test prediction
        sample_customer = {
            'customerID': 'demo-customer-001',
            'tenure': 12,
            'MonthlyCharges': 65.5,
            'TotalCharges': 786.0,
            'InternetService': 'Fiber optic',
            'TechSupport': 'No',
            'Contract': 'Month-to-month'
        }
        
        result = predictor.predict_single(sample_customer)
        print(f"Demo prediction: {result['customer_id']} -> {result['prediction_label']} ({result['churn_probability']:.1%})")
        
        return predictor
    except Exception as e:
        print(f"Prediction demo failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Customer Churn Prediction Pipeline')
    parser.add_argument('--skip-data', action='store_true', help='Skip data preparation')
    parser.add_argument('--skip-clv', action='store_true', help='Skip CLV analysis')
    parser.add_argument('--skip-models', action='store_true', help='Skip model training')
    parser.add_argument('--skip-predictions', action='store_true', help='Skip prediction demo')
    
    args = parser.parse_args()
    
    # Run pipeline steps
    df = None
    if not args.skip_data:
        df, preprocessor = run_data_preparation()
    else:
        print("Skipping data preparation...")

    if not args.skip_clv and df is not None:
        run_clv_analysis(df)
    elif not args.skip_clv:
        print("Cannot run CLV analysis without data. Run without --skip-data or provide existing data.")
    
    if not args.skip_models:
        trainer = run_model_training()
    
    if not args.skip_predictions:
        predictor = run_predictions()
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Next steps:")
    print("1. Run the Streamlit app: streamlit run app/app.py")
    print("2. Check model performance in the app")
    print("3. Use ChurnPredictor for making predictions on new data")
    print("="*50)

if __name__ == "__main__":
    main()