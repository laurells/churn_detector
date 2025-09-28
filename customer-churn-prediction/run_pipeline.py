#!/usr/bin/env python3
"""
Main pipeline runner for Customer Churn Prediction project
"""

import argparse
from src.data.preprocessing import DataPreprocessor
from src.clv.analysis import CLVAnalyzer
from src.models.train import ModelTrainer

def run_data_preparation():
    """Run data preparation pipeline"""
    print("=== DATA PREPARATION ===")
    preprocessor = DataPreprocessor()
    splits, original_df = preprocessor.run_pipeline()
    return original_df

def run_clv_analysis(df):
    """Run CLV analysis"""
    print("\n=== CLV ANALYSIS ===")
    analyzer = CLVAnalyzer(df)
    results = analyzer.analyze_clv_segments()
    
    print("CLV Analysis Results:")
    for insight in results['insights']:
        print(f"• {insight}")
    
    return analyzer

def run_model_training():
    """Run model training pipeline"""
    print("\n=== MODEL TRAINING ===")
    trainer = ModelTrainer()
    trainer.run_training_pipeline()
    return trainer

def main():
    parser = argparse.ArgumentParser(description='Customer Churn Prediction Pipeline')
    parser.add_argument('--skip-data', action='store_true', help='Skip data preparation')
    parser.add_argument('--skip-clv', action='store_true', help='Skip CLV analysis')
    parser.add_argument('--skip-models', action='store_true', help='Skip model training')
    
    args = parser.parse_args()
    
    # Run pipeline steps
    if not args.skip_data:
        df = run_data_preparation()
    else:
        # Load existing data for CLV analysis
        pass
    
    if not args.skip_clv and not args.skip_data:
        run_clv_analysis(df)
    
    if not args.skip_models:
        run_model_training()
    
    print("\n=== PIPELINE COMPLETED ===")
    print("Next steps:")
    print("1. Run the Streamlit app: streamlit run app/app.py")
    print("2. Check model performance in the app")
    print("3. Review business insights in CLV analysis tab")

if __name__ == "__main__":
    main()