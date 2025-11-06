"""
Customer Lifetime Value analysis module.
Analyzes CLV distribution and churn patterns across customer segments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
from pathlib import Path


class CLVAnalyzer:
    """Enhanced CLV analyzer with business-focused insights."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize CLV analyzer with preprocessed dataframe.
        
        Args:
            df: DataFrame with CLV already computed (MonthlyCharges × ExpectedTenure)
        """
        self.df = df.copy()
        self.analysis_results = {}
        
        # Ensure CLV exists
        if 'clv' not in self.df.columns:
            raise ValueError("CLV must be computed before analysis. Run DataPreprocessor first.")
        
        print(f"CLV Analyzer initialized with {len(self.df)} customers")
        print(f"CLV range: ${self.df['clv'].min():.2f} - ${self.df['clv'].max():.2f}")

    def compute_clv_quartiles(self) -> pd.DataFrame:
        """
        Split customers into CLV quartiles: Low/Medium/High/Premium
        
        Returns:
            DataFrame with clv_quartile column added
        """
        # Create quartiles with explicit labels
        self.df['clv_quartile'] = pd.qcut(
            self.df['clv'], 
            q=4,
            labels=['Low', 'Medium', 'High', 'Premium'],
            duplicates='drop'  # Handle edge cases with duplicate bin edges
        )
        
        # Get quartile boundaries for reporting
        quartile_bounds = self.df.groupby('clv_quartile', observed=True)['clv'].agg(['min', 'max', 'mean'])
        
        print("\nCLV Quartile Boundaries:")
        print(quartile_bounds.to_string())
        
        self.analysis_results['quartile_bounds'] = quartile_bounds
        
        return self.df

    def analyze_churn_by_clv(self) -> pd.DataFrame:
        """
        Compute churn rate by CLV quartile with detailed statistics.
        
        Returns:
            DataFrame with churn metrics by quartile
        """
        # Ensure quartiles exist
        if 'clv_quartile' not in self.df.columns:
            self.compute_clv_quartiles()
        
        # Convert Churn to numeric for calculations
        self.df['churn_numeric'] = (self.df['Churn'] == 'Yes').astype(int)
        
        # Comprehensive churn analysis by quartile
        churn_analysis = self.df.groupby('clv_quartile', observed=True).agg({
            'churn_numeric': [
                ('churn_rate', 'mean'),
                ('churned_count', 'sum'),
                ('total_customers', 'count')
            ],
            'clv': [
                ('avg_clv', 'mean'),
                ('median_clv', 'median')
            ]
        }).round(4)
        
        # Flatten column names
        churn_analysis.columns = ['_'.join(col).strip() for col in churn_analysis.columns.values]
        churn_analysis = churn_analysis.rename(columns={
            'churn_numeric_churn_rate': 'churn_rate',
            'churn_numeric_churned_count': 'churned_count',
            'churn_numeric_total_customers': 'total_customers',
            'clv_avg_clv': 'avg_clv',
            'clv_median_clv': 'median_clv'
        })
        
        # Add percentage format for display
        churn_analysis['churn_rate_pct'] = (churn_analysis['churn_rate'] * 100).round(2)
        
        print("\n" + "="*70)
        print("CHURN RATE BY CLV QUARTILE")
        print("="*70)
        print(churn_analysis[['total_customers', 'churned_count', 'churn_rate_pct', 'avg_clv']].to_string())
        print("="*70)
        
        self.analysis_results['churn_by_clv'] = churn_analysis
        
        return churn_analysis

    def create_visualizations(self, save_path: Path = None) -> Tuple[plt.Figure, plt.Figure]:
        """
        Create 2 clear charts for CLV analysis as required.
        
        Chart 1: Churn Rate by CLV Quartile (Bar Chart)
        Chart 2: Customer Distribution & Churn by CLV Quartile (Combined)
        
        Args:
            save_path: Optional path to save figures
            
        Returns:
            Tuple of (fig1, fig2)
        """
        # Ensure analysis is complete
        if 'churn_by_clv' not in self.analysis_results:
            self.analyze_churn_by_clv()
        
        churn_data = self.analysis_results['churn_by_clv']
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        
        # ========================
        # CHART 1: Churn Rate by CLV Quartile
        # ========================
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        quartiles = churn_data.index
        churn_rates = churn_data['churn_rate_pct']
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']  # Red to Blue gradient
        
        bars = ax1.bar(quartiles, churn_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add horizontal line for overall churn rate
        overall_churn = (self.df['Churn'] == 'Yes').mean() * 100
        ax1.axhline(overall_churn, color='red', linestyle='--', linewidth=2, 
                   label=f'Overall Churn Rate: {overall_churn:.1f}%')
        
        ax1.set_xlabel('CLV Quartile', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Churn Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Churn Rate by Customer Lifetime Value Quartile', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.set_ylim(0, max(churn_rates) * 1.2)
        ax1.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # ========================
        # CHART 2: Customer Distribution & Churn Comparison
        # ========================
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left subplot: Customer count by quartile
        customer_counts = churn_data['total_customers']
        bars2a = ax2a.bar(quartiles, customer_counts, color=colors, alpha=0.7, 
                         edgecolor='black', linewidth=1.2)
        
        for bar in bars2a:
            height = bar.get_height()
            ax2a.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height):,}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2a.set_xlabel('CLV Quartile', fontsize=11, fontweight='bold')
        ax2a.set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
        ax2a.set_title('Customer Distribution by CLV Quartile', 
                      fontsize=12, fontweight='bold')
        ax2a.grid(axis='y', alpha=0.3)
        
        # Right subplot: Churned vs Retained
        churned = churn_data['churned_count']
        retained = churn_data['total_customers'] - churn_data['churned_count']
        
        x = np.arange(len(quartiles))
        width = 0.35
        
        bars2b1 = ax2b.bar(x - width/2, churned, width, label='Churned', 
                          color='#d62728', alpha=0.8, edgecolor='black')
        bars2b2 = ax2b.bar(x + width/2, retained, width, label='Retained', 
                          color='#2ca02c', alpha=0.8, edgecolor='black')
        
        ax2b.set_xlabel('CLV Quartile', fontsize=11, fontweight='bold')
        ax2b.set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
        ax2b.set_title('Churned vs Retained Customers by CLV', 
                      fontsize=12, fontweight='bold')
        ax2b.set_xticks(x)
        ax2b.set_xticklabels(quartiles)
        ax2b.legend(loc='upper right', fontsize=10)
        ax2b.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figures if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            fig1.savefig(save_path / 'clv_churn_rate.png', dpi=300, bbox_inches='tight')
            fig2.savefig(save_path / 'clv_distribution.png', dpi=300, bbox_inches='tight')
            print(f"\nVisualizations saved to: {save_path}")
        
        return fig1, fig2

    def generate_insights(self) -> Dict[str, list]:
        """
        Generate key insights and business recommendations from CLV analysis.
        
        Returns:
            Dictionary with 'insights' and 'recommendations' lists
        """
        if 'churn_by_clv' not in self.analysis_results:
            self.analyze_churn_by_clv()
        
        churn_data = self.analysis_results['churn_by_clv']
        
        # Key Statistical Insights
        insights = [
            f"Total Customers Analyzed: {len(self.df):,}",
            f"Average CLV: ${self.df['clv'].mean():.2f}",
            f"CLV Range: ${self.df['clv'].min():.2f} - ${self.df['clv'].max():.2f}",
            f"Expected Tenure Assumption: {self.df['expected_tenure'].iloc[0]} months",
            "",
            "Churn Rates by Quartile:",
            f"   • Low CLV: {churn_data.loc['Low', 'churn_rate_pct']:.1f}% "
            f"(Avg CLV: ${churn_data.loc['Low', 'avg_clv']:.0f})",
            f"   • Medium CLV: {churn_data.loc['Medium', 'churn_rate_pct']:.1f}% "
            f"(Avg CLV: ${churn_data.loc['Medium', 'avg_clv']:.0f})",
            f"   • High CLV: {churn_data.loc['High', 'churn_rate_pct']:.1f}% "
            f"(Avg CLV: ${churn_data.loc['High', 'avg_clv']:.0f})",
            f"   • Premium CLV: {churn_data.loc['Premium', 'churn_rate_pct']:.1f}% "
            f"(Avg CLV: ${churn_data.loc['Premium', 'avg_clv']:.0f})",
        ]
        
        # Business Recommendations
        premium_churn = churn_data.loc['Premium', 'churn_rate_pct']
        low_churn = churn_data.loc['Low', 'churn_rate_pct']
        
        recommendations = [
            f"Priority Action: Premium segment has {premium_churn:.1f}% churn rate - "
            f"these customers represent highest value (${churn_data.loc['Premium', 'avg_clv']:.0f} CLV)",
            
            f"Retention Strategy: Low CLV segment has {low_churn:.1f}% churn - "
            f"implement cost-effective retention tactics",
            
            "Root Cause Analysis: Investigate why high-value customers churn - "
            "conduct exit interviews and satisfaction surveys",
            
            "Proactive Outreach: Develop early warning system for Premium customers "
            "showing churn signals (e.g., decreased usage, support tickets)",
            
            f"Revenue Impact: Reducing Premium churn by 10% could save "
            f"${int(churn_data.loc['Premium', 'churned_count'] * 0.10 * churn_data.loc['Premium', 'avg_clv']):,} in CLV",
            
            "Loyalty Programs: Consider VIP treatment for Premium segment - "
            "dedicated support, exclusive offers, tenure rewards"
        ]
        
        self.analysis_results['insights'] = insights
        self.analysis_results['recommendations'] = recommendations
        
        return {
            'insights': insights,
            'recommendations': recommendations
        }

    def print_summary_report(self):
        """Print comprehensive CLV analysis summary report."""
        if 'insights' not in self.analysis_results:
            self.generate_insights()
        
        print("\n" + "="*70)
        print("CLV ANALYSIS SUMMARY REPORT")
        print("="*70)
        
        print("\nKEY INSIGHTS:")
        for insight in self.analysis_results['insights']:
            print(insight)
        
        print("\n" + "-"*70)
        print("\n💼 BUSINESS RECOMMENDATIONS:")
        for i, rec in enumerate(self.analysis_results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*70)

    def run_full_analysis(self, save_path: Path = None):
        """
        Execute complete CLV analysis pipeline.
        
        Args:
            save_path: Optional path to save visualizations
            
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "="*70)
        print("EXECUTING FULL CLV ANALYSIS")
        print("="*70)
        
        # Step 1: Compute quartiles
        print("\n[1/4] Computing CLV quartiles...")
        self.compute_clv_quartiles()
        
        # Step 2: Analyze churn by quartile
        print("\n[2/4] Analyzing churn patterns...")
        self.analyze_churn_by_clv()
        
        # Step 3: Create visualizations
        print("\n[3/4] Creating visualizations...")
        fig1, fig2 = self.create_visualizations(save_path)
        
        # Step 4: Generate insights
        print("\n[4/4] Generating insights and recommendations...")
        self.generate_insights()
        
        # Print summary
        self.print_summary_report()
        
        print("\nCLV Analysis Complete!")
        
        return self.analysis_results
