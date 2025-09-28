"""
Customer Lifetime Value analysis module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict


class CLVAnalyzer:
    """Enhanced CLV analyzer with business-focused insights."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analysis_results = {}

    def analyze_clv_segments(self) -> Dict:
        """Analyze CLV segments and churn patterns"""
        # Create CLV quartiles
        self.df['clv_quartile'] = pd.qcut(self.df['clv'], q=4,
                                        labels=['Low', 'Medium', 'High', 'Premium'])

        # Churn rate by CLV segment
        churn_by_clv = self.df.groupby('clv_quartile')['Churn'].agg(['mean', 'count']).round(3)
        churn_by_clv.columns = ['churn_rate', 'customer_count']

        self.analysis_results['churn_by_clv'] = churn_by_clv

        # Key insights
        insights = [
            f"Premium CLV customers have {churn_by_clv.loc['Premium', 'churn_rate']*100:.1f}% churn rate",
            f"Low CLV customers have {churn_by_clv.loc['Low', 'churn_rate']*100:.1f}% churn rate",
            f"CLV range: ${self.df['clv'].min():.0f} - ${self.df['clv'].max():.0f}",
            f"Expected tenure assumption: {self.df['expected_tenure'].iloc[0]} months"
        ]

        self.analysis_results['insights'] = insights

        return self.analysis_results

    def create_visualizations(self) -> Tuple[plt.Figure, plt.Figure]:
        """Create CLV analysis visualizations"""
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=self.df, x='clv_quartile', y='Churn', ax=ax1)
        ax1.set_title('Churn Rate by CLV Segment')
        ax1.set_ylabel('Churn Rate')

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        self.df['clv'].hist(bins=50, ax=ax2)
        ax2.set_title('CLV Distribution')
        ax2.set_xlabel('Customer Lifetime Value ($)')
        ax2.set_ylabel('Frequency')

        return fig1, fig2

    def generate_business_recommendations(self) -> list:
        """Generate actionable business insights"""
        churn_by_clv = self.analysis_results['churn_by_clv']

        recommendations = [
            f"Focus retention efforts on Premium segment (${self.df[self.df['clv_quartile'] == 'Premium']['clv'].mean():.0f} CLV)",
            f"High CLV customers churn at {churn_by_clv.loc['High', 'churn_rate']*100:.1f}% - investigate root causes",
            f"Low CLV segment has {churn_by_clv.loc['Low', 'churn_rate']*100:.1f}% churn - consider cost-effective retention",
            "Monitor monthly-to-total ratio: high ratio may indicate new customer risk"
        ]

        return recommendations
