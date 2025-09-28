"""
Visualization module for customer churn prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go


class ChurnVisualizer:
    """Class for creating visualizations related to churn analysis."""

    def __init__(self, style: str = 'default'):
        """Initialize visualizer with plotting style."""
        plt.style.use(style)
        self.style = style

    def plot_churn_distribution(self, df: pd.DataFrame, churn_column: str = 'churn') -> plt.Figure:
        """Plot churn distribution."""
        plt.figure(figsize=(8, 6))
        churn_counts = df[churn_column].value_counts()
        churn_counts.plot(kind='bar')
        plt.title('Churn Distribution')
        plt.xlabel('Churn Status')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['No Churn', 'Churn'])
        return plt.gcf()

    def plot_numerical_distributions(self, df: pd.DataFrame, numerical_columns: List[str],
                                   hue: Optional[str] = None) -> plt.Figure:
        """Plot distributions of numerical features."""
        n_cols = len(numerical_columns)
        n_rows = (n_cols + 2) // 3  # Ceiling division

        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(numerical_columns):
            row, col_idx = divmod(i, 3)
            if hue:
                sns.histplot(data=df, x=col, hue=hue, ax=axes[row, col_idx], multiple="stack")
            else:
                df[col].hist(ax=axes[row, col_idx], bins=30)

            axes[row, col_idx].set_title(f'Distribution of {col}')

        # Hide empty subplots
        for i in range(n_cols, n_rows * 3):
            row, col_idx = divmod(i, 3)
            axes[row, col_idx].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_correlation_heatmap(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> plt.Figure:
        """Plot correlation heatmap."""
        if columns:
            corr_df = df[columns].corr()
        else:
            corr_df = df.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Heatmap')
        return plt.gcf()

    def plot_feature_importance_plotly(self, feature_names: List[str], importance_scores: np.ndarray,
                                     title: str = "Feature Importance") -> go.Figure:
        """Create interactive feature importance plot with Plotly."""
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = importance_scores[sorted_idx]

        fig = go.Figure(data=[
            go.Bar(
                x=sorted_importance,
                y=sorted_features,
                orientation='h',
                marker=dict(color='lightblue')
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(feature_names) * 20)
        )

        return fig

    def plot_churn_by_category(self, df: pd.DataFrame, category_column: str,
                             churn_column: str = 'churn') -> plt.Figure:
        """Plot churn rate by categorical feature."""
        churn_rate = df.groupby(category_column)[churn_column].mean().sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        churn_rate.plot(kind='bar')
        plt.title(f'Churn Rate by {category_column}')
        plt.xlabel(category_column)
        plt.ylabel('Churn Rate')
        plt.xticks(rotation=45)
        return plt.gcf()

    def plot_customer_lifetime_value_distribution(self, df: pd.DataFrame,
                                                clv_column: str = 'clv') -> plt.Figure:
        """Plot customer lifetime value distribution."""
        plt.figure(figsize=(10, 6))
        df[clv_column].hist(bins=50)
        plt.title('Customer Lifetime Value Distribution')
        plt.xlabel('Customer Lifetime Value')
        plt.ylabel('Frequency')
        return plt.gcf()
