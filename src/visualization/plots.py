"""Visualization utilities for anomaly detection and explanations."""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from matplotlib.figure import Figure
from omegaconf import DictConfig


class AnomalyVisualizer:
    """Visualization class for anomaly detection results."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Setup matplotlib and seaborn styles."""
        plt.style.use(self.config.visualization.style)
        sns.set_palette("husl")
    
    def plot_anomaly_scores(
        self, 
        scores: np.ndarray, 
        anomalies: np.ndarray,
        threshold: float,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot anomaly scores with threshold.
        
        Args:
            scores: Anomaly scores
            anomalies: Boolean array indicating anomalies
            threshold: Detection threshold
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.visualization.figure_size)
        
        # Plot scores
        normal_indices = ~anomalies
        anomaly_indices = anomalies
        
        ax.scatter(
            np.where(normal_indices)[0], 
            scores[normal_indices], 
            c='blue', 
            alpha=0.6, 
            label='Normal',
            s=50
        )
        ax.scatter(
            np.where(anomaly_indices)[0], 
            scores[anomaly_indices], 
            c='red', 
            alpha=0.8, 
            label='Anomaly',
            s=50
        )
        
        # Plot threshold line
        ax.axhline(y=threshold, color='black', linestyle='--', alpha=0.7, label=f'Threshold ({threshold:.3f})')
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Detection Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_distributions(
        self, 
        X: np.ndarray, 
        anomalies: np.ndarray,
        feature_names: List[str],
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot feature distributions for normal vs anomalous samples.
        
        Args:
            X: Feature matrix
            anomalies: Boolean array indicating anomalies
            feature_names: Names of features
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_features = X.shape[1]
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Plot distributions
            normal_data = X[~anomalies, i]
            anomaly_data = X[anomalies, i]
            
            ax.hist(normal_data, bins=20, alpha=0.6, label='Normal', color='blue')
            ax.hist(anomaly_data, bins=20, alpha=0.6, label='Anomaly', color='red')
            
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {feature_names[i]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_explanation_heatmap(
        self, 
        explanations: np.ndarray, 
        feature_names: List[str],
        sample_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot explanation heatmap.
        
        Args:
            explanations: Explanation values
            feature_names: Names of features
            sample_indices: Indices of samples to plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if sample_indices is not None:
            explanations = explanations[sample_indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap
        im = ax.imshow(explanations, cmap='RdBu_r', aspect='auto')
        
        # Set labels
        ax.set_xlabel('Features')
        ax.set_ylabel('Samples')
        ax.set_title('Explanation Heatmap')
        
        # Set tick labels
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        
        if sample_indices is not None:
            ax.set_yticks(range(len(sample_indices)))
            ax.set_yticklabels(sample_indices)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Explanation Value')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(
        self, 
        explanations: np.ndarray, 
        feature_names: List[str],
        method: str = "mean",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot feature importance from explanations.
        
        Args:
            explanations: Explanation values
            feature_names: Names of features
            method: Aggregation method ('mean', 'median', 'max')
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if method == "mean":
            importance = np.mean(np.abs(explanations), axis=0)
        elif method == "median":
            importance = np.median(np.abs(explanations), axis=0)
        elif method == "max":
            importance = np.max(np.abs(explanations), axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        sorted_importance = importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=self.config.visualization.figure_size)
        
        bars = ax.bar(range(len(sorted_importance)), sorted_importance)
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title(f'Feature Importance ({method})')
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_importance)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_shap_summary(
        self, 
        shap_values: np.ndarray, 
        X: np.ndarray, 
        feature_names: List[str],
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values
            X: Input data
            feature_names: Names of features
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(10, 8))
        
        # Create SHAP summary plot
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=feature_names,
            show=False
        )
        
        plt.title('SHAP Summary Plot')
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_explanation_comparison(
        self, 
        explanations_dict: Dict[str, np.ndarray], 
        feature_names: List[str],
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot comparison of different explanation methods.
        
        Args:
            explanations_dict: Dictionary of explanation methods and values
            feature_names: Names of features
            sample_idx: Index of sample to compare
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_methods = len(explanations_dict)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 6))
        
        if n_methods == 1:
            axes = [axes]
        
        for i, (method_name, explanations) in enumerate(explanations_dict.items()):
            ax = axes[i]
            
            # Get explanation for the sample
            if explanations.ndim == 2:
                sample_explanation = explanations[sample_idx]
            else:
                sample_explanation = explanations
            
            # Create bar plot
            bars = ax.bar(range(len(sample_explanation)), sample_explanation)
            ax.set_xlabel('Features')
            ax.set_ylabel('Explanation Value')
            ax.set_title(f'{method_name} (Sample {sample_idx})')
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            
            # Color bars by sign
            for j, bar in enumerate(bars):
                if sample_explanation[j] >= 0:
                    bar.set_color('red')
                else:
                    bar.set_color('blue')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_evaluation_metrics(
        self, 
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot evaluation metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.visualization.figure_size)
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values)
        ax.set_ylabel('Score')
        ax.set_title('Evaluation Metrics')
        ax.set_ylim(0, 1)
        
        # Color bars by value
        colors = plt.cm.viridis(np.array(metric_values))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
        
        return fig
