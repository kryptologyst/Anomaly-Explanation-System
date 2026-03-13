"""Evaluation metrics for anomaly detection and explanations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_score
from omegaconf import DictConfig


class EvaluationMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric value."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the metric name."""
        pass


class FaithfulnessMetric(EvaluationMetric):
    """Faithfulness evaluation for explanations."""
    
    def __init__(self, method: str = "deletion"):
        """
        Initialize faithfulness metric.
        
        Args:
            method: Faithfulness method ('deletion', 'insertion')
        """
        self.method = method
    
    @property
    def name(self) -> str:
        """Get metric name."""
        return f"faithfulness_{self.method}"
    
    def compute(
        self, 
        X: np.ndarray, 
        explanations: np.ndarray, 
        model: Any,
        threshold: float = 0.5
    ) -> float:
        """
        Compute faithfulness using deletion or insertion method.
        
        Args:
            X: Input data
            explanations: Explanation values
            model: Trained model
            threshold: Threshold for feature importance
            
        Returns:
            Faithfulness score (AUC)
        """
        if self.method == "deletion":
            return self._deletion_faithfulness(X, explanations, model, threshold)
        elif self.method == "insertion":
            return self._insertion_faithfulness(X, explanations, model, threshold)
        else:
            raise ValueError(f"Unknown faithfulness method: {self.method}")
    
    def _deletion_faithfulness(
        self, 
        X: np.ndarray, 
        explanations: np.ndarray, 
        model: Any,
        threshold: float
    ) -> float:
        """Compute deletion-based faithfulness."""
        original_scores = model.predict(X)
        
        # Sort features by importance
        feature_importance = np.abs(explanations)
        sorted_indices = np.argsort(feature_importance, axis=1)[:, ::-1]
        
        # Progressive deletion
        deletion_scores = []
        for i in range(X.shape[1]):
            X_perturbed = X.copy()
            
            # Remove top i features
            for j in range(X.shape[0]):
                indices_to_remove = sorted_indices[j, :i+1]
                X_perturbed[j, indices_to_remove] = 0  # Set to mean or zero
            
            perturbed_scores = model.predict(X_perturbed)
            score_change = np.abs(original_scores - perturbed_scores)
            deletion_scores.append(np.mean(score_change))
        
        # Compute AUC
        x_axis = np.arange(len(deletion_scores)) / len(deletion_scores)
        auc = np.trapz(deletion_scores, x_axis)
        
        return auc
    
    def _insertion_faithfulness(
        self, 
        X: np.ndarray, 
        explanations: np.ndarray, 
        model: Any,
        threshold: float
    ) -> float:
        """Compute insertion-based faithfulness."""
        # Start with zeros
        X_baseline = np.zeros_like(X)
        
        # Sort features by importance
        feature_importance = np.abs(explanations)
        sorted_indices = np.argsort(feature_importance, axis=1)[:, ::-1]
        
        # Progressive insertion
        insertion_scores = []
        for i in range(X.shape[1]):
            X_perturbed = X_baseline.copy()
            
            # Add top i features
            for j in range(X.shape[0]):
                indices_to_add = sorted_indices[j, :i+1]
                X_perturbed[j, indices_to_add] = X[j, indices_to_add]
            
            perturbed_scores = model.predict(X_perturbed)
            insertion_scores.append(np.mean(perturbed_scores))
        
        # Compute AUC
        x_axis = np.arange(len(insertion_scores)) / len(insertion_scores)
        auc = np.trapz(insertion_scores, x_axis)
        
        return auc


class StabilityMetric(EvaluationMetric):
    """Stability evaluation for explanations."""
    
    def __init__(self, method: str = "correlation"):
        """
        Initialize stability metric.
        
        Args:
            method: Stability method ('correlation', 'kendall', 'spearman')
        """
        self.method = method
    
    @property
    def name(self) -> str:
        """Get metric name."""
        return f"stability_{self.method}"
    
    def compute(
        self, 
        explanations1: np.ndarray, 
        explanations2: np.ndarray
    ) -> float:
        """
        Compute stability between two sets of explanations.
        
        Args:
            explanations1: First set of explanations
            explanations2: Second set of explanations
            
        Returns:
            Stability score
        """
        if self.method == "correlation":
            return self._correlation_stability(explanations1, explanations2)
        elif self.method == "kendall":
            return self._kendall_stability(explanations1, explanations2)
        elif self.method == "spearman":
            return self._spearman_stability(explanations1, explanations2)
        else:
            raise ValueError(f"Unknown stability method: {self.method}")
    
    def _correlation_stability(
        self, 
        explanations1: np.ndarray, 
        explanations2: np.ndarray
    ) -> float:
        """Compute correlation-based stability."""
        # Flatten explanations
        flat1 = explanations1.flatten()
        flat2 = explanations2.flatten()
        
        # Compute correlation
        correlation = np.corrcoef(flat1, flat2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _kendall_stability(
        self, 
        explanations1: np.ndarray, 
        explanations2: np.ndarray
    ) -> float:
        """Compute Kendall tau-based stability."""
        flat1 = explanations1.flatten()
        flat2 = explanations2.flatten()
        
        tau, _ = kendalltau(flat1, flat2)
        return tau if not np.isnan(tau) else 0.0
    
    def _spearman_stability(
        self, 
        explanations1: np.ndarray, 
        explanations2: np.ndarray
    ) -> float:
        """Compute Spearman rho-based stability."""
        flat1 = explanations1.flatten()
        flat2 = explanations2.flatten()
        
        rho, _ = spearmanr(flat1, flat2)
        return rho if not np.isnan(rho) else 0.0


class UtilityMetric(EvaluationMetric):
    """Utility evaluation for explanations."""
    
    def __init__(self, method: str = "simplicity"):
        """
        Initialize utility metric.
        
        Args:
            method: Utility method ('simplicity', 'completeness')
        """
        self.method = method
    
    @property
    def name(self) -> str:
        """Get metric name."""
        return f"utility_{self.method}"
    
    def compute(self, explanations: np.ndarray) -> float:
        """
        Compute utility of explanations.
        
        Args:
            explanations: Explanation values
            
        Returns:
            Utility score
        """
        if self.method == "simplicity":
            return self._simplicity_score(explanations)
        elif self.method == "completeness":
            return self._completeness_score(explanations)
        else:
            raise ValueError(f"Unknown utility method: {self.method}")
    
    def _simplicity_score(self, explanations: np.ndarray) -> float:
        """Compute simplicity score (sparsity)."""
        # Count non-zero explanations
        non_zero_count = np.count_nonzero(np.abs(explanations) > 1e-6)
        total_count = explanations.size
        
        # Simplicity is inverse of sparsity
        sparsity = non_zero_count / total_count
        return 1.0 - sparsity
    
    def _completeness_score(self, explanations: np.ndarray) -> float:
        """Compute completeness score."""
        # Completeness is based on how much of the prediction is explained
        total_explanation = np.sum(np.abs(explanations), axis=1)
        
        # Normalize by maximum possible explanation
        max_explanation = np.max(total_explanation)
        if max_explanation == 0:
            return 0.0
        
        completeness = total_explanation / max_explanation
        return np.mean(completeness)


class AnomalyDetectionMetrics:
    """Metrics for anomaly detection performance."""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        threshold: float
    ) -> Dict[str, float]:
        """
        Compute anomaly detection metrics.
        
        Args:
            y_true: True anomaly labels
            y_scores: Anomaly scores
            threshold: Detection threshold
            
        Returns:
            Dictionary of metrics
        """
        y_pred = (y_scores > threshold).astype(int)
        
        # Basic metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC metrics
        try:
            auc_roc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc_roc = 0.5  # Random performance
        
        try:
            auc_pr = average_precision_score(y_true, y_scores)
        except ValueError:
            auc_pr = 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }


class ExplanationEvaluator:
    """Comprehensive evaluation of explanations."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize explanation evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.metrics = {
            "faithfulness": FaithfulnessMetric("deletion"),
            "stability": StabilityMetric("correlation"),
            "utility": UtilityMetric("simplicity")
        }
    
    def evaluate_explanations(
        self,
        X: np.ndarray,
        explanations: np.ndarray,
        model: Any,
        X_reference: Optional[np.ndarray] = None,
        explanations_reference: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate explanations comprehensively.
        
        Args:
            X: Input data
            explanations: Explanation values
            model: Trained model
            X_reference: Reference data for stability
            explanations_reference: Reference explanations for stability
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Faithfulness
        if "faithfulness" in self.config.evaluation.metrics:
            faithfulness_score = self.metrics["faithfulness"].compute(
                X, explanations, model
            )
            results["faithfulness"] = faithfulness_score
        
        # Stability
        if "stability" in self.config.evaluation.metrics and explanations_reference is not None:
            stability_score = self.metrics["stability"].compute(
                explanations, explanations_reference
            )
            results["stability"] = stability_score
        
        # Utility
        if "utility" in self.config.evaluation.metrics:
            utility_score = self.metrics["utility"].compute(explanations)
            results["utility"] = utility_score
        
        return results
    
    def cross_validation_evaluation(
        self,
        X: np.ndarray,
        model: Any,
        explainer: Any,
        cv_folds: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation evaluation of explanations.
        
        Args:
            X: Input data
            model: Trained model
            explainer: Explanation method
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary of CV results
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.config.seed)
        
        cv_results = {
            "faithfulness": [],
            "stability": [],
            "utility": []
        }
        
        explanations_folds = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            
            # Get explanations for test set
            explanations = explainer.explain(X_test, model)
            explanations_folds.append(explanations)
            
            # Evaluate explanations
            eval_results = self.evaluate_explanations(X_test, explanations, model)
            
            for metric, score in eval_results.items():
                cv_results[metric].append(score)
        
        # Compute stability across folds
        if len(explanations_folds) > 1:
            stability_scores = []
            for i in range(len(explanations_folds) - 1):
                stability = self.metrics["stability"].compute(
                    explanations_folds[i], explanations_folds[i + 1]
                )
                stability_scores.append(stability)
            
            cv_results["stability"] = stability_scores
        
        return cv_results
