"""Explanation methods for anomaly detection models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import shap
import lime
import lime.tabular
from captum.attr import IntegratedGradients, GradientShap, Saliency
from captum.attr import visualization as viz
from sklearn.base import BaseEstimator
from omegaconf import DictConfig


class Explainer(ABC):
    """Abstract base class for explainers."""
    
    @abstractmethod
    def explain(self, X: np.ndarray, model: Any) -> np.ndarray:
        """Generate explanations for the given data and model."""
        pass
    
    @abstractmethod
    def explain_instance(self, instance: np.ndarray, model: Any) -> np.ndarray:
        """Generate explanation for a single instance."""
        pass


class SHAPExplainer(Explainer):
    """SHAP-based explainer for anomaly detection."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize SHAP explainer.
        
        Args:
            config: Explanation configuration
        """
        self.config = config
        self.explainer: Optional[shap.Explainer] = None
        self.background_data: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, model: Any) -> "SHAPExplainer":
        """
        Fit the SHAP explainer to background data.
        
        Args:
            X: Background data for SHAP
            model: Trained model
            
        Returns:
            Self for method chaining
        """
        # Sample background data if too large
        if len(X) > self.config.explainer.background_samples:
            indices = np.random.choice(
                len(X), 
                self.config.explainer.background_samples, 
                replace=False
            )
            self.background_data = X[indices]
        else:
            self.background_data = X
        
        # Create explainer based on method
        if self.config.explainer.method == "kernel":
            self.explainer = shap.KernelExplainer(
                model.predict, 
                self.background_data
            )
        elif self.config.explainer.method == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif self.config.explainer.method == "deep":
            self.explainer = shap.DeepExplainer(model, self.background_data)
        else:
            raise ValueError(f"Unknown SHAP method: {self.config.explainer.method}")
        
        return self
    
    def explain(self, X: np.ndarray, model: Any) -> np.ndarray:
        """
        Generate SHAP explanations for the data.
        
        Args:
            X: Input data
            model: Trained model
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            self.fit(X, model)
        
        # Limit samples for computational efficiency
        if len(X) > self.config.explainer.max_samples:
            indices = np.random.choice(
                len(X), 
                self.config.explainer.max_samples, 
                replace=False
            )
            X_sample = X[indices]
        else:
            X_sample = X
        
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        return shap_values
    
    def explain_instance(self, instance: np.ndarray, model: Any) -> np.ndarray:
        """
        Generate SHAP explanation for a single instance.
        
        Args:
            instance: Single data instance
            model: Trained model
            
        Returns:
            SHAP values for the instance
        """
        if self.explainer is None:
            raise ValueError("Explainer must be fitted before explaining instances")
        
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        return shap_values.flatten()


class LIMEExplainer(Explainer):
    """LIME-based explainer for anomaly detection."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize LIME explainer.
        
        Args:
            config: Explanation configuration
        """
        self.config = config
        self.explainer: Optional[lime.tabular.LimeTabularExplainer] = None
        
    def fit(self, X: np.ndarray, model: Any) -> "LIMEExplainer":
        """
        Fit the LIME explainer to background data.
        
        Args:
            X: Background data for LIME
            model: Trained model
            
        Returns:
            Self for method chaining
        """
        self.explainer = lime.tabular.LimeTabularExplainer(
            X,
            feature_names=self.config.explainer.feature_names,
            mode='regression',  # For anomaly scores
            discretize_continuous=True
        )
        return self
    
    def explain(self, X: np.ndarray, model: Any) -> List[Dict[str, Any]]:
        """
        Generate LIME explanations for the data.
        
        Args:
            X: Input data
            model: Trained model
            
        Returns:
            List of LIME explanations
        """
        if self.explainer is None:
            self.fit(X, model)
        
        explanations = []
        for i in range(min(len(X), self.config.explainer.max_samples)):
            explanation = self.explainer.explain_instance(
                X[i], 
                model.predict,
                num_features=len(self.config.explainer.feature_names)
            )
            explanations.append(explanation.as_list())
        
        return explanations
    
    def explain_instance(self, instance: np.ndarray, model: Any) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance: Single data instance
            model: Trained model
            
        Returns:
            LIME explanation for the instance
        """
        if self.explainer is None:
            raise ValueError("Explainer must be fitted before explaining instances")
        
        explanation = self.explainer.explain_instance(
            instance, 
            model.predict,
            num_features=len(self.config.explainer.feature_names)
        )
        
        return explanation.as_list()


class CaptumExplainer(Explainer):
    """Captum-based explainer for PyTorch models."""
    
    def __init__(self, config: DictConfig, method: str = "integrated_gradients"):
        """
        Initialize Captum explainer.
        
        Args:
            config: Explanation configuration
            method: Captum method ('integrated_gradients', 'gradient_shap', 'saliency')
        """
        self.config = config
        self.method = method
        self.explainer: Optional[Any] = None
        
    def fit(self, X: np.ndarray, model: Any) -> "CaptumExplainer":
        """
        Fit the Captum explainer.
        
        Args:
            X: Background data
            model: PyTorch model
            
        Returns:
            Self for method chaining
        """
        if self.method == "integrated_gradients":
            self.explainer = IntegratedGradients(model)
        elif self.method == "gradient_shap":
            baseline = torch.zeros_like(torch.FloatTensor(X[:1]))
            self.explainer = GradientShap(model)
        elif self.method == "saliency":
            self.explainer = Saliency(model)
        else:
            raise ValueError(f"Unknown Captum method: {self.method}")
        
        return self
    
    def explain(self, X: np.ndarray, model: Any) -> np.ndarray:
        """
        Generate Captum explanations for the data.
        
        Args:
            X: Input data
            model: PyTorch model
            
        Returns:
            Attribution values
        """
        if self.explainer is None:
            self.fit(X, model)
        
        X_tensor = torch.FloatTensor(X)
        attributions = self.explainer.attribute(X_tensor)
        
        return attributions.detach().numpy()
    
    def explain_instance(self, instance: np.ndarray, model: Any) -> np.ndarray:
        """
        Generate Captum explanation for a single instance.
        
        Args:
            instance: Single data instance
            model: PyTorch model
            
        Returns:
            Attribution values for the instance
        """
        if self.explainer is None:
            raise ValueError("Explainer must be fitted before explaining instances")
        
        instance_tensor = torch.FloatTensor(instance).unsqueeze(0)
        attribution = self.explainer.attribute(instance_tensor)
        
        return attribution.detach().numpy().flatten()


class CounterfactualExplainer(Explainer):
    """Counterfactual explanation generator."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize counterfactual explainer.
        
        Args:
            config: Explanation configuration
        """
        self.config = config
        
    def explain(self, X: np.ndarray, model: Any) -> List[Dict[str, Any]]:
        """
        Generate counterfactual explanations.
        
        Args:
            X: Input data
            model: Trained model
            
        Returns:
            List of counterfactual explanations
        """
        explanations = []
        
        for i in range(min(len(X), self.config.explainer.max_samples)):
            counterfactual = self._generate_counterfactual(X[i], model)
            explanations.append({
                "original": X[i].tolist(),
                "counterfactual": counterfactual["instance"].tolist(),
                "changes": counterfactual["changes"],
                "distance": counterfactual["distance"]
            })
        
        return explanations
    
    def explain_instance(self, instance: np.ndarray, model: Any) -> Dict[str, Any]:
        """
        Generate counterfactual explanation for a single instance.
        
        Args:
            instance: Single data instance
            model: Trained model
            
        Returns:
            Counterfactual explanation
        """
        counterfactual = self._generate_counterfactual(instance, model)
        return {
            "original": instance.tolist(),
            "counterfactual": counterfactual["instance"].tolist(),
            "changes": counterfactual["changes"],
            "distance": counterfactual["distance"]
        }
    
    def _generate_counterfactual(
        self, 
        instance: np.ndarray, 
        model: Any
    ) -> Dict[str, Any]:
        """
        Generate a counterfactual for the given instance.
        
        Args:
            instance: Input instance
            model: Trained model
            
        Returns:
            Counterfactual information
        """
        # Simple counterfactual generation by perturbing features
        original_score = model.predict(instance.reshape(1, -1))[0]
        
        # Try different perturbations
        best_counterfactual = None
        best_distance = float('inf')
        
        for _ in range(100):  # Try 100 random perturbations
            # Generate random perturbation
            perturbation = np.random.normal(0, 0.1, instance.shape)
            candidate = instance + perturbation
            
            # Ensure candidate is within reasonable bounds
            candidate = np.clip(candidate, -3, 3)
            
            # Check if this makes it non-anomalous
            candidate_score = model.predict(candidate.reshape(1, -1))[0]
            
            if candidate_score < original_score:  # Less anomalous
                distance = np.linalg.norm(candidate - instance)
                if distance < best_distance:
                    best_distance = distance
                    best_counterfactual = candidate
        
        if best_counterfactual is None:
            # Fallback: return original with small perturbation
            best_counterfactual = instance + np.random.normal(0, 0.01, instance.shape)
            best_distance = np.linalg.norm(best_counterfactual - instance)
        
        # Calculate changes
        changes = []
        for i, (orig, cf) in enumerate(zip(instance, best_counterfactual)):
            if abs(cf - orig) > 0.01:  # Significant change
                changes.append({
                    "feature": self.config.explainer.feature_names[i],
                    "original": float(orig),
                    "counterfactual": float(cf),
                    "change": float(cf - orig)
                })
        
        return {
            "instance": best_counterfactual,
            "changes": changes,
            "distance": float(best_distance)
        }


def create_explainer(explainer_type: str, config: DictConfig) -> Explainer:
    """
    Create explainer based on type.
    
    Args:
        explainer_type: Type of explainer
        config: Configuration object
        
    Returns:
        Explainer instance
    """
    explainers = {
        "shap": SHAPExplainer,
        "lime": LIMEExplainer,
        "integrated_gradients": lambda cfg: CaptumExplainer(cfg, "integrated_gradients"),
        "gradient_shap": lambda cfg: CaptumExplainer(cfg, "gradient_shap"),
        "saliency": lambda cfg: CaptumExplainer(cfg, "saliency"),
        "counterfactual": CounterfactualExplainer,
    }
    
    if explainer_type not in explainers:
        raise ValueError(f"Unknown explainer type: {explainer_type}")
    
    explainer_class = explainers[explainer_type]
    return explainer_class(config)
