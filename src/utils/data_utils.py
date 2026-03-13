"""Data loading and preprocessing utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataLoader:
    """Data loading and preprocessing class."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration object containing data settings
        """
        self.config = config
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: Optional[List[str]] = None
        self.target_names: Optional[List[str]] = None
        
    def load_iris_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Load the Iris dataset.
        
        Returns:
            Tuple of (features, targets, feature_names, target_names)
        """
        data = load_iris()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int32)
        feature_names = data.feature_names
        target_names = data.target_names.tolist()
        
        return X, y, feature_names, target_names
    
    def generate_synthetic_data(
        self, 
        n_samples: int = 1000,
        n_features: int = 4,
        n_classes: int = 2,
        n_clusters_per_class: int = 1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Generate synthetic dataset for testing.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features
            n_classes: Number of classes
            n_clusters_per_class: Number of clusters per class
            random_state: Random seed
            
        Returns:
            Tuple of (features, targets, feature_names, target_names)
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            n_redundant=0,
            n_informative=n_features,
            random_state=random_state
        )
        
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        target_names = [f"class_{i}" for i in range(n_classes)]
        
        return X, y, feature_names, target_names
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Load data based on configuration.
        
        Returns:
            Tuple of (features, targets, feature_names, target_names)
        """
        if self.config.data.name == "iris":
            return self.load_iris_data()
        elif self.config.data.name == "synthetic":
            return self.generate_synthetic_data()
        else:
            raise ValueError(f"Unknown dataset: {self.config.data.name}")
    
    def preprocess_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data with scaling and train/test split.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Store feature and target names
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        if self.target_names is None:
            self.target_names = [f"class_{i}" for i in range(len(np.unique(y)))]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
            stratify=y
        )
        
        # Apply scaling if configured
        if self.config.data.normalize or self.config.data.standardize:
            if self.config.data.standardize:
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """
        Get feature metadata for the dataset.
        
        Returns:
            Dictionary containing feature metadata
        """
        if self.feature_names is None:
            return {}
        
        metadata = {
            "features": {
                name: {
                    "type": "numerical",
                    "description": f"Feature {name}",
                    "sensitive": False,
                    "monotonic": None
                }
                for name in self.feature_names
            },
            "targets": {
                "type": "categorical",
                "classes": self.target_names,
                "description": "Target classes"
            }
        }
        
        return metadata
    
    def save_metadata(self, output_path: Union[str, Path]) -> None:
        """
        Save feature metadata to file.
        
        Args:
            output_path: Path to save metadata
        """
        metadata = self.get_feature_metadata()
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load feature metadata from file.
        
        Args:
            input_path: Path to load metadata from
            
        Returns:
            Dictionary containing feature metadata
        """
        with open(input_path, 'r') as f:
            return json.load(f)
