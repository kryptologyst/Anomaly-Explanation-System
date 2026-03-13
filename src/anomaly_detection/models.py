"""Anomaly detection models using PyTorch."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """Fit the anomaly detector to the data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores for the data."""
        pass
    
    @abstractmethod
    def detect(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Detect anomalies based on threshold."""
        pass


class AutoencoderDetector(AnomalyDetector):
    """Autoencoder-based anomaly detector."""
    
    def __init__(self, config: DictConfig, device: torch.device):
        """
        Initialize autoencoder detector.
        
        Args:
            config: Model configuration
            device: PyTorch device
        """
        self.config = config
        self.device = device
        self.model: Optional[Autoencoder] = None
        self.scaler: Optional[StandardScaler] = None
        
    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        """
        Fit the autoencoder to the data.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        # Normalize data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Initialize model
        self.model = Autoencoder(
            input_dim=X.shape[1],
            hidden_dims=self.config.model.hidden_dims,
            activation=self.config.model.activation,
            output_activation=self.config.model.output_activation
        ).to(self.device)
        
        # Train model
        self._train_model(X_tensor)
        
        return self
    
    def _train_model(self, X: torch.Tensor) -> None:
        """Train the autoencoder model."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.model.learning_rate
        )
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.config.model.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = self.model(X)
            loss = criterion(reconstructed, X)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict reconstruction errors (anomaly scores).
        
        Args:
            X: Input data
            
        Returns:
            Reconstruction errors
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Normalize data
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Get reconstructions
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_error = F.mse_loss(
                reconstructed, X_tensor, reduction='none'
            ).mean(dim=1).cpu().numpy()
        
        return reconstruction_error
    
    def detect(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Detect anomalies based on reconstruction error threshold.
        
        Args:
            X: Input data
            threshold: Anomaly threshold
            
        Returns:
            Boolean array indicating anomalies
        """
        scores = self.predict(X)
        return scores > threshold


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detector."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize Isolation Forest detector.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model: Optional[IsolationForest] = None
        
    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest to the data.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        self.model = IsolationForest(
            contamination=self.config.anomaly.contamination,
            random_state=self.config.seed
        )
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores (negative values indicate anomalies)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return -self.model.decision_function(X)
    
    def detect(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Detect anomalies.
        
        Args:
            X: Input data
            threshold: Anomaly threshold
            
        Returns:
            Boolean array indicating anomalies
        """
        scores = self.predict(X)
        return scores > threshold


class OneClassSVMDetector(AnomalyDetector):
    """One-Class SVM anomaly detector."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize One-Class SVM detector.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model: Optional[OneClassSVM] = None
        
    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        """
        Fit the One-Class SVM to the data.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        self.model = OneClassSVM(
            nu=self.config.anomaly.contamination,
            kernel='rbf',
            gamma='scale'
        )
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores (negative values indicate anomalies)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return -self.model.decision_function(X)
    
    def detect(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Detect anomalies.
        
        Args:
            X: Input data
            threshold: Anomaly threshold
            
        Returns:
            Boolean array indicating anomalies
        """
        scores = self.predict(X)
        return scores > threshold


class Autoencoder(nn.Module):
    """Autoencoder neural network."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        output_activation: str = "sigmoid"
    ):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            output_activation: Output activation function name
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        prev_dim = hidden_dims_reversed[0]
        
        for hidden_dim in hidden_dims_reversed[1:]:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(self._get_activation(output_activation))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module."""
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed tensor
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded tensor
        """
        return self.encoder(x)


def create_anomaly_detector(
    detector_type: str, 
    config: DictConfig, 
    device: torch.device
) -> AnomalyDetector:
    """
    Create anomaly detector based on type.
    
    Args:
        detector_type: Type of detector ('autoencoder', 'isolation_forest', 'one_class_svm')
        config: Configuration object
        device: PyTorch device
        
    Returns:
        Anomaly detector instance
    """
    detectors = {
        "autoencoder": AutoencoderDetector,
        "isolation_forest": IsolationForestDetector,
        "one_class_svm": OneClassSVMDetector,
    }
    
    if detector_type not in detectors:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    detector_class = detectors[detector_type]
    
    if detector_type == "autoencoder":
        return detector_class(config, device)
    else:
        return detector_class(config)
