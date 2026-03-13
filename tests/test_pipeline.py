"""Test suite for the Anomaly Explanation System."""

import pytest
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.anomaly_detection.models import create_anomaly_detector, AutoencoderDetector
from src.explanation.explainers import create_explainer, SHAPExplainer
from src.metrics.evaluation import AnomalyDetectionMetrics, ExplanationEvaluator
from src.utils.device import get_device, set_seed
from src.utils.data_utils import DataLoader


class TestAnomalyDetectors:
    """Test anomaly detection models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        return X
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = OmegaConf.create({
            "model": {
                "name": "autoencoder",
                "hidden_dims": [32, 16],
                "learning_rate": 0.001,
                "epochs": 10,
                "batch_size": 16,
                "activation": "relu",
                "output_activation": "sigmoid"
            },
            "anomaly": {
                "threshold_method": "percentile",
                "threshold_value": 95.0,
                "contamination": 0.1
            },
            "seed": 42,
            "device": "cpu"
        })
        return config
    
    def test_autoencoder_detector(self, sample_data, config):
        """Test autoencoder detector."""
        device = get_device("cpu")
        detector = AutoencoderDetector(config, device)
        
        # Test fitting
        detector.fit(sample_data)
        assert detector.model is not None
        assert detector.scaler is not None
        
        # Test prediction
        scores = detector.predict(sample_data)
        assert len(scores) == len(sample_data)
        assert all(scores >= 0)  # Reconstruction error should be non-negative
        
        # Test detection
        anomalies = detector.detect(sample_data, threshold=0.1)
        assert len(anomalies) == len(sample_data)
        assert anomalies.dtype == bool
    
    def test_isolation_forest_detector(self, sample_data, config):
        """Test isolation forest detector."""
        config.model.name = "isolation_forest"
        detector = create_anomaly_detector("isolation_forest", config, get_device("cpu"))
        
        # Test fitting
        detector.fit(sample_data)
        assert detector.model is not None
        
        # Test prediction
        scores = detector.predict(sample_data)
        assert len(scores) == len(sample_data)
        
        # Test detection
        anomalies = detector.detect(sample_data, threshold=0.0)
        assert len(anomalies) == len(sample_data)
        assert anomalies.dtype == bool
    
    def test_one_class_svm_detector(self, sample_data, config):
        """Test one-class SVM detector."""
        config.model.name = "one_class_svm"
        detector = create_anomaly_detector("one_class_svm", config, get_device("cpu"))
        
        # Test fitting
        detector.fit(sample_data)
        assert detector.model is not None
        
        # Test prediction
        scores = detector.predict(sample_data)
        assert len(scores) == len(sample_data)
        
        # Test detection
        anomalies = detector.detect(sample_data, threshold=0.0)
        assert len(anomalies) == len(sample_data)
        assert anomalies.dtype == bool


class TestExplainers:
    """Test explanation methods."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(50, 4)
        return X
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        class MockModel:
            def predict(self, X):
                return np.random.randn(len(X))
        
        return MockModel()
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = OmegaConf.create({
            "explainer": {
                "name": "shap",
                "method": "kernel",
                "background_samples": 20,
                "max_samples": 50,
                "feature_names": ["f1", "f2", "f3", "f4"]
            },
            "seed": 42
        })
        return config
    
    def test_shap_explainer(self, sample_data, mock_model, config):
        """Test SHAP explainer."""
        explainer = SHAPExplainer(config)
        
        # Test fitting
        explainer.fit(sample_data, mock_model)
        assert explainer.explainer is not None
        assert explainer.background_data is not None
        
        # Test explanation
        explanations = explainer.explain(sample_data, mock_model)
        assert explanations.shape[0] <= len(sample_data)
        assert explanations.shape[1] == sample_data.shape[1]
        
        # Test single instance explanation
        explanation = explainer.explain_instance(sample_data[0], mock_model)
        assert len(explanation) == sample_data.shape[1]


class TestEvaluationMetrics:
    """Test evaluation metrics."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(50, 4)
        return X
    
    @pytest.fixture
    def sample_explanations(self):
        """Create sample explanations for testing."""
        np.random.seed(42)
        explanations = np.random.randn(50, 4)
        return explanations
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        class MockModel:
            def predict(self, X):
                return np.random.randn(len(X))
        
        return MockModel()
    
    def test_anomaly_detection_metrics(self):
        """Test anomaly detection metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.1, 0.6, 0.2])
        threshold = 0.5
        
        metrics = AnomalyDetectionMetrics.compute_metrics(y_true, y_scores, threshold)
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc_roc" in metrics
        assert "auc_pr" in metrics
        
        # Check metric ranges
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["auc_roc"] <= 1
        assert 0 <= metrics["auc_pr"] <= 1
    
    def test_explanation_evaluator(self, sample_data, sample_explanations, mock_model):
        """Test explanation evaluator."""
        config = OmegaConf.create({
            "evaluation": {
                "metrics": ["faithfulness", "utility"]
            },
            "seed": 42
        })
        
        evaluator = ExplanationEvaluator(config)
        results = evaluator.evaluate_explanations(
            sample_data, sample_explanations, mock_model
        )
        
        assert "faithfulness" in results
        assert "utility" in results
        
        # Check that scores are reasonable
        assert isinstance(results["faithfulness"], float)
        assert isinstance(results["utility"], float)


class TestDataUtils:
    """Test data utilities."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = OmegaConf.create({
            "data": {
                "name": "iris",
                "test_size": 0.3,
                "random_state": 42,
                "normalize": True,
                "standardize": True
            }
        })
        return config
    
    def test_data_loader(self, config):
        """Test data loader."""
        loader = DataLoader(config)
        
        # Test loading iris data
        X, y, feature_names, target_names = loader.load_iris_data()
        
        assert X.shape[1] == 4  # Iris has 4 features
        assert len(y) == len(X)
        assert len(feature_names) == 4
        assert len(target_names) == 3  # Iris has 3 classes
        
        # Test preprocessing
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        
        # Test metadata
        metadata = loader.get_feature_metadata()
        assert "features" in metadata
        assert "targets" in metadata


class TestDeviceUtils:
    """Test device utilities."""
    
    def test_get_device(self):
        """Test device selection."""
        # Test auto device selection
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        # Test CPU device
        device = get_device("cpu")
        assert device.type == "cpu"
        
        # Test invalid device
        with pytest.raises(RuntimeError):
            get_device("invalid_device")
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy seed
        np.random.seed(42)
        val1 = np.random.rand()
        
        set_seed(42)
        val2 = np.random.rand()
        
        assert val1 == val2


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full pipeline integration."""
        from src.pipeline import AnomalyExplanationPipeline
        
        # Create minimal config
        config = OmegaConf.create({
            "data": {
                "name": "iris",
                "test_size": 0.3,
                "random_state": 42,
                "normalize": True,
                "standardize": True
            },
            "model": {
                "name": "autoencoder",
                "hidden_dims": [32, 16],
                "learning_rate": 0.001,
                "epochs": 5,  # Reduced for testing
                "batch_size": 16,
                "activation": "relu",
                "output_activation": "sigmoid"
            },
            "anomaly": {
                "threshold_method": "percentile",
                "threshold_value": 95.0,
                "contamination": 0.1
            },
            "explainer": {
                "name": "shap",
                "method": "kernel",
                "background_samples": 20,
                "max_samples": 30,
                "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            },
            "evaluation": {
                "metrics": ["faithfulness", "utility"]
            },
            "visualization": {
                "figure_size": [8, 6],
                "dpi": 100,
                "style": "default"
            },
            "seed": 42,
            "device": "cpu"
        })
        
        # Save config temporarily
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(OmegaConf.to_container(config), f)
            config_path = f.name
        
        try:
            # Initialize pipeline
            pipeline = AnomalyExplanationPipeline(config_path)
            
            # Test individual components
            pipeline.load_data()
            assert pipeline.X_train is not None
            assert pipeline.X_test is not None
            
            pipeline.train_detector()
            assert pipeline.detector is not None
            
            pipeline.detect_anomalies()
            assert pipeline.anomaly_scores is not None
            assert pipeline.anomalies is not None
            
            pipeline.explain_anomalies()
            assert pipeline.explanations is not None
            
            pipeline.evaluate_results()
            # Evaluation results may be None if metrics fail
            
        finally:
            # Clean up
            import os
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__])
