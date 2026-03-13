"""Main pipeline for anomaly detection and explanation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.anomaly_detection.models import create_anomaly_detector
from src.explanation.explainers import create_explainer
from src.metrics.evaluation import AnomalyDetectionMetrics, ExplanationEvaluator
from src.utils.data_utils import DataLoader
from src.utils.device import setup_reproducibility
from src.visualization.plots import AnomalyVisualizer


class AnomalyExplanationPipeline:
    """Main pipeline for anomaly detection and explanation."""
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = OmegaConf.load(config_path)
        self.device = setup_reproducibility(self.config)
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.visualizer = AnomalyVisualizer(self.config)
        self.evaluator = ExplanationEvaluator(self.config)
        
        # Model and explainer will be initialized during training
        self.detector: Optional[Any] = None
        self.explainer: Optional[Any] = None
        
        # Data
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.target_names: Optional[List[str]] = None
        
        # Results
        self.anomaly_scores: Optional[np.ndarray] = None
        self.anomalies: Optional[np.ndarray] = None
        self.explanations: Optional[np.ndarray] = None
        self.evaluation_results: Optional[Dict[str, float]] = None
    
    def load_data(self) -> None:
        """Load and preprocess data."""
        print("Loading data...")
        
        # Load raw data
        X, y, feature_names, target_names = self.data_loader.load_data()
        self.feature_names = feature_names
        self.target_names = target_names
        
        # Preprocess data
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.preprocess_data(X, y)
        
        print(f"Data loaded: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test samples")
        print(f"Features: {self.feature_names}")
    
    def train_detector(self) -> None:
        """Train the anomaly detector."""
        print("Training anomaly detector...")
        
        # Create detector
        detector_type = self.config.model.name
        self.detector = create_anomaly_detector(detector_type, self.config, self.device)
        
        # Train detector
        self.detector.fit(self.X_train)
        
        print(f"Detector trained: {detector_type}")
    
    def detect_anomalies(self) -> None:
        """Detect anomalies in test data."""
        print("Detecting anomalies...")
        
        if self.detector is None:
            raise ValueError("Detector must be trained before anomaly detection")
        
        # Get anomaly scores
        self.anomaly_scores = self.detector.predict(self.X_test)
        
        # Determine threshold
        threshold = self._determine_threshold(self.anomaly_scores)
        
        # Detect anomalies
        self.anomalies = self.detector.detect(self.X_test, threshold)
        
        n_anomalies = np.sum(self.anomalies)
        print(f"Detected {n_anomalies} anomalies out of {len(self.anomaly_scores)} samples")
        print(f"Threshold: {threshold:.4f}")
    
    def _determine_threshold(self, scores: np.ndarray) -> float:
        """Determine anomaly detection threshold."""
        method = self.config.anomaly.threshold_method
        
        if method == "percentile":
            threshold = np.percentile(scores, self.config.anomaly.threshold_value)
        elif method == "iqr":
            q1, q3 = np.percentile(scores, [25, 75])
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
        elif method == "statistical":
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            threshold = mean_score + 2 * std_score
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        return threshold
    
    def explain_anomalies(self) -> None:
        """Generate explanations for anomalies."""
        print("Generating explanations...")
        
        if self.detector is None:
            raise ValueError("Detector must be trained before explanation")
        
        # Create explainer
        explainer_type = self.config.explainer.name
        self.explainer = create_explainer(explainer_type, self.config)
        
        # Fit explainer
        self.explainer.fit(self.X_train, self.detector)
        
        # Generate explanations
        self.explanations = self.explainer.explain(self.X_test, self.detector)
        
        print(f"Explanations generated using {explainer_type}")
    
    def evaluate_results(self) -> None:
        """Evaluate detection and explanation results."""
        print("Evaluating results...")
        
        # Anomaly detection metrics
        if self.anomaly_scores is not None and self.anomalies is not None:
            # Create synthetic ground truth for evaluation
            # In practice, you would have real anomaly labels
            y_true = self._create_synthetic_labels()
            
            detection_metrics = AnomalyDetectionMetrics.compute_metrics(
                y_true, self.anomaly_scores, 
                np.percentile(self.anomaly_scores, 95)
            )
            
            print("Anomaly Detection Metrics:")
            for metric, value in detection_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Explanation evaluation
        if self.explanations is not None:
            self.evaluation_results = self.evaluator.evaluate_explanations(
                self.X_test, self.explanations, self.detector
            )
            
            print("Explanation Evaluation:")
            for metric, value in self.evaluation_results.items():
                print(f"  {metric}: {value:.4f}")
    
    def _create_synthetic_labels(self) -> np.ndarray:
        """Create synthetic ground truth labels for evaluation."""
        # This is a placeholder - in practice you would have real labels
        # For now, we'll use the detected anomalies as "ground truth"
        return self.anomalies.astype(int)
    
    def visualize_results(self, output_dir: Union[str, Path]) -> None:
        """Generate and save visualizations."""
        print("Generating visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.anomaly_scores is not None and self.anomalies is not None:
            # Plot anomaly scores
            threshold = self._determine_threshold(self.anomaly_scores)
            self.visualizer.plot_anomaly_scores(
                self.anomaly_scores, self.anomalies, threshold,
                save_path=output_dir / "anomaly_scores.png"
            )
            
            # Plot feature distributions
            self.visualizer.plot_feature_distributions(
                self.X_test, self.anomalies, self.feature_names,
                save_path=output_dir / "feature_distributions.png"
            )
        
        if self.explanations is not None:
            # Plot explanation heatmap
            self.visualizer.plot_explanation_heatmap(
                self.explanations, self.feature_names,
                save_path=output_dir / "explanation_heatmap.png"
            )
            
            # Plot feature importance
            self.visualizer.plot_feature_importance(
                self.explanations, self.feature_names,
                save_path=output_dir / "feature_importance.png"
            )
            
            # Plot SHAP summary if using SHAP
            if self.config.explainer.name == "shap":
                self.visualizer.plot_shap_summary(
                    self.explanations, self.X_test, self.feature_names,
                    save_path=output_dir / "shap_summary.png"
                )
        
        if self.evaluation_results is not None:
            # Plot evaluation metrics
            self.visualizer.plot_evaluation_metrics(
                self.evaluation_results,
                save_path=output_dir / "evaluation_metrics.png"
            )
        
        print(f"Visualizations saved to {output_dir}")
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save results to files."""
        print("Saving results...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save anomaly scores
        if self.anomaly_scores is not None:
            np.save(output_dir / "anomaly_scores.npy", self.anomaly_scores)
        
        # Save anomaly labels
        if self.anomalies is not None:
            np.save(output_dir / "anomalies.npy", self.anomalies)
        
        # Save explanations
        if self.explanations is not None:
            np.save(output_dir / "explanations.npy", self.explanations)
        
        # Save evaluation results
        if self.evaluation_results is not None:
            with open(output_dir / "evaluation_results.json", 'w') as f:
                json.dump(self.evaluation_results, f, indent=2)
        
        # Save configuration
        with open(output_dir / "config.yaml", 'w') as f:
            OmegaConf.save(self.config, f)
        
        print(f"Results saved to {output_dir}")
    
    def run_full_pipeline(self, output_dir: Union[str, Path] = "results") -> None:
        """Run the complete pipeline."""
        print("Starting Anomaly Explanation Pipeline")
        print("=" * 50)
        
        try:
            # Load data
            self.load_data()
            
            # Train detector
            self.train_detector()
            
            # Detect anomalies
            self.detect_anomalies()
            
            # Generate explanations
            self.explain_anomalies()
            
            # Evaluate results
            self.evaluate_results()
            
            # Generate visualizations
            self.visualize_results(output_dir)
            
            # Save results
            self.save_results(output_dir)
            
            print("=" * 50)
            print("Pipeline completed successfully!")
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline results."""
        summary = {
            "config": OmegaConf.to_container(self.config),
            "data_info": {
                "n_train": len(self.X_train) if self.X_train is not None else 0,
                "n_test": len(self.X_test) if self.X_test is not None else 0,
                "n_features": len(self.feature_names) if self.feature_names is not None else 0,
                "feature_names": self.feature_names,
            },
            "detector_info": {
                "type": self.config.model.name,
                "device": str(self.device),
            },
            "explainer_info": {
                "type": self.config.explainer.name,
                "method": self.config.explainer.method,
            },
            "results": {
                "n_anomalies": int(np.sum(self.anomalies)) if self.anomalies is not None else 0,
                "anomaly_rate": float(np.mean(self.anomalies)) if self.anomalies is not None else 0,
                "evaluation_metrics": self.evaluation_results,
            }
        }
        
        return summary
