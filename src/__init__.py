"""Anomaly Explanation System - A modern XAI framework for anomaly detection and explanation."""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"

from src.pipeline import AnomalyExplanationPipeline
from src.anomaly_detection.models import create_anomaly_detector
from src.explanation.explainers import create_explainer
from src.metrics.evaluation import AnomalyDetectionMetrics, ExplanationEvaluator
from src.utils.device import get_device, set_seed
from src.utils.data_utils import DataLoader

__all__ = [
    "AnomalyExplanationPipeline",
    "create_anomaly_detector",
    "create_explainer", 
    "AnomalyDetectionMetrics",
    "ExplanationEvaluator",
    "get_device",
    "set_seed",
    "DataLoader",
]
