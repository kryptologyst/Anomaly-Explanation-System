# Anomaly Explanation System

Comprehensive Explainable AI (XAI) system for anomaly detection and explanation, designed for research and educational purposes.

## ⚠️ DISCLAIMER

**IMPORTANT**: This system is designed for research and educational purposes only. XAI outputs may be unstable or misleading and should not be used for regulated decisions without human review. The explanations provided by this system are not guaranteed to be accurate or reliable for critical decision-making.

## Features

- **Multiple Anomaly Detection Methods**: Autoencoder, Isolation Forest, One-Class SVM
- **Comprehensive Explanation Methods**: SHAP, LIME, Integrated Gradients, Counterfactuals
- **Robust Evaluation Framework**: Faithfulness, stability, and utility metrics
- **Interactive Demo**: Streamlit-based web interface
- **Modern Architecture**: PyTorch-based with device fallback (CUDA → MPS → CPU)
- **Reproducible Research**: Deterministic seeding and comprehensive logging
- **Production-Ready**: Type hints, comprehensive testing, and CI/CD setup

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Anomaly-Explanation-System.git
cd Anomaly-Explanation-System

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.pipeline import AnomalyExplanationPipeline

# Initialize pipeline with configuration
pipeline = AnomalyExplanationPipeline("configs/config.yaml")

# Run complete analysis
pipeline.run_full_pipeline("results")

# Get summary
summary = pipeline.get_summary()
print(summary)
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/streamlit_app.py
```

## Project Structure

```
anomaly-explanation-system/
├── src/                          # Source code
│   ├── anomaly_detection/        # Anomaly detection models
│   ├── explanation/              # Explanation methods
│   ├── metrics/                  # Evaluation metrics
│   ├── visualization/            # Plotting utilities
│   ├── utils/                    # Utility functions
│   └── pipeline.py               # Main pipeline
├── configs/                      # Configuration files
├── data/                         # Data directory
│   ├── raw/                      # Raw data
│   └── processed/                # Processed data
├── demo/                         # Demo applications
├── tests/                        # Test suite
├── assets/                       # Generated assets
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
├── notebooks/                    # Jupyter notebooks
└── pyproject.toml               # Project configuration
```

## Configuration

The system uses YAML-based configuration with OmegaConf. Key configuration sections:

### Data Configuration
```yaml
data:
  name: iris                    # Dataset name
  test_size: 0.3               # Test set size
  random_state: 42             # Random seed
  normalize: true              # Enable normalization
  standardize: true            # Enable standardization
```

### Model Configuration
```yaml
model:
  name: autoencoder            # Model type
  hidden_dims: [64, 32, 16]   # Hidden layer dimensions
  learning_rate: 0.001         # Learning rate
  epochs: 100                  # Training epochs
  batch_size: 16              # Batch size
```

### Anomaly Detection Configuration
```yaml
anomaly:
  threshold_method: percentile  # Threshold method
  threshold_value: 95.0         # Threshold value
  contamination: 0.1           # Expected anomaly rate
```

### Explanation Configuration
```yaml
explainer:
  name: shap                   # Explainer type
  method: kernel              # SHAP method
  background_samples: 100     # Background samples
  max_samples: 1000          # Max samples for explanation
```

## Supported Methods

### Anomaly Detection
- **Autoencoder**: Deep learning-based reconstruction error
- **Isolation Forest**: Tree-based isolation
- **One-Class SVM**: Support vector-based detection

### Explanation Methods
- **SHAP**: SHapley Additive exPlanations (Kernel, Tree, Deep)
- **LIME**: Local Interpretable Model-agnostic Explanations
- **Integrated Gradients**: Gradient-based attribution
- **Counterfactuals**: Counterfactual explanation generation

### Evaluation Metrics
- **Faithfulness**: Deletion and insertion-based faithfulness
- **Stability**: Correlation, Kendall tau, Spearman rho
- **Utility**: Simplicity and completeness scores

## API Reference

### AnomalyExplanationPipeline

Main pipeline class for running complete analysis.

```python
class AnomalyExplanationPipeline:
    def __init__(self, config_path: Union[str, Path])
    def load_data(self) -> None
    def train_detector(self) -> None
    def detect_anomalies(self) -> None
    def explain_anomalies(self) -> None
    def evaluate_results(self) -> None
    def visualize_results(self, output_dir: Union[str, Path]) -> None
    def save_results(self, output_dir: Union[str, Path]) -> None
    def run_full_pipeline(self, output_dir: Union[str, Path] = "results") -> None
    def get_summary(self) -> Dict[str, Any]
```

### Anomaly Detectors

```python
# Create detector
detector = create_anomaly_detector("autoencoder", config, device)

# Train detector
detector.fit(X_train)

# Predict anomaly scores
scores = detector.predict(X_test)

# Detect anomalies
anomalies = detector.detect(X_test, threshold)
```

### Explainers

```python
# Create explainer
explainer = create_explainer("shap", config)

# Fit explainer
explainer.fit(X_train, model)

# Generate explanations
explanations = explainer.explain(X_test, model)

# Explain single instance
explanation = explainer.explain_instance(instance, model)
```

## Examples

### Basic Anomaly Detection

```python
from src.anomaly_detection.models import AutoencoderDetector
from src.utils.device import get_device

# Setup
device = get_device("auto")
config = load_config()

# Create and train detector
detector = AutoencoderDetector(config, device)
detector.fit(X_train)

# Detect anomalies
scores = detector.predict(X_test)
anomalies = detector.detect(X_test, threshold=0.1)
```

### SHAP Explanations

```python
from src.explanation.explainers import SHAPExplainer

# Create explainer
explainer = SHAPExplainer(config)
explainer.fit(X_train, detector)

# Generate explanations
explanations = explainer.explain(X_test, detector)

# Visualize
import shap
shap.summary_plot(explanations, X_test, feature_names)
```

### Evaluation

```python
from src.metrics.evaluation import ExplanationEvaluator

# Evaluate explanations
evaluator = ExplanationEvaluator(config)
results = evaluator.evaluate_explanations(X_test, explanations, detector)

print(f"Faithfulness: {results['faithfulness']:.4f}")
print(f"Stability: {results['stability']:.4f}")
print(f"Utility: {results['utility']:.4f}")
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black src/
ruff check src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py
```

### Code Quality

The project uses:
- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks for quality checks

## Limitations and Considerations

1. **Explanation Instability**: XAI methods can produce unstable explanations across different runs
2. **Model Dependency**: Explanation quality depends on the underlying model performance
3. **Computational Cost**: Some explanation methods (e.g., SHAP) can be computationally expensive
4. **Ground Truth**: Evaluation metrics may not reflect real-world explanation quality
5. **Domain Specificity**: Methods may not generalize across different domains

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{anomaly_explanation_system,
  title={Anomaly Explanation System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Anomaly-Explanation-System}
}
```

## Support

For questions, issues, or contributions, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed description
4. Contact the maintainers

## Changelog

### Version 1.0.0
- Initial release
- Support for multiple anomaly detection methods
- Comprehensive explanation framework
- Interactive Streamlit demo
- Robust evaluation metrics
- Modern PyTorch-based architecture
# Anomaly-Explanation-System
