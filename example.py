#!/usr/bin/env python3
"""Simple example script demonstrating the Anomaly Explanation System."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import AnomalyExplanationPipeline
from src.utils.device import get_device_info


def main():
    """Run a simple example of the anomaly explanation system."""
    print("🔍 Anomaly Explanation System - Simple Example")
    print("=" * 50)
    
    # Check system info
    device_info = get_device_info()
    print(f"Available devices: {[k for k, v in device_info.items() if v]}")
    
    # Initialize pipeline
    config_path = "configs/config.yaml"
    if not Path(config_path).exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Please ensure configs/config.yaml exists")
        return
    
    print(f"📁 Using configuration: {config_path}")
    
    try:
        # Create pipeline
        pipeline = AnomalyExplanationPipeline(config_path)
        
        # Run analysis
        print("🚀 Running anomaly detection and explanation...")
        pipeline.run_full_pipeline("example_results")
        
        # Show results
        summary = pipeline.get_summary()
        print("\n📊 Results Summary:")
        print(f"  Dataset: {summary['data_info']['feature_names']}")
        print(f"  Detector: {summary['detector_info']['type']}")
        print(f"  Explainer: {summary['explainer_info']['type']}")
        print(f"  Anomalies detected: {summary['results']['n_anomalies']}")
        print(f"  Anomaly rate: {summary['results']['anomaly_rate']:.1%}")
        
        if summary['results']['evaluation_metrics']:
            print("  Evaluation metrics:")
            for metric, score in summary['results']['evaluation_metrics'].items():
                print(f"    {metric}: {score:.4f}")
        
        print(f"\n✅ Analysis completed! Results saved to: example_results/")
        print("\n💡 Next steps:")
        print("  - Run 'streamlit run demo/streamlit_app.py' for interactive demo")
        print("  - Check example_results/ for generated visualizations")
        print("  - Modify configs/config.yaml to try different settings")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your configuration and dependencies")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
