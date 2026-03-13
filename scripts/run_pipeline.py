#!/usr/bin/env python3
"""Simple script to run the Anomaly Explanation System."""

import argparse
import sys
from pathlib import Path

from src.pipeline import AnomalyExplanationPipeline


def main():
    """Main function for running the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the Anomaly Explanation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --config configs/config.yaml
  python scripts/run_pipeline.py --config configs/config.yaml --output results/
  python scripts/run_pipeline.py --help
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Please ensure the config file exists or specify a different path.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"Configuration: {config_path}")
        print(f"Output directory: {output_dir}")
        print("Starting pipeline...")
    
    try:
        # Initialize and run pipeline
        pipeline = AnomalyExplanationPipeline(config_path)
        pipeline.run_full_pipeline(output_dir)
        
        if args.verbose:
            print("\nPipeline completed successfully!")
            print(f"Results saved to: {output_dir}")
            
            # Print summary
            summary = pipeline.get_summary()
            print(f"\nSummary:")
            print(f"  Dataset: {summary['data_info']['feature_names']}")
            print(f"  Detector: {summary['detector_info']['type']}")
            print(f"  Explainer: {summary['explainer_info']['type']}")
            print(f"  Anomalies detected: {summary['results']['n_anomalies']}")
            print(f"  Anomaly rate: {summary['results']['anomaly_rate']:.1%}")
        
    except Exception as e:
        print(f"Error: Pipeline failed with exception: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
