"""Streamlit demo for the Anomaly Explanation System."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pipeline import AnomalyExplanationPipeline
from src.utils.device import get_device_info


def load_config() -> Dict[str, Any]:
    """Load configuration for the demo."""
    return {
        "data": {
            "name": "iris",
            "test_size": 0.3,
            "random_state": 42,
            "normalize": True,
            "standardize": True
        },
        "model": {
            "name": "autoencoder",
            "hidden_dims": [64, 32, 16],
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 16
        },
        "anomaly": {
            "threshold_method": "percentile",
            "threshold_value": 95.0,
            "contamination": 0.1
        },
        "explainer": {
            "name": "shap",
            "method": "kernel",
            "background_samples": 100,
            "max_samples": 1000,
            "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        },
        "evaluation": {
            "metrics": ["faithfulness", "stability", "utility"]
        },
        "visualization": {
            "figure_size": [10, 6],
            "dpi": 300,
            "style": "seaborn-v0_8"
        },
        "seed": 42,
        "device": "auto"
    }


def create_sample_data() -> Tuple[np.ndarray, List[str]]:
    """Create sample data for the demo."""
    # Generate synthetic data similar to Iris
    np.random.seed(42)
    n_samples = 150
    n_features = 4
    
    # Normal data (similar to Iris)
    normal_data = np.random.multivariate_normal(
        mean=[5.8, 3.0, 3.8, 1.2],
        cov=[[0.7, 0.0, 0.0, 0.0],
             [0.0, 0.2, 0.0, 0.0],
             [0.0, 0.0, 0.7, 0.0],
             [0.0, 0.0, 0.0, 0.2]],
        size=n_samples
    )
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=10, replace=False)
    normal_data[anomaly_indices] += np.random.normal(0, 2, (10, n_features))
    
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    return normal_data, feature_names


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Anomaly Explanation System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Anomaly Explanation System")
    st.markdown("""
    **DISCLAIMER**: This system is for research and educational purposes only. 
    XAI outputs may be unstable or misleading and should not be used for regulated decisions without human review.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data selection
    data_option = st.sidebar.selectbox(
        "Dataset",
        ["Iris", "Synthetic"],
        help="Choose the dataset to analyze"
    )
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Anomaly Detector",
        ["autoencoder", "isolation_forest", "one_class_svm"],
        help="Choose the anomaly detection method"
    )
    
    # Explainer selection
    explainer_option = st.sidebar.selectbox(
        "Explanation Method",
        ["shap", "lime", "integrated_gradients", "counterfactual"],
        help="Choose the explanation method"
    )
    
    # Threshold configuration
    threshold_method = st.sidebar.selectbox(
        "Threshold Method",
        ["percentile", "iqr", "statistical"],
        help="Method for determining anomaly threshold"
    )
    
    threshold_value = st.sidebar.slider(
        "Threshold Value",
        min_value=80.0,
        max_value=99.0,
        value=95.0,
        step=1.0,
        help="Threshold percentile for anomaly detection"
    )
    
    # Run analysis button
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Running anomaly detection and explanation..."):
            try:
                # Load configuration
                config = load_config()
                config["data"]["name"] = "iris" if data_option == "Iris" else "synthetic"
                config["model"]["name"] = model_option
                config["explainer"]["name"] = explainer_option
                config["anomaly"]["threshold_method"] = threshold_method
                config["anomaly"]["threshold_value"] = threshold_value
                
                # Create temporary config file
                config_path = Path("temp_config.yaml")
                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
                
                # Initialize pipeline
                pipeline = AnomalyExplanationPipeline(config_path)
                
                # Run pipeline
                pipeline.run_full_pipeline("temp_results")
                
                # Store results in session state
                st.session_state.pipeline = pipeline
                st.session_state.config = config
                
                st.success("Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    
    # Display results if available
    if hasattr(st.session_state, 'pipeline'):
        pipeline = st.session_state.pipeline
        config = st.session_state.config
        
        st.header("Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview", "Anomaly Detection", "Explanations", "Evaluation", "System Info"
        ])
        
        with tab1:
            st.subheader("Analysis Overview")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Samples",
                    len(pipeline.X_test) if pipeline.X_test is not None else 0
                )
            
            with col2:
                n_anomalies = int(np.sum(pipeline.anomalies)) if pipeline.anomalies is not None else 0
                st.metric(
                    "Anomalies Detected",
                    n_anomalies
                )
            
            with col3:
                anomaly_rate = float(np.mean(pipeline.anomalies)) if pipeline.anomalies is not None else 0
                st.metric(
                    "Anomaly Rate",
                    f"{anomaly_rate:.1%}"
                )
            
            with col4:
                n_features = len(pipeline.feature_names) if pipeline.feature_names is not None else 0
                st.metric(
                    "Features",
                    n_features
                )
            
            # Configuration summary
            st.subheader("Configuration")
            config_df = pd.DataFrame([
                {"Parameter": "Dataset", "Value": config["data"]["name"]},
                {"Parameter": "Detector", "Value": config["model"]["name"]},
                {"Parameter": "Explainer", "Value": config["explainer"]["name"]},
                {"Parameter": "Threshold Method", "Value": config["anomaly"]["threshold_method"]},
                {"Parameter": "Threshold Value", "Value": config["anomaly"]["threshold_value"]},
            ])
            st.dataframe(config_df, use_container_width=True)
        
        with tab2:
            st.subheader("Anomaly Detection Results")
            
            if pipeline.anomaly_scores is not None and pipeline.anomalies is not None:
                # Anomaly scores plot
                fig_scores = go.Figure()
                
                # Normal samples
                normal_indices = ~pipeline.anomalies
                fig_scores.add_trace(go.Scatter(
                    x=np.where(normal_indices)[0],
                    y=pipeline.anomaly_scores[normal_indices],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=8, opacity=0.6)
                ))
                
                # Anomalous samples
                anomaly_indices = pipeline.anomalies
                fig_scores.add_trace(go.Scatter(
                    x=np.where(anomaly_indices)[0],
                    y=pipeline.anomaly_scores[anomaly_indices],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=8, opacity=0.8)
                ))
                
                # Threshold line
                threshold = pipeline._determine_threshold(pipeline.anomaly_scores)
                fig_scores.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="black",
                    annotation_text=f"Threshold: {threshold:.3f}"
                )
                
                fig_scores.update_layout(
                    title="Anomaly Scores",
                    xaxis_title="Sample Index",
                    yaxis_title="Anomaly Score",
                    height=500
                )
                
                st.plotly_chart(fig_scores, use_container_width=True)
                
                # Feature distributions
                if pipeline.X_test is not None and pipeline.feature_names is not None:
                    st.subheader("Feature Distributions")
                    
                    # Create subplots for each feature
                    fig_dist = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=pipeline.feature_names,
                        vertical_spacing=0.1
                    )
                    
                    for i, feature_name in enumerate(pipeline.feature_names):
                        row = i // 2 + 1
                        col = i % 2 + 1
                        
                        # Normal data
                        normal_data = pipeline.X_test[~pipeline.anomalies, i]
                        fig_dist.add_trace(
                            go.Histogram(
                                x=normal_data,
                                name='Normal',
                                opacity=0.6,
                                marker_color='blue'
                            ),
                            row=row, col=col
                        )
                        
                        # Anomaly data
                        anomaly_data = pipeline.X_test[pipeline.anomalies, i]
                        fig_dist.add_trace(
                            go.Histogram(
                                x=anomaly_data,
                                name='Anomaly',
                                opacity=0.6,
                                marker_color='red'
                            ),
                            row=row, col=col
                        )
                    
                    fig_dist.update_layout(
                        title="Feature Distributions: Normal vs Anomaly",
                        height=600,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab3:
            st.subheader("Explanations")
            
            if pipeline.explanations is not None and pipeline.feature_names is not None:
                # Explanation heatmap
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=pipeline.explanations,
                    x=pipeline.feature_names,
                    colorscale='RdBu',
                    colorbar=dict(title="Explanation Value")
                ))
                
                fig_heatmap.update_layout(
                    title="Explanation Heatmap",
                    xaxis_title="Features",
                    yaxis_title="Samples",
                    height=500
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Feature importance
                importance = np.mean(np.abs(pipeline.explanations), axis=0)
                
                fig_importance = go.Figure(data=go.Bar(
                    x=pipeline.feature_names,
                    y=importance,
                    marker_color='lightblue'
                ))
                
                fig_importance.update_layout(
                    title="Feature Importance (Mean Absolute Explanation)",
                    xaxis_title="Features",
                    yaxis_title="Importance",
                    height=400
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Individual sample explanations
                st.subheader("Individual Sample Explanations")
                
                sample_idx = st.selectbox(
                    "Select Sample",
                    range(len(pipeline.explanations)),
                    help="Choose a sample to view its explanation"
                )
                
                if sample_idx is not None:
                    sample_explanation = pipeline.explanations[sample_idx]
                    is_anomaly = pipeline.anomalies[sample_idx] if pipeline.anomalies is not None else False
                    
                    st.write(f"**Sample {sample_idx}** - {'Anomaly' if is_anomaly else 'Normal'}")
                    
                    fig_sample = go.Figure(data=go.Bar(
                        x=pipeline.feature_names,
                        y=sample_explanation,
                        marker_color=['red' if x >= 0 else 'blue' for x in sample_explanation]
                    ))
                    
                    fig_sample.update_layout(
                        title=f"Explanation for Sample {sample_idx}",
                        xaxis_title="Features",
                        yaxis_title="Explanation Value",
                        height=400
                    )
                    
                    st.plotly_chart(fig_sample, use_container_width=True)
        
        with tab4:
            st.subheader("Evaluation Metrics")
            
            if pipeline.evaluation_results is not None:
                # Evaluation metrics
                metrics_df = pd.DataFrame([
                    {"Metric": metric, "Score": f"{score:.4f}"}
                    for metric, score in pipeline.evaluation_results.items()
                ])
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # Metrics visualization
                fig_metrics = go.Figure(data=go.Bar(
                    x=list(pipeline.evaluation_results.keys()),
                    y=list(pipeline.evaluation_results.values()),
                    marker_color='lightgreen'
                ))
                
                fig_metrics.update_layout(
                    title="Evaluation Metrics",
                    xaxis_title="Metrics",
                    yaxis_title="Score",
                    height=400
                )
                
                st.plotly_chart(fig_metrics, use_container_width=True)
            else:
                st.info("No evaluation results available. Run the analysis to see metrics.")
        
        with tab5:
            st.subheader("System Information")
            
            # Device information
            device_info = get_device_info()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Device Information**")
                for device, available in device_info.items():
                    if isinstance(available, bool):
                        status = "✅ Available" if available else "❌ Not Available"
                        st.write(f"- {device.title()}: {status}")
                    else:
                        st.write(f"- {device}: {available}")
            
            with col2:
                st.write("**Pipeline Summary**")
                summary = pipeline.get_summary()
                
                st.write(f"- **Dataset**: {summary['data_info']['feature_names']}")
                st.write(f"- **Detector**: {summary['detector_info']['type']}")
                st.write(f"- **Explainer**: {summary['explainer_info']['type']}")
                st.write(f"- **Device**: {summary['detector_info']['device']}")
    
    else:
        st.info("👈 Configure the analysis parameters in the sidebar and click 'Run Analysis' to get started.")
        
        # Show sample data preview
        st.subheader("Sample Data Preview")
        sample_data, feature_names = create_sample_data()
        
        df = pd.DataFrame(sample_data, columns=feature_names)
        st.dataframe(df.head(10), use_container_width=True)
        
        st.write(f"**Dataset Shape**: {sample_data.shape}")
        st.write(f"**Features**: {', '.join(feature_names)}")


if __name__ == "__main__":
    main()
