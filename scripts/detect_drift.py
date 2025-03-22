#!/usr/bin/env python3
"""
Detect drift in the Fashion MNIST dataset and generate drift reports.
This script is run by GitHub Actions during the CI/CD pipeline.
"""

import os
import sys
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import mlflow
import mlflow.sklearn

# Add the root directory to the path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_fashion_mnist, preprocess_data
from utils.drift_detection import detect_drift, monitor_drift_over_time

# Create reports/drift directory if it doesn't exist
os.makedirs("reports/drift", exist_ok=True)

# Configure MLflow
mlflow.set_tracking_uri('file:./mlruns')
experiment_name = f"fashion_mnist_drift_detection_{int(time.time())}"
mlflow.set_experiment(experiment_name)

def main():
    print("Loading Fashion MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    
    print("Loading model artifacts...")
    model_path = os.path.join("models", "fashion_mnist_model.joblib")
    pipeline_path = os.path.join("models", "feature_pipeline.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
        print("Error: Model artifacts not found. Run train_model.py first.")
        sys.exit(1)
    
    model = joblib.load(model_path)
    feature_pipeline = joblib.load(pipeline_path)
    
    # Preprocess the data
    print("Preprocessing test data...")
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test)
    
    # Detect drift
    print("Detecting data drift...")
    # Test with different drift intensities
    drift_intensities = [0.1, 0.2, 0.3]
    
    all_drift_results = {}
    
    for intensity in drift_intensities:
        print(f"Testing with drift intensity: {intensity}")
        drift_results = detect_drift(model, X_test_processed, y_test, drift_intensity=intensity)
        all_drift_results[str(intensity)] = {
            "accuracy_drop": float(drift_results["accuracy_drop"]),
            "drift_score": float(drift_results["drift_score"]),
            "drift_detected": bool(drift_results["drift_detected"])
        }
    
    # Save drift detection results
    drift_path = os.path.join("reports", "drift", "drift_detection.json")
    with open(drift_path, 'w') as f:
        json.dump(all_drift_results, f, indent=2)
    
    # Monitor drift over time
    print("Monitoring drift over time...")
    drift_progression = np.linspace(0, 0.5, 10)  # Gradually increasing drift
    monitoring_results = monitor_drift_over_time(
        model, X_test_processed, y_test, 
        n_batches=10, 
        drift_progression=drift_progression
    )
    
    # Save monitoring results
    monitoring_path = os.path.join("reports", "drift", "drift_monitoring.json")
    # Convert numpy arrays to lists for JSON serialization
    monitoring_results_serializable = {
        "timestamps": monitoring_results["timestamps"],
        "drift_intensity": monitoring_results["drift_intensity"].tolist(),
        "accuracy": [float(x) for x in monitoring_results["accuracy"]],
        "precision": [float(x) for x in monitoring_results["precision"]],
        "recall": [float(x) for x in monitoring_results["recall"]],
        "f1": [float(x) for x in monitoring_results["f1"]],
        "drift_scores": [float(x) for x in monitoring_results["drift_scores"]]
    }
    
    with open(monitoring_path, 'w') as f:
        json.dump(monitoring_results_serializable, f, indent=2)
    
    # Generate visualization of drift over time
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(monitoring_results["drift_intensity"], monitoring_results["accuracy"], 'b-o', label='Accuracy')
    plt.plot(monitoring_results["drift_intensity"], monitoring_results["precision"], 'g-o', label='Precision')
    plt.plot(monitoring_results["drift_intensity"], monitoring_results["recall"], 'r-o', label='Recall')
    plt.plot(monitoring_results["drift_intensity"], monitoring_results["f1"], 'y-o', label='F1')
    plt.xlabel('Drift Intensity')
    plt.ylabel('Score')
    plt.title('Model Performance vs. Drift Intensity')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(monitoring_results["drift_intensity"], monitoring_results["drift_scores"], 'c-o')
    plt.xlabel('Drift Intensity')
    plt.ylabel('Drift Score')
    plt.title('Drift Score vs. Drift Intensity')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save visualization
    drift_viz_path = os.path.join("reports", "drift", "drift_visualization.png")
    plt.savefig(drift_viz_path)
    
    # Log drift detection results to MLflow
    with mlflow.start_run() as run:
        # Log metrics
        for intensity, results in all_drift_results.items():
            mlflow.log_metric(f"accuracy_drop_{intensity}", results["accuracy_drop"])
            mlflow.log_metric(f"drift_score_{intensity}", results["drift_score"])
        
        # Log average drift metrics across intensities
        avg_accuracy_drop = np.mean([results["accuracy_drop"] for results in all_drift_results.values()])
        avg_drift_score = np.mean([results["drift_score"] for results in all_drift_results.values()])
        mlflow.log_metric("avg_accuracy_drop", avg_accuracy_drop)
        mlflow.log_metric("avg_drift_score", avg_drift_score)
        
        # Log artifacts
        mlflow.log_artifact(drift_path)
        mlflow.log_artifact(monitoring_path)
        mlflow.log_artifact(drift_viz_path)
        
        print(f"Drift detection results logged to MLflow with run_id: {run.info.run_id}")
    
    # Print summary
    print("\nDrift Detection Summary:")
    for intensity, results in all_drift_results.items():
        status = "DETECTED" if results["drift_detected"] else "NOT DETECTED"
        print(f"Intensity {intensity}: Accuracy Drop = {results['accuracy_drop']:.4f}, Drift Score = {results['drift_score']:.4f}, Status = {status}")
    
    print(f"\nDrift reports saved to reports/drift/ directory")

if __name__ == "__main__":
    main()