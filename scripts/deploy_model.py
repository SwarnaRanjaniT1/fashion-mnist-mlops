#!/usr/bin/env python3
"""
Deploy the trained model to a production environment.
This script is run by GitHub Actions during the CI/CD pipeline.
"""

import os
import sys
import json
import mlflow
import mlflow.sklearn
import joblib
import datetime
import time

# Add the root directory to the path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure MLflow tracking server if available from environment
if "MLFLOW_TRACKING_URI" in os.environ and os.environ["MLFLOW_TRACKING_URI"]:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
else:
    mlflow.set_tracking_uri('file:./mlruns')

# Set experiment name
experiment_name = f"fashion_mnist_deployment_{int(time.time())}"
mlflow.set_experiment(experiment_name)

def main():
    print("Starting model deployment process...")
    
    # Load model and artifacts
    model_path = os.path.join("models", "fashion_mnist_model.joblib")
    pipeline_path = os.path.join("models", "feature_pipeline.joblib")
    metadata_path = os.path.join("models", "model_metadata.json")
    
    # Check if model artifacts exist
    if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
        print("Error: Model artifacts not found. Run train_model.py first.")
        sys.exit(1)
    
    # Load model and metadata
    model = joblib.load(model_path)
    feature_pipeline = joblib.load(pipeline_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load evaluation metrics if available
    try:
        metrics_path = os.path.join("reports", "metrics.json")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {"accuracy": "unknown", "precision": "unknown", "recall": "unknown", "f1_score": "unknown"}
    
    # Create a deployment marker
    deployment_info = {
        "model_type": metadata["model_type"],
        "training_date": metadata["training_date"],
        "deployment_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": "v1.0",  # In real scenario, this would be incremented
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        }
    }
    
    # Save deployment info
    os.makedirs("models/production", exist_ok=True)
    deployment_path = os.path.join("models", "production", "deployment_info.json")
    
    with open(deployment_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    # Copy the model to production directory
    production_model_path = os.path.join("models", "production", "fashion_mnist_model.joblib")
    production_pipeline_path = os.path.join("models", "production", "feature_pipeline.joblib")
    
    joblib.dump(model, production_model_path)
    joblib.dump(feature_pipeline, production_pipeline_path)
    
    # Log model to MLflow with additional production tags
    with mlflow.start_run() as run:
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log artifacts
        mlflow.log_artifact(model_path, "production_model")
        mlflow.log_artifact(pipeline_path, "production_pipeline")
        mlflow.log_artifact(deployment_path, "deployment_info")
        
        # Set tags for production model
        mlflow.set_tag("model_stage", "Production")
        mlflow.set_tag("deployment_timestamp", deployment_info["deployment_date"])
        mlflow.set_tag("model_version", deployment_info["model_version"])
        
        # Log metrics if available
        if metrics["accuracy"] != "unknown":
            mlflow.log_metrics({
                "prod_accuracy": float(metrics["accuracy"]),
                "prod_precision": float(metrics["precision"]),
                "prod_recall": float(metrics["recall"]),
                "prod_f1": float(metrics["f1_score"])
            })
        
        print(f"Model deployed to production with run_id: {run.info.run_id}")
    
    print("\nDeployment Summary:")
    print(f"Model Type: {deployment_info['model_type']}")
    print(f"Deployment Date: {deployment_info['deployment_date']}")
    print(f"Model Version: {deployment_info['model_version']}")
    print(f"Metrics:")
    print(f"  - Accuracy: {deployment_info['metrics']['accuracy']}")
    print(f"  - Precision: {deployment_info['metrics']['precision']}")
    print(f"  - Recall: {deployment_info['metrics']['recall']}")
    print(f"  - F1 Score: {deployment_info['metrics']['f1_score']}")
    
    print("\nModel artifacts copied to models/production/ directory")
    print("Deployment process completed successfully")

if __name__ == "__main__":
    main()