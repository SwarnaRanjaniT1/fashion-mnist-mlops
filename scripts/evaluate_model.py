#!/usr/bin/env python3
"""
Evaluate the trained model on the Fashion MNIST test set and generate evaluation reports.
This script is run by GitHub Actions during the CI/CD pipeline.
"""

import os
import sys
import joblib
import json
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import time

# Add the root directory to the path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_fashion_mnist, preprocess_data, get_class_names
from utils.explainability import explain_model

# Create reports directory if it doesn't exist
os.makedirs("reports", exist_ok=True)

# Configure MLflow
mlflow.set_tracking_uri('file:./mlruns')
experiment_name = f"fashion_mnist_evaluation_{int(time.time())}"
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
    
    # Evaluate the model
    print("Evaluating model performance...")
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    
    y_pred = model.predict(X_test_processed)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Generate evaluation report
    class_names = get_class_names()
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Save classification report
    report_path = os.path.join("reports", "classification_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save confusion matrix visualization
    cm_fig, cm_ax = plt.subplots(figsize=(12, 10))
    im = cm_ax.imshow(cm, interpolation='nearest', cmap='Blues')
    cm_ax.set_title('Confusion Matrix')
    plt.colorbar(im, ax=cm_ax)
    
    tick_marks = np.arange(len(class_names))
    cm_ax.set_xticks(tick_marks)
    cm_ax.set_yticks(tick_marks)
    cm_ax.set_xticklabels(class_names, rotation=45, ha='right')
    cm_ax.set_yticklabels(class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cm_ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    cm_ax.set_ylabel('True label')
    cm_ax.set_xlabel('Predicted label')
    plt.tight_layout()
    
    cm_path = os.path.join("reports", "confusion_matrix.png")
    plt.savefig(cm_path)
    
    # Generate summary metrics report
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metrics_path = os.path.join("reports", "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Log with MLflow
    with mlflow.start_run() as run:
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        
        # Log artifacts
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(metrics_path)
        
        print(f"Evaluation results logged to MLflow with run_id: {run.info.run_id}")
    
    # Print summary
    print("\nModel Evaluation Summary:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nDetailed reports saved to reports/ directory")

if __name__ == "__main__":
    main()