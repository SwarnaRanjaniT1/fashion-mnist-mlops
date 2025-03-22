import time
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from utils.data_loader import load_fashion_mnist, preprocess_data
from utils.feature_engineering import apply_feature_engineering
from utils.drift_detection import detect_drift, monitor_drift_over_time
import os

def main():
    """Run drift evaluation and log results to MLflow."""
    print("Loading Fashion MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    
    print("Preprocessing data...")
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test)
    
    print("Applying feature engineering...")
    X_train_fe, X_test_fe, _ = apply_feature_engineering(X_train, X_test)
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri('file:./mlruns')
    experiment_name = f"drift_evaluation_{int(time.time())}"
    mlflow.set_experiment(experiment_name)
    
    # Train a simple model
    print("Training baseline model...")
    sample_size = min(5000, len(X_train_fe))
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_fe[:sample_size], y_train[:sample_size])
    
    with mlflow.start_run(run_name="drift_evaluation"):
        # Test drift detection with different intensities
        drift_intensities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        results = []
        
        for intensity in drift_intensities:
            print(f"Testing drift with intensity {intensity}...")
            drift_results = detect_drift(model, X_test_fe, y_test, drift_intensity=intensity)
            
            results.append({
                "drift_intensity": intensity,
                "accuracy": drift_results['drift_metrics']['accuracy'],
                "drift_score": drift_results['drift_score'],
                "drift_detected": drift_results['drift_detected']
            })
            
            # Log to MLflow
            mlflow.log_metric(f"accuracy_at_drift_{intensity}", drift_results['drift_metrics']['accuracy'])
            mlflow.log_metric(f"drift_score_at_drift_{intensity}", drift_results['drift_score'])
        
        # Create summary dataframe
        results_df = pd.DataFrame(results)
        
        # Generate plots
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Drift Intensity')
        ax1.set_ylabel('Accuracy', color=color)
        ax1.plot(results_df['drift_intensity'], results_df['accuracy'], 'o-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Drift Score', color=color)
        ax2.plot(results_df['drift_intensity'], results_df['drift_score'], 'o-', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Impact of Data Drift on Model Performance')
        plt.tight_layout()
        
        # Save plot
        drift_plot_path = "drift_evaluation.png"
        plt.savefig(drift_plot_path)
        plt.close()
        
        # Log plot as artifact
        mlflow.log_artifact(drift_plot_path)
        
        # Clean up
        if os.path.exists(drift_plot_path):
            os.remove(drift_plot_path)
        
        # Monitor drift over time
        print("Simulating drift progression over time...")
        monitoring_results = monitor_drift_over_time(
            model, 
            X_test_fe, 
            y_test, 
            n_batches=5, 
            drift_progression=[0.0, 0.1, 0.2, 0.3, 0.4]
        )
        
        # Log time series metrics
        for i, timestamp in enumerate(monitoring_results['timestamps']):
            mlflow.log_metric(f"time_accuracy_{i}", monitoring_results['accuracy'][i])
            mlflow.log_metric(f"time_drift_score_{i}", monitoring_results['drift_scores'][i])
        
        # Create time series plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(monitoring_results['timestamps'])), 
                monitoring_results['accuracy'], 'o-', label='Accuracy')
        ax.plot(range(len(monitoring_results['timestamps'])), 
                monitoring_results['drift_scores'], 'o-', label='Drift Score')
        
        ax.set_xticks(range(len(monitoring_results['timestamps'])))
        ax.set_xticklabels([t.split()[0] for t in monitoring_results['timestamps']], rotation=45)
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Over Time with Increasing Drift')
        ax.legend()
        
        plt.tight_layout()
        
        # Save time series plot
        time_plot_path = "drift_time_series.png"
        plt.savefig(time_plot_path)
        plt.close()
        
        # Log plot as artifact
        mlflow.log_artifact(time_plot_path)
        
        # Clean up
        if os.path.exists(time_plot_path):
            os.remove(time_plot_path)
    
    print("Drift evaluation complete!")

if __name__ == "__main__":
    main()