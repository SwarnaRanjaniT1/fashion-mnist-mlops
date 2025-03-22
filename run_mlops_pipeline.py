#!/usr/bin/env python3
"""
MLOps Pipeline for Fashion MNIST dataset

This script runs a complete MLOps pipeline for the Fashion MNIST dataset, including:
- Data loading
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Explainability
- AutoML Model Selection
- Hyperparameter Optimization
- Model Performance Monitoring
- Drift Detection
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"mlops_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import utility functions
from utils.data_loader import load_fashion_mnist, preprocess_data, get_class_names
from utils.eda import perform_eda
from utils.feature_engineering import apply_feature_engineering
from utils.explainability import explain_model
from utils.model_selection import select_model_with_automl
from utils.hyperparameter_optimization import optimize_hyperparameters
from utils.model_monitoring import track_model_performance
from utils.drift_detection import detect_drift, monitor_drift_over_time

# Initialize MLflow
mlflow.set_tracking_uri('file:./mlruns')
experiment_name = f"fashion_mnist_mlops_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.set_experiment(experiment_name)
logger.info(f"MLflow experiment: {experiment_name}")

# Create directories for reports and models if they don't exist
os.makedirs('reports/eda', exist_ok=True)
os.makedirs('reports/explainability', exist_ok=True)
os.makedirs('reports/model_performance', exist_ok=True)
os.makedirs('models', exist_ok=True)

def save_figure(fig, directory, filename):
    """Save a matplotlib figure to a specified directory with a given filename."""
    path = os.path.join(directory, filename)
    fig.savefig(path)
    logger.info(f"Figure saved to {path}")
    return path

def main():
    # Start MLflow run
    with mlflow.start_run(run_name="complete_pipeline") as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run with ID: {run_id}")
        
        # 1. Data Loading & EDA
        logger.info("MILESTONE 1: Data Loading & EDA")
        
        # Load Fashion MNIST dataset
        logger.info("Loading Fashion MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
        logger.info(f"Dataset loaded: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        logger.info(f"Dataset loaded: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")
        
        # Get class names
        class_names = get_class_names()
        
        # Display class distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        class_counts = [(y_train == i).sum() for i in range(10)]
        ax.bar(class_names, class_counts)
        ax.set_title('Class Distribution in Training Set')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        class_dist_path = save_figure(fig, 'reports/eda', 'class_distribution.png')
        
        # Display sample images
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        axs = axs.flatten()
        for i in range(10):
            idx = np.where(y_train == i)[0][0]
            axs[i].imshow(X_train[idx], cmap='gray')
            axs[i].set_title(class_names[i])
            axs[i].axis('off')
        plt.tight_layout()
        samples_path = save_figure(fig, 'reports/eda', 'sample_images.png')
        
        # Perform EDA
        logger.info("Generating automated EDA report...")
        eda_report = perform_eda(X_train, y_train)
        
        # Save EDA report summary
        # Define a recursive function to make objects JSON serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                # Handle pandas Series or DataFrame
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [make_serializable(item) for item in obj]
            else:
                return obj
            
        # Create a serializable version of the EDA report
        serializable_report = make_serializable(eda_report)
        
        # Save to JSON file
        with open('reports/eda/eda_summary.json', 'w') as f:
            json.dump(serializable_report, f, indent=2)
        logger.info("EDA completed and saved to reports/eda/")
        
        # 2. Feature Engineering & Explainability
        logger.info("MILESTONE 2: Feature Engineering & Explainability")
        
        # Perform feature engineering
        logger.info("Engineering features...")
        X_train_processed, X_test_processed, feature_pipeline = apply_feature_engineering(X_train, X_test)
        logger.info(f"Feature engineering complete: X_train_processed.shape={X_train_processed.shape}")
        
        # Save the first 5 samples for reference
        with open('reports/eda/processed_features_sample.csv', 'w') as f:
            pd.DataFrame(X_train_processed[:5]).to_csv(f, index=False)
        
        # PCA visualization of processed features
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_processed[:1000])  # Use subset for visualization
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                            c=y_train[:1000], alpha=0.6, cmap='viridis')
        ax.set_title('PCA visualization of processed features')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        plt.tight_layout()
        pca_path = save_figure(fig, 'reports/eda', 'pca_visualization.png')
        
        # Model explainability
        logger.info("Training a simple model for explainability...")
        from sklearn.ensemble import RandomForestClassifier
        simple_model = RandomForestClassifier(n_estimators=50, random_state=42)
        simple_model.fit(X_train_processed[:5000], y_train[:5000])  # Train on subset for speed
        
        # Generate explainability visualizations
        logger.info("Generating explainability visualizations...")
        explain_model(simple_model, X_train_processed, y_train)
        logger.info("Explainability visualizations generated in reports/explainability/")
        
        # 3. Model Selection & Hyperparameter Tuning
        logger.info("MILESTONE 3: Model Selection & Hyperparameter Tuning")
        
        # Run AutoML
        logger.info("Running AutoML to find the best model...")
        best_model, all_models = select_model_with_automl(
            X_train_processed, y_train, X_test_processed, y_test
        )
        logger.info(f"Best model found: {type(best_model).__name__}")
        
        # Plot model comparison
        model_names = [type(model).__name__ for model in all_models.keys()]
        test_scores = [score['test_score'] for score in all_models.values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, test_scores, color='skyblue')
        
        # Function to add labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(bars)
        ax.set_title('Model Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Test Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)
        plt.tight_layout()
        model_comp_path = save_figure(fig, 'reports/model_performance', 'model_comparison.png')
        
        # Run hyperparameter optimization
        logger.info("Optimizing hyperparameters for the best model...")
        optimized_model, optimization_history = optimize_hyperparameters(
            best_model, X_train_processed, y_train, X_test_processed, y_test
        )
        logger.info("Hyperparameter optimization completed")
        
        # Save optimized model
        model_path = os.path.join('models', f'optimized_model_{run_id}.pkl')
        import joblib
        joblib.dump(optimized_model, model_path)
        logger.info(f"Optimized model saved to {model_path}")
        
        # Visualization of optimization progress
        best_values = [trial.value if trial.value is not None else 0 for trial in optimization_history]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(best_values) + 1), best_values, 'o-', color='blue')
        ax.set_title('Hyperparameter Optimization Progress')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Objective Value (higher is better)')
        ax.grid(True)
        plt.tight_layout()
        opt_path = save_figure(fig, 'reports/model_performance', 'optimization_progress.png')
        
        # Performance metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        y_pred = optimized_model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion Matrix')
        plt.colorbar(im, ax=ax)
        
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        plt.tight_layout()
        cm_path = save_figure(fig, 'reports/model_performance', 'confusion_matrix.png')
        
        # Save classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('reports/model_performance/classification_report.csv')
        
        # 4. Model Monitoring & Drift Detection
        logger.info("MILESTONE 4: Model Monitoring & Drift Detection")
        
        # Track model performance using MLflow
        logger.info("Tracking model performance with MLflow...")
        run_id, metrics = track_model_performance(
            optimized_model, X_train_processed, y_train, X_test_processed, y_test
        )
        logger.info(f"Model tracked in MLflow run: {run_id}")
        logger.info(f"Model metrics: {metrics}")
        
        # Detect drift
        logger.info("Detecting and simulating data drift...")
        drift_results = detect_drift(optimized_model, X_test_processed, y_test)
        
        # Monitor drift over time
        logger.info("Monitoring drift over time...")
        drift_monitoring = monitor_drift_over_time(
            optimized_model, X_test_processed, y_test, n_batches=5
        )
        
        # Plot drift monitoring results
        accuracies = [result['accuracy'] for result in drift_monitoring['results']]
        drift_intensities = drift_monitoring['drift_intensities']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(drift_intensities, accuracies, 'o-', label='Accuracy')
        ax.set_title('Model Performance Under Increasing Drift')
        ax.set_xlabel('Drift Intensity')
        ax.set_ylabel('Accuracy')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        drift_path = save_figure(fig, 'reports/model_performance', 'drift_monitoring.png')
        
        # Create a comprehensive report
        logger.info("Creating comprehensive MLOps pipeline report...")
        with open('mlops_report.md', 'w') as f:
            f.write("# Fashion MNIST MLOps Pipeline Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 1. Data Loading & EDA\n")
            f.write(f"Training set: {X_train.shape[0]} samples\n")
            f.write(f"Test set: {X_test.shape[0]} samples\n")
            f.write(f"![Class Distribution](reports/eda/class_distribution.png)\n")
            f.write(f"![Sample Images](reports/eda/sample_images.png)\n\n")
            
            f.write("## 2. Feature Engineering & Explainability\n")
            f.write(f"Engineered features shape: {X_train_processed.shape}\n")
            f.write(f"![PCA Visualization](reports/eda/pca_visualization.png)\n")
            f.write(f"![Feature Importance](reports/explainability/feature_importance.png)\n\n")
            
            f.write("## 3. Model Selection & Hyperparameter Tuning\n")
            f.write(f"Best model: {type(optimized_model).__name__}\n")
            f.write(f"Test accuracy: {accuracy:.4f}\n")
            f.write(f"![Model Comparison](reports/model_performance/model_comparison.png)\n")
            f.write(f"![Confusion Matrix](reports/model_performance/confusion_matrix.png)\n\n")
            
            f.write("## 4. Model Monitoring & Drift Detection\n")
            f.write(f"MLflow run ID: {run_id}\n")
            f.write(f"![Drift Monitoring](reports/model_performance/drift_monitoring.png)\n")
        
        logger.info("MLOps pipeline completed successfully!")
        logger.info("Comprehensive report created: mlops_report.md")

if __name__ == "__main__":
    main()