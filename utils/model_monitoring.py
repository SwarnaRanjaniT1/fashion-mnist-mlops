import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import json
import os

def track_model_performance(model, X_train, y_train, X_test, y_test):
    """
    Track model performance using MLflow
    
    Args:
        model: Trained model
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        
    Returns:
        tuple: (run_id, metrics)
    """
    # Create a unique experiment name based on model type and timestamp
    experiment_name = f"fashion_mnist_{type(model).__name__}_{int(time.time())}"
    
    # Set up MLflow tracking
    mlflow.set_experiment(experiment_name)
    
    # Start an MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Log model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for param_name, param_value in params.items():
                # Convert non-serializable params to strings
                if not isinstance(param_value, (int, float, str, bool, type(None))):
                    params[param_name] = str(param_value)
            
            mlflow.log_params(params)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # Log metrics
        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }
        
        mlflow.log_metrics(metrics)
        
        # Log confusion matrix as a figure
        cm = confusion_matrix(y_test, y_pred_test)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion Matrix')
        plt.colorbar(im, ax=ax)
        
        # Add class labels
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
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
        
        # Save figure to disk temporarily and log it
        temp_fig_path = "temp_confusion_matrix.png"
        plt.savefig(temp_fig_path)
        mlflow.log_artifact(temp_fig_path)
        
        # Clean up temporary file
        if os.path.exists(temp_fig_path):
            os.remove(temp_fig_path)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        # Display tracking info in Streamlit
        st.success(f"Model tracking initiated with MLflow")
        st.write(f"Run ID: {run_id}")
        st.write(f"Experiment Name: {experiment_name}")
        
        # Display performance metrics
        st.subheader("Performance Metrics")
        
        # Create columns for metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Train Accuracy", f"{train_accuracy:.4f}")
        
        with col2:
            st.metric("Test Accuracy", f"{test_accuracy:.4f}")
        
        with col3:
            st.metric("Precision", f"{test_precision:.4f}")
        
        with col4:
            st.metric("Recall", f"{test_recall:.4f}")
        
        with col5:
            st.metric("F1 Score", f"{test_f1:.4f}")
        
        # Display the confusion matrix
        st.pyplot(fig)
        
        # If the model provides feature importances, log them
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            
            # Get top N features
            n_top_features = min(20, len(feature_importances))
            top_indices = np.argsort(feature_importances)[-n_top_features:]
            
            # Plot feature importances
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh([f"Feature {i}" for i in top_indices], 
                   feature_importances[top_indices])
            ax.set_title("Feature Importances")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            
            # Save and log feature importance plot
            temp_feat_path = "temp_feature_importances.png"
            plt.savefig(temp_feat_path)
            mlflow.log_artifact(temp_feat_path)
            
            # Clean up temporary file
            if os.path.exists(temp_feat_path):
                os.remove(temp_feat_path)
            
            # Display feature importances in Streamlit
            st.subheader("Feature Importances")
            st.pyplot(fig)
    
    return run_id, metrics
