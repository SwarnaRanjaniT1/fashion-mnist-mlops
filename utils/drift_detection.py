import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import wasserstein_distance
import time

def detect_drift(model, X_test, y_test, drift_intensity=0.3):
    """
    Detect data drift in the test dataset and evaluate its impact on model performance
    
    Args:
        model: Trained model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        drift_intensity (float): Intensity of simulated drift (0 to 1)
        
    Returns:
        dict: Drift detection results
    """
    # First, get baseline performance on original test data
    y_pred_original = model.predict(X_test)
    
    original_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_original),
        'precision': precision_score(y_test, y_pred_original, average='weighted'),
        'recall': recall_score(y_test, y_pred_original, average='weighted'),
        'f1': f1_score(y_test, y_pred_original, average='weighted')
    }
    
    # Simulate drift in the test data
    X_test_drifted = simulate_drift(X_test, intensity=drift_intensity)
    
    # Get performance on drifted data
    y_pred_drifted = model.predict(X_test_drifted)
    
    drift_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_drifted),
        'precision': precision_score(y_test, y_pred_drifted, average='weighted'),
        'recall': recall_score(y_test, y_pred_drifted, average='weighted'),
        'f1': f1_score(y_test, y_pred_drifted, average='weighted')
    }
    
    # Calculate feature-level drift using Wasserstein distance
    feature_drift_scores = {}
    for i in range(X_test.shape[1]):
        feature_drift_scores[i] = wasserstein_distance(X_test[:, i], X_test_drifted[:, i])
    
    # Calculate overall drift score as the average of the accuracy drop and feature-level drift
    accuracy_drop = original_metrics['accuracy'] - drift_metrics['accuracy']
    avg_feature_drift = np.mean(list(feature_drift_scores.values()))
    
    # Normalize the drift score to a [0, 1] range where higher means more drift
    drift_score = 0.5 * (accuracy_drop / original_metrics['accuracy']) + 0.5 * avg_feature_drift
    
    # Determine if drift is significant enough to require action
    # Typical threshold values could range from 0.05 to 0.2 depending on the application
    drift_threshold = 0.1
    drift_detected = drift_score > drift_threshold
    
    # Return comprehensive drift detection results
    drift_results = {
        'original_metrics': original_metrics,
        'drift_metrics': drift_metrics,
        'accuracy_drop': accuracy_drop,
        'feature_drift_scores': feature_drift_scores,
        'avg_feature_drift': avg_feature_drift,
        'drift_score': drift_score,
        'drift_threshold': drift_threshold,
        'drift_detected': drift_detected
    }
    
    return drift_results

def simulate_drift(X, intensity=0.3, random_seed=None):
    """
    Simulate data drift by adding noise and applying transformations
    
    Args:
        X (np.ndarray): Original data
        intensity (float): Intensity of drift (0 to 1)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        np.ndarray: Drifted data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create a copy of the data
    X_drifted = X.copy()
    
    # Add Gaussian noise
    noise_scale = intensity * X.std()
    X_drifted += np.random.normal(0, noise_scale, X.shape)
    
    # Apply scaling drift to a portion of features
    n_features = X.shape[1]
    n_features_to_drift = int(intensity * n_features)
    
    if n_features_to_drift > 0:
        features_to_drift = np.random.choice(n_features, n_features_to_drift, replace=False)
        for feature_idx in features_to_drift:
            # Scale the feature (either up or down)
            scaling_factor = np.random.uniform(1 - intensity, 1 + intensity)
            X_drifted[:, feature_idx] *= scaling_factor
    
    # Apply shifting drift to a different portion of features
    n_features_to_shift = int(intensity * n_features)
    
    if n_features_to_shift > 0:
        features_to_shift = np.random.choice(n_features, n_features_to_shift, replace=False)
        for feature_idx in features_to_shift:
            # Shift the feature (either up or down)
            shift_amount = intensity * (X[:, feature_idx].max() - X[:, feature_idx].min())
            shift_direction = np.random.choice([-1, 1])
            X_drifted[:, feature_idx] += shift_direction * shift_amount
    
    # Optionally apply correlation drift by mixing some features
    if intensity > 0.5 and n_features > 5:
        n_pairs = int(intensity * min(5, n_features // 2))
        for _ in range(n_pairs):
            i, j = np.random.choice(n_features, 2, replace=False)
            mix_ratio = intensity * np.random.uniform(0.1, 0.3)
            X_drifted[:, i] = (1 - mix_ratio) * X_drifted[:, i] + mix_ratio * X_drifted[:, j]
            X_drifted[:, j] = (1 - mix_ratio) * X_drifted[:, j] + mix_ratio * X_drifted[:, i]
    
    return X_drifted

def monitor_drift_over_time(model, X_test, y_test, n_batches=5, drift_progression=None):
    """
    Monitor data drift and model performance over time with progressively increasing drift
    
    Args:
        model: Trained model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        n_batches (int): Number of time points to simulate
        drift_progression (list): Optional list of drift intensities for each batch
        
    Returns:
        dict: Drift monitoring results over time
    """
    # Set default drift progression if not provided
    if drift_progression is None:
        drift_progression = np.linspace(0, 0.5, n_batches)
    else:
        n_batches = len(drift_progression)
    
    # Prepare results containers
    timestamps = [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + i*86400)) for i in range(n_batches)]
    accuracy_over_time = []
    precision_over_time = []
    recall_over_time = []
    f1_over_time = []
    drift_scores_over_time = []
    
    # Monitor drift over simulated time
    for batch_idx, drift_intensity in enumerate(drift_progression):
        # Simulate drift for the current batch
        X_batch_drifted = simulate_drift(X_test, intensity=drift_intensity, random_seed=batch_idx)
        
        # Make predictions
        y_pred = model.predict(X_batch_drifted)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate drift score
        feature_drift_scores = {}
        for i in range(X_test.shape[1]):
            feature_drift_scores[i] = wasserstein_distance(X_test[:, i], X_batch_drifted[:, i])
        
        avg_feature_drift = np.mean(list(feature_drift_scores.values()))
        drift_score = avg_feature_drift  # Simplified drift score
        
        # Store results
        accuracy_over_time.append(accuracy)
        precision_over_time.append(precision)
        recall_over_time.append(recall)
        f1_over_time.append(f1)
        drift_scores_over_time.append(drift_score)
    
    # Return monitoring results
    monitoring_results = {
        'timestamps': timestamps,
        'drift_intensity': drift_progression,
        'accuracy': accuracy_over_time,
        'precision': precision_over_time,
        'recall': recall_over_time,
        'f1': f1_over_time,
        'drift_scores': drift_scores_over_time
    }
    
    return monitoring_results
