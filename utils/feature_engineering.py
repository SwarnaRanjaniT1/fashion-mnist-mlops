import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.ndimage import zoom
import streamlit as st

def apply_feature_engineering(X_train, X_test):
    """
    Apply feature engineering to the raw Fashion MNIST dataset
    
    Args:
        X_train (np.ndarray): Training images
        X_test (np.ndarray): Test images
        
    Returns:
        tuple: (X_train_processed, X_test_processed, feature_pipeline)
    """
    # Convert images to 2D arrays if they're not already
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    
    # Create a feature engineering pipeline
    feature_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # Note: PCA is commented out as it might remove important information
        # Uncomment if dimensionality reduction is needed
        # ('pca', PCA(n_components=0.95))
    ])
    
    # Fit and transform the training data
    X_train_processed = feature_pipeline.fit_transform(X_train_2d)
    
    # Transform the test data
    X_test_processed = feature_pipeline.transform(X_test_2d)
    
    return X_train_processed, X_test_processed, feature_pipeline

def extract_advanced_features(X, feature_type='hog'):
    """
    Extract advanced features from Fashion MNIST images
    
    Args:
        X (np.ndarray): Input images (n_samples, 28, 28)
        feature_type (str): Type of features to extract ('hog', 'lbp', or 'both')
        
    Returns:
        np.ndarray: Extracted features
    """
    # Ensure X is in the right shape
    if X.ndim == 2:
        # Single image, reshape to (1, 28, 28)
        X = X.reshape(1, 28, 28)
    elif X.ndim == 3 and X.shape[1] == 28 and X.shape[2] == 28:
        # Already in the right shape (n_samples, 28, 28)
        pass
    elif X.ndim == 3 and X.shape[0] == 28 and X.shape[1] == 28:
        # Single image in wrong orientation, reshape to (1, 28, 28)
        X = X.reshape(1, 28, 28)
    else:
        # Reshape to (n_samples, 28, 28)
        X = X.reshape(-1, 28, 28)
    
    n_samples = X.shape[0]
    
    if feature_type == 'hog' or feature_type == 'both':
        from skimage.feature import hog
        
        # Initialize array to store HOG features
        hog_features = np.zeros((n_samples, 324))  # 324 is the typical size for (28,28) with these parameters
        
        # Extract HOG features for each image
        for i in range(n_samples):
            hog_features[i] = hog(
                X[i],
                orientations=9,
                pixels_per_cell=(4, 4),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=False
            )
        
        if feature_type == 'hog':
            return hog_features
    
    if feature_type == 'lbp' or feature_type == 'both':
        from skimage.feature import local_binary_pattern
        
        # Initialize array to store LBP features
        lbp_features = np.zeros((n_samples, 26 * 26))
        
        # Parameters for LBP
        radius = 1
        n_points = 8 * radius
        
        # Extract LBP features for each image
        for i in range(n_samples):
            lbp = local_binary_pattern(X[i], n_points, radius, method='uniform')
            # The LBP will be 26x26 for a 28x28 image with radius=1
            lbp_features[i] = lbp.reshape(-1)
        
        if feature_type == 'lbp':
            return lbp_features
    
    # If feature_type is 'both', concatenate HOG and LBP features
    if feature_type == 'both':
        return np.hstack((hog_features, lbp_features))
    
    # Default return if feature_type is not recognized
    return X.reshape(n_samples, -1)
