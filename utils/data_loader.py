import os
import numpy as np
import pandas as pd
import struct
import gzip
import streamlit as st
import kagglehub
from sklearn.model_selection import train_test_split

def load_fashion_mnist_from_kaggle():
    """
    Load the Fashion MNIST dataset from Kaggle using kagglehub
    
    Returns:
        tuple: ((X_train, y_train), (X_test, y_test))
    """
    try:
        # Download the dataset files from Kaggle
        print("Downloading Fashion MNIST dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("zalando-research/fashionmnist")
        
        # Path to training and test files
        train_images_path = os.path.join(dataset_path, "train-images-idx3-ubyte.gz")
        train_labels_path = os.path.join(dataset_path, "train-labels-idx1-ubyte.gz")
        test_images_path = os.path.join(dataset_path, "t10k-images-idx3-ubyte.gz")
        test_labels_path = os.path.join(dataset_path, "t10k-labels-idx1-ubyte.gz")
        
        # Load training images
        with gzip.open(train_images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            X_train = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        
        # Load training labels
        with gzip.open(train_labels_path, 'rb') as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            y_train = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Load test images
        with gzip.open(test_images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            X_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        
        # Load test labels
        with gzip.open(test_labels_path, 'rb') as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            y_test = np.frombuffer(f.read(), dtype=np.uint8)
        
        return (X_train, y_train), (X_test, y_test)
    
    except Exception as e:
        if st._is_running_with_streamlit:
            st.error(f"Error loading Fashion MNIST dataset from Kaggle: {str(e)}")
            st.error("Please ensure you have proper kaggle credentials or check your internet connection.")
        else:
            print(f"Error loading Fashion MNIST dataset from Kaggle: {str(e)}")
        raise

def load_fashion_mnist():
    """
    Load the Fashion MNIST dataset
    
    Returns:
        tuple: ((X_train, y_train), (X_test, y_test))
    """
    if st._is_running_with_streamlit:
        with st.spinner("Fetching Fashion MNIST dataset (this may take a moment)..."):
            return load_fashion_mnist_from_kaggle()
    else:
        return load_fashion_mnist_from_kaggle()

def preprocess_data(X_train, X_test):
    """
    Preprocess the Fashion MNIST dataset
    
    Args:
        X_train (np.ndarray): Training data
        X_test (np.ndarray): Test data
        
    Returns:
        tuple: (X_train_processed, X_test_processed)
    """
    # Normalize pixel values to [0, 1]
    X_train_scaled = X_train.astype('float32') / 255.0
    X_test_scaled = X_test.astype('float32') / 255.0
    
    # Reshape to (n_samples, n_features)
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
    
    return X_train_reshaped, X_test_reshaped

def get_class_names():
    """
    Get the class names for Fashion MNIST
    
    Returns:
        list: Class names
    """
    return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
