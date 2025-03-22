"""
Download the Fashion MNIST dataset from Kaggle using kagglehub.
This script can be run independently to download the dataset before running the full pipeline.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the data loader
from utils.data_loader import load_fashion_mnist, get_class_names

def main():
    print("Downloading Fashion MNIST dataset using kagglehub...")
    
    try:
        # Load the dataset
        (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
        
        # Print dataset information
        print(f"\nDataset successfully downloaded!")
        print(f"Training set: {X_train.shape[0]} images, each {X_train.shape[1]}x{X_train.shape[2]}")
        print(f"Test set: {X_test.shape[0]} images, each {X_test.shape[1]}x{X_test.shape[2]}")
        
        # Get class names
        class_names = get_class_names()
        print(f"\nClasses: {class_names}")
        
        # Display sample images (optional)
        display_sample_images = True
        if display_sample_images:
            # Create output directory for sample images
            os.makedirs("reports/samples", exist_ok=True)
            
            # Display and save a few sample images
            plt.figure(figsize=(10, 10))
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(X_train[i], cmap='gray')
                plt.title(class_names[y_train[i]])
                plt.axis('off')
            plt.tight_layout()
            plt.savefig("reports/samples/fashion_mnist_samples.png")
            print("\nSample images saved to reports/samples/fashion_mnist_samples.png")
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())