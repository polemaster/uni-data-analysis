import os
import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def load_and_process_train_data():
    """
    Downloads data, reshapes images, and performs One-Hot Encoding.
    Returns:
        x_train, y_train_bin: Processed training data
        x_val, y_val_bin: Processed validation data
        lb: The fitted LabelBinarizer (needed for the test set)
        num_classes: Integer count of classes
    """
    print("Downloading/Loading Sign Language MNIST dataset...")
    # This downloads to the local cache directory
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")
    
    train_path = os.path.join(path, "sign_mnist_train/sign_mnist_train.csv")
    val_path = os.path.join(path, "sign_mnist_test/sign_mnist_test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Split labels and features
    y_train = train_df['label']
    x_train = train_df.drop('label', axis=1)
    
    y_val = val_df['label']
    x_val = val_df.drop('label', axis=1)

    # Reshape to (Batch, 28, 28, 1)
    x_train = x_train.values.reshape(-1, 28, 28, 1)
    x_val = x_val.values.reshape(-1, 28, 28, 1)

    # One-Hot Encoding
    lb = LabelBinarizer()
    y_train_bin = lb.fit_transform(y_train)
    y_val_bin = lb.transform(y_val)
    
    num_classes = y_train_bin.shape[1]
    
    print(f"Training Data Loaded: {x_train.shape}")
    print(f"Validation Data Loaded: {x_val.shape}")
    print(f"Number of classes: {num_classes}")

    return x_train, y_train_bin, x_val, y_val_bin, lb, num_classes