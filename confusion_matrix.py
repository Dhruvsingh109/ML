import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def generate_confusion_matrix(y_true, y_pred, labels):
    """
    Generate an n*n confusion matrix.

    Parameters:
    y_true (list or array): Ground truth labels.
    y_pred (list or array): Predicted labels.
    labels (list): Unique class labels.

    Returns:
    DataFrame: Confusion matrix as a pandas DataFrame.
    """
    # Generate confusion matrix
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # Convert to DataFrame for better visualization
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    return matrix_df

def visualize_confusion_matrix(matrix_df):
    """
    Visualize the confusion matrix using a heatmap.

    Parameters:
    matrix_df (DataFrame): Confusion matrix as a pandas DataFrame.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Confusion Matrix Heatmap", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("Actual Labels", fontsize=14)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Simulate a large dataset with random predictions
    np.random.seed(42)
    n_classes = 5  # You can change the number of classes
    data_size = 100000  # Number of samples

    # Generate random ground truth labels and predictions
    labels = [f"Class_{i}" for i in range(n_classes)]
    y_true = np.random.choice(labels, size=data_size)
    y_pred = np.random.choice(labels, size=data_size)

    # Generate confusion matrix
    confusion_df = generate_confusion_matrix(y_true, y_pred, labels)
    print("Confusion Matrix:")
    print(confusion_df)

    # Visualize confusion matrix
    visualize_confusion_matrix(confusion_df)
