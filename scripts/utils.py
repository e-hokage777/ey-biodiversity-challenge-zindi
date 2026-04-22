from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import os


def split_dataset(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def clean_training_data(X_train, y_train):
    duplicated_indices = X_train.duplicated()
    Xt = X_train[~duplicated_indices]
    yt = y_train[~duplicated_indices]
    return Xt, yt


def get_x_and_y(path: str):
    df = pd.read_csv(path)
    y = df["Occurrence Status"]
    X = df.drop(["latitude", "longitude", "Occurrence Status"], axis=1)
    return X, y


def validate_classification(model, X, y, target_names=None):
    """
    Print the classification report and display the confusion matrix for the given model and data.

    Parameters
    ----------
    model : sklearn estimator
        The trained model or pipeline to evaluate.
    X : array-like
        Feature matrix.
    y : array-like
        True labels.
    target_names : list of str, optional
        Names of the classes for the confusion matrix.
    """
    y_pred = model.predict(X)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=target_names))

    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


def get_test_data(path: str):
    df = pd.read_csv(path)
    df = df.drop(["latitude", "longitude"], axis=1)
    return df


def make_submission(path: str, model, save_path: str):
    ids = pd.read_csv("../data/Test.csv")["ID"]
    X = get_test_data(path)
    predictions = model.predict(X)
    sub = pd.DataFrame({"ID": ids, "Occurrence Status": predictions})
    sub.to_csv(save_path, index=False)
