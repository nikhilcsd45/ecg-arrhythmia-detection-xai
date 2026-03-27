
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.lstm import LSTMModel

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.data.loader import load_dataset
from src.utils.label_encoder import encode_labels, filter_data
from src.preprocessing.preprocess import preprocess_batch, reshape_for_cnn
from src.models.cnn import CNNModel

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)



def prepare_data():
    X, y = load_dataset()

    X, y = filter_data(X, y)
    y = encode_labels(y)

    X = preprocess_batch(X)
    X = reshape_for_cnn(X)

    return X, y


def plot_confusion_matrix(cm, title):
    labels = ["N", "S", "V", "F", "Q"]

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=labels,
                yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_f1_scores(y_test, preds, title):
    f1_per_class = f1_score(y_test, preds, average=None)

    plt.figure()
    plt.bar(range(len(f1_per_class)), f1_per_class)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.show()


def evaluate_model(model_path, X_test, y_test):
    print("\n==============================")
    print(f"Evaluating: {model_path}")
    print("==============================")

    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).numpy()

    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')

    print("\n📊 Results:")
    print("Accuracy:", acc)
    print("F1 Score:", f1)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)

    print("\n📊 Confusion Matrix:")
    print(cm)

    # 🔥 Plots
    plot_confusion_matrix(cm, f"Confusion Matrix - {model_path}")
    plot_f1_scores(y_test, preds, f"F1 Score per Class - {model_path}")


def main():
    print("Loading data...")
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_test = torch.tensor(X_test).float()

    # 🔥 Compare both models
    evaluate_model("models/cnn_baseline.pth", X_test, y_test)
    evaluate_model("models/cnn_weighted.pth", X_test, y_test)


if __name__ == "__main__":
    main()