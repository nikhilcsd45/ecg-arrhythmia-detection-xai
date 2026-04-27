import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.data.loader import load_dataset
from src.utils.label_encoder import encode_labels, filter_data
from src.preprocessing.preprocess import prepare_for_cnn
from src.models.cnn import CNNModel


# =========================================================
# 📁 Setup
# =========================================================
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ["N", "S", "V", "F", "Q"]


# =========================================================
# 📊 Data Preparation
# =========================================================
def prepare_data():
    print("\n🔄 Preparing data...")

    X, y = load_dataset(segment_length=180)
    X, y = filter_data(X, y)
    y = encode_labels(y)

    X = prepare_for_cnn(X)

    print(f" Data ready | Samples: {len(X)}")

    return X, y


# =========================================================
#  Visualization Functions
# =========================================================
def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=LABELS,
        yticklabels=LABELS,
        cmap="Blues"
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_f1_scores(y_test, preds, title, save_path):
    f1_per_class = f1_score(y_test, preds, average=None)

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(f1_per_class)), f1_per_class)

    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.xticks(range(len(f1_per_class)), LABELS)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
#  Model Evaluation
# =========================================================
def evaluate_model(model_path, X_test, y_test):
    model_name = os.path.basename(model_path).replace(".pth", "")

    print("\n" + "=" * 50)
    print(f" Evaluating Model: {model_name}")
    print("=" * 50)

    # Load model
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Prediction
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).numpy()

    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    # ================= PRINT =================
    print("\n PERFORMANCE")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Score     : {f1:.4f}")

    print("\n CLASSIFICATION REPORT")
    print(report)

    print("\n CONFUSION MATRIX")
    print(cm)

    # ================= SAVE =================
    save_results(model_name, acc, f1, report, cm, y_test, preds)

    print(f"\n Results saved for {model_name}")


# =========================================================
# 💾 Save Results
# =========================================================
def save_results(model_name, acc, f1, report, cm, y_test, preds):

    # Save text report
    report_path = os.path.join(OUTPUT_DIR, f"{model_name}_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"MODEL: {model_name}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Accuracy     : {acc:.6f}\n")
        f.write(f"F1 Score     : {f1:.6f}\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write(report + "\n")

        f.write("CONFUSION MATRIX\n")
        f.write(np.array2string(cm))

    # Save plots
    plot_confusion_matrix(
        cm,
        f"Confusion Matrix - {model_name}",
        os.path.join(OUTPUT_DIR, f"{model_name}_cm.png")
    )

    plot_f1_scores(
        y_test,
        preds,
        f"F1 Score per Class - {model_name}",
        os.path.join(OUTPUT_DIR, f"{model_name}_f1.png")
    )


# =========================================================
# 🏁 Main Function
# =========================================================
def main():
    print("\n Starting Evaluation Pipeline...")

    X, y = prepare_data()

    # Train-Test Split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_test = torch.tensor(X_test).float()

    # Evaluate Models
    evaluate_model("models/cnn_baseline.pth", X_test, y_test)
    evaluate_model("models/cnn_weighted.pth", X_test, y_test)

    print("\n All evaluations completed!")


if __name__ == "__main__":
    main()
