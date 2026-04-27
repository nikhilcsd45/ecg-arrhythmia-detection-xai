import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

os.makedirs("results", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath("results/.matplotlib"))

import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from src.models.cnn_trans import CNNTransformer
from src.data.loader import load_dataset
from src.utils.label_encoder import encode_labels, filter_data
from src.preprocessing.preprocess import prepare_for_cnn_transformer

OUTPUT_DIR = "results"
MODEL_PATH = "models/cnn_trans.pth"
BATCH_SIZE = 32
LABELS = ["N", "S", "V", "F", "Q"]


def prepare_data():
    print("Loading segmented data...")
    X, y = load_dataset(segment_length=180)

    X, y = filter_data(X, y)
    y = encode_labels(y)

    X = prepare_for_cnn_transformer(X)

    print(f"Data ready | Samples: {len(X)}")
    return X, y


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        title="CNN-Transformer Confusion Matrix",
        xlabel="Predicted",
        ylabel="Actual",
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
    )

    threshold = cm.max() / 2
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            ax.text(col, row, cm[row, col], ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_f1_scores(y_test, preds, save_path):
    f1_per_class = f1_score(
        y_test,
        preds,
        average=None,
        labels=range(len(LABELS)),
        zero_division=0,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(f1_per_class)), f1_per_class)
    ax.set(
        title="CNN-Transformer F1 Score per Class",
        xlabel="Class",
        ylabel="F1 Score",
        xticks=range(len(f1_per_class)),
        xticklabels=LABELS,
        ylim=(0, 1),
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_report(acc, f1, report, cm):
    report_path = os.path.join(OUTPUT_DIR, "cnn_trans_report.txt")

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("CNN-Transformer Results\n")
        report_file.write("=" * 50 + "\n\n")
        report_file.write(f"Accuracy: {acc:.6f}\n")
        report_file.write(f"F1 Score: {f1:.6f}\n\n")
        report_file.write("Classification Report\n")
        report_file.write(report + "\n")
        report_file.write("Confusion Matrix\n")
        report_file.write(str(cm))

    return report_path

def evaluate():
    X, y = prepare_data()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNTransformer(num_classes=5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(
        TensorDataset(X_test, y_test_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Evaluating {len(X_test)} samples on {device}...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    report = classification_report(
        all_labels,
        all_preds,
        labels=range(len(LABELS)),
        target_names=LABELS,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(LABELS)))

    print("\n CNN-Transformer Results:")
    print("Accuracy:", acc)
    print("F1 Score:", f1)

    print("\n Classification Report:")
    print(report)

    print("\n Confusion Matrix:")
    print(cm)

    report_path = save_report(acc, f1, report, cm)
    cm_path = os.path.join(OUTPUT_DIR, "cnn_trans_cm.png")
    f1_path = os.path.join(OUTPUT_DIR, "cnn_trans_f1.png")
    plot_confusion_matrix(cm, cm_path)
    plot_f1_scores(all_labels, all_preds, f1_path)

    print(f"\nSaved report: {report_path}")
    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved F1 plot: {f1_path}")

if __name__ == "__main__":
    evaluate()
