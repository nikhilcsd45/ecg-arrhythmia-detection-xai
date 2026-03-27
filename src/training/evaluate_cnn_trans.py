import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from src.models.cnn_trans import CNNTransformer
from src.preprocessing.preprocess import preprocess_batch, reshape_for_cnn_transformer

def plot_confusion_matrix(cm):
    labels = ["N", "S", "V", "F", "Q"]
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title("CNN-Transformer Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_f1_scores(y_test, preds):
    f1_per_class = f1_score(y_test, preds, average=None)
    plt.figure()
    plt.bar(range(len(f1_per_class)), f1_per_class)
    plt.title("CNN-Transformer F1 Score per Class")
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.show()

def evaluate():
    # 🔹 Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 Load model
    model = CNNTransformer(num_classes=5).to(device)
    model.load_state_dict(torch.load("models/cnn_trans.pth"))
    model.eval()

    # 🔹 Example test data (REPLACE with real data)
    X_test = np.random.randn(20, 187)
    y_test = np.random.randint(0, 5, 20)

    # 🔹 Preprocessing
    X_test = preprocess_batch(X_test)
    X_test = reshape_for_cnn_transformer(X_test)

    # 🔹 Convert to tensor
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    # 🔹 Evaluation
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')

    print("\n📊 CNN-Transformer Results:")
    print("Accuracy:", acc)
    print("F1 Score:", f1)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)

    print("\n📊 Confusion Matrix:")
    print(cm)

    # 🔥 Visualizations
    plot_confusion_matrix(cm)
    plot_f1_scores(y_test, preds)

if __name__ == "__main__":
    evaluate()