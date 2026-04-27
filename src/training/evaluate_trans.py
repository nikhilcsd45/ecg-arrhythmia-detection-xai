from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.transformer import ECGTransformer
from src.data.loader import load_dataset
from src.utils.label_encoder import encode_labels, filter_data
from src.preprocessing import prepare_for_transformer


def prepare_data():
    X, y = load_dataset(segment_length=180)

    X, y = filter_data(X, y)
    y = encode_labels(y)

    X = prepare_for_transformer(X)

    return X, y


def evaluate():
    X, y = prepare_data()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    dataset = TensorDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECGTransformer(num_classes=5).to(device)
    model.load_state_dict(torch.load("models/transformer.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)

            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Precision is ill-defined*")
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        report = classification_report(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

    print("\n📊 Transformer Results:")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}\n")

    print("📊 Classification Report:")
    print(report)

    print("📊 Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    evaluate()
