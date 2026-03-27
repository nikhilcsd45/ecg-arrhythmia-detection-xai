import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.data.loader import load_dataset
from src.utils.label_encoder import encode_labels, filter_data
from src.preprocessing.preprocess import preprocess_batch, reshape_for_cnn
from src.models.cnn import CNNModel


# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001


def compute_class_weights(y):
    class_counts = np.bincount(y)
    total = len(y)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float)


def prepare_data():
    X, y = load_dataset()

    X, y = filter_data(X, y)
    y = encode_labels(y)

    X = preprocess_batch(X)
    X = reshape_for_cnn(X)

    return X, y


def train_model(use_weights=False, save_path="models/model.pth"):
    print("\n==============================")
    print("Training:", "WITH weights" if use_weights else "WITHOUT weights")
    print("==============================")

    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).long()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CNNModel()

    if use_weights:
        class_weights = compute_class_weights(y_train.numpy())
        print("Class Weights:", class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"Model saved at {save_path}")


if __name__ == "__main__":
    # Train baseline model
    train_model(use_weights=False, save_path="models/cnn_baseline.pth")

    # Train weighted model
    train_model(use_weights=True, save_path="models/cnn_weighted.pth")