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
from src.preprocessing.preprocess import prepare_for_lstm
from src.models.lstm import LSTMModel


BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
NUM_CLASSES = 5


def compute_class_weights(y):
    class_counts = np.bincount(y, minlength=NUM_CLASSES)
    total = len(y)
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    nonzero_classes = class_counts > 0
    weights[nonzero_classes] = total / (NUM_CLASSES * class_counts[nonzero_classes])
    return torch.tensor(weights, dtype=torch.float)


def prepare_data():
    X, y = load_dataset(segment_length=180)

    X, y = filter_data(X, y)
    y = encode_labels(y)

    X = prepare_for_lstm(X)

    return X, y


def train():
    print("Loading data...")
    X, y = prepare_data()

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).long()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel()

    #  Use class weights
    class_weights = compute_class_weights(y_train.numpy())
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training LSTM...")

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
    torch.save(model.state_dict(), "models/lstm.pth")

    print("LSTM Model saved!")


if __name__ == "__main__":
    train()
