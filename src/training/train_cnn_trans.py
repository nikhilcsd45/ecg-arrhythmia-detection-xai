import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.data.loader import load_dataset
from src.utils.label_encoder import encode_labels, filter_data
from src.preprocessing.preprocess import prepare_for_cnn_transformer
from src.models.cnn_trans import CNNTransformer

BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001


def prepare_data():
    X, y = load_dataset(segment_length=180)

    X, y = filter_data(X, y)
    y = encode_labels(y)

    X = prepare_for_cnn_transformer(X)

    return X, y


def train():
    print("Loading segmented data...")
    X, y = prepare_data()

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNTransformer(num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Training CNN-Transformer...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_trans.pth")
    print("CNN-Transformer model saved!")


if __name__ == "__main__":
    train()
