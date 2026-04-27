import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.models.transformer import ECGTransformer
from src.data.loader import load_dataset
from src.utils.label_encoder import encode_labels, filter_data
from src.preprocessing import prepare_for_transformer





BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001





def prepare_data():
    X, y = load_dataset(segment_length=180)

    X, y = filter_data(X, y)
    y = encode_labels(y)

    X = prepare_for_transformer(X)

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
    model = ECGTransformer(num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Training Transformer...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/transformer.pth")

    print("Transformer model saved!")



if __name__ == "__main__":
    train()
