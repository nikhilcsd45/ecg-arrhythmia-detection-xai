import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.preprocess import preprocess_batch, reshape_for_cnn_transformer

# 🔹 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from src.models.cnn_trans import CNNTransformer
# 🔹 Model
model = CNNTransformer(num_classes=5).to(device)

# 🔹 Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔹 Example dataset (REPLACE with your real loader)
X_train = np.random.randn(100, 187)   # dummy ECG signals
y_train = np.random.randint(0, 5, 100)

# 🔹 Preprocessing
X_train = preprocess_batch(X_train)
X_train = reshape_for_cnn_transformer(X_train)

# 🔹 Convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# 🔹 Training loop
epochs = 10
batch_size = 16

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i in range(0, len(X_train), batch_size):
        signals = X_train[i:i+batch_size].to(device)
        labels = y_train[i:i+batch_size].to(device)

        outputs = model(signals)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 🔹 Save model
torch.save(model.state_dict(), "models/cnn_trans.pth")
print("Model saved successfully!")