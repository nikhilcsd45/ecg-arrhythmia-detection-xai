import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models.transformer import ECGTransformer
from src.preprocessing import preprocess_batch, reshape_for_transformer




# ========================
# 🔹 LOAD YOUR DATA HERE
# ========================

# Example (replace with your dataset)
X = np.random.randn(1000, 187)
y = np.random.randint(0, 5, 1000)

# ========================
# 🔹 PREPROCESSING
# ========================

X = preprocess_batch(X)
X = reshape_for_transformer(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# ========================
# 🔹 MODEL
# ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ECGTransformer(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ========================
# 🔹 TRAINING LOOP
# ========================

epochs = 10

for epoch in range(epochs):
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

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")




# ========================
# 🔹 SAVE MODEL
# ========================




torch.save(model.state_dict(), "models/transformer.pth")

print("✅ Transformer trained and saved!")