import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # ✅ x is already (batch, 180, 1)

        out, _ = self.lstm(x)

        out = out[:, -1, :]  # last timestep

        out = self.fc(out)

        return out
