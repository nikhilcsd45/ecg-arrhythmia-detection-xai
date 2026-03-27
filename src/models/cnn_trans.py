import torch
import torch.nn as nn

class CNNTransformer(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNTransformer, self).__init__()

        # 🔹 CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 🔹 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # 🔹 Classifier
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 1)

        x = x.permute(0, 2, 1)     # (batch, 1, seq_len)

        x = self.cnn(x)            # (batch, 64, new_seq)

        x = x.permute(0, 2, 1)     # (batch, new_seq, 64)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.fc(x)

        return x