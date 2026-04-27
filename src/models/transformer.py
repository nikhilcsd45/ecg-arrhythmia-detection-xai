import torch.nn as nn


class ECGTransformer(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Step 1: Project input (1 → 64)
        self.embedding = nn.Linear(1, 64)

        # Step 2: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Step 3: Classification
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 1)

        x = self.embedding(x)          # (batch, seq_len, 64)

        x = self.transformer(x)        # attention

        x = x.mean(dim=1)              # global pooling

        x = self.fc(x)

        return x
