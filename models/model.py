import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, hidden1, hidden2, latent, dropout_rate=0.1):
        super().__init__()
        self.seq_len = seq_len

        self.encoder = nn.Sequential(
            nn.LSTM(input_dim, hidden1, batch_first=True),
            nn.LSTM(hidden1, hidden2, batch_first=True),
            nn.LSTM(hidden2, latent, batch_first=True)
        )

        self.decoder = nn.Sequential(
            nn.LSTM(latent, hidden2, batch_first=True),
            nn.LSTM(hidden2, hidden1, batch_first=True),
            nn.LSTM(hidden1, input_dim, batch_first=True)
        )

        self.output_head = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out, _ = self.encoder[0](x)
        out = self.dropout(out)
        out, _ = self.encoder[1](out)
        out = self.dropout(out)
        out, _ = self.encoder[2](out)
        out = self.dropout(out)

        out, _ = self.decoder[0](out)
        out = self.dropout(out)
        out, _ = self.decoder[1](out)
        out = self.dropout(out)
        out, _ = self.decoder[2](out)

        return self.output_head(out)
