import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, hidden1, hidden2, latent, dropout_rate=0.1):
        super().__init__()
        self.seq_len = seq_len

        self.encoder1 = nn.LSTM(input_dim, hidden1, batch_first=True)
        self.encoder2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.encoder3 = nn.LSTM(hidden2, latent, batch_first=True)

        self.decoder1 = nn.LSTM(latent, hidden2, batch_first=True)
        self.decoder2 = nn.LSTM(hidden2, hidden1, batch_first=True)
        self.decoder3 = nn.LSTM(hidden1, input_dim, batch_first=True)

        self.output_head = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out, _ = self.encoder1(x)
        out = self.dropout(out)
        out, _ = self.encoder2(out)
        out = self.dropout(out)
        out, _ = self.encoder3(out)
        out = self.dropout(out)

        out, _ = self.decoder1(out)
        out = self.dropout(out)
        out, _ = self.decoder2(out)
        out = self.dropout(out)
        out, _ = self.decoder3(out)

        return self.output_head(out)
