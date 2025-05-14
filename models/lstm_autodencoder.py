import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 중인 하드웨어:", device)

    df = pd.read_csv("data/BTC_5min_1year_data.csv")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

    features = df[['open', 'high', 'low', 'close', 'volume']].values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    SEQ_LEN = 60
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i:i+seq_len])
        return np.array(X), np.array(y)

    X, y = create_sequences(features_scaled, SEQ_LEN)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    split = int(len(X_tensor) * 0.8)
    X_train, X_test = X_tensor[:split], X_tensor[split:]
    y_train, y_test = y_tensor[:split], y_tensor[split:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    class LSTMAutoencoder(nn.Module):
        def __init__(self, input_dim, seq_len, hidden1, hidden2, latent, dropout_rate=0.2):
            super().__init__()
            self.seq_len = seq_len
            self.encoder1 = nn.LSTM(input_dim, hidden1, batch_first=True)
            self.encoder2 = nn.LSTM(hidden1, hidden2, batch_first=True)
            self.encoder3 = nn.LSTM(hidden2, latent, batch_first=True)
            self.dropout = nn.Dropout(dropout_rate)
            self.decoder1 = nn.LSTM(latent, hidden2, batch_first=True)
            self.decoder2 = nn.LSTM(hidden2, hidden1, batch_first=True)
            self.decoder3 = nn.LSTM(hidden1, input_dim, batch_first=True)

        def forward(self, x):
            _, (h1, _) = self.encoder1(x)
            h1 = self.dropout(h1.permute(1,0,2).repeat(1, self.seq_len, 1))
            _, (h2, _) = self.encoder2(h1)
            h2 = self.dropout(h2.permute(1,0,2).repeat(1, self.seq_len, 1))
            _, (h3, _) = self.encoder3(h2)
            latent = self.dropout(h3.permute(1,0,2).repeat(1, self.seq_len, 1))
            out1, _ = self.decoder1(latent)
            out2, _ = self.decoder2(self.dropout(out1))
            out3, _ = self.decoder3(self.dropout(out2))
            return out3

    model = LSTMAutoencoder(input_dim=X.shape[2], seq_len=SEQ_LEN, hidden1=128, hidden2=64, latent=32).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    EPOCHS = 20
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(batch_X)
                loss = criterion(output, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                val_loss += criterion(output, batch_y).item()
        val_losses.append(val_loss / len(test_loader))

    print(f"최종 검증 손실: {val_losses[-1]:.6f}")

    def calc_recon_errors(data_loader):
        model.eval()
        errors = []
        with torch.no_grad():
            for batch_X, _ in data_loader:
                batch_X = batch_X.to(device)
                reconstructed = model(batch_X)
                err = torch.mean(torch.abs(reconstructed - batch_X), dim=(1,2)).cpu().numpy()
                errors.extend(err)
        return np.array(errors)

    train_recon_errors = calc_recon_errors(train_loader)
    test_recon_errors = calc_recon_errors(test_loader)

    threshold = np.percentile(train_recon_errors, 95)
    print(f"Threshold (95 percentile): {threshold:.6f}")
    print(f"Train MAE: min={train_recon_errors.min():.6f}, max={train_recon_errors.max():.6f}")
    print(f"Test MAE: min={test_recon_errors.min():.6f}, max={test_recon_errors.max():.6f}")

    anomalies = test_recon_errors > threshold
    print(f"이상치 개수: {np.sum(anomalies)}")

    print("히스토그램 저장 중...")
    plt.figure(figsize=(12, 6))
    bins = 100

    train_hist, bin_edges = np.histogram(train_recon_errors, bins=bins)
    test_hist, _ = np.histogram(test_recon_errors, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = (bin_edges[1] - bin_edges[0]) * 0.9

    for i in tqdm(range(len(bin_centers)), desc="히스토그램 그리기"):
        plt.bar(bin_centers[i], train_hist[i], width=bar_width, color="blue", alpha=0.5)
        plt.bar(bin_centers[i], test_hist[i], width=bar_width, color="orange", alpha=0.5)

    plt.axvline(threshold, color='red', linestyle='--', label="Threshold")
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("MAE")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/modles/result/lstm_autoencoder.png")
    print("그래프 저장 완료: recon_error_hist.png")

if __name__ == "__main__":
    main()
