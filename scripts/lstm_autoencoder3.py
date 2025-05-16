import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ✅ 단일 LSTM 기반 Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=128, latent_dim=32, num_layers=2, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len

        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=latent_dim,
                               num_layers=num_layers, dropout=dropout,
                               batch_first=True, bidirectional=False)

        self.decoder = nn.LSTM(input_size=latent_dim, hidden_size=input_dim,
                               num_layers=num_layers, dropout=dropout,
                               batch_first=True, bidirectional=False)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        latent = h_n[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder(latent)
        return out

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ 사용 디바이스:", device)

    # ✅ 데이터 로딩
    df = pd.read_csv("data/BTCUSDT_2024.csv")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

    # ✅ 차분 및 정규화
    features = df[['open', 'high', 'low', 'close', 'volume']].values
    diff = np.diff(features, axis=0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(diff)

    # ✅ 시퀀스 생성
    SEQ_LEN = 24
    def create_sequences(data, seq_len):
        return np.array([data[i:i+seq_len] for i in range(len(data) - seq_len)])

    X = create_sequences(scaled, SEQ_LEN)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # ✅ 데이터 분할
    split = int(len(X_tensor) * 0.8)
    X_train, X_test = X_tensor[:split], X_tensor[split:]
    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=128, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, X_test), batch_size=128, shuffle=False)

    # ✅ 모델 및 학습 설정
    model = LSTMAutoencoder(input_dim=5, seq_len=SEQ_LEN).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, _ in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]"):
            batch_X = batch_X.to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(batch_X)
                loss = criterion(output, batch_X)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}")

    # ✅ reconstruction error 계산
    def calc_recon_errors(loader):
        model.eval()
        errors = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(device)
                output = model(batch_X)
                error = torch.mean(torch.abs(output - batch_X), dim=(1, 2)).cpu().numpy()
                errors.extend(error)
        return np.array(errors)

    train_errors = calc_recon_errors(train_loader)
    test_errors = calc_recon_errors(test_loader)

    # ✅ 이상치 기준 및 저장
    threshold = np.percentile(train_errors, 95)
    anomalies = test_errors > threshold
    anomaly_indices = np.where(anomalies)[0]

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_autoencoder_diff.pt")
    np.save("models/test_recon_errors_diff.npy", test_errors)
    np.save("models/anomaly_indices_diff.npy", anomaly_indices)
    print(f"✅ 이상치 {len(anomaly_indices)}건 탐지됨, 모델 저장 완료")

if __name__ == "__main__":
    main()
