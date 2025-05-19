import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.model import LSTMAutoencoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# -----------------------------
# 1. 설정
# -----------------------------
SEQ_LEN = 60
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. 데이터 불러오기 및 전처리
# -----------------------------
df = pd.read_csv("data/BTCUSDT_2024.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
features = df[['open', 'high', 'low', 'close', 'volume']].values

class SlidingDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        return x, x

# ✅ 차분
diffed = np.diff(features, axis=0)

# ✅ 저장된 스케일러 불러오기
min_vals, max_vals = np.load("models/minmax_scaler.npy", allow_pickle=True)
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = 0, 1 / (max_vals - min_vals + 1e-8)
scaler.data_min_ = min_vals
scaler.data_max_ = max_vals

# ✅ 스케일링
features_scaled = scaler.transform(diffed)

# ✅ 데이터셋 구성
dataset = SlidingDataset(features_scaled, SEQ_LEN)
split = int(len(dataset) * 0.8)
train_dataset = torch.utils.data.Subset(dataset, range(split))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -----------------------------
# 3. 모델 로드
# -----------------------------
model = LSTMAutoencoder(input_dim=features_scaled.shape[1],
                        seq_len=SEQ_LEN,
                        hidden1=128,
                        hidden2=64,
                        latent=32).to(DEVICE)
model.load_state_dict(torch.load("models/best_lstm_autoencoder.pt", map_location=DEVICE))
model.eval()

# -----------------------------
# 4. 에러 계산 및 저장
# -----------------------------
recon_errors = []

with torch.no_grad():
    for x, _ in train_loader:
        x = x.to(DEVICE)
        recon = model(x)
        mae = torch.mean(torch.abs(recon - x), dim=(1, 2))  # sample별 MAE
        recon_errors.extend(mae.cpu().numpy())

recon_errors = np.array(recon_errors)
np.save("models/train_recon_errors.npy", recon_errors)
print("✅ train_recon_errors.npy 저장 완료")
