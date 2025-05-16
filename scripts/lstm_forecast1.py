import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 사용 디바이스:", device)

# ✅ 데이터 로딩 (차분 없이 원본)
df = pd.read_csv("data/BTCUSDT_2025.csv")
features = df[['open', 'high', 'low', 'close', 'volume']].values

# ✅ Min-Max 정규화 (차분 없이)
min_vals = features.min(axis=0)
max_vals = features.max(axis=0)
scaled = (features - min_vals) / (max_vals - min_vals + 1e-8)

# ✅ 정규화 기준 저장 (API 예측 등에서 사용)
np.save("min_vals_nodiff.npy", min_vals)
np.save("max_vals_nodiff.npy", max_vals)

# ✅ 시퀀스 생성
WINDOW_SIZE = 24  # 2시간
def create_sequence_and_target(data, window):
    x, y = [], []
    for i in range(len(data) - window):
        x.append(data[i:i+window])
        y.append(data[i+window][3])  # 다음 close (절대값 예측)
    return np.array(x), np.array(y).reshape(-1, 1)

X_np, y_np = create_sequence_and_target(scaled, WINDOW_SIZE)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

# ✅ 데이터셋 분할
dataset = TensorDataset(X, y)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

# ✅ 모델 정의 (LSTM 2층 + Dense Head)
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        return self.fc2(h)

# ✅ 모델 생성 및 학습 설정
model = LSTMForecaster().to(device)
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 학습 루프
EPOCHS = 50
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for batch_x, batch_y in tqdm(train_loader, desc=f"[Epoch {epoch}/{EPOCHS}]"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_output = model(val_x)
            val_loss += criterion(val_output, val_y).item() * val_x.size(0)

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss/len(train_ds):.6f} | Val Loss: {val_loss/len(val_ds):.6f}")

# ✅ 모델 저장
torch.save(model.state_dict(), "btc_5min_ws24_nodiff.pt")
print("✅ 모델 저장 완료: btc_5min_ws24_nodiff.pt")
