import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 디바이스:", device)

# ✅ 모델 정의
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

# ✅ 데이터 로딩
df = pd.read_csv("data/BTCUSDT_2025.csv")
features = df[['open', 'high', 'low', 'close', 'volume']].values
timestamps = pd.to_datetime(df['timestamp'])

# ✅ 정규화 기준 (2024년 기준)
min_vals = np.load("models/min_vals_nodiff.npy")
max_vals = np.load("models/max_vals_nodiff.npy")
scaled = (features - min_vals) / (max_vals - min_vals + 1e-8)

# ✅ 시퀀스 생성
WINDOW_SIZE = 24
def create_sequences(data, window):
    x, y = [], []
    for i in range(len(data) - window):
        x.append(data[i:i+window])
        y.append(data[i+window][3])  # close
    return np.array(x), np.array(y)

X_np, y_np = create_sequences(scaled, WINDOW_SIZE)
X = torch.tensor(X_np, dtype=torch.float32).to(device)

# ✅ 모델 로드 및 예측
model = LSTMForecaster().to(device)
model.load_state_dict(torch.load("models/forecast1.pt"))
model.eval()
with torch.no_grad():
    y_pred = model(X).cpu().numpy().flatten()

# ✅ 역정규화
y_true_denorm = y_np * (max_vals[3] - min_vals[3]) + min_vals[3]
y_pred_denorm = y_pred * (max_vals[3] - min_vals[3]) + min_vals[3]

# ✅ 예측 오차 계산
error = np.abs(y_pred_denorm - y_true_denorm)

# ✅ 1. 3σ 기준
mean_e = np.mean(error)
std_e = np.std(error)
thresh_std = mean_e + 3 * std_e
anomalies_std = np.where(error > thresh_std)[0]

# ✅ 2. IQR 기준
Q1 = np.percentile(error, 25)
Q3 = np.percentile(error, 75)
IQR = Q3 - Q1
thresh_iqr = Q3 + 1.5 * IQR
anomalies_iqr = np.where(error > thresh_iqr)[0]

# ✅ 3. 절대 오차 기준 ($1000 이상)
thresh_fixed = 1000
anomalies_fixed = np.where(error > thresh_fixed)[0]

# ✅ 결과 출력
print(f"🔎 3σ 이상치: {len(anomalies_std)}개 | threshold = {thresh_std:.2f}")
print(f"🔎 IQR 이상치: {len(anomalies_iqr)}개 | threshold = {thresh_iqr:.2f}")
print(f"🔎 고정 이상치: {len(anomalies_fixed)}개 | threshold = {thresh_fixed}")

# ✅ 시각화 함수 (공통)
def plot_anomaly_graph(title, anomalies, color):
    plt.figure(figsize=(16, 6))
    plt.plot(timestamps[WINDOW_SIZE:], y_true_denorm, label="Actual", alpha=0.6)
    plt.plot(timestamps[WINDOW_SIZE:], y_pred_denorm, label="Predicted", alpha=0.8)
    plt.scatter(timestamps[WINDOW_SIZE:][anomalies], y_true_denorm[anomalies],
                color=color, s=20, label="Anomaly", alpha=0.8)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ✅ 기준별 개별 그래프 출력
plot_anomaly_graph("Anomaly Detection (3σ Threshold)", anomalies_std, 'red')
plot_anomaly_graph("Anomaly Detection (IQR Threshold)", anomalies_iqr, 'blue')
plot_anomaly_graph("Anomaly Detection ($1000+ Error)", anomalies_fixed, 'green')
