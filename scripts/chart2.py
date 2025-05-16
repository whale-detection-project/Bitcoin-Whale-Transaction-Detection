import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib.dates as mdates

# ✅ 한글 폰트 설정 (Windows 전용)
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ✅ 오차 비율 계산
error_signed = y_pred_denorm - y_true_denorm
relative_error = error_signed / y_true_denorm
threshold = 0.01  # 3% 이상 오차

# ✅ 이상치 인덱스
anomaly_up = np.where(relative_error < -threshold)[0]   # 예측보다 실제값이 훨씬 큼 → 급등
anomaly_down = np.where(relative_error > threshold)[0]  # 예측보다 실제값이 훨씬 작음 → 급락

print(f"🔺 급등 이상치: {len(anomaly_up)}개")
print(f"🔻 급락 이상치: {len(anomaly_down)}개")

# ✅ 시각화
plt.figure(figsize=(16, 6))
plt.plot(timestamps[WINDOW_SIZE:], y_true_denorm, label="Actual", alpha=0.6)
plt.plot(timestamps[WINDOW_SIZE:], y_pred_denorm, label="Predicted", alpha=0.8)

# 급등 이상치: 빨간색
plt.scatter(timestamps[WINDOW_SIZE:][anomaly_up], y_true_denorm[anomaly_up],
            color='red', s=20, label="Anomaly: Surge (↑)")
# 급락 이상치: 파란색
plt.scatter(timestamps[WINDOW_SIZE:][anomaly_down], y_true_denorm[anomaly_down],
            color='blue', s=20, label="Anomaly: Drop (↓)")

plt.title("BTC Forecast - Anomaly Detection (↑ 급등 / ↓ 급락, >3%)")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
