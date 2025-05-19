import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model import LSTMAutoencoder
import logging

# 설정
SEQ_LEN = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = LSTMAutoencoder(input_dim=5, seq_len=SEQ_LEN, hidden1=128, hidden2=64, latent=32).to(DEVICE)
model.load_state_dict(torch.load("best_lstm_autoencoder.pt", map_location=DEVICE))
model.eval()

# 스케일러 로드
min_vals, max_vals = np.load("minmax_scaler.npy", allow_pickle=True)
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = 0, 1 / (max_vals - min_vals + 1e-8)
scaler.data_min_ = min_vals
scaler.data_max_ = max_vals

# threshold 로드
train_errors = np.load("train_recon_errors.npy")
q1, q3 = np.percentile(train_errors, [25, 75])
iqr = q3 - q1
lower_thres = q1 - 1.5 * iqr
upper_thres = q3 + 1.5 * iqr

# window 저장소
window = []

# 이상치 탐지 함수
def detect_anomaly(new_candle):
    global window
    window.append(new_candle)

    if len(window) < SEQ_LEN + 1:
        return  # 59개 + 1개 안 되면 return

    # 차분
    window_np = np.array(window)
    diffed = np.diff(window_np, axis=0)

    # 스케일링
    scaled = scaler.transform(diffed[-SEQ_LEN:])

    # 모델 예측
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        recon = model(x)
    error = torch.mean(torch.abs(recon - x)).item()

    # 이상치 판단
    if error < lower_thres or error > upper_thres:
        logging.warning(f"🚨 Anomaly Detected! MAE={error:.6f}")
    else:
        logging.info(f"✅ Normal | MAE={error:.6f}")

    # sliding window 유지 (가장 오래된 값 제거)
    window.pop(0)
