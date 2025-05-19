import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.model import LSTMAutoencoder
import logging
from logs.log_config import setup_logger
setup_logger()

# 기본 설정
SEQ_LEN = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 모델 로드
model = LSTMAutoencoder(input_dim=5, seq_len=SEQ_LEN, hidden1=128, hidden2=64, latent=32).to(DEVICE)
model.load_state_dict(torch.load("models/best_lstm_autoencoder.pt", map_location=DEVICE))
model.eval()

logging.info("✅ LSTM Autoencoder 모델 로드 완료")

# 스케일러 로드
min_vals, max_vals = np.load("models/minmax_scaler.npy", allow_pickle=True)
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = 0, 1 / (max_vals - min_vals + 1e-8)
scaler.data_min_ = min_vals
scaler.data_max_ = max_vals

logging.info("✅ MinMaxScaler 파라미터 로드 완료")

# threshold 계산
train_errors = np.load("models/train_recon_errors.npy")
q1, q3 = np.percentile(train_errors, [25, 75])
iqr = q3 - q1
lower_thres = q1 - 1.5 * iqr
upper_thres = q3 + 1.5 * iqr

logging.info(f"✅ IQR 임계치 설정 완료 | lower: {lower_thres:.6f}, upper: {upper_thres:.6f}")

# 실시간 윈도우
window = []

# 이상 탐지 함수
def detect_anomaly(new_candle):
    global window
    window.append(new_candle)

    if len(window) < SEQ_LEN + 1:
        logging.info(f"📉 캔들 수신됨 (윈도우 길이: {len(window)}) - 아직 이상 탐지 안 함")
        return  # 아직 윈도우 준비 안됨

    # 차분 후 정규화
    window_np = np.array(window)
    diffed = np.diff(window_np, axis=0)
    scaled = scaler.transform(diffed[-SEQ_LEN:])

    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        recon = model(x)
    error = torch.mean(torch.abs(recon - x)).item()

    # 이상 탐지 로그
    if error < lower_thres or error > upper_thres:
        logging.warning(f"🚨 이상치 감지됨! MAE = {error:.6f}")
    else:
        logging.info(f"정상 데이터 | MAE = {error:.6f}")

    # 슬라이딩 윈도우 유지
    window.pop(0)
