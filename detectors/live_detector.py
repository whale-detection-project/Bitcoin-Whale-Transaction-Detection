import os
import logging
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from models.model import LSTMAutoencoder
from logs.log_config import setup_logger
from logs.shared_log import log_buffer

# ————————— 설정 및 초기화 —————————
setup_logger()
logging.info("🔄 애플리케이션 시작 중...")

SEQ_LEN       = 60
VOL_Z_THRESH  = 4.0       # 볼륨 로그-z 스코어 임계치
PRICE_RET_TH  = 0.005     # 종가 변화율 임계치 (0.5%)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"🖥️  디바이스: {DEVICE}")

# ————————— 모델 로드 —————————
model = LSTMAutoencoder(input_dim=5,
                        seq_len=SEQ_LEN,
                        hidden1=128,
                        hidden2=64,
                        latent=32).to(DEVICE)

model_path = "models/best_lstm_autoencoder.pt"
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
logging.info(f"✅ LSTM Autoencoder 모델 로드 완료 ({model_path})")

# ————————— 스케일러 로드 —————————
min_vals, max_vals = np.load("models/minmax_scaler.npy", allow_pickle=True)
scaler = MinMaxScaler()
# MinMaxScaler 내부 파라미터 직접 설정
scaler.min_, scaler.scale_     = 0, 1 / (max_vals - min_vals + 1e-8)
scaler.data_min_, scaler.data_max_ = min_vals, max_vals
logging.info("✅ MinMaxScaler 파라미터 로드 완료")

# ————————— IQR 임계치 계산 —————————
train_errors = np.load("models/train_recon_errors.npy")
q1, q3       = np.percentile(train_errors, [25, 75])
iqr          = q3 - q1
lower_thres  = q1 - 2.0 * iqr
upper_thres  = q3 + 2.0 * iqr
logging.info(f"✅ IQR 임계치 설정 완료 | lower: {lower_thres:.6f}, upper: {upper_thres:.6f}")

# ————————— 실시간 윈도우 버퍼 —————————
window = []

# ————————— 이상 탐지 함수 —————————
def detect_anomaly(new_candle: np.ndarray):
    """
    new_candle: [open, high, low, close, volume] 순의 1D numpy array
    """
    global window
    window.append(new_candle)

    # 윈도우가 차기 전까지는 탐지하지 않음
    if len(window) < SEQ_LEN + 1:
        logging.info(f"📉 캔들 수신 (윈도우 길이: {len(window)}) — 대기 중")
        return

    # 1) 모델 기반 재구성 오차 계산
    arr       = np.array(window)
    diffed    = np.diff(arr, axis=0)               # 차분
    seq       = diffed[-SEQ_LEN:]                  # 마지막 SEQ_LEN개
    scaled    = scaler.transform(seq)              # 정규화
    x_tensor  = torch.tensor(scaled, dtype=torch.float32)\
                    .unsqueeze(0).to(DEVICE)      # (1, SEQ_LEN, 5)
    with torch.no_grad():
        recon    = model(x_tensor)
    mae_error = torch.mean(torch.abs(recon - x_tensor)).item()
    iqr_anomaly = (mae_error < lower_thres) or (mae_error > upper_thres)

    # 2) 볼륨 스파이크 + 종가 변화율 계산
    vols      = arr[-SEQ_LEN:, 4]                  # 최근 SEQ_LEN 거래량
    log_vols  = np.log1p(vols)
    mu, sigma = log_vols.mean(), log_vols.std(ddof=0)
    last_log  = np.log1p(arr[-1, 4])
    z_score   = (last_log - mu) / (sigma + 1e-8)

    prev_close = arr[-2, 3]
    last_close = arr[-1, 3]
    price_ret  = (last_close - prev_close) / (prev_close + 1e-8)

    vol_spike           = abs(z_score) > VOL_Z_THRESH
    price_spike         = abs(price_ret) > PRICE_RET_TH
    vol_price_anomaly   = vol_spike and price_spike

    # 3) 최종 이상치 판정: IQR 모델 OR (볼륨+종가 스파이크)
    final_anomaly = iqr_anomaly or vol_price_anomaly

    # 4) 로그 버퍼에 저장 (FastAPI → 프론트엔드)
    log_msg = {
        "mae":               mae_error,
        "iqr_anomaly":       bool(iqr_anomaly),
        "vol_z_score":       float(z_score),
        "price_ret":         float(price_ret),
        "vol_price_anomaly": bool(vol_price_anomaly),
        "anomaly":           final_anomaly
    }
    log_buffer.append(log_msg)

    # 5) 콘솔 출력
    if final_anomaly:
        logging.warning(
            f"🚨 이상치 감지! MAE={mae_error:.6f} | "
            f"vol_z={z_score:.2f} | ret={price_ret*100:.2f}%"
        )
    else:
        logging.info(f"✅ 정상 캔들 | MAE={mae_error:.6f}")

    # 6) 윈도우 슬라이드
    window.pop(0)

