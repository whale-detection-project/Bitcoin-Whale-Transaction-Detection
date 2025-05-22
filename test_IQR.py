import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
from models.model import LSTMAutoencoder
from logs.log_config import setup_logger
from logs.shared_log import log_buffer

# ————————— 설정 및 초기화 —————————
setup_logger()
logging.info("🔄 백테스트 스크립트 시작")

# 파라미터
CSV_PATH      = "BTCUSDT_2025.csv"
SEQ_LEN       = 60
VOL_Z_THRESH  = 4.0       # 로그-z 임계치
PRICE_RET_TH  = 0.005     # 종가 변화율 임계치
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ————————— 모델 & 스케일러 로드 —————————
# 1) Autoencoder
model = LSTMAutoencoder(input_dim=5, seq_len=SEQ_LEN, hidden1=128, hidden2=64, latent=32)
model.load_state_dict(torch.load("models/best_lstm_autoencoder.pt", map_location=DEVICE))
model.to(DEVICE).eval()
logging.info("✅ 모델 로드 완료")

# 2) MinMaxScaler 파라미터
min_vals, max_vals = np.load("models/minmax_scaler.npy", allow_pickle=True)
scaler = MinMaxScaler()
scaler.min_, scaler.scale_       = 0, 1/(max_vals - min_vals + 1e-8)
scaler.data_min_, scaler.data_max_ = min_vals, max_vals
logging.info("✅ Scaler 파라미터 로드 완료")



# 3) IQR 임계치
train_errors = np.load("models/train_recon_errors.npy")
q1, q3       = np.percentile(train_errors, [25, 75])
iqr          = q3 - q1
lower_thres  = q1 - 1.5 * iqr
upper_thres  = q3 + 1.5 * iqr
logging.info(f"✅ IQR 임계치: lower={lower_thres:.6f}, upper={upper_thres:.6f}")

# ————————— CSV 로드 —————————
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join("data", CSV_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
logging.info(f"✅ 데이터 로드 완료: {len(df)}개 행")

# ————————— 이상치 탐지 백테스트 —————————
results = []
window = []

for ts, row in df.iterrows():
    candle = row[['open','high','low','close','volume']].values
    window.append(candle)

    if len(window) < SEQ_LEN + 1:
        results.append({
            'timestamp': ts,
            'close': row['close'],
            'mae': np.nan,
            'vol_z': np.nan,
            'price_ret': np.nan,
            'iqr_anomaly': False,
            'vol_price_anomaly': False,
            'anomaly': False
        })
        continue

    arr = np.array(window[-(SEQ_LEN+1):])
    # — 모델 재구성 오차
    diffed    = np.diff(arr, axis=0)[-SEQ_LEN:]
    scaled    = scaler.transform(diffed)
    x_tensor  = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        recon = model(x_tensor)
    mae = torch.mean(torch.abs(recon - x_tensor)).item()
    iqr_anom = (mae < lower_thres) or (mae > upper_thres)

    # — 볼륨+가격 스파이크
    vols      = arr[-SEQ_LEN:, 4]
    log_vols  = np.log1p(vols)
    mu, sigma = log_vols.mean(), log_vols.std(ddof=0)
    last_log  = np.log1p(arr[-1, 4])
    z_score   = (last_log - mu) / (sigma + 1e-8)

    prev_c    = arr[-2, 3]
    last_c    = arr[-1, 3]
    price_ret = (last_c - prev_c) / (prev_c + 1e-8)

    vol_spike    = abs(z_score) > VOL_Z_THRESH
    price_spike  = abs(price_ret) > PRICE_RET_TH
    vol_price_anom = vol_spike and price_spike

    final_anom = iqr_anom or vol_price_anom

    results.append({
        'timestamp': ts,
        'close': last_c,
        'mae': mae,
        'vol_z': z_score,
        'price_ret': price_ret,
        'iqr_anomaly': iqr_anom,
        'vol_price_anomaly': vol_price_anom,
        'anomaly': final_anom
    })

    window.pop(0)

# ——————— DataFrame & 시각화 ———————
res_df = pd.DataFrame(results).set_index('timestamp')

# IQR 기반 이상치 개수 로깅
iqr_count = int(res_df['iqr_anomaly'].sum())
logging.info(f"🔍 IQR 이상치 개수: {iqr_count}")

# 볼륨+가격 스파이크 개수 로깅
vol_count = int(res_df['vol_price_anomaly'].sum())
logging.info(f"🔍 볼륨+가격 이상치 개수: {vol_count}")

# 1) IQR 기반 이상치만
plt.figure(figsize=(12,6))
plt.plot(res_df.index, res_df['close'], alpha=0.6, label='Close')
mask_iqr = res_df['iqr_anomaly']
plt.scatter(
    res_df.index[mask_iqr],
    res_df['close'][mask_iqr],
    color='blue', s=10, marker='o',    # 점 크기 10으로 조정
    label=f'IQR Anomaly ({iqr_count})'
)
plt.title("BTC 2025 — IQR 기반 이상치")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()

# 2) 볼륨스파이크 + 가격변동 이상치만
plt.figure(figsize=(12,6))
plt.plot(res_df.index, res_df['close'], alpha=0.6, label='Close')
mask_vol = res_df['vol_price_anomaly']
plt.scatter(
    res_df.index[mask_vol],
    res_df['close'][mask_vol],
    color='orange', s=10, marker='x',  # 점 크기 10으로 조정
    label=f'Volume+Price Spike ({vol_count})'
)
plt.title("BTC 2025 — Volume Spike + Price Move 이상치")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()


# ——————— 결과 요약 ———————
total = len(res_df)
anom  = res_df['anomaly'].sum()
logging.info(f"🔍 전체 캔들: {total}, 이상치: {anom} ({anom/total*100:.2f}%)")
