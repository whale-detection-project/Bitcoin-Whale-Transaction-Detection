# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

# # ✅ 단일 LSTM Autoencoder
# class LSTMAutoencoder(nn.Module):
#     def __init__(self, input_dim, seq_len, hidden_dim=128, latent_dim=32, num_layers=2, dropout=0.3):
#         super().__init__()
#         self.seq_len = seq_len
#         self.encoder = nn.LSTM(input_size=input_dim, hidden_size=latent_dim,
#                                num_layers=num_layers, dropout=dropout,
#                                batch_first=True, bidirectional=False)
#         self.decoder = nn.LSTM(input_size=latent_dim, hidden_size=input_dim,
#                                num_layers=num_layers, dropout=dropout,
#                                batch_first=True, bidirectional=False)

#     def forward(self, x):
#         _, (h_n, _) = self.encoder(x)
#         latent = h_n[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
#         out, _ = self.decoder(latent)
#         return out

# # ✅ 설정
# SEQ_LEN = 24
# RATE_THRESHOLD = 0.015  # 3% 급변 기준

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # ✅ 데이터 로딩
#     df = pd.read_csv("data/BTCUSDT_2025.csv")
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     df.set_index("timestamp", inplace=True)
#     raw_close = df["close"].values
#     timestamps = df.index

#     # ✅ 차분 및 정규화
#     features = df[['open', 'high', 'low', 'close', 'volume']].values
#     diff = np.diff(features, axis=0)
#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(diff)

#     # ✅ 모델 로딩
#     model = LSTMAutoencoder(5, SEQ_LEN).to(device)
#     model.load_state_dict(torch.load("models/lstm_autoencoder_diff.pt"))
#     model.eval()

#     # ✅ reconstruction error 계산 (슬라이딩 윈도우)
#     buffer = []
#     recon_errors = []
#     with torch.no_grad():
#         for i in range(len(scaled)):
#             buffer.append(scaled[i])
#             if len(buffer) < SEQ_LEN:
#                 recon_errors.append(np.nan)
#                 continue
#             if len(buffer) > SEQ_LEN:
#                 buffer.pop(0)
#             seq = torch.tensor(np.array(buffer), dtype=torch.float32).unsqueeze(0).to(device)
#             output = model(seq)
#             error = torch.mean(torch.abs(output - seq)).item()
#             recon_errors.append(error)

#     # ✅ 시계열 정렬
#     aligned_timestamps = timestamps[1:]
#     aligned_close = raw_close[1:]
#     recon_errors = np.array(recon_errors)

#     # ✅ 변화율 계산 (← 재정렬 및 정합성 보정 포함)
#     price_series = pd.Series(aligned_close)
#     pct_change = price_series.pct_change().shift(-1).fillna(0)  # shift(-1)로 정렬 맞춤
#     rate_anomaly_mask = pct_change.abs() > RATE_THRESHOLD

#     # ✅ IQR threshold 계산 (reconstruction error 기반)
#     series = pd.Series(recon_errors)
#     q1 = series.rolling(144, center=True, min_periods=1).quantile(0.25)
#     q3 = series.rolling(144, center=True, min_periods=1).quantile(0.75)
#     iqr = q3 - q1
#     iqr_threshold = (q3 + 1.5 * iqr).fillna(method='bfill').fillna(method='ffill')
#     iqr_threshold = iqr_threshold.apply(lambda x: x if x > 0.01 else 0.01)
#     recon_anomaly_mask = series > iqr_threshold

#     # ✅ 이상치 병합: 하나라도 이상이면 True
#     final_anomaly_mask = recon_anomaly_mask.fillna(False) | rate_anomaly_mask
#     anomaly_indices = np.where(final_anomaly_mask)[0]

#     print("🔍 변화율 이상치 수:", rate_anomaly_mask.sum())
#     print("🔍 Reconstruction 이상치 수:", recon_anomaly_mask.sum())
#     print("🔍 병합된 이상치 수:", final_anomaly_mask.sum())
#     # ✅ 시각화
#     os.makedirs("results", exist_ok=True)
#     plt.figure(figsize=(16, 6))
#     plt.plot(aligned_timestamps, aligned_close, label="BTC Close Price", linewidth=0.8)
#     plt.plot(aligned_timestamps, iqr_threshold, linestyle='--', color='green', label="IQR Threshold (Error)")
#     plt.scatter(aligned_timestamps[anomaly_indices], aligned_close[anomaly_indices],
#                 color="red", s=20, label=f"Anomalies ({len(anomaly_indices)}건)")

#     plt.title("BTC Price Anomaly Detection (Autoencoder + Change Rate)")
#     plt.xlabel("Time")
#     plt.ylabel("Price (USD)")
#     plt.grid(True)
#     plt.legend()
#     plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("results/btc_price_anomaly_combined.png")
#     plt.show()
#     print(f"✅ 완료: 이상치 {len(anomaly_indices)}건 탐지됨 → results/btc_price_anomaly_combined.png")

# if __name__ == "__main__":
#     main()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ✅ 설정
SEQ_LEN = 24
RATE_THRESHOLD = 0.015  # 1.5% 급변 기준

def main():
    # ✅ 데이터 로딩
    df = pd.read_csv("data/BTCUSDT_2025.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    close_prices = df["close"].values
    timestamps = df.index

    # ✅ 시계열 정렬 맞추기 (diff 적용으로 인해 1칸 밀림 보정)
    aligned_close = close_prices[1:]
    aligned_timestamps = timestamps[1:]

    # ✅ 변화율 계산 및 이상치 마스크
    pct_change = pd.Series(aligned_close).pct_change().shift(-1).fillna(0)
    rate_anomaly_mask = pct_change.abs() > RATE_THRESHOLD
    anomaly_indices = np.where(rate_anomaly_mask)[0]

    print(f"✅ 변화율 이상치 감지됨: {len(anomaly_indices)}건")

    # ✅ 시각화
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(16, 6))
    plt.plot(aligned_timestamps, aligned_close, label="BTC Close Price", linewidth=0.8)
    plt.scatter(aligned_timestamps[anomaly_indices], aligned_close[anomaly_indices],
                color="blue", label=f"Rate Anomaly ({len(anomaly_indices)}건)", s=20)

    plt.title("BTC Price Anomaly Detection (Change Rate Only)")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/btc_price_anomaly_rate_only.png")
    plt.show()
    print("✅ 시각화 완료: results/btc_price_anomaly_rate_only.png")

if __name__ == "__main__":
    main()

