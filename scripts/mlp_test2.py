"""
Test script ·  PyTorch MLP + 거리 컷오프(τ)
────────────────────────────────────────────
• 스케일러 · τ · PyTorch 모델 로드
• 테스트셋 전처리 → Unknown(4) 판정 → 예측 · 평가
• 전체 결과 CSV + Unknown 전용 CSV **둘 다** 저장
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

# ───────────────────────────── 1. 경로 설정
MODEL_DIR = Path("model")
DATA_PATH = Path("dataset/1000btc_test.csv")
OUT_DIR   = Path("test"); OUT_DIR.mkdir(exist_ok=True)

FEATURES = ["input_count", "output_count", "max_output_ratio", "max_input_ratio"]

# ───────────────────────────── 2. 모델 정의 & 로드
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, 4)
        )
    def forward(self, x): return self.net(x)

model = MLP()
model.load_state_dict(torch.load(MODEL_DIR / "mlp4_torch.pt", map_location="cpu"))
model.eval()

# ───────────────────────────── 3. 전처리 정보 로드
scaler = joblib.load(MODEL_DIR / "scaler_np.pkl")
centers_tau = joblib.load(MODEL_DIR / "km_tau.pkl")
mean_, std_ = scaler["mean"], scaler["std"]
centers, tau = centers_tau["centers"], centers_tau["tau"]

# ───────────────────────────── 4. 데이터 로드 & 전처리
df = pd.read_csv(DATA_PATH)
X_raw = df[FEATURES].values
X_log = np.log1p(X_raw)
X_scaled = (X_log - mean_) / std_

# ───────────────────────────── 5. 참값 생성 (KMeans 중심 기준)
y_true = np.argmin(cdist(X_scaled, centers), axis=1)

# ───────────────────────────── 6. MLP 예측 + τ 컷오프
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    mlp_pred = model(X_tensor).argmax(1).numpy()

d_min = cdist(X_scaled, centers).min(1)
y_pred = np.where(d_min > tau, 4, mlp_pred)  # Unknown=4

# ───────────────────────────── 7. 평가 지표 (inlier 기준)
unknown_mask = (y_pred == 4)
idx_in = ~unknown_mask

print(f"🟡 Unknown(4) 샘플 수 : {unknown_mask.sum()} / {len(df)}"
      f"  ({unknown_mask.mean()*100:.2f} %)")

print("\n📊  Balanced Accuracy (inlier):",
      balanced_accuracy_score(y_true[idx_in], y_pred[idx_in]))

print("\n📊  Confusion Matrix (inlier)\n",
      confusion_matrix(y_true[idx_in], y_pred[idx_in]))

print("\n📊  Classification Report (inlier)\n",
      classification_report(y_true[idx_in], y_pred[idx_in], digits=4))

# ───────────────────────────── 8-A. 전체 결과 저장
out_all = pd.DataFrame({
    "idx": df.index,
    "true_cluster": y_true,
    "pred_cluster": y_pred,
    "d_min": d_min
})
out_all.to_csv(OUT_DIR / "mlp4_tau_predictions.csv", index=False)
print(f"\n✅ 전체 CSV 저장 → {OUT_DIR/'mlp4_tau_predictions.csv'}")

# ───────────────────────────── 8-B. Unknown 전용 CSV
df_unknown = df.loc[unknown_mask].assign(
    predicted_cluster=y_pred[unknown_mask],
    d_min=d_min[unknown_mask]
)
df_unknown.to_csv(OUT_DIR / "mlp4_tau_unknowns.csv", index=False)
print(f"✅ Unknown CSV 저장 → {OUT_DIR/'mlp4_tau_unknowns.csv'}")
