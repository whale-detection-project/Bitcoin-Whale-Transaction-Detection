"""
Test script ·  MLP + 거리 컷오프(τ)  ·  scikit-learn 1.6.x
──────────────────────────────────────────────────────────
• scaler · KMeans · MLP · τ(컷오프) 로드
• 테스트셋 전처리 → Unknown(4) 판정 → 예측 · 평가
• 전체 결과 CSV + Unknown 전용 CSV **둘 다** 저장
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

# ───────────────────────────── 1. 경로 설정
MODEL_DIR = Path("model")
DATA_PATH = Path("dataset/1000btc_test.csv")
OUT_DIR   = Path("test"); OUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "input_count", "output_count",
    "max_output_ratio", "max_input_ratio"
]

# ───────────────────────────── 2. 아티팩트 로드
scaler        = joblib.load(MODEL_DIR / "scaler4.pkl")
kmeans        = joblib.load(MODEL_DIR / "kmeans4.pkl")
mlp           = joblib.load(MODEL_DIR / "mlp_model4.pkl")
centers, tau  = joblib.load(MODEL_DIR / "km_tau.pkl")     # (centers, τ)

# ───────────────────────────── 3. 데이터 로드 & 전처리
df        = pd.read_csv(DATA_PATH)
X_log     = df[FEATURES].apply(np.log1p)
X_scaled  = scaler.transform(X_log)

# ───────────────────────────── 4. 컷오프 + MLP 예측
d_min   = cdist(X_scaled, centers, metric="euclidean").min(axis=1)
mlp_pred  = mlp.predict(X_scaled)
y_pred    = np.where(d_min > tau, 4, mlp_pred)            # 4 = Unknown

# ───────────────────────────── 5. 참값·Unknown 통계
y_true       = kmeans.predict(X_scaled)                   # 0–3
unknown_mask = (y_pred == 4)

print(f"🟡 Unknown(4) 샘플 수 : {unknown_mask.sum()} / {len(df)}"
      f"  ({unknown_mask.mean()*100:.2f} %)")

# ───────── 평가: Unknown 제외 (inlier만)
idx_in = ~unknown_mask
print("\n📊  Balanced Accuracy (inlier):",
      balanced_accuracy_score(y_true[idx_in], y_pred[idx_in]))

print("\n📊  Confusion Matrix (inlier)\n",
      confusion_matrix(y_true[idx_in], y_pred[idx_in]))

print("\n📊  Classification Report (inlier)\n",
      classification_report(y_true[idx_in], y_pred[idx_in], digits=4))

# ───────────────────────────── 6-A. 전체 결과 CSV
Path(OUT_DIR, "mlp_tau_predictions.csv").write_text(
    pd.DataFrame({
        "idx"          : df.index,
        "true_cluster" : y_true,
        "pred_cluster" : y_pred,
        "d_min"        : d_min
    }).to_csv(index=False)
)
print(f"\n✅ 전체 CSV 저장 → {OUT_DIR/'mlp_tau_predictions.csv'}")

# ───────────────────────────── 6-B. Unknown 전용 CSV
unknown_df = df.loc[unknown_mask].assign(
    predicted_cluster=y_pred[unknown_mask],
    d_min=d_min[unknown_mask]
)
Path(OUT_DIR, "mlp_tau_unknowns.csv").write_text(
    unknown_df.to_csv(index=False)
)
print(f"✅ Unknown CSV 저장 → {OUT_DIR/'mlp_tau_unknowns.csv'}")
