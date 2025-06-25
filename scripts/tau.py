import joblib, numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist

MODEL_DIR  = Path("model")
DATA_PATH  = Path("dataset/1000btc_train.csv")    # 학습 때 썼던 파일
FEATURES   = ["input_count","output_count",
              "max_output_ratio","max_input_ratio"]

# 1) 아티팩트 로드
scaler  = joblib.load(MODEL_DIR / "scaler4.pkl")
kmeans  = joblib.load(MODEL_DIR / "kmeans4.pkl")      # 4-클러스터 KMeans
centers = kmeans.cluster_centers_

# 2) 학습 데이터 전처리 (로그 + 스케일)
df        = pd.read_csv(DATA_PATH)
X_log     = df[FEATURES].apply(np.log1p)
X_scaled  = scaler.transform(X_log)

# 3) 각 샘플의 최소 중심-거리 계산
d_min = cdist(X_scaled, centers, metric="euclidean").min(axis=1)

# 4) 컷오프 τ  (예: 99.5-percentile → 이론상 최대 오류 ≈ 0.5 %)
tau = np.percentile(d_min, 99.5)

# 5) 저장  (tuple 형식: (centers, tau))
out_path = MODEL_DIR / "km_tau.pkl"
joblib.dump((centers, tau), out_path)
print(f"✅ km_tau.pkl 저장 완료 → {out_path.resolve()}")
