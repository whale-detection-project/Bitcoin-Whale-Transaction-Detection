"""
Oversampling + MLPClassifier  (scikit-learn 1.6.x)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler   # NEW
import joblib
from pathlib import Path

# ──────────────────────────────── 1. 데이터 로드
df = pd.read_csv("dataset/1000btc_train.csv")
FEATURES = ["input_count", "output_count", "max_output_ratio", "max_input_ratio"]

# ──────────────────────────────── 2. 로그 변환 + 스케일링
X_log = df[FEATURES].apply(np.log1p)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# ──────────────────────────────── 3. 임시 라벨 (KMeans)
kmeans        = KMeans(n_clusters=4, random_state=42, n_init="auto")
df["label"]   = kmeans.fit_predict(X_scaled)
y             = df["label"].to_numpy()

# ──────────────────────────────── 4. Oversampling으로 클래스 균형화
ros    = RandomOverSampler(random_state=42)
X_bal, y_bal = ros.fit_resample(X_scaled, y)

# ──────────────────────────────── 5. 학습/검증 분리
X_tr, X_val, y_tr, y_val = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# ──────────────────────────────── 6. MLP 정의 & 학습 (sample_weight X)
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),   # 4개 층
    activation="relu",
    solver="adam",
    batch_size=256,
    learning_rate_init=1e-3,
    alpha=1e-4,          # L2 정규화
    max_iter=400,        
    early_stopping=True,
    n_iter_no_change=25, # 조기 종료 지연 폭
    random_state=42,
    verbose=True
)

mlp.fit(X_tr, y_tr)

# ──────────────────────────────── 7. 검증 성능
print("\n📊  Validation metrics")
print(classification_report(y_val, mlp.predict(X_val), digits=4))

# ──────────────────────────────── 8. PCA (시각화 용도)
pca = PCA(n_components=2, random_state=42).fit(X_scaled)

# ──────────────────────────────── 9. 모델·전처리기 저장
MODEL_DIR = Path("model"); MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(mlp,    MODEL_DIR / "mlp_model.pkl4")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl4")
joblib.dump(kmeans, MODEL_DIR / "kmeans.pkl4")
joblib.dump(pca,    MODEL_DIR / "pca.pkl4")

print("✅ 모델·스케일러·KMeans·PCA 저장 완료 →", MODEL_DIR.resolve())
