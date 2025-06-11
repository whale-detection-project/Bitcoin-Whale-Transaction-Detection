import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import xgboost as xgb
import os

# 🔹 1. 테스트 데이터 로드
df = pd.read_csv("dataset/1000btc_test.csv")

# 🔹 2. 피처 정의
features = [
    'input_count', 'output_count', 'max_output_ratio'
    , 'max_input_ratio'
]

# 🔹 3. 전처리: 로그 변환
X_log = df[features].apply(lambda x: np.log1p(x))

# 🔹 4. 전처리 도구 불러오기 (train 기준)
scaler = joblib.load("model/scaler.pkl")
kmeans = joblib.load("model/kmeans.pkl")
pca = joblib.load("model/pca.pkl")

# 🔹 5. 정규화 및 클러스터 라벨 부여 (train 기준으로)
X_scaled = scaler.transform(X_log)
df['cluster_label'] = kmeans.predict(X_scaled)

# 🔹 6. PCA 2D 시각화 데이터 생성
X_pca = pca.transform(X_scaled)
df['pca1'], df['pca2'] = X_pca[:, 0], X_pca[:, 1]

# 🔹 7. 모델 로드 (XGBoost)
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("model/xgb_model.json")

# ✅ 전체 데이터로 예측
X = X_scaled
y_true = df['cluster_label']
y_pred = xgb_model.predict(X)

# 🔹 8. 결과 저장 폴더 생성
os.makedirs("test", exist_ok=True)
with open("test/eval_result.txt", "w", encoding="utf-8") as f:
    f.write("📊 [Confusion Matrix]\n")
    f.write(str(confusion_matrix(y_true, y_pred)) + "\n\n")

    f.write("📋 [Classification Report]\n")
    f.write(classification_report(y_true, y_pred) + "\n")

    acc = accuracy_score(y_true, y_pred)
    f.write(f"🎯 [Accuracy Score]\n{acc * 100:.2f}%\n\n")

    f.write("📊 [Test Prediction Distribution]\n")
    unique, counts = np.unique(y_pred, return_counts=True)
    total = sum(counts)
    for label, count in zip(unique, counts):
        f.write(f"cluster {label}: {count:,}건 ({(count / total) * 100:.2f}%)\n")
    
    f.write("\n🔍 [클러스터 중심값 복원 (원래 단위)]\n")
    cluster_centers_scaled = kmeans.cluster_centers_
    cluster_centers_log = scaler.inverse_transform(cluster_centers_scaled)
    cluster_centers_orig = np.expm1(cluster_centers_log)  # log1p → 원복
    df_centers = pd.DataFrame(cluster_centers_orig, columns=features)
    df_centers.index.name = "cluster"
    f.write(df_centers.to_string(float_format='{:,.7f}'.format) + "\n")

    f.write("\n📌 클러스터 해석\n")
    f.write("Cluster 0 : 소수 입력 → 중간 다수 출력, 지갑 리밸런싱 추정\n")
    f.write("Cluster 1 : 단일 입력 → 단일 출력, 콜드월렛 or 고정 전송\n")
    f.write("Cluster 2 : 다수 입력 → 소수 출력, 입력 병합 / Mixing 준비\n")
    f.write("Cluster 3 : 소수 입력 → 다수 출력, 세탁 의심 or 거래소 출금\n")

# 🔹 9. 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['pca1'], y=df['pca2'], hue=df['cluster_label'], palette='tab10', s=10)
plt.title("🖼 PCA Visualization with Cluster Labels")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster", loc='upper right')
plt.tight_layout()
plt.savefig("test/pca_visualization.png", dpi=300)
plt.close()
