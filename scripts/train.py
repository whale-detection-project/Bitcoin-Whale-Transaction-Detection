import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 🔹 1. 학습 데이터 로드
train_df = pd.read_csv("dataset/1000btc_train.csv")
features = ['input_count', 'output_count', 'max_output_ratio', 'fee_per_max_ratio', 'max_input_ratio']

# 🔹 2. 로그 변환 + 정규화
X_train_log = train_df[features].apply(lambda x: np.log1p(x))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_log)

# 🔹 3. 클러스터 라벨 생성
kmeans = KMeans(n_clusters=4, random_state=42)
train_df['cluster_label'] = kmeans.fit_predict(X_train_scaled)

# 🔹 4. PCA 학습 (시각화 및 실시간 탐지 용도)
pca = PCA(n_components=2)
pca.fit(X_train_scaled)

# 🔹 5. 학습 준비
X = X_train_scaled
y = train_df['cluster_label']

# 🔹 6. 클래스별 수동 가중치
manual_class_weights = {0: 4.0, 1: 0.5, 2: 4.0, 3: 3.0}
sample_weights = np.array([manual_class_weights[label] for label in y])

# 🔹 7. 모델 학습
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    tree_method='hist'
)
xgb_model.fit(X, y, sample_weight=sample_weights)

# 🔹 8. 모델 및 전처리기 저장
joblib.dump(xgb_model, "model/xgb_model.pkl")
xgb_model.save_model("model/xgb_model.json")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(kmeans, "model/kmeans.pkl")
joblib.dump(pca, "model/pca.pkl")

print("✅ 모델, 스케일러, 클러스터링, PCA 저장 완료")
