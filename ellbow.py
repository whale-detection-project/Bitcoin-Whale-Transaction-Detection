import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 🔹 1. 데이터 로딩
train_df = pd.read_csv("dataset/1000btc_train.csv")
features = ['input_count', 'output_count', 'max_output_ratio', 'max_input_ratio']

# 🔹 2. 전처리 (로그 변환 + 정규화)
X_log = train_df[features].apply(lambda x: np.log1p(x))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# 🔹 3. 엘보우 방법으로 최적 k 찾기
wcss = []  # 클러스터 내 제곱합 (within-cluster sum of squares)

K_RANGE = range(1, 11)  # k를 1부터 10까지 실험

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # inertia_ = WCSS

# 기울기 기반 자동 탐지
wcss_diff = np.diff(wcss)
wcss_diff2 = np.diff(wcss_diff)
optimal_k = np.argmax(wcss_diff2) + 2

plt.figure(figsize=(8, 4))
plt.plot(K_RANGE, wcss, marker='o')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f"Optimal k = {optimal_k}")
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.legend()
plt.grid(True)
plt.show()

print(f"📌 자동 탐지된 최적의 클러스터 수: k = {optimal_k}")

