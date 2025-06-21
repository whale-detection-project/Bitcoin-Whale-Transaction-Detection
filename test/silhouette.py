import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm # 진행 상황 표시를 위해 tqdm 추가

# 🔹 1. 학습 데이터 로드
train_df = pd.read_csv("dataset/1000btc_train.csv")
features = ['input_count', 'output_count', 'max_output_ratio', 'max_input_ratio']

# 🔹 2. 로그 변환 + 정규화
X_train_log = train_df[features].apply(lambda x: np.log1p(x))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_log)

# --- 💡 실루엣 분석을 위한 K-Means 최적 K 탐색 및 시각화 ---
print("--- 실루엣 분석을 통한 최적 K 탐색 시작 ---")
silhouette_scores = []
K_RANGE = range(2, 11) # 클러스터 개수 2부터 10까지 테스트

for k in tqdm(K_RANGE, desc="Calculating silhouette scores"):
    # n_init='auto'를 사용하면 K-Means 초기화가 더 견고해집니다.
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels_temp = kmeans_temp.fit_predict(X_train_scaled)
    score = silhouette_score(X_train_scaled, labels_temp)
    silhouette_scores.append(score)
    # tqdm.write()를 사용하면 진행률 표시줄을 방해하지 않고 출력할 수 있습니다.
    tqdm.write(f"k={k}, silhouette score={score:.4f}")

# 실루엣 점수 시각화
plt.figure(figsize=(10, 6))
plt.plot(K_RANGE, silhouette_scores, marker='o')
plt.title('Silhouette Score for Different k values')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.xticks(K_RANGE)
plt.show()
print("--- 실루엣 분석 완료 ---")
# -------------------------------------------------------------

print("\n✅ 실루엣 분석이 완료되었습니다. 위에 표시된 그래프를 확인하세요.")