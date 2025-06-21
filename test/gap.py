# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# def compute_gap_statistic(X, n_refs=10, max_k=10):
#     gaps = []
#     deviations = []
#     wcss = []

#     shape = X.shape
#     tops = X.max(axis=0)
#     bottoms = X.min(axis=0)
#     dists = np.diag(tops - bottoms)

#     for k in tqdm(range(1, max_k + 1), desc="Calculating Gap Statistic"):
#         km = KMeans(n_clusters=k, random_state=42)
#         km.fit(X)
#         orig_wcss = np.log(km.inertia_)
#         wcss.append(orig_wcss)

#         ref_inertias = []
#         for _ in range(n_refs):
#             random_data = np.random.rand(*shape)
#             random_data = random_data @ dists + bottoms
#             km_ref = KMeans(n_clusters=k, random_state=42)
#             km_ref.fit(random_data)
#             ref_inertias.append(np.log(km_ref.inertia_))

#         gap = np.mean(ref_inertias) - orig_wcss
#         gaps.append(gap)
#         deviations.append(np.std(ref_inertias) * np.sqrt(1 + 1.0 / n_refs))

#     return np.array(gaps), np.array(deviations), np.array(wcss)

# # 🔹 데이터 로딩 및 전처리
# df = pd.read_csv("dataset/1000btc_train.csv")
# features = ['input_count', 'output_count', 'max_output_ratio', 'max_input_ratio']
# X_log = df[features].apply(lambda x: np.log1p(x))
# X_scaled = StandardScaler().fit_transform(X_log)

# # 🔹 Gap Statistic 계산
# gaps, deviations, wcss = compute_gap_statistic(X_scaled, n_refs=10, max_k=10)

# # 🔹 최적 k 선택 (gap[i] >= gap[i+1] - dev[i+1])
# optimal_k = next((k for k in range(0, len(gaps) - 1)
#                   if gaps[k] >= gaps[k + 1] - deviations[k + 1]), len(gaps) - 1) + 1

# print(f"📌 최적의 클러스터 수 (k): {optimal_k}")

# # 🔹 시각화
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(gaps) + 1), gaps, marker='o', label="Gap Statistic")
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Gap Value")
# plt.title("Gap Statistic for Optimal k")
# plt.grid(True)
# plt.axvline(optimal_k, color='r', linestyle='--', label=f"Optimal k = {optimal_k}")
# plt.legend()
# plt.show()

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import DBSCAN
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # 🔹 1. 데이터 로딩 및 전처리
# df = pd.read_csv("dataset/1000btc_train.csv")
# features = ['input_count', 'output_count', 'max_output_ratio', 'max_input_ratio']
# X_log = df[features].apply(lambda x: np.log1p(x))
# X_scaled = StandardScaler().fit_transform(X_log)

# # 🔹 2. DBSCAN 적용
# db = DBSCAN(eps=0.5, min_samples=5)  # eps와 min_samples는 조절 필요
# labels = db.fit_predict(X_scaled)

# # 🔹 3. 결과 확인
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise = list(labels).count(-1)

# print(f"📦 추정된 클러스터 수: {n_clusters}")
# print(f"❌ 이상치(노이즈) 데이터 수: {n_noise}")

# # 🔹 4. 시각화 (PCA 2D)
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# plt.figure(figsize=(8, 5))
# unique_labels = set(labels)
# for label in unique_labels:
#     idx = labels == label
#     plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
#                 label=f"Cluster {label}" if label != -1 else "Noise", alpha=0.6)

# plt.title(f"DBSCAN Clustering Result (k ≠ required)")
# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.metrics import calinski_harabasz_score
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # 🔹 1. 데이터 로딩 및 전처리
# df = pd.read_csv("dataset/1000btc_train.csv")
# features = ['input_count', 'output_count', 'max_output_ratio', 'max_input_ratio']
# X_log = df[features].apply(lambda x: np.log1p(x))
# X_scaled = StandardScaler().fit_transform(X_log)

# # 🔹 2. CH Score 계산
# ch_scores = []
# K_RANGE = range(2, 11)

# for k in tqdm(K_RANGE, desc="Calculating CH scores"):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X_scaled)
#     score = calinski_harabasz_score(X_scaled, labels)
#     ch_scores.append(score)
#     print(f"k={k}, CH Score={score:.2f}")

# # 🔹 3. 시각화
# optimal_k = K_RANGE[np.argmax(ch_scores)]

# plt.figure(figsize=(8, 4))
# plt.plot(K_RANGE, ch_scores, marker='o')
# plt.title("Calinski-Harabasz Index by k")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("CH Score")
# plt.axvline(optimal_k, linestyle='--', color='r', label=f"Optimal k = {optimal_k}")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# print(f"📌 Calinski-Harabasz 기준 최적의 k: {optimal_k}")

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import DBSCAN
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # ✅ 1. 데이터 로딩 + 샘플링 (RAM 절약)
# df = pd.read_csv("dataset/1000btc_train.csv")
# features = ['input_count', 'output_count', 'max_output_ratio', 'max_input_ratio']

# # 너무 많으면 1~2만개 정도만 사용
# if len(df) > 10000:
#     df = df.sample(n=10000, random_state=42)

# # ✅ 2. 로그 변환 + 정규화
# X_log = df[features].apply(lambda x: np.log1p(x))
# X_scaled = StandardScaler().fit_transform(X_log)

# # ✅ 3. DBSCAN 적용 (RAM 효율 높고 빠름)
# db = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)  # n_jobs=-1 → 모든 CPU 코어 사용
# labels = db.fit_predict(X_scaled)

# # ✅ 4. 요약 출력
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise = np.sum(labels == -1)

# print(f"📦 클러스터 수: {n_clusters}")
# print(f"❌ 노이즈 수: {n_noise}")

# # ✅ 5. 시각화 (PCA 2D)
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# plt.figure(figsize=(8, 5))
# unique_labels = set(labels)
# for label in unique_labels:
#     idx = labels == label
#     plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
#                 label=f"Cluster {label}" if label != -1 else "Noise", alpha=0.6)

# plt.title("DBSCAN Clustering (샘플링 + RAM 최적화)")
# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import tqdm

# 🔹 1. 데이터 로딩 및 전처리
df = pd.read_csv("dataset/1000btc_train.csv")
features = ['input_count', 'output_count', 'max_output_ratio', 'max_input_ratio']

# ✅ RAM 보호용 샘플링 (1만 행 제한)
df = df.sample(n=min(10000, len(df)), random_state=42)

X_log = df[features].apply(lambda x: np.log1p(x))
X_scaled = StandardScaler().fit_transform(X_log)

# ✅ 실루엣 점수 기반 최적 k 탐색
silhouette_scores = []
K_RANGE = range(2, 7)

for k in tqdm(K_RANGE, desc="Calculating Silhouette scores"):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    # 실루엣 계산 샘플 제한 (2000개 이하로)
    if len(X_scaled) > 5000:
        X_sampled, labels_sampled = resample(X_scaled, labels, n_samples=50000, random_state=42)
    else:
        X_sampled, labels_sampled = X_scaled, labels

    score = silhouette_score(X_sampled, labels_sampled)
    silhouette_scores.append(score)
    print(f"k={k}, Silhouette Score={score:.4f}")

# ✅ 최적 k 선택
optimal_k = K_RANGE[np.argmax(silhouette_scores)]
print(f"\n📌 실루엣 기준 최적 k: {optimal_k}")

# 🔹 시각화
plt.figure(figsize=(8, 4))
plt.plot(K_RANGE, silhouette_scores, marker='o')
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.axvline(optimal_k, linestyle='--', color='r', label=f"Optimal k = {optimal_k}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
