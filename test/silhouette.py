# ------------------------------------------------------------
# 0) 라이브러리
# ------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans          # ⚡️빠른 K-Means
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------------------------------------
# 1) 데이터 로드 & 기본 전처리
# ------------------------------------------------------------
DATA_PATH = "dataset/1000btc_train.csv"              # 경로만 맞춰 주세요
features = ['input_count', 'output_count',
            'max_output_ratio', 'max_input_ratio']

df = pd.read_csv(DATA_PATH, usecols=features)

# 로그 변환 → 0 처리 주의
X_log = np.log1p(df[features])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# ------------------------------------------------------------
# 2) MiniBatchKMeans + 실루엣 점수 계산
# ------------------------------------------------------------
K_RANGE = range(2, 11)               # 2 ≤ k ≤ 10
SAMPLE_SIZE = 10_000                 # silhouette 샘플 수 (8~10k면 충분)

sil_scores = []

print("\n--- 실루엣 분석 시작 ---")
for k in tqdm(K_RANGE, desc="k loop"):
    mbk = MiniBatchKMeans(
        n_clusters=k,
        batch_size=4096,             # GPU쓰면 8192↑로 키워도 OK
        random_state=42
    ).fit(X_scaled)

    labels = mbk.labels_

    # O(N²) 방지용: sample_size 옵션
    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=SAMPLE_SIZE,
        random_state=42
    )
    sil_scores.append(score)
    tqdm.write(f"k={k}\t sil={score:.4f}")

print("--- 실루엣 분석 완료 ---")

# ------------------------------------------------------------
# 3) 결과 시각화
# ------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(list(K_RANGE), sil_scores, marker='o')
plt.title("Silhouette score vs k")
plt.xlabel("k (number of clusters)")
plt.ylabel("Silhouette score")
plt.xticks(list(K_RANGE))
plt.grid(True, alpha=.3)
plt.show()

# ------------------------------------------------------------
# 4) 최적 k 자동 선택(예: 최고점 or knee)
# ------------------------------------------------------------
best_k = K_RANGE[int(np.argmax(sil_scores))]
print(f"\n✅ 추천 k = {best_k}  (silhouette 최대)")
