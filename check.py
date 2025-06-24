import json
from collections import Counter

# 1. JSON 파일 열기
with open('logs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 클러스터 라벨 수집
cluster_labels = [item['predicted_cluster'] for item in data]

# 3. 빈도수 계산
cluster_count = Counter(cluster_labels)
total = len(cluster_labels)

# 4. 결과 출력
print("🔍 클러스터 분포 통계:")
for cluster_id, count in sorted(cluster_count.items()):
    percentage = (count / total) * 100
    print(f"  - Cluster {cluster_id}: {count}건 ({percentage:.2f}%)")

print(f"\n📊 전체 데이터 수: {total}건")
