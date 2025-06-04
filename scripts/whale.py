import pandas as pd
from tqdm import tqdm

# tqdm 활성화
tqdm.pandas()

# 데이터 로드
print("📥 데이터 불러오는 중...")
df = pd.read_csv("labeled_whales.csv")

# 라벨 정의
whale_labels = {
    0: 'normal',                  
    1: 'one_input_whale',         
    2: 'distributed_whale',       
    3: 'dust_merging_whale',      
    4: 'fast_transfer_whale',     
    5: 'clean_hide_whale'         
}

# 고래만 필터링 (whale_type != 0)
print("🔍 고래 거래 필터링 중...")
whales_only = df[df['whale_type'] != 0]

# 타입별 개수 및 비율 계산
print("📊 통계 계산 중...")
whale_counts = whales_only['whale_type'].value_counts().sort_index()
total = whale_counts.sum()
whale_stats = pd.DataFrame({
    'type_code': whale_counts.index,
    'type_name': [whale_labels.get(i, f"type_{i}") for i in whale_counts.index],
    'count': whale_counts.values,
    'percentage': (whale_counts.values / total * 100).round(2)
})

# 통계 출력
print("\n[🐋 Whale Type 분류된 거래 통계]")
print(whale_stats.to_string(index=False))

# 고래 거래만 저장
print("\n💾 whale_only.csv 저장 중...")
for _ in tqdm(range(1), desc="Saving CSV"):
    whales_only.to_csv("whale_only.csv", index=False)

print("✅ 완료: whale_only.csv 저장됨")
