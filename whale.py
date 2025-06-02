import pandas as pd

# 데이터 로드
df = pd.read_csv("labeled_whales.csv")

# 라벨 정의
whale_labels = {
    0: 'normal',                  # 일반 거래
    1: 'one_input_whale',         # 단일 수신형 고래
    2: 'distributed_whale',       # 분산 전송형 고래
    3: 'dust_merging_whale',      # 잔돈 합치기형 고래
    4: 'fast_transfer_whale',     # 급행 전송형 고래
    5: 'clean_hide_whale'         # 세탁/위장형 고래
}

# 고래만 필터링 (whale_type != 0)
whales_only = df[df['whale_type'] != 0]

# 타입별 개수 및 비율 계산
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

# 고래 데이터만 CSV로 저장
whales_only.to_csv("whale_only.csv", index=False)
print("\n✅ 고래 거래만 whale_only.csv로 저장 완료!")
