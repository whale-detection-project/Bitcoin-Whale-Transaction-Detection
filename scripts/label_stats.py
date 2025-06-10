"""
- Bitcoin 거래 데이터를 기반으로 고래 유형을 분류하고 라벨링 한후 
  통계를 계산하는 통합 스크립트.

- 입력: data/BTC_1m.csv
- 출력:
    - labeled_whales.csv : 고래 유형 라벨링 결과
    - whale_only.csv : 고래 거래만 추출한 결과
    - 콘솔 : 고래 유형별 통계 출력
"""

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

# 고래 라벨 정의
whale_labels = {
    0: 'normal',
    1: 'less_output_whale', #판매자 5명이상 구매자 2명이하인 경우
    2: 'less_input_whale', #판매자 2명이하 구매자 5명이상인 경우
    3: 'less_to_less_whale', # 판매자 2명이하 구매자 2명이하
    4: 'dust_merging_whale', #판매자 10명이상 구매자 10명이상
    5: 'fast_transfer_whale', #수수료가 전채 아웃풋밸류의 1퍼이상인경우
    #5: 'clean_hide_whale'
}

def classify_whale(row):
    """
    한 행(row)의 거래 데이터를 기반으로 고래 유형을 규칙에 따라 분류합니다.
    
    분류 기준:
    - 0 (normal): 일반 거래
    - 1 (less_output_whale): 다수의 입력 → 소수의 출력 (보통 1~2)
    - 2 (less_input_whale): 소수 입력 → 다수의 출력 (분산 목적)
    - X (dust_merging_whale): 50개 이상 잔돈 입력 → 합쳐서 전송 - 탐지 표본이 너무 적어 제거
    - 3 (fast_transfer_whale): 수수료 비율이 높은 빠른 전송 거래
    - X (clean_hide_whale): 0 출력값을 포함한 은폐성 높은 거래 - 탐지 표본이 너무 적어 제거
    """
    input_count = row['input_count']
    output_count = row['output_count']
    total_input = row['total_input_value']
    total_output = row['total_output_value']  # total_input
    max_input = row['max_input_value']
    max_output = row['max_output_value']
    max_output_ratio = row['max_output_ratio']
    fee_ratio = row['fee_per_max_ratio']
    #has_zero = row['has_zero_output']

    if total_input < 1e10: #and max_output < 3e9 :
        return 0
        
    # ───── 고액/고래 후보만 여기서 분기 ─────
    if  input_count >= 5 and output_count <= 2 and max_output_ratio > 0.9:
        return 1 #less_output_whale
    elif input_count <=2 and output_count >= 5 and max_output_ratio < 0.3:
        return 2 #less_input_whale
    elif input_count <= 2 and output_count <=2:
        return 3 #
    elif input_count >= 10 and output_count >= 10 and max_input < (0.1 * total_input):
        return 4 #dust_merging_whale
    elif fee_ratio > 0.01:
        return 5 #fast_transfer_whale
    #elif has_zero and output_count > 5:
        #return 5 #clean_hide_whale
    else:
        return 0

def label_whales(input_path="data/raw.csv", output_path="data/labeled_whales.csv"):
    """
    거래 데이터를 불러와서 고래 유형을 라벨링한 후 CSV로 저장합니다.
    """
    print("📥 거래 데이터 로딩 중...")
    df = pd.read_csv(input_path)
    df['whale_type'] = df.progress_apply(classify_whale, axis=1)
    df.to_csv(output_path, index=False)
    print(f"✅ 라벨링 결과 저장 완료 → {output_path}")
    return df  # 이후 통계를 위해 반환

def compute_whale_stats(df, output_path="data/whale_only.csv"):
    """
    라벨링된 데이터프레임에서 고래 거래만 추출하여 저장하고 통계를 출력합니다.
    """
    print("🔍 고래 거래 필터링 중...")
    whales_only = df[df['whale_type'] != 0]

    print("📊 고래 통계 계산 중...")
    whale_counts = whales_only['whale_type'].value_counts().sort_index()
    total = whale_counts.sum()
    stats = pd.DataFrame({
        'type_code': whale_counts.index,
        'type_name': [whale_labels[i] for i in whale_counts.index],
        'count': whale_counts.values,
        'percentage': (whale_counts.values / total * 100).round(2)
    })

    print("\n[🐋 Whale Type 분류 통계]")
    print(stats.to_string(index=False))

    print(f"\n💾 고래 거래 저장 중 → {output_path}")
    whales_only.to_csv(output_path, index=False)
    print("✅ 완료")

def main():
    df = label_whales()
    compute_whale_stats(df)

if __name__ == "__main__":
    main()
