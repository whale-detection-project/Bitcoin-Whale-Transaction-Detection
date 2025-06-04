import pandas as pd
from tqdm import tqdm

# tqdm 설정
tqdm.pandas()

# 라벨 매핑
whale_labels = {
    0: 'normal',
    1: 'one_input_whale',
    2: 'distributed_whale',
    3: 'dust_merging_whale',
    4: 'fast_transfer_whale',
    5: 'clean_hide_whale'
}

# 예측된 CSV 로드
df = pd.read_csv("predicted_whales.csv")
df['has_zero_output'] = df['has_zero_output'].astype(bool)

# 라벨링 기준값 (필요 시 사용 가능)
input_99 = df['total_input_value'].quantile(0.99)
output_99 = df['max_output_value'].quantile(0.99)

# 규칙 기반 라벨링 함수
def classify_whale(row):
    input_count = row['input_count']
    output_count = row['output_count']
    total_input = row['total_input_value']
    max_input = row['max_input_value']
    max_output = row['max_output_value']
    max_output_ratio = row['max_output_ratio']
    fee_ratio = row['fee_per_max_ratio']
    has_zero = row['has_zero_output']

    if total_input < 5e9 and max_output < 1e8:
        return 0
    if input_count > 5 and output_count == 1 and max_output_ratio > 0.9:
        return 1
    elif input_count == 1 and output_count >= 10 and max_output_ratio < 0.3:
        return 2
    elif input_count >= 100 and output_count <= 2 and max_input < (0.1 * total_input):
        return 3
    elif fee_ratio > 0.01:
        return 4
    elif has_zero and output_count > 5:
        return 5
    else:
        return 0

# 규칙 기반 라벨 부여
df['true_whale_type_code'] = df.progress_apply(classify_whale, axis=1)
df['true_whale_type'] = df['true_whale_type_code'].map(whale_labels)

# 비교
df['match'] = df['predicted_whale_type'] == df['true_whale_type']

# 정확도
accuracy = df['match'].mean() * 100
print(f"\n✅ 규칙 기반 라벨과 모델 예측 일치율: {accuracy:.2f}%")

# 불일치 저장
df_mismatches = df[df['match'] == False]
df_mismatches.to_csv("prediction_vs_rule_mismatches.csv", index=False)
print("🔍 불일치 사례는 prediction_vs_rule_mismatches.csv 로 저장됨")
