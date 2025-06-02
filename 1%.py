import pandas as pd

# CSV 불러오기
df = pd.read_csv("data/BTC_1m.csv")  # ← 경로 수정 필요

def classify_whale(row):
    input_count = row['input_count']
    output_count = row['output_count']
    total_input = row['total_input_value']
    max_input = row['max_input_value']
    max_output = row['max_output_value']
    max_output_ratio = row['max_output_ratio']
    fee_ratio = row['fee_per_max_ratio']
    has_zero = row['has_zero_output']
    
    input_99 = df['total_input_value'].quantile(0.99)
    output_99 = df['max_output_value'].quantile(0.99)

    # 고래로 간주할 최소 금액 조건 (10 BTC 이상)
    #if total_input < 1e8 and max_output < 1e8:
     #   return 0  # 금액 너무 작으면 일반 거래로 간주
     
     # 상위 1% 이상일 때만 고래로 간주
    if total_input < input_99 and max_output < output_99:
        return 0  # 일반 거래


    # 1. 단일 수신형 고래
    if input_count > 5 and output_count == 1 and max_output_ratio > 0.9:
        return 1

    # 2. 분산 전송형 고래
    elif input_count == 1 and output_count >= 10 and max_output_ratio < 0.3:
        return 2

    # 3. 잔돈 합치기형 고래
    elif input_count >= 100 and output_count <= 2 and max_input < (0.1 * total_input):
        return 3

    # 4. 급행 전송형 고래
    elif fee_ratio > 0.3:
        return 4

    # 5. 세탁/위장형 고래
    elif has_zero and output_count > 5 and max_output_ratio < 0.4:
        return 5

    # 0. 일반 거래
    else:
        return 0

# 라벨링 수행
df['whale_type'] = df.apply(classify_whale, axis=1)

# 결과 저장
df.to_csv("labeled_whales.csv", index=False)
print("완료: whale_type 라벨링 결과가 labeled_whales.csv로 저장되었습니다.")
