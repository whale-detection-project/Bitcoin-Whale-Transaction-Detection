import pandas as pd
from tqdm import tqdm

# 📁 CSV 로드
df = pd.read_csv("data/whale_train.csv")

# tqdm 설정
tqdm.pandas()

# 🐋 고래 유형 분류 함수
def classify_whale(row):
    if row['input_count'] >= 10 and row['output_count'] <= 2:
        return 0  # 다수입력 단일출력형
    
    elif row['input_count'] <= 2 and row['output_count'] >= 10:
        return 1  # 단일입력 다수출력형
    
    elif (
    row['input_count'] >= 20 and
    row['output_count'] <= 5 ):
        return 2  # 잔돈합치기형
        
    elif row.get('fee_per_max_ratio', 0) > 0.000001:
        return 3  # 급행전송형
    
    elif row['fee_per_max_ratio'] < 0.000001 and row['max_output_ratio'] < 0.2:
        return 4  # 은닉형
    
    else:
        return 5  # 기타/미분류형

# 🏷️ tqdm 적용 라벨 생성
print("📌 고래 유형 분류 중...")
df['whale_type'] = df.progress_apply(classify_whale, axis=1)

# 🗂️ 유형 라벨 이름 매핑
whale_label_map = {
    0: '0: 다수입력 단일출력형 (less_output_whale)',
    1: '1: 단일입력 다수출력형 (less_input_whale)',
    2: '2: 잔돈합치기형 (dust_merging_whale)',
    3: '3: 급행전송형 (fast_transfer_whale)',
    4: '4: 은닉전송형 (clean_hide_whale)',
    5: '5: 기타/미분류형 (unknown_whale)'
}

# 📊 통계 출력
print("\n📊 고래 유형별 분포:")
counts = df['whale_type'].value_counts(normalize=True).sort_index()
for idx, ratio in counts.items():
    print(f"• {whale_label_map.get(idx, idx)} → {ratio:.2%}")

# 💾 저장
df.to_csv("data/labeled_whales.csv", index=False)
print("\n✅ 라벨링 완료 → data/labeled_whales.csv 저장됨")
