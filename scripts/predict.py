import pandas as pd
import joblib

# 🔽 예측 대상 데이터 로드
df = pd.read_csv("data/test.csv")

# 사용될 피처 목록
features = [
    'input_count', 'output_count', 'total_input_value',
    'max_input_value', 'max_output_value', 'max_output_ratio',
    'fee_per_max_ratio', 'has_zero_output'
]

# 전처리
X = df[features].copy()
X['has_zero_output'] = X['has_zero_output'].astype(int)

# ✅ 저장된 모델 불러오기 (일반 거래 포함된 모델)
model = joblib.load("dog.joblib")

# 예측
y_pred = model.predict(X)

# 예측 라벨 이름 매핑 (0 포함)
label_mapping = {
    0: 'normal',
    1: 'one_input_whale',
    2: 'distributed_whale',
    3: 'dust_merging_whale',
    4: 'fast_transfer_whale',
    5: 'clean_hide_whale'
}
df['predicted_whale_type'] = [label_mapping.get(label, f"type_{label}") for label in y_pred]

# 저장
df.to_csv("predicted_whales.csv", index=False)
print("✅ 예측 결과가 predicted_whales.csv로 저장되었습니다.")
