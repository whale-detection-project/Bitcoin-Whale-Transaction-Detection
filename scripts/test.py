"""
비트코인 거래 데이터를 기반으로
- 학습된 모델로 고래 유형 예측을 수행하고,
  규칙 기반 라벨링 결과와 비교하여 검증하는 통합 스크립트입니다.

- 입력: data/test.csv
- 출력: 
    - predicted_whales.csv : 예측 결과 저장
    - prediction_vs_rule_mismatches.csv : 예측과 규칙 기반 라벨 불일치 저장
- 모델: model/dog.joblib (XGBoost 모델)
"""

import pandas as pd
import joblib
from tqdm import tqdm

# tqdm 설정
tqdm.pandas()

# 라벨 매핑
label_mapping = {
    0: 'normal',
    1: 'less_output_whale',
    2: 'less_input_whale',
    #3: 'dust_merging_whale',
    3: 'fast_transfer_whale',
    #5: 'clean_hide_whale'
}

def load_model(model_path="dog.joblib"):
    """ 저장된 모델 불러오기 """
    return joblib.load(model_path)

def preprocess_features(df):
    """ 예측용 피처 전처리 """
    features = [
        'input_count', 'output_count', 'total_input_value',
        'max_input_value', 'max_output_value', 'max_output_ratio',
        'fee_per_max_ratio', 'has_zero_output'
    ]
    X = df[features].copy()
    X['has_zero_output'] = X['has_zero_output'].astype(int)
    return X

def predict_whale_type(df, model):
    """ 모델 예측 결과를 데이터프레임에 추가 """
    X = preprocess_features(df)
    y_pred = model.predict(X)
    df['predicted_whale_type'] = [label_mapping.get(label, f"type_{label}") for label in y_pred]
    return df

def classify_whale(row):
    """ 규칙 기반 고래 분류 함수 """
    input_count = row['input_count']
    output_count = row['output_count']
    total_input = row['total_input_value']
    max_input = row['max_input_value']
    max_output = row['max_output_value']
    max_output_ratio = row['max_output_ratio']
    fee_ratio = row['fee_per_max_ratio']
    has_zero = bool(row['has_zero_output'])

    if total_input < 5e9 and max_output < 1e8:
        return 0
    if input_count >= 5 and output_count <= 2 and max_output_ratio > 0.9:
        return 1 #less_output_whale
    elif input_count <=2 and output_count >= 5 and max_output_ratio < 0.3:
        return 2 #less_input_whale
    #elif input_count >= 50 and output_count <= 2 and max_input < (0.03 * total_input):
        #return 3 #dust_merging_whale
    elif fee_ratio > 0.01:
        return 3 #fast_transfer_whale
    #elif has_zero and output_count > 5:
        #return 5 #clean_hide_whale
    else:
        return 0

def validate_prediction(df):
    """ 규칙 기반 라벨 생성 및 예측과 비교 """
    df['true_whale_type_code'] = df.progress_apply(classify_whale, axis=1)
    df['true_whale_type'] = df['true_whale_type_code'].map(label_mapping)
    df['match'] = df['predicted_whale_type'] == df['true_whale_type']
    return df

def main():
    print("📥 테스트 데이터 로드 중...")
    df = pd.read_csv("data/test.csv")

    print("🤖 모델 불러오는 중...")
    model = load_model("model/dog.joblib")

    print("🔮 예측 수행 중...")
    df = predict_whale_type(df, model)

    print("📐 규칙 기반 검증 중...")
    df = validate_prediction(df)

    accuracy = df['match'].mean() * 100
    print(f"\n✅ 모델 예측과 규칙 기반 라벨 일치율: {accuracy:.2f}%")

    df.to_csv("test/predicted_whales.csv", index=False)
    print("💾 예측 결과 saved → predicted_whales.csv")

    mismatches = df[df['match'] == False]
    mismatches.to_csv("test/prediction_vs_rule_mismatches.csv", index=False)
    print("🔍 불일치 결과 saved → prediction_vs_rule_mismatches.csv")

if __name__ == "__main__":
    main()
