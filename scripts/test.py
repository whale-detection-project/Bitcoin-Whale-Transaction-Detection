"""
비트코인 거래 데이터를 기반으로
- 학습된 모델로 고래 유형 예측을 수행하고,
  규칙 기반 라벨링 결과와 비교하여 검증하는 통합 스크립트입니다.

- 입력: data/test.csv
- 출력: 
    - predicted_whales.csv : 예측 결과 저장
    - prediction_vs_rule_mismatches.csv : 예측과 규칙 기반 라벨 불일치 저장
- 모델: model/dog.joblib (XGBodost 모델)
"""

import pandas as pd
import joblib
from tqdm import tqdm

# ───────────────────────────────
# 전역 설정
# ───────────────────────────────
tqdm.pandas()

label_mapping = {
    0: 'normal',
    1: 'less_output_whale',
    2: 'less_input_whale',
    3: 'less_to_less_whale',
    4: 'dust_merging_whale',
    5: 'fast_transfer_whale',
    #5: 'clean_hide_whale'
}

# ───────────────────────────────
# 1. 모델 & 전처리
# ───────────────────────────────
def load_model(path="model/dog.joblib"):
    return joblib.load(path)

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = [
        'input_count', 'output_count', 'total_input_value',
        'max_input_value', 'max_output_value', 'max_output_ratio',
        'fee_per_max_ratio', 'has_zero_output'
    ]
    X = df[feats].copy()
    X['has_zero_output'] = X['has_zero_output'].astype(int)
    return X

def predict_whale_type(df: pd.DataFrame, model) -> pd.DataFrame:
    X = preprocess_features(df)
    y_pred = model.predict(X)
    df['predicted_whale_type'] = [
        label_mapping.get(code, f"type_{code}") for code in y_pred
    ]
    return df

# ───────────────────────────────
# 2. 규칙 기반 라벨링

# ───────────────────────────────
def classify_whale(row) -> int:
    """
    규칙 기반 고래 유형 분류기
    ------------------------------------------------------------------
    0 normal                : 소액 필터(≤ 5 BTC & 최대 출력 ≤ 1 BTC)
    1 less_output_whale     : 다수 입력(≥5) → 소수 출력(≤2) & 편중(>0.9)
    2 less_input_whale      : 소수 입력(≤2) → 다수 출력(≥5) & 분산(<0.3)
    3 less_to_less_whale    : 소수 입력(≤2) → 소수 출력(≤2)
    4 dust_merging_whale    : 다수↔다수(≥10) + 잔돈 합치기(max_in < 0.1·total_in)
    5 fast_transfer_whale   : 높은 수수료(fee_ratio > 1 %)
    ------------------------------------------------------------------
    """
    ic  = row['input_count']
    oc  = row['output_count']
    tot = row['total_input_value']
    max_in  = row['max_input_value']
    max_out = row['max_output_value']
    ratio   = row['max_output_ratio']
    fee_r   = row['fee_per_max_ratio']
    # has_zero = row['has_zero_output']  # 필요 시 사용

    # 0️⃣ 초기 필터: 총 입력이 5e9 (50 BTC) 미만이거나 최대 출력이 3e9 (30 BTC) 미만인 경우 일반 거래로 분류
    if tot < 1e10: #and max_out < 3e9:
        return 0
        
    # ───── 고액/고래 후보만 여기서 분기 ─────
    # 1️⃣ 다수 입력 → 소수 출력 (less_output_whale)
    if ic >= 5 and oc <= 2 and ratio > 0.9:
        return 1 

    # 2️⃣ 소수 입력 → 다수 출력 (less_input_whale)
    elif ic <= 2 and oc >= 5 and ratio < 0.3:
        return 2 

    # 3️⃣ 소수 입력 → 소수 출력 (특정 고래 유형에 해당하지 않는 고액 거래)
    elif ic <= 2 and oc <= 2:
        return 3 

    # 4️⃣ 잔돈 합치기 (dust_merging_whale)
    elif ic >= 10 and oc >= 10 and max_in < (0.1 * tot):
        return 4 

    # 5️⃣ 빠른 전송 (고수수료) (fast_transfer_whale)
    elif fee_r > 0.01:
        return 5 

    # 그 외 조건에 해당하지 않는 경우 일반 거래로 분류
    else:
        return 0


def validate_prediction(df: pd.DataFrame) -> pd.DataFrame:
    df['true_whale_type_code'] = df.progress_apply(classify_whale, axis=1)
    df['true_whale_type']      = df['true_whale_type_code'].map(label_mapping)
    df['match']                = df['predicted_whale_type'] == df['true_whale_type']
    return df

# ───────────────────────────────
# 3. 클래스별 정확도 & 오류 수
# ───────────────────────────────
def get_class_metrics(df: pd.DataFrame):
    """
    각 클래스별
        accuracy (%)  : 일치율
        wrong_count   : 틀린 행 수
        total         : 총 샘플 수
    반환
    """
    metrics = {}
    for code, name in label_mapping.items():
        cls_df   = df[df['true_whale_type_code'] == code]
        total    = len(cls_df)
        wrong    = int((~cls_df['match']).sum())
        acc      = None if total == 0 else (1 - wrong / total) * 100
        metrics[name] = {'accuracy': acc, 'wrong_count': wrong, 'total': total}
    return metrics

def print_class_metrics(metrics: dict):
    print("\n📊 테스트 데이터 개수 : 1000000개")
    print("\n📊 클래스별 일치율 및 오류 건수")
    for cls, m in metrics.items():
        if m['total'] == 0:
            print(f"   • {cls}: (샘플 없음)")
        else:
            print(f"   • {cls}: {m['accuracy']:.2f}%  | 고래수: {m['total']}건 / 불일치: {m['wrong_count']}건 ")

# ───────────────────────────────
# 4. 결과 저장
# ───────────────────────────────
def save_results(df: pd.DataFrame):
    # 전체 예측은 그대로 모두 저장
    df.to_csv("test/predicted_whales.csv", index=False)

    # CSV에서 제외할 컬럼
    drop_cols = ['tx_hash', 'block_timestamp', 'has_zero_output']

    # ① 예측 ≠ 규칙  (불일치)
    df_mismatch = df[~df['match']].drop(columns=drop_cols, errors='ignore')
    df_mismatch.to_csv("test/prediction_vs_rule_mismatches.csv", index=False)

    # ② 예측 = 규칙 & 고래(정상 제외)  (정확히 탐지된 고래)
    df_correct_whales = df[df['match'] & (df['true_whale_type'] != 'normal')] \
                          .drop(columns=drop_cols, errors='ignore')
    df_correct_whales.to_csv("test/correctly_detected_whales.csv", index=False)

    print("\n💾 CSV 저장 완료:")
    print("   • test/predicted_whales.csv")
    print("   • test/prediction_vs_rule_mismatches.csv       (컬럼 3개 제거)")
    print("   • test/correctly_detected_whales.csv           (컬럼 3개 제거)")



# ───────────────────────────────
# 5. 메인
# ───────────────────────────────
def main():
    print("📥 데이터 로드 중...")
    df = pd.read_csv("data/test2.csv")

    print("🤖 모델 로드 중...")
    model = load_model()

    print("🔮 예측 중...")
    df = predict_whale_type(df, model)

    print("📐 규칙 기반 검증 중...")
    df = validate_prediction(df)

    metrics = get_class_metrics(df)
    print_class_metrics(metrics)

    save_results(df)

if __name__ == "__main__":
    main()