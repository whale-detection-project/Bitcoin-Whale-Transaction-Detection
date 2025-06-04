"""
XGBoost를 이용해 Bitcoin 고래 거래 유형을 분류하는 모델을 학습하는 스크립트입니다.

- 입력 파일: labeled_whales.csv (규칙 기반으로 분류된 고래 라벨 포함)

- 출력:
    - 콘솔에 모델 분류 성능 출력
    - 모델 파일 저장
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import numpy as np
import joblib

# 🐋 고래 라벨 정의 (0 포함)
whale_labels = {
    0: 'normal',
    1: 'less_output_whale', #소수의 출력 고래
    2: 'less_input_whale', #소수의 입력 고래
    #3: 'dust_merging_whale', #잔돈 처리 고래
    3: 'fast_transfer_whale', #빠른 전송 고래
    #5: 'clean_hide_whale' #세탁/숨김 고래
}

def load_and_preprocess_data(csv_path: str):
    """
    라벨링된 CSV 파일을 로드하고 피처와 라벨을 전처리합니다.

    매개변수:
        csv_path (str): 입력 CSV 파일 경로

    반환값:
        X (pd.DataFrame): 피처 데이터
        y (pd.Series): 인코딩된 라벨 데이터
    """
    df = pd.read_csv(csv_path)

    label_map = {k: k for k in whale_labels.keys()}
    df['whale_type_encoded'] = df['whale_type'].map(label_map)

    features = [
        'input_count', 'output_count', 'total_input_value',
        'max_input_value', 'max_output_value', 'max_output_ratio',
        'fee_per_max_ratio', 'has_zero_output'
    ]
    X = df[features].copy()
    X['has_zero_output'] = X['has_zero_output'].astype(int)
    y = df['whale_type_encoded']

    return X, y

def train_and_evaluate_model(X, y, model_path="dog.joblib"):
    """
    XGBoost 모델을 학습하고 테스트 데이터로 성능을 평가한 뒤 저장합니다.

    매개변수:
        X (pd.DataFrame): 피처 데이터
        y (pd.Series): 라벨 데이터
        model_path (str): 저장할 모델 경로 (.joblib)
    """
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 모델 학습
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 평가 리포트 출력
    labels = sorted(whale_labels.keys())
    target_names = [whale_labels[i] for i in labels]

    print("\n[📊 Classification Report]")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))

    # 모델 저장
    joblib.dump(model, model_path)
    print(f"\n✅ 모델이 '{model_path}' 파일로 저장되었습니다.")

def main():
    """
    전체 실행 함수:
    - labeled_whales.csv 로부터 데이터를 불러오고
    - 모델을 학습 및 평가하며
    - 저장된 모델을 생성합니다.
    """
    X, y = load_and_preprocess_data("data/labeled_whales.csv")
    train_and_evaluate_model(X, y, model_path="model/dog.joblib")

if __name__ == "__main__":
    main()
