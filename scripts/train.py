import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import numpy as np
import joblib

# CSV 로드
df = pd.read_csv("labeled_whales.csv")

# 🐋 고래 라벨 정의 (0 포함)
whale_labels = {
    0: 'normal',
    1: 'one_input_whale',
    2: 'distributed_whale',
    3: 'dust_merging_whale',
    4: 'fast_transfer_whale',
    5: 'clean_hide_whale'
}

# 라벨 인코딩 (0~5 → 0~5 그대로)
label_map = {k: k for k in whale_labels.keys()}
reverse_map = {v: k for k, v in label_map.items()}
df['whale_type_encoded'] = df['whale_type'].map(label_map)

# 피처 및 라벨 정의
features = [
    'input_count', 'output_count', 'total_input_value',
    'max_input_value', 'max_output_value', 'max_output_ratio',
    'fee_per_max_ratio', 'has_zero_output'
]
X = df[features].copy()
X['has_zero_output'] = X['has_zero_output'].astype(int)
y = df['whale_type_encoded']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost 모델 학습
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)

# 전체 레이블 및 이름
labels = sorted(whale_labels.keys())  # [0, 1, 2, 3, 4, 5]
target_names = [whale_labels[i] for i in labels]

print("\n[📊 Classification Report]")
print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))

# ✅ 모델 저장
joblib.dump(model, "dog.joblib")
print("\n✅ 모델이 'xgb_whale_classifier_with_normal.joblib' 파일로 저장되었습니다.")
