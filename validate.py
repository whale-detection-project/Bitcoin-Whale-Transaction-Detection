#!/usr/bin/env python3
# validate_whale_lgbm.py
# ----------------------------------------------------------
#  whale_lgbm.pkl  을  원본 CSV 일부(20 %)로 검증
# ----------------------------------------------------------
import joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, confusion_matrix)

# ---------- 설정 ----------
CSV_PATH   = Path("data/BTC_2024_binary.csv")
MODEL_PATH = Path("whale_lgbm.pkl")
TEST_FRAC  = 0.20                # 검증용 비율

FEATURES = [
    "fee", "fee_per_max_ratio",
    "input_count", "output_count",
    "max_output_value", "max_input_value"
]

# ---------- 1. 데이터 로드 & 샘플 ----------
df = pd.read_csv(CSV_PATH)

test_df = df.sample(frac=TEST_FRAC, random_state=777)
X_test  = test_df[FEATURES].apply(np.log1p)    # ★ 학습과 동일한 log1p
y_test  = test_df["whale_any"]

# ---------- 2. 모델 로드 ----------
model = joblib.load(MODEL_PATH)

# ---------- 3. 예측 ----------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)           # 필요 시 threshold 조정

# ---------- 4. 지표 ----------
print(classification_report(y_test, y_pred, digits=4))
print("ROC-AUC :", roc_auc_score(y_test, y_prob))
print("PR-AUC  :", average_precision_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- 5. 샘플 결과 미리보기 ----------
print("\nSample predictions (prob, true):")
print(list(zip(y_prob[:10], y_test.values[:10])))
