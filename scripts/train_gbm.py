# train_lgbm_to_torch.py
# ----------------------------------------------------------
# 1) LightGBM 이진 분류 학습
# 2) Hummingbird 로 PyTorch 모듈 변환(GPU 지원)
# 3) TorchScript 저장 & 간단 추론
# ----------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from hummingbird.ml import convert
import torch
import joblib, os, pathlib

# ---------- 0. 설정 ----------
CSV_PATH   = "data/BTC_2024_binary.csv"
OUT_LGBM   = "whale_lgbm.pkl"
OUT_TORCH  = "whale_lgbm.pt"          # TorchScript 저장 파일
FEATURES = [
    "fee", "fee_per_max_ratio",
    "input_count", "output_count",
    "max_output_value", "max_input_value"
]

# ---------- 1. 데이터 로드 ----------
df = pd.read_csv(CSV_PATH)
X  = df[FEATURES].apply(np.log1p).values      # log1p 정규화 → ndarray
y  = df["whale_any"].values.astype(np.int32)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- 2. LightGBM 학습 ----------
lgbm = LGBMClassifier(
    objective      = "binary",
    n_estimators   = 600,
    learning_rate  = 0.03,
    num_leaves     = 31,
    class_weight   = "balanced",
    subsample      = 0.8,
    colsample_bytree = 0.8,
    random_state   = 42
)
lgbm.fit(X_tr, y_tr)

prob = lgbm.predict_proba(X_val)[:, 1]
print("LightGBM ROC-AUC:", roc_auc_score(y_val, prob))

joblib.dump(lgbm, OUT_LGBM)
print(f"[√] LightGBM 모델 저장 → {OUT_LGBM}")

# ---------- 3. PyTorch 변환 ----------
torch_model = convert(lgbm, "torch", device="cpu")     # device="cuda" 가능
# Hummingbird가 nn.Module 래퍼를 반환
torch_script = torch.jit.script(torch_model.model)
torch.jit.save(torch_script, OUT_TORCH)
print(f"[√] TorchScript 저장 → {OUT_TORCH}")

# ---------- 4. 추론 예시 ----------
sample = torch.tensor(X_val[:5])           # (batch, feat)
with torch.no_grad():
    torch_prob = torch_script(sample)[:, 1]   # Positive 클래스 확률
print("Torch 예측 확률 :", torch_prob.tolist())
