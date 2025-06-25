import joblib
import math
import pandas as pd
from datetime import datetime
from .config import FEATURES

def load_models():
    """모델과 전처리기를 로드합니다."""
    scaler = joblib.load("model/scaler.pkl")
    xgb_model = joblib.load("model/xgb_model.pkl")
    pca = joblib.load("model/pca.pkl")
    return scaler, xgb_model, pca

def predict_cluster(data_dict, scaler, model, pca):
    """주어진 데이터로 클러스터를 예측합니다."""
    try:
        for f in FEATURES:
            if f not in data_dict or data_dict[f] is None or data_dict[f] < 0:
                raise ValueError(f"Invalid or missing feature: {f} = {data_dict.get(f)}")

        X_dict = {f: [math.log1p(data_dict[f])] for f in FEATURES}
        X_df = pd.DataFrame(X_dict)
        X_scaled = scaler.transform(X_df)
        label = int(model.predict(X_scaled)[0])
        embedding = pca.transform(X_scaled)[0].tolist()

        # 사람이 보기 쉬운 시간 형식으로 변경
        human_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "predicted_cluster": label,
            "pca_embedding": embedding,
            "timestamp": human_time  # ← 문자열로 저장됨
        }
    except Exception as e:
        return {"error": str(e)}