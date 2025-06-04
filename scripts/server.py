# FastAPI 기반 실시간 고래 탐지 서버

import joblib
import pandas as pd
from fastapi import FastAPI, WebSocket

app = FastAPI()

MODEL_PATH = "model/xgb_whale_classifier_with_normal.joblib"

# 모델 로드
model = joblib.load(MODEL_PATH)

LABELS = {
    0: "normal",
    1: "less_output_whale",
    2: "less_input_whale",
    3: "fast_transfer_whale",
}

def preprocess(tx: dict) -> pd.DataFrame:
    """전처리: 트랜잭션 dict -> 모델 입력 데이터프레임"""
    df = pd.DataFrame([tx])
    df["has_zero_output"] = df["has_zero_output"].astype(int)
    features = [
        "input_count", "output_count", "total_input_value",
        "max_input_value", "max_output_value", "max_output_ratio",
        "fee_per_max_ratio", "has_zero_output",
    ]
    return df[features]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        tx = await websocket.receive_json()
        X = preprocess(tx)
        pred = model.predict(X)[0]
        label = LABELS.get(int(pred), f"type_{pred}")
        await websocket.send_json({"whale_type": label})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
