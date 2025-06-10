import websocket
import json
import joblib
import time
import math
import numpy as np
import pandas as pd
import logging
from threading import Lock, Thread
from fastapi import FastAPI
import uvicorn
from pymongo import MongoClient
from datetime import datetime



# 🔹 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# 🔹 MongoDB 설정
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["whale_detection"]
collection = db["whale_logs"]

# 🔹 모델 로드
scaler = joblib.load("model/scaler.pkl")
xgb_model = joblib.load("model/xgb_model.pkl")
pca = joblib.load("model/pca.pkl")

features = ['input_count', 'output_count', 'max_output_ratio', 'fee_per_max_ratio', 'max_input_ratio']
latest_result = {}
lock = Lock()

from datetime import datetime  # 이미 import 했을 수도 있음

def predict_cluster(data_dict, scaler, model, pca):
    try:
        for f in features:
            if f not in data_dict or data_dict[f] is None or data_dict[f] < 0:
                raise ValueError(f"Invalid or missing feature: {f} = {data_dict.get(f)}")

        X_dict = {f: [math.log1p(data_dict[f])] for f in features}
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



# 🔹 WebSocket 처리
def on_message(ws, message):
    try:
        data = json.loads(message)
        tx = data.get("x", {})

        total_input_value = sum(i.get("prev_out", {}).get("value", 0) for i in tx.get("inputs", [])) / 1e8
        if total_input_value < 10:
            return

        input_count = len(tx.get("inputs", []))
        output_list = tx.get("out", [])
        output_count = len(output_list)
        output_values = [o.get("value", 0) for o in output_list]
        total_output_value = sum(output_values)
        max_output = max(output_values, default=0)

        fee = (total_input_value * 1e8 - total_output_value) if total_output_value else 0
        fee_per_max_ratio = fee / max_output if max_output else 0
        max_output_ratio = max_output / total_output_value if total_output_value else 0

        max_input = max([i.get("prev_out", {}).get("value", 0) for i in tx.get("inputs", [])], default=0)
        max_input_ratio = max_input / (total_input_value * 1e8) if total_input_value > 0 else 0

        tx_data = {
            "input_count": input_count,
            "output_count": output_count,
            "max_output_ratio": max_output_ratio,
            "fee_per_max_ratio": fee_per_max_ratio,
            "max_input_ratio": max_input_ratio,
        }

        result = predict_cluster(tx_data, scaler, xgb_model, pca)

        if "error" in result:
            logging.error(f"❌ 예측 실패: {result['error']}")
            logging.error(f"입력 tx_data: {tx_data}")
            return

        with lock:
            latest_result.clear()
            latest_result.update(result)

        result_for_db = {
            **result,
            **tx_data,
            "total_input_value": total_input_value,
        }

        collection.insert_one(result_for_db)

        logging.info(
            f"🚨 고래 거래 감지 → 클러스터 {result['predicted_cluster']}, 총 입력: {total_input_value:.2f} BTC\n"
            f"입력수: {input_count}개, 출력수: {output_count}개, "
            f"max_output_ratio: {max_output_ratio:.4f}, "
            f"fee_per_max_ratio: {fee_per_max_ratio:.4f}, "
            f"max_input_ratio: {max_input_ratio:.4f}"
        )

    except Exception as e:
        logging.error(f"❌ WebSocket 처리 오류: {e}")


def on_open(ws):
    logging.info("🔗 Blockchain.com WebSocket 연결됨")
    ws.send(json.dumps({"op": "unconfirmed_sub"}))

def on_close(ws): logging.info("🔌 연결 종료")
def on_error(ws, error): logging.error(f"❌ 오류 발생: {error}")

def run_websocket():
    ws = websocket.WebSocketApp(
        "wss://ws.blockchain.info/inv",
        on_open=on_open,
        on_close=on_close,
        on_error=on_error,
        on_message=on_message
    )
    ws.run_forever()

# 🔹 FastAPI 엔드포인트
app = FastAPI()

# FastAPI 시작 이벤트에 웹소켓 시작 추가
@app.on_event("startup")
async def startup_event():
    logging.info("🚀 서버 시작: 웹소켓 연결을 시작합니다...")
    Thread(target=start_websocket, daemon=True).start()

@app.get("/detect")
def get_latest_result():
    with lock:
        if not latest_result:
            return {"message": "아직 탐지된 고래 거래 없음"}
        return latest_result

# 🔹 실행
def start_websocket():
    logging.info("🔄 웹소켓 스레드 시작")
    run_websocket()

if __name__ == "__main__":
    # 독립 실행 시에도 웹소켓 시작
    Thread(target=start_websocket, daemon=True).start()
    # 직접 실행할 때는 uvicorn 서버 시작
    uvicorn.run(app, host="0.0.0.0", port=8000)
