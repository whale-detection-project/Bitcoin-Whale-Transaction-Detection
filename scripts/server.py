from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from datetime import datetime
import joblib
import pandas as pd
import logging, asyncio, websockets, json
from contextlib import asynccontextmanager, suppress  # ⬅️ Lifespan용

# ──────────────────────────────── 로깅 ──────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ──────────────────────────────── 모델 & DB ─────────────────────────── #
model = joblib.load("model/dog.joblib")
LABELS = {0: "normal", 1: "less_output_whale", 2: "less_input_whale", 3: "fast_transfer_whale"}

client = MongoClient("mongodb://localhost:27017")
db = client["whale_detection"]
collection = db["detections"]

# ──────────────────────────────── 전처리 ─────────────────────────────── #
def preprocess(tx: dict) -> pd.DataFrame:
    df = pd.DataFrame([tx])
    df["has_zero_output"] = df["has_zero_output"].astype(int)
    feats = [
        "input_count", "output_count", "total_input_value",
        "max_input_value", "max_output_value", "max_output_ratio",
        "fee_per_max_ratio", "has_zero_output",
    ]
    return df[feats]

# ──────────────────────────────── WS 리스너 ─────────────────────────── #
async def listen_to_bitcoin_ws():
    uri = "wss://ws.blockchain.info/inv"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"op": "unconfirmed_sub"}))
        logger.info("🟢 비트코인 WS 연결")

        async for msg in ws:
            data = json.loads(msg).get("x", {})
            inputs, outputs = data.get("inputs", []), data.get("out", [])
            if not inputs or not outputs:
                continue

            tx = {
                "input_count": len(inputs),
                "output_count": len(outputs),
                "total_input_value": sum(i.get("prev_out", {}).get("value", 0) for i in inputs),
                "max_input_value": max(i.get("prev_out", {}).get("value", 0) for i in inputs),
                "max_output_value": max(o.get("value", 0) for o in outputs),
                "max_output_ratio": max(o.get("value", 0) for o in outputs) /
                                    sum(o.get("value", 1) for o in outputs),
                "fee_per_max_ratio": data.get("fee", 0) /
                                     (max(o.get("value", 1) for o in outputs) or 1),
                "has_zero_output": any(o.get("value", 1) == 0 for o in outputs),
                "timestamp": datetime.utcnow().isoformat(),
            }

            pred_code = int(model.predict(preprocess(tx))[0])
            tx["whale_type"] = LABELS.get(pred_code, f"type_{pred_code}")

            if tx["whale_type"] != "normal":          # ✅ 고래만 저장
                collection.insert_one(tx)
                logger.info(f"🐳 Whale saved → {tx}")
            else:
                logger.debug("Normal tx skipped")

# ──────────────────────────────── Lifespan ─────────────────────────── #
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(listen_to_bitcoin_ws())
    try:
        yield                                  # ⬅️ startup 끝
    finally:                                   # ⬅️ shutdown 시작
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

app = FastAPI(lifespan=lifespan)

# ───────────────────────────── API 엔드포인트 ────────────────────────── #
@app.get("/api/detected_whales")
def get_detected_whales():
    docs = list(collection.find({}, {"_id": 0}))
    return JSONResponse(content=docs)
