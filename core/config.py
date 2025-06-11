# core/config.py
import logging
import os
import sys
from pymongo import MongoClient

# ✅ 중복 핸들러 제거 후 로깅 설정
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
handler.setFormatter(formatter)
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)

# ✅ MongoDB 설정

mongo_uri = os.getenv("MONGODB_URI", "mongodb://mongodb:27017")
mongo_client = MongoClient(mongo_uri)
db = mongo_client["whale_detection"]
collection = db["whale_logs"]

# ✅ 상수 설정
FEATURES = ['input_count', 'output_count', 'max_output_ratio', 'fee_per_max_ratio', 'max_input_ratio']
WEBSOCKET_URL = "wss://ws.blockchain.info/inv"
MIN_INPUT_VALUE = 1000  # BTC
