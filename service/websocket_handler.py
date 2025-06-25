import websocket
import json
import logging
import time
import numpy as np
import torch
from threading import Lock
from core.config import collection, WEBSOCKET_URL, FEATURES
from core.transaction_processor import process_transaction
from datetime import datetime
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class WebSocketHandler:
    def __init__(self, scaler_dict, mlp_model, pca, centers, tau):
        self.scaler     = scaler_dict      # {'mean': np.array, 'std': np.array}
        self.model      = mlp_model        # PyTorch 모델
        self.pca        = pca              # PCA (optional)
        self.centers    = centers          # (4, D)
        self.tau        = tau              # float

        self.latest_result = {}
        self.lock = Lock()

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            tx = data.get("x", {})

            total_input_value = sum(
                i.get("prev_out", {}).get("value", 0) for i in tx.get("inputs", [])
            ) / 1e8
            if total_input_value < 200:
                return

            tx_processed = process_transaction(tx)
            tx_data = {k: v for k, v in tx_processed.items() if k in FEATURES}

            # 전처리
            X_np = np.array([[np.log1p(tx_data[f]) for f in FEATURES]])
            X_scaled = (X_np - self.scaler["mean"]) / self.scaler["std"]

            # 거리 기반 Unknown 판별
            d_min = cdist(X_scaled, self.centers).min(axis=1)[0]
            if d_min > self.tau:
                cluster = 4  # Unknown
            else:
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                    out = self.model(X_tensor)
                    cluster = int(torch.argmax(out, dim=1).item())

            # PCA 임베딩 (optional)
            embedding = self.pca.transform(X_scaled)[0].tolist()

            result = {
                "predicted_cluster": cluster,
                "pca_embedding": embedding,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with self.lock:
                self.latest_result.clear()
                self.latest_result.update(result)

            if hasattr(self, "callback"):
                self.callback({**result, **tx_processed})

            collection.insert_one({**result, **tx_processed})

            logger.info(
                f"🚨 고래 거래 감지 → 클러스터 {cluster}, "
                f"총 입력: {tx_processed['total_input_value']:.2f} BTC\n"
                f"입력수: {tx_processed['input_count']}개, 출력수: {tx_processed['output_count']}개, "
                f"max_output_ratio: {tx_processed['max_output_ratio']:.4f}, "
                f"fee_per_max_ratio: {tx_processed['fee_per_max_ratio']:.8f}, "
                f"max_input_ratio: {tx_processed['max_input_ratio']:.4f}"
            )

        except Exception as e:
            logger.error(f"❌ WebSocket 처리 오류: {e}")

    def on_open(self, ws):
        logger.info("🔗 Blockchain.com WebSocket 연결됨")
        ws.send(json.dumps({"op": "unconfirmed_sub"}))
        self.retry_count = 0

    def on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"🔌 WebSocket 연결 종료됨: code={close_status_code}, msg={close_msg}")

    def on_error(self, ws, error):
        logger.error(f"❌ WebSocket 오류 발생: {error}")

    def set_callback(self, callback):
        self.callback = callback

    def _connect_once(self):
        ws = websocket.WebSocketApp(
            WEBSOCKET_URL,
            on_open=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message,
        )
        ws.run_forever(ping_interval=30, ping_timeout=10)

    def run_websocket(self, base_delay=5, max_delay=300):
        self.retry_count = 0
        while True:
            try:
                logger.info("🔁 WebSocket 연결 시도 중...")
                self._connect_once()
            except Exception as e:
                logger.error(f"❌ WebSocket 예외 발생: {e}")

            delay = min(base_delay * (2 ** self.retry_count), max_delay)
            logger.info(f"⏳ {delay}초 후 재연결 시도 예정...")
            time.sleep(delay)
            self.retry_count += 1

    def get_latest_result(self):
        with self.lock:
            return (
                self.latest_result.copy()
                if self.latest_result
                else {"message": "아직 탐지된 고래 거래 없음"}
            )
