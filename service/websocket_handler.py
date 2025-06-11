# service/websocket_handler.py
import websocket
import json
import logging
from threading import Lock
from core.config import collection, WEBSOCKET_URL, MIN_INPUT_VALUE
from core.model_loader import predict_cluster
from core.transaction_processor import process_transaction

logger = logging.getLogger(__name__)

class WebSocketHandler:
    def __init__(self, scaler, xgb_model, pca):
        self.scaler = scaler
        self.xgb_model = xgb_model
        self.pca = pca
        self.latest_result = {}
        self.lock = Lock()

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            tx = data.get("x", {})

            total_input_value = sum(i.get("prev_out", {}).get("value", 0) for i in tx.get("inputs", [])) / 1e8
            if total_input_value < MIN_INPUT_VALUE:
                return

            tx_processed = process_transaction(tx)
            tx_data = {k: v for k, v in tx_processed.items() if k != "total_input_value"}

            result = predict_cluster(tx_data, self.scaler, self.xgb_model, self.pca)

            if "error" in result:
                logger.error(f"❌ 예측 실패: {result['error']}")
                logger.error(f"입력 tx_data: {tx_data}")
                return

            with self.lock:
                self.latest_result.clear()
                self.latest_result.update(result)
            
            if hasattr(self, "callback"):
                self.callback({**result, **tx_processed})


            collection.insert_one({**result, **tx_processed})

            logger.info(
                f"🚨 고래 거래 감지 → 클러스터 {result['predicted_cluster']}, 총 입력: {tx_processed['total_input_value']:.2f} BTC\n"
                f"입력수: {tx_processed['input_count']}개, 출력수: {tx_processed['output_count']}개, "
                f"max_output_ratio: {tx_processed['max_output_ratio']:.4f}, "
                f"fee_per_max_ratio: {tx_processed['fee_per_max_ratio']:.4f}, "
                f"max_input_ratio: {tx_processed['max_input_ratio']:.4f}"
            )

        except Exception as e:
            logger.error(f"❌ WebSocket 처리 오류: {e}")

    def on_open(self, ws):
        logger.info("🔗 Blockchain.com WebSocket 연결됨")
        ws.send(json.dumps({"op": "unconfirmed_sub"}))

    def on_close(self, ws, close_status_code, close_msg):
        logger.info(f"🔌 연결 종료: code={close_status_code}, msg={close_msg}")

        
    def set_callback(self, callback):
        self.callback = callback
   

    def on_error(self, ws, error):
        logger.error(f"❌ 오류 발생: {error}")

    def run_websocket(self):
        ws = websocket.WebSocketApp(
            WEBSOCKET_URL,
            on_open=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message
        )
        ws.run_forever(ping_interval=30, ping_timeout=10)

    def get_latest_result(self):
        with self.lock:
            return self.latest_result.copy() if self.latest_result else {"message": "아직 탐지된 고래 거래 없음"}
