import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import websocket
import json
import logging
from logs.log_config import setup_logger
setup_logger()
from detectors.live_detector import detect_anomaly

def on_message(ws, message):
    msg = json.loads(message)

    # 올바른 구조: kline 데이터는 "k" 키에 있음
    if "k" not in msg:
        # print("⚠️ 'k' 키 없음! 메시지:", msg)
        return

    kline = msg["k"]

    if not kline["x"]:  # 캔들 미완성 상태는 무시
        return
    
    logging.info(f"✅ 캔들 수신: {kline}")

    candle = [
        float(kline["o"]),
        float(kline["h"]),
        float(kline["l"]),
        float(kline["c"]),
        float(kline["v"]),
    ]

    detect_anomaly(candle)


def on_error(ws, error):
    print("WebSocket 에러:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket 종료")

def on_open(ws):
    payload = {
        "method": "SUBSCRIBE",
        "params": ["btcusdt@kline_5m"],
        "id": 1
    }
    ws.send(json.dumps(payload))
    print("✅ Binance WebSocket 연결됨 (5분봉)")

if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/ws/btcusdt@kline_5m",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()
