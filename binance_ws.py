import websocket
import json
from detectors.live_detector import detect_anomaly

def on_message(ws, message):
    msg = json.loads(message)
    if "k" not in msg["data"]:
        return

    kline = msg["data"]["k"]
    if not kline["x"]:  # 캔들 닫힘이 아닐 경우 무시
        return

    candle = [
        float(kline["o"]),
        float(kline["h"]),
        float(kline["l"]),
        float(kline["c"]),
        float(kline["v"]),
    ]
    detect_anomaly(candle)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    payload = {
        "method": "SUBSCRIBE",
        "params": ["btcusdt@kline_5m"],
        "id": 1
    }
    ws.send(json.dumps(payload))

if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/ws/btcusdt@kline_5m",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open,
    )
    ws.run_forever()
