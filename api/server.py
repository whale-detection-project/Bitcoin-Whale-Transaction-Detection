import logging
import json                                
import asyncio                              
import httpx
from threading import Thread
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from service.websocket_handler import WebSocketHandler
from core.config import collection
from core.model_loader import load_models
from core.schemas import WhaleTransactionList, AddressInfo

logger = logging.getLogger(__name__)


class APIServer:
    def __init__(self):
        self.app = FastAPI(
            title="Whale Detection API",
            description="""
            이 API는 블록체인 네트워크에서 고래 거래를 감지하고,
            실시간으로 클러스터링 결과를 제공합니다.
            - WebSocket을 통해 실시간 거래 데이터를 수신합니다.
            - 탐지되는 고래의 최소 거래단위는 200BTC입니다.
            - 거래 데이터를 클러스터링하여 고래 거래를 식별합니다.
            - SSE(Server-Sent Events)를 통해 클라이언트에 실시간 알림을 전송합니다.
            - MongoDB에 거래 로그를 저장합니다.
            """,
            version="1.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc",
            openapi_url="/api/openapi.json"
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        )
        self.scaler, self.xgb_model, self.pca = load_models()
        self.websocket_handler = WebSocketHandler(self.scaler, self.xgb_model, self.pca)
        self.subscribers = set()
        self.main_loop = None                
        self.setup_routes()

    def setup_routes(self):
        @self.app.on_event("startup")
        async def startup_event():
            self.main_loop = asyncio.get_running_loop()    
            Thread(target=self.start_websocket, daemon=True).start()

        @self.app.get(
            "/api/stream",
            summary="SSE 실시간 고래 탐지 알림",
            description=(
            "클라이언트가 SSE를 통해 실시간으로 고래 거래 알림을 받을 수 있습니다.\n\n"
            "쿼리파라미터 `min_input_value`를 통해 알림 최소 기준값을 지정할 수 있습니다.\n\n"
            "### 예시 요청\n"
            "`GET /api/stream?min_input_value=1000`  *(기본값 1000)*\n\n"
            "### 예시 전송 메시지 (JSON)\n"
            "```json\n"
            "data: {\n"
            '  "cluster": 2,\n'
            '  "btc": 1234.56,\n'
            '  "input_count": 3,\n'
            '  "output_count": 4,\n'
            '  "max_output_ratio": 0.78,\n'
            '  "max_input_ratio": 0.91,\n'
            '  "fee_per_max_ratio": 0.000032,\n'
            '  "timestamp": "2025-06-21T16:45:00",\n'
            '  "max_input_address": "1ABCDxyz...",\n'
            '  "max_output_address": "bc1qWErty..."\n'
            "}\n"
            "```"
            )
        )
        async def stream(request: Request, min_input_value: float = 1000):
            async def event_generator():
                queue = asyncio.Queue()
                subscriber = (queue, min_input_value)
                self.subscribers.add(subscriber)
                try:
                    while True:
                        if await request.is_disconnected():
                            break
                        try:
                            message = await asyncio.wait_for(queue.get(), timeout=15.0)
                            yield f"data: {message}\n\n"
                        except asyncio.TimeoutError:
                            yield ": keep-alive\n\n"
                finally:
                    self.subscribers.discard(subscriber)

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        @self.app.get(
            "/api/logs",
            response_model=WhaleTransactionList,
            summary="최신순으로 고래 탐지 로그 N건 조회",
            description="MongoDB에 저장된 최근 N개 고래 거래 로그를 반환합니다. 기본 20건"
        )
        def get_logs(limit: int = 20):
            try:
                cursor = collection.find().sort("_id", -1).limit(limit)
                logs = list(cursor)
                for log in logs:
                    log["_id"] = str(log["_id"])
                return JSONResponse(content={"logs": logs})
            except Exception:
                logger.exception("❌ 로그 조회 오류")
                return JSONResponse(status_code=500, content={"error": "로그 조회 실패"})

        @self.app.get(
            "/api/whales",
            summary="특정 BTC 이상 고래 거래 조회",
            response_model=WhaleTransactionList,
            description="`total_input_value`가 특정값 이상인 고래 거래를 조회합니다."
        )
        def get_whales(min_value: float = 1000.0, limit: int = 10):
            try:
                cursor = collection.find({"total_input_value": {"$gte": min_value}}).sort("_id", -1).limit(limit)
                logs = list(cursor)
                for log in logs:
                    log["_id"] = str(log["_id"])
                return JSONResponse(content={"logs": logs})
            except Exception:
                logger.exception("❌ 고래 거래 조회 오류")
                return JSONResponse(status_code=500, content={"error": "고래 거래 조회 실패"})
            
        @self.app.get(
            "/api/address-info",
            response_model=AddressInfo,
            summary="지갑 주소 정보 조회",
            description="blockchain.info API를 사용해 주소의 총 입출금, 잔고 등을 조회하고 거래소 주소로 추정되는지 여부를 반환합니다."
        )
        async def address_info(address: str):
            try:
                url = f"https://blockchain.info/rawaddr/{address}"
                async with httpx.AsyncClient() as client:
                    res = await client.get(url, timeout=10)
                    if res.status_code != 200:
                        return JSONResponse(status_code=404, content={"error": "주소 조회 실패"})
                    data = res.json()
                    
                    # 사토시 → BTC 변환
                    total_received = data.get("total_received", 0) / 1e8
                    total_sent = data.get("total_sent", 0) / 1e8
                    final_balance = data.get("final_balance", 0) / 1e8
                    n_tx = data.get("n_tx", 0)

                    # 거래소 주소 추정 로직 (간단 기준)
                    is_exchange_like = (
                        n_tx > 1000 and 
                        total_received > 10000 and 
                        final_balance < (total_received * 0.05)  # 잔액이 거의 없음 → hot wallet 형태
                    )

                    return {
                        "address": address,
                        "total_received_btc": total_received,
                        "total_sent_btc": total_sent,
                        "final_balance_btc": final_balance,
                        "tx_count": n_tx,
                        "is_exchange_like": is_exchange_like
                    }

            except Exception:
                logger.exception("❌ 주소 정보 조회 오류")
                return JSONResponse(status_code=500, content={"error": "주소 정보 조회 실패"})

    def start_websocket(self):
        def on_whale_detected(result):
            message = {
                "cluster": result['predicted_cluster'],
                "btc": result['total_input_value'],
                "input_count": result.get('input_count'),
                "output_count": result.get('output_count'),
                "max_output_ratio": result.get('max_output_ratio'),
                "max_input_ratio": result.get('max_input_ratio'),
                "fee_per_max_ratio": result.get('fee_per_max_ratio'),
                "timestamp": result.get('timestamp'),
                "max_input_address": result.get('max_input_address'),
                "max_output_address": result.get('max_output_address')
            }
            json_msg = json.dumps(message, separators=(",", ":"))     

           
            if self.main_loop:
                for queue, min_input_value in self.subscribers.copy():
                    if result["total_input_value"] >= min_input_value:
                        self.main_loop.call_soon_threadsafe(queue.put_nowait, json_msg)

        self.websocket_handler.set_callback(on_whale_detected)
        self.websocket_handler.run_websocket()

    def get_app(self):
        return self.app
