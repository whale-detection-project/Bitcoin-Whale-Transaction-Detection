import logging
import asyncio
from threading import Thread
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from service.websocket_handler import WebSocketHandler
from core.config import collection
from core.model_loader import load_models

logger = logging.getLogger(__name__)

class APIServer:
    def __init__(self):
        self.app = FastAPI(
            title="Whale Detection API",
            description="""
            이 API는 블록체인 네트워크에서 고래 거래를 감지하고,
            실시간으로 클러스터링 결과를 제공합니다.
            - WebSocket을 통해 실시간 거래 데이터를 수신합니다.
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
        self.subscribers = set()  # (queue, min_input_value) 튜플로 저장
        self.setup_routes()

    def setup_routes(self):
        @self.app.on_event("startup")
        async def startup_event():
            Thread(target=self.start_websocket, daemon=True).start()

        @self.app.get(
            "/api/stream",
            summary="SSE 실시간 고래 탐지 알림",
            description="""
            클라이언트가 SSE를 통해 실시간으로 고래 거래 알림을 받을 수 있습니다.
            쿼리파라미터 `min_input_value`를 통해 알림 최소 기준값을 지정할 수 있습니다.
            예시 메시지 형식:
            {
              "cluster": 3,
              "btc": 2450.12,
              "input_count": 2,
              "output_count": 5,
              "max_output_ratio": 0.76,
              "max_input_ratio": 0.95,
              "fee_per_max_ratio": 0.012,
              "timestamp": "2025-06-12T09:45:00"
            }
            """
        )
        async def stream(request: Request, min_input_value: float = 1000):
            async def event_generator():
                queue = asyncio.Queue()
                subscriber = (queue, min_input_value)
                self.subscribers.add(subscriber)
                logger.info(f"🟢 SSE 연결됨 (min_input_value={min_input_value})")

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
                    logger.info("🔴 SSE 연결 해제됨")

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        @self.app.get(
            "/api/logs",
            summary="고래 탐지 로그 10건 조회",
            description="저장된 최근 10개의 로그를 반환합니다.",
            responses={
                200: {
                    "description": "저장된 최근 10개의 로그를 반환합니다.",
                    "content": {
                        "application/json": {
                            "example": {
                                "logs": [
                                    {
                                        "_id": "60c72b2f9b1e8b001c8e4d3a",
                                        "predicted_cluster": 1,
                                        "pca_embedding": [-0.2146, -0.3442],
                                        "timestamp": "2025-06-11 17:34:03",
                                        "input_count": 2,
                                        "output_count": 2,
                                        "max_output_ratio": 0.9962,
                                        "fee_per_max_ratio": 1.12e-8,
                                        "max_input_ratio": 0.9981,
                                        "total_input_value": 2450.12
                                    }
                                ]
                            }
                        }
                    }
                },
                500: {"description": "로그 조회 실패"}
            }
        )
        def get_logs(limit: int = 10):
            try:
                cursor = collection.find().sort("_id", -1).limit(limit)
                logs = list(cursor)
                for log in logs:
                    log["_id"] = str(log["_id"])
                return JSONResponse(content={"logs": logs})
            except Exception as e:
                logger.error(f"❌ 로그 조회 오류: {e}")
                return JSONResponse(status_code=500, content={"error": "로그 조회 실패"})

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
                "timestamp": result.get('timestamp')
            }
            logger.info(f"📣 SSE 브로드캐스트: {message}")

            for queue, min_input_value in self.subscribers.copy():
                try:
                    if result["total_input_value"] >= min_input_value:
                        queue.put_nowait(message)
                except Exception as e:
                    logger.warning(f"SSE 전송 실패: {e}")

        self.websocket_handler.set_callback(on_whale_detected)
        self.websocket_handler.run_websocket()

    def get_app(self):
        return self.app
