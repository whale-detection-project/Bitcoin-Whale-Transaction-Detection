# api/server.py
import logging
import asyncio
from threading import Thread
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from service.websocket_handler import WebSocketHandler
from core.config import collection
from core.model_loader import load_models

logger = logging.getLogger(__name__)

class APIServer:
    def __init__(self):
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], #나중에 수정
            allow_methods=["*"],
            allow_headers=["*"]
        )
        self.scaler, self.xgb_model, self.pca = load_models()
        self.websocket_handler = WebSocketHandler(self.scaler, self.xgb_model, self.pca)
        self.subscribers = set()
        self.setup_routes()

    def setup_routes(self):
        @self.app.on_event("startup")
        async def startup_event():
            Thread(target=self.start_websocket, daemon=True).start()

        # ✅ SSE 알림 엔드포인트
        @self.app.get("/stream")
        async def stream(request: Request):
            async def event_generator():
                queue = asyncio.Queue()
                self.subscribers.add(queue)
                logger.info("🟢 클라이언트 SSE 연결됨")
                try:
                    while True:
                        if await request.is_disconnected():
                            break
                        try:
                            message = await asyncio.wait_for(queue.get(), timeout=15.0)
                            yield f"data: {message}\n\n"
                        except asyncio.TimeoutError:
                            yield ": keep-alive\n\n"  # SSE 연결 유지용
                finally:
                    self.subscribers.discard(queue)
                    logger.info("🔴 클라이언트 SSE 연결 해제됨")

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        
        # ✅ 최신 로그 n개 조회 엔드포인트
        @self.app.get("/logs")
        def get_logs(limit: int = 10):
            try:
            # 최신 로그 기준으로 최근 N개 조회
                cursor = collection.find().sort("_id", -1).limit(limit)
                logs = list(cursor)
                for log in logs:
                    log["_id"] = str(log["_id"])  # ObjectId를 문자열로 변환 (JSON 직렬화 위해)
                return JSONResponse(content={"logs": logs})
            except Exception as e:
                logger.error(f"❌ 로그 조회 오류: {e}")
                return JSONResponse(status_code=500, content={"error": "로그 조회 실패"})

    # api/server.py
import logging
import asyncio
from threading import Thread
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from service.websocket_handler import WebSocketHandler
from core.config import collection
from core.model_loader import load_models

logger = logging.getLogger(__name__)

class APIServer:
    def __init__(self):
        self.app = FastAPI()
        self.scaler, self.xgb_model, self.pca = load_models()
        self.websocket_handler = WebSocketHandler(self.scaler, self.xgb_model, self.pca)
        self.subscribers = set()
        self.setup_routes()

    def setup_routes(self):
        @self.app.on_event("startup")
        async def startup_event():
            Thread(target=self.start_websocket, daemon=True).start()

        # ✅ SSE 스트리밍 알림 엔드포인트
        @self.app.get("/stream")
        async def stream(request: Request):
            async def event_generator():
                queue = asyncio.Queue()
                self.subscribers.add(queue)
                logger.info("🟢 클라이언트 SSE 연결됨")
                try:
                    while True:
                        if await request.is_disconnected():
                            break
                        try:
                            message = await asyncio.wait_for(queue.get(), timeout=15.0)
                            yield f"data: {message}\n\n"
                        except asyncio.TimeoutError:
                            yield ": keep-alive\n\n"  # SSE 연결 유지용
                finally:
                    self.subscribers.discard(queue)
                    logger.info("🔴 클라이언트 SSE 연결 해제됨")

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        
        @self.app.get("/logs")
        def get_logs(limit: int = 10):
            try:
            # 최신 로그 기준으로 최근 N개 조회
                cursor = collection.find().sort("_id", -1).limit(limit)
                logs = list(cursor)
                for log in logs:
                    log["_id"] = str(log["_id"])  # ObjectId를 문자열로 변환 (JSON 직렬화 위해)
                return JSONResponse(content={"logs": logs})
            except Exception as e:
                logger.error(f"❌ 로그 조회 오류: {e}")
                return JSONResponse(status_code=500, content={"error": "로그 조회 실패"})

    def start_websocket(self):
        # 고래 탐지 시 → SSE로 메시지 push
        def on_whale_detected(result):
            message = {
                "cluster": result['predicted_cluster'],
                "btc": result['total_input_value'],
                "input_count": result.get('input_count'),
                "output_count": result.get('output_count'),
                "max_output_ratio": result.get('max_output_ratio'),
                "max_input_ratio": result.get('max_input_ratio'),
                "fee_per_max_ratio": result.get('fee_per_max_ratio'),
                "timestamp": result.get('timestamp')  # 있으면 포함
            }
            logger.info(f"📣 SSE 브로드캐스트: {message}")
            for queue in self.subscribers.copy():
                try:
                    queue.put_nowait(message)
                except Exception as e:
                    logger.warning(f"SSE 전송 실패: {e}")

        self.websocket_handler.set_callback(on_whale_detected)
        self.websocket_handler.run_websocket()

    def get_app(self):
        return self.app
