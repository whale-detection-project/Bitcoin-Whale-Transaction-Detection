# api/server.py
import logging
from threading import Thread
from fastapi import FastAPI
from service.websocket_handler import WebSocketHandler
from core.model_loader import load_models

logger = logging.getLogger(__name__)

class APIServer:
    def __init__(self):
        self.app = FastAPI()
        self.scaler, self.xgb_model, self.pca = load_models()
        self.websocket_handler = WebSocketHandler(self.scaler, self.xgb_model, self.pca)
        self.setup_routes()

    def setup_routes(self):
        @self.app.on_event("startup")
        async def startup_event():
            Thread(target=self.start_websocket, daemon=True).start()

        @self.app.get("/detect")
        def get_latest_result():
            return self.websocket_handler.get_latest_result()

    def start_websocket(self):
        self.websocket_handler.run_websocket()

    def get_app(self):
        return self.app
