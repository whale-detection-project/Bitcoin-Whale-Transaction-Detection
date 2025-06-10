# main.py
import uvicorn
from api.server import APIServer
import core.config  # 로딩 시 로깅 설정됨

api_server = APIServer()
app = api_server.get_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
