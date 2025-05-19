from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from shared_log import log_buffer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 원하는 프론트 도메인만 지정
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/logs")
def get_logs():
    return list(log_buffer)
