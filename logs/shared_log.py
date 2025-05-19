# shared_log.py
from collections import deque

log_buffer = deque(maxlen=120)  # 최근 120개 결과 저장
