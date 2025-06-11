# 📌 Python 3.9 기반 이미지 사용
FROM python:3.9

# 📌 작업 디렉토리 생성
WORKDIR /app

# 📌 requirements.txt 복사 → 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 📌 전체 프로젝트 코드 복사
COPY . .

# 📌 FastAPI 앱 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
