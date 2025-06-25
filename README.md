# 🐋 Bitcoin Whale Transaction Detection - Backend

**비트코인 고래 거래 탐지 시스템 (FastAPI 기반 백엔드)**
온체인 트래널션 데이터를 기반으로 고래의 이상 거래 유형을 실시간으로 분류 및 탐지합니다.

---

## 포함 경로

```
프로젝트/
└── api/               #FastAPI 엔드포인트트
    core/              # 모델 로더, 설정, 스키마
    dataset/           # 학습 및 테스트 데이터셋
    model/             # 학습된 MLP 모델 및 보조 파일
    scripts/           # 학습 및 테스팅 스크립트
    service/           # WebSocket 핸드러
    test/              # 테스트 파일 및 결과과
    main.py/            # 서버 실행
    README.md
```

---

## 🚀 주요 기능

### 고래 유형 자동 분류

* `input_count`, `output_count`, `max_input_ratio`, `max_output_ratio` 기본
* KMeans 로 클러스터링 후, MLP 모델로 실시간 분류 수행

### 실시간 탐지 & 스트림링

* Binance WebSocket 기반 실시간 트랜잭잭션 수신
* 조건변수 (`input_value ≥ X`) 필터링
* SSE(Server-Sent Events)로 프론트엔드 대시보드로 실시간 전송

### RESTful API 제공

* `/api/stream`: SSE 실시간 고래 탐지 이벤트 스트림
* `/api/logs`: MongoDB에서 경로 고래 거래 로그 조회
* `/api/whales`: 특정 금액 이상 고래 거래 필터링 조회
* `/api/address-info`: blockchain.info API를 이용해 주소의 정보 조회

---

## 🐋 고래 유형 정의 (`cluster_label`)

| 라벨 | 설명 |
|------|------|
| 0 | **입력 2+, 출력 5+** → 중간 다수 출력형 / 지갑 리밸런싱 추정 |
| 1 | **단일 입력 → 단일 출력** → 콜드월렛 이체 or 소수 정형 전송 |
| 2 | **입력 20+ → 출력 1~2** → 입력 병합형, Mixing 전 조짐 |
| 3 | **입력 1~2 → 출력 20+** → 다중 분산형, 세탁 / 거래소 출금 의심 |
| 4 | **이상치로 감지된 하위 99.5 이내의 값** → 기존 클러스터명에는 속할 수 있지만 클러스터 밀집도가 떨어지는 유형 |

---

## ⚙️ 실행 방법

### 1. 의존성성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 학습 및 테스팅 (Optional)

```bash
# KMeans + MLP 학습
python scripts/train2.py

# 테스팅 수행
python scripts/test2.py
```

### 3. 배포시 core/config.py MongoDB 설정 필요

### 4. 탐지 기준설정
```config.py
   def on_message(self, ws, message):
        try:
            data = json.loads(message)
            tx = data.get("x", {})

            total_input_value = sum(
                i.get("prev_out", {}).get("value", 0) for i in tx.get("inputs", [])
            ) / 1e8
            if total_input_value < 200: # 탐지 기준
                return
```

### 5. FastAPI 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 📘 API 문서 (Docs)

* Docs: [`/api/docs`](http://localhost:8000/api/docs) 
* Redoc: [`/api/redoc`](http://localhost:8000/api/redoc)
* (배포용) Docs: [`wt-backend.store/api/docs`](https://wt-backend.store/api/docs)
---

## 참고 사항

* 고래 거래 데이터는 MongoDB에 자동 저장됩니다.
* `/model/`은 학습된 MLP 모델(`mlp4_torch.pt`), 스케일러, PCA, KMeans 중심점 등이 포함됩니다.

---

## 🔧 개발 규칙

### 커밋밋 메시지 형식

* 형식: `type: subject`
* 예시: `feat: SSE 고래 알림 엔드포인트 추가`

| 타입       | 설명             |
| -------- | -------------- |
| feat     | 기능 추가          |
| fix      | 버그 수정          |
| docs     | 문서 수정          |
| style    | 스타일(작성법) 수정    |
| refactor | 바이코 변경 (기능 무관) |
| test     | 테스트 추가         |
| chore    | 설정/빌드 관련       |

### 브랜치 규칙

* `main`: 배포용 안정 브랜치
* `develop`: 기능 통합 브랜치
* `feature/*`: 기능 개발 브랜치
* `fix/*`: 버그 수정 브랜치

### PR 규칙

* PR 제목: 커밋 메시지 형식 구조
* PR 본문: 변경 사항 규정
* 발생한 충돌 해결 + 테스트 통과 필요
