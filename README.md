# 🐋 비트코인 고래 거래 이상탐지 프로젝트

LSTM을 활용한 비트코인 고래 거래의 이상 패턴 탐지 시스템입니다. K-Means 클러스터링과 시계열 분석을 통해 수상한 거래 패턴을 식별하고 분류합니다.

## 📋 프로젝트 개요

### 주요 목표

- **K-Means 클러스터링**: 고래 거래 유형 분류 및 패턴 분석
- **LSTM 이상탐지**: 시계열 기반 거래 이상 패턴 탐지
- **실시간 모니터링**: 지속적인 거래 패턴 감시 시스템 구축

### 핵심 특징

- 💰 1,000 BTC 이상 대규모 거래 집중 분석
- 🤖 AI 기반 복합 패턴 탐지
- 📊 실시간 시각화 및 알림 시스템
- 🔍 다차원 위험도 평가

## 📁 프로젝트 구조

```
LSTM_Crypto_Anomaly_Detection/
├── data/                          # 데이터 파일
│   └── 1000btc.csv               # 비트코인 고래 거래 데이터
├── analysis/                      # 분석 모듈
│   ├── exploratory/              # 탐색적 데이터 분석
│   │   ├── core_eda.py          # 핵심 EDA 스크립트
│   │   ├── simple_validation.py  # 간단 검증 스크립트
│   │   └── results/             # 분석 결과
│   ├── preprocessing/            # 데이터 전처리
│   └── modeling/                # 모델링
├── config/                       # 설정 파일
│   └── config.yaml              # 프로젝트 설정
├── docs/                         # 문서
│   ├── core_eda_guide.md        # 핵심 EDA 가이드
│   └── why_we_need_this_analysis.md  # 분석 필요성 설명
├── logs/                         # 로그 파일
├── scripts/                      # 실행 스크립트
├── results/                      # 최종 결과
└── requirements.txt              # 필수 패키지
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/username/LSTM_Crypto_Anomaly_Detection.git
cd LSTM_Crypto_Anomaly_Detection

# 패키지 설치
pip install -r requirements.txt
```

### 2. 핵심 EDA 실행

```bash
# 모델링에 필요한 핵심 분석 실행
python analysis/exploratory/core_eda.py
```

### 3. 간단 검증 실행 (발표용)

```bash
# 대중이 이해하기 쉬운 검증 분석
python analysis/exploratory/simple_validation.py
```

### 4. 결과 확인

```bash
# 핵심 EDA 결과
cat analysis/exploratory/results/core_eda_report.txt

# 발표용 요약 확인
cat analysis/exploratory/simple_validation/presentation_summary.txt
```

## 📊 핵심 EDA 분석 결과

### 분석 대상

- **총 거래**: 888,942건 (1,000 BTC 이상)
- **이상 거래율**: 83.2%
- **최적 클러스터 수**: 2개

### 상세 분석 결과

#### 1. 특성 상관관계 분석

- ✅ 다중공선성 없음 (모든 상관계수 < 0.7)
- ✅ 4개 특성 모두 클러스터링에 적합

#### 2. 클러스터링 최적화

- 🎯 **최적 클러스터 수**: 2개
- 📈 **실루엣 점수**: 0.683
- 💡 **권장사항**: K-Means 적용 전 특성 정규화 필요

#### 3. 시계열 데이터 품질 확인

- ✅ 모든 특성 정상성 확인 (Stationarity)
- ✅ LSTM 입력 데이터로 적합
- 💡 **권장사항**: 비정상 시계열의 경우 차분 적용

#### 4. 이상탐지 임계값 최적화

- 🎯 **현재 임계값**: 50 (F1-Score: 0.999)
- 🎯 **최적 임계값**: 55.0 (F1-Score: 0.999)
- 💡 **성능**: 매우 높은 정확도 달성

## 🎯 간단 검증 결과 (발표용)

### 핵심 발견사항

- **총 분석 거래**: 888,942건
- **고위험 거래**: 115,516건 (13.0%)
- **새벽 거래**: 135,292건 (15.2%)

### 수상한 패턴들

- 🚨 **극도 집중 (99% 이상)**: 687,389건 (77.3%)
- 🚨 **거의 무료 수수료**: 808,440건 (90.9%)
- 🚨 **새벽 시간대 (2-5시)**: 135,292건 (15.2%)
- 🚨 **광범위 분산 (100곳 이상)**: 6,527건 (0.7%)

### 거래 유형 분류

- **정상적 거래**: 15,512건 (1.7%)
- **집중형 수상 거래**: 763,932건 (85.9%)
- **저수수료 수상 거래**: 96,999건 (10.9%)
- **분산형 수상 거래**: 6,527건 (0.7%)

### 발표용 핵심 메시지

#### 🎤 30초 버전

"88만 건 고래 거래 중 13.0%가 고위험 패턴을 보였습니다."

#### 🎤 3분 버전

- 전체 거래의 77%가 한 곳으로만 극도 집중
- 91%가 거의 무료 수수료 사용
- 15%가 새벽 시간대에 집중
- → **AI 기반 복합 분석 필요성 입증**

#### 🎤 질의응답 대비

- **Q: 정말 문제인가?** → A: 전체의 13%가 고위험 패턴
- **Q: AI가 꼭 필요한가?** → A: 복합 패턴은 단순 규칙으로 불가능
- **Q: 효과가 있나?** → A: 99.9% 정확도로 탐지 가능

## 🐋 고래 유형 분류

### 분류 기준

| 유형        | 집중도 | 수수료 | 시간대    | 위험도    |
| ----------- | ------ | ------ | --------- | --------- |
| 정상 고래   | < 80%  | 정상   | 정상 시간 | 낮음      |
| 집중형 고래 | > 95%  | 낮음   | 새벽      | 높음      |
| 분산형 고래 | < 50%  | 높음   | 다양      | 중간      |
| 의심 고래   | > 99%  | 극저   | 새벽      | 매우 높음 |

## 📈 다음 단계

### 1. K-Means 클러스터링 구현

- **클러스터 수**: 2개
- **사용 특성**: 4개 (모든 특성)
- **전처리**: StandardScaler 적용

### 2. LSTM 이상탐지 모델 개발

- **시퀀스 길이**: 7-14일
- **이상 임계값**: 55.0
- **검증 방법**: 시계열 교차검증

### 3. 모델 통합 및 평가

- **앙상블 방법**: K-Means + LSTM
- **평가 지표**: F1-Score, 정밀도, 재현율
- **실시간 모니터링**: 스트리밍 데이터 처리

## 📖 사용자 가이드

### 초보자용

1. 📚 [핵심 EDA 가이드](docs/core_eda_guide.md) 읽기
2. 📚 [분석 필요성 설명](docs/why_we_need_this_analysis.md) 확인
3. 🔧 환경 설정 후 핵심 EDA 실행
4. 📊 결과 해석 및 다음 단계 진행

### 커스터마이징

- `config/config.yaml`: 분석 매개변수 조정
- 클러스터 수, 임계값, 시각화 옵션 변경 가능

### 분석 결과 활용

- K-Means 클러스터링을 위한 최적 매개변수 제공
- LSTM 모델을 위한 데이터 품질 확인
- 이상탐지 임계값 최적화 결과 활용

## 🤝 기여 방법

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.

## 📞 문의

- 이메일: contact@example.com
- 이슈: [GitHub Issues](https://github.com/username/LSTM_Crypto_Anomaly_Detection/issues)
