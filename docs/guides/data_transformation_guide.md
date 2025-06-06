# 🔄 데이터 변환 가이드: 원본 → 최적화 데이터셋

## 📊 개요

이 문서는 원본 비트코인 트랜잭션 데이터(`1000btc.csv`)에서 고래 분류용 최적화 데이터셋(`optimized_whale_dataset.csv`)으로의 변환 과정을 상세히 설명합니다.

---

## 🏗️ 비트코인 트랜잭션 구조와 데이터 연관관계

### 💰 비트코인 트랜잭션 기본 구조

```
트랜잭션 = {
    입력들(Inputs) → [Address1: 500 BTC, Address2: 300 BTC, ...]
                   ↓
    출력들(Outputs) → [Address3: 200 BTC, Address4: 150 BTC, Address5: 449 BTC]
                    ↓
    수수료(Fee) = 총입력 - 총출력 = 1 BTC
}
```

### 📋 원본 데이터 컬럼 상세 설명

| 컬럼명               | 설명                     | 트랜잭션 연관도 | 비즈니스 의미                        |
| -------------------- | ------------------------ | --------------- | ------------------------------------ |
| `tx_hash`            | 트랜잭션 고유 해시값     | **1.0**         | 블록체인상 트랜잭션 식별자           |
| `block_timestamp`    | 블록 생성 시간           | **1.0**         | 트랜잭션 발생 시점                   |
| `input_count`        | 입력 주소 개수           | **1.0**         | 얼마나 많은 주소에서 돈을 모았는가   |
| `output_count`       | 출력 주소 개수           | **1.0**         | 얼마나 많은 주소로 돈을 보냈는가     |
| `total_input_value`  | 총 입력 금액 (satoshi)   | **1.0**         | 모든 입력의 합계                     |
| `total_output_value` | 총 출력 금액 (satoshi)   | **1.0**         | 모든 출력의 합계                     |
| `fee`                | 수수료 (satoshi)         | **1.0**         | `total_input - total_output`         |
| `max_output_value`   | 최대 출력 금액 (satoshi) | **1.0**         | 가장 큰 출력 금액                    |
| `max_input_value`    | 최대 입력 금액 (satoshi) | **1.0**         | 가장 큰 입력 금액                    |
| `max_output_ratio`   | 최대 출력 비율           | **0.9**         | `max_output / total_output` (집중도) |
| `fee_per_max_ratio`  | 수수료/최대출력 비율     | **0.7**         | 파생 지표                            |

### 🎯 거래 행동 패턴 해석

#### 1. **수집형 고래** (Collector Whale)

```
Input 많음 + Output 적음 = 여러 곳에서 모아서 한 곳으로
[A:100₿][B:200₿][C:300₿] → [D:599₿] + 1₿ 수수료
```

- **연관도**: `input_count` (1.0), `output_count` (1.0)
- **의미**: 분산된 자산을 집중시키는 패턴

#### 2. **분산형 고래** (Distributor Whale)

```
Input 적음 + Output 많음 = 한 곳에서 여러 곳으로
[A:1000₿] → [B:200₿][C:300₿][D:100₿][E:399₿] + 1₿ 수수료
```

- **연관도**: `output_count` (1.0), `max_output_ratio` (0.9)
- **의미**: 대량 자산을 분산시키는 패턴

#### 3. **급행형 고래** (Express Whale)

```
높은 수수료 = 빠른 처리 원함
[A:1000₿] → [B:995₿] + 5₿ 수수료 (일반적 0.1₿의 50배!)
```

- **연관도**: `fee` (1.0)
- **의미**: 속도를 위해 높은 수수료 지불

#### 4. **집중형 고래** (Focused Whale)

```
집중도 높음 = 대부분을 한 곳으로
[A:1000₿] → [B:999₿][C:0.9₿] + 0.1₿ 수수료 (집중도: 99.9%)
```

- **연관도**: `max_output_ratio` (1.0)
- **의미**: 대부분 금액을 한 주소로 집중

#### 5. **거대형 고래** (Mega Whale)

```
거래량 자체가 압도적
[A:10000₿] → [B:9999₿] + 1₿ 수수료
```

- **연관도**: `total_input_value`, `total_output_value` (1.0)
- **의미**: 시장에 큰 영향을 줄 수 있는 규모

---

## 🔄 데이터 변환 과정

### Step 1: 데이터 로드 및 단위 변환

```python
# 원본 데이터 로드
df = pd.read_csv('data/1000btc.csv')

# Satoshi → BTC 변환 (1 BTC = 100,000,000 satoshi)
df['total_output_btc'] = df['total_output_value'] / 100000000
df['total_input_btc'] = df['total_input_value'] / 100000000
df['fee_btc'] = df['fee'] / 100000000
df['max_output_btc'] = df['max_output_value'] / 100000000
```

### Step 2: 피처 엔지니어링

```python
# 총 거래량 계산 (Input과 Output 중 최대값)
df['total_volume_btc'] = df[['total_input_btc', 'total_output_btc']].max(axis=1)

# 집중도 명확화 (기존 컬럼 이름 변경)
df['concentration'] = df['max_output_ratio']
```

### Step 3: 거래 행동 패턴 분류

```python
def classify_whale_behavior(row):
    # 분류 기준 임계값
    volume_p99 = df['total_volume_btc'].quantile(0.99)  # 상위 1%
    volume_median = df['total_volume_btc'].median()     # 중앙값
    input_p80 = df['input_count'].quantile(0.80)       # 상위 20%
    output_p85 = df['output_count'].quantile(0.85)     # 상위 15%
    fee_p95 = df['fee_btc'].quantile(0.95)             # 상위 5%

    # 우선순위 기반 분류
    if row['total_volume_btc'] >= volume_p99:
        return 4  # 거대형 고래
    elif row['fee_btc'] >= fee_p95:
        return 2  # 급행형 고래
    elif row['input_count'] >= input_p80 and row['output_count'] <= 2:
        return 0  # 수집형 고래
    elif row['output_count'] >= output_p85 and row['concentration'] < 0.8:
        return 1  # 분산형 고래
    elif row['concentration'] > 0.99 and row['total_volume_btc'] > volume_median:
        return 3  # 집중형 고래
    else:
        return 3  # 집중형 고래 (기본값)

df['whale_class'] = df.apply(classify_whale_behavior, axis=1)
```

### Step 4: 최종 데이터셋 구성

```python
# 최종 피처 선택
feature_cols = [
    'total_volume_btc',    # 총 거래량 (연관도: 1.0)
    'input_count',         # 입력 개수 (연관도: 1.0)
    'output_count',        # 출력 개수 (연관도: 1.0)
    'concentration',       # 집중도 (연관도: 0.9)
    'fee_btc'              # 수수료 (연관도: 1.0)
]

X = df[feature_cols]
y = df['whale_class']
```

---

## 📊 변환 결과 비교

### 원본 데이터 (1000btc.csv)

```
컬럼 수: 11개
데이터 수: 888,942건
형태: Raw blockchain data
단위: Satoshi (1₿ = 100,000,000 satoshi)
목적: 일반적인 분석용
```

### 최적화 데이터셋 (optimized_whale_dataset.csv)

```
컬럼 수: 8개 (피처 5개 + 라벨 3개)
데이터 수: 888,942건
형태: Feature engineered data
단위: BTC (사용자 친화적)
목적: Random Forest 거래 행동 패턴 분류
```

---

## 🔍 컬럼 변환 매핑

| 원본 컬럼            | 변환 후            | 변환 방법               | 연관도  |
| -------------------- | ------------------ | ----------------------- | ------- |
| `tx_hash`            | **제거**           | 식별자라서 ML에 불필요  | -       |
| `block_timestamp`    | **제거**           | 시간 정보 불필요        | -       |
| `input_count`        | `input_count`      | 그대로 유지             | **1.0** |
| `output_count`       | `output_count`     | 그대로 유지             | **1.0** |
| `total_input_value`  | `total_volume_btc` | max(input,output) / 1e8 | **1.0** |
| `total_output_value` | `total_volume_btc` | max(input,output) / 1e8 | **1.0** |
| `fee`                | `fee_btc`          | / 100000000             | **1.0** |
| `max_output_value`   | **제거**           | 집중도 계산에만 사용    | **1.0** |
| `max_input_value`    | **제거**           | 사용 안함               | **1.0** |
| `max_output_ratio`   | `concentration`    | 이름 변경               | **0.9** |
| `fee_per_max_ratio`  | **제거**           | 사용 안함               | **0.7** |
| **신규**             | `whale_class`      | 행동 패턴 분류 (0-4)    | **1.0** |
| **신규**             | `class_name`       | 한글 라벨명             | **1.0** |
| **신규**             | `class_weight`     | 최적 가중치             | **1.0** |

---

## 🎯 최적화 효과

### 1. **데이터 품질 향상**

- ✅ **단위 통일**: Satoshi → BTC (해석 용이)
- ✅ **노이즈 제거**: 불필요한 컬럼 제거
- ✅ **피처 집중**: 거래 행동 패턴 관련 피처만 선택

### 2. **머신러닝 성능 향상**

- ✅ **클래스 라벨링**: 실제 거래 행동 패턴 기반
- ✅ **Class Weight 최적화**: 희귀도 기반 가중치 (16.9% 성능 향상)
- ✅ **과적합 방지**: 핵심 피처만 선택

### 3. **비즈니스 가치 증대**

- ✅ **해석 가능성**: 거래 패턴별 명확한 의미
- ✅ **실용성**: Random Forest 모델에 최적화
- ✅ **확장성**: 새로운 데이터 적용 가능

---

## 📈 사용 예시

### 데이터 로드

```python
import pandas as pd

# 최적화된 데이터셋 로드
df = pd.read_csv('analysis/step1_results/class_weight_results/optimized_whale_dataset.csv')

# 피처와 라벨 분리
features = ['total_volume_btc', 'input_count', 'output_count', 'concentration', 'fee_btc']
X = df[features]
y = df['whale_class']
class_weights = df.set_index('whale_class')['class_weight'].drop_duplicates().to_dict()
```

### 모델 훈련

```python
from sklearn.ensemble import RandomForestClassifier

# 최적 설정으로 모델 훈련
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight=class_weights,  # 최적화된 가중치 사용
    max_depth=8,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features='sqrt'
)

model.fit(X, y)
```

---

## 🔧 트러블슈팅

### Q1: 왜 일부 컬럼이 제거되었나요?

**A**: 머신러닝 모델의 성능과 해석 가능성을 위해 노이즈가 될 수 있는 컬럼들을 제거했습니다.

- `tx_hash`: 식별자로 패턴 학습에 무의미
- `block_timestamp`: 시간 정보는 이번 분석에서 불필요
- `max_output_value`, `max_input_value`: 집중도 계산에만 사용

### Q2: 연관도 0~1은 어떻게 계산했나요?

**A**: 비트코인 트랜잭션 구조와의 직접적 연관성을 기준으로 평가:

- **1.0**: 트랜잭션에서 직접 추출되는 값
- **0.9**: 직접 값들로 계산된 비율 (max_output_ratio)
- **0.7**: 2차 파생 지표 (fee_per_max_ratio)

### Q3: 원본 데이터도 필요한가요?

**A**: 용도에 따라 다릅니다:

- **거래 행동 패턴 분류**: 최적화된 데이터셋 사용
- **시계열 분석**: 원본 데이터의 `block_timestamp` 필요
- **트랜잭션 추적**: 원본 데이터의 `tx_hash` 필요

---

## 📝 결론

이 변환 과정을 통해 **원본 블록체인 데이터**를 **Random Forest 기반 고래 거래 행동 패턴 분류**에 최적화된 형태로 성공적으로 변환했습니다.

**핵심 성과**:

- 📊 **16.9% 성능 향상** (F1-Score: 0.5081)
- 🎯 **실제 거래 패턴 기반** 분류 체계 구축
- 🚀 **즉시 사용 가능한** 최적화된 데이터셋 제공
