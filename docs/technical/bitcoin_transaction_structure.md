# 🔗 비트코인 트랜잭션 구조 상세 가이드

## 📊 개요

이 문서는 비트코인 트랜잭션의 기술적 구조와 우리 데이터셋의 각 컬럼이 어떻게 블록체인에서 추출되는지 설명합니다.

---

## 🏗️ 비트코인 트랜잭션 기본 구조

### 🔗 블록체인 레벨 구조

```
블록체인
├── 블록 #800000
│   ├── 블록 헤더
│   │   ├── Previous Block Hash
│   │   ├── Merkle Root
│   │   ├── Timestamp ← block_timestamp 추출
│   │   └── Nonce
│   └── 트랜잭션들
│       ├── 트랜잭션 #1 ← tx_hash 생성
│       ├── 트랜잭션 #2
│       └── 트랜잭션 #3
```

### 💰 개별 트랜잭션 구조

```json
{
  "txid": "a1b2c3d4...",           // ← tx_hash
  "version": 2,
  "locktime": 0,
  "vin": [                         // ← 입력들 (Inputs)
    {
      "txid": "prev_tx_hash",
      "vout": 0,
      "scriptSig": {...},
      "value": 50000000000          // 500 BTC in satoshi
    },
    {
      "txid": "prev_tx_hash2",
      "vout": 1,
      "scriptSig": {...},
      "value": 30000000000          // 300 BTC in satoshi
    }
  ],
  "vout": [                        // ← 출력들 (Outputs)
    {
      "value": 20000000000,        // 200 BTC in satoshi
      "n": 0,
      "scriptPubKey": {...}
    },
    {
      "value": 15000000000,        // 150 BTC in satoshi
      "n": 1,
      "scriptPubKey": {...}
    },
    {
      "value": 44900000000,        // 449 BTC in satoshi
      "n": 2,
      "scriptPubKey": {...}
    }
  ]
}
```

---

## 📋 데이터 컬럼 생성 과정

### 1. **직접 추출 컬럼들** (연관도: 1.0)

#### `tx_hash`

```python
# 트랜잭션 전체 데이터의 SHA256 해시
tx_data = serialize_transaction(tx)
tx_hash = sha256(sha256(tx_data)).hex()
```

- **생성 방법**: 트랜잭션 전체 데이터의 이중 SHA256 해시
- **용도**: 블록체인상 트랜잭션 고유 식별자
- **예시**: `"a1b2c3d4e5f6789..."`

#### `block_timestamp`

```python
# 블록 헤더에서 추출
block_timestamp = block_header['timestamp']
```

- **생성 방법**: 블록 헤더의 타임스탬프 필드
- **형태**: UNIX 타임스탬프 (초 단위)
- **예시**: `1609459200` (2021-01-01 00:00:00 UTC)

#### `input_count`

```python
# 입력 배열의 길이
input_count = len(transaction['vin'])
```

- **생성 방법**: `vin` 배열의 길이
- **의미**: 이 트랜잭션이 소비하는 이전 출력들의 개수
- **예시**: `5` (5개 주소에서 비트코인을 받음)

#### `output_count`

```python
# 출력 배열의 길이
output_count = len(transaction['vout'])
```

- **생성 방법**: `vout` 배열의 길이
- **의미**: 이 트랜잭션이 생성하는 새로운 출력들의 개수
- **예시**: `3` (3개 주소로 비트코인을 보냄)

#### `total_input_value`

```python
# 모든 입력 값의 합계
total_input_value = 0
for input_tx in transaction['vin']:
    # 이전 트랜잭션에서 해당 출력의 값을 찾아서 합산
    prev_output = get_previous_output(input_tx['txid'], input_tx['vout'])
    total_input_value += prev_output['value']
```

- **생성 방법**: 모든 입력이 참조하는 이전 출력값들의 합계
- **단위**: Satoshi (1 BTC = 100,000,000 satoshi)
- **예시**: `80000000000` (800 BTC)

#### `total_output_value`

```python
# 모든 출력 값의 합계
total_output_value = sum(output['value'] for output in transaction['vout'])
```

- **생성 방법**: 현재 트랜잭션의 모든 출력값 합계
- **단위**: Satoshi
- **예시**: `79900000000` (799 BTC)

#### `fee`

```python
# 입력 총합 - 출력 총합
fee = total_input_value - total_output_value
```

- **생성 방법**: 입력과 출력의 차이
- **의미**: 채굴자에게 지불하는 수수료
- **예시**: `100000000` (1 BTC)

#### `max_output_value`

```python
# 가장 큰 출력 값
max_output_value = max(output['value'] for output in transaction['vout'])
```

- **생성 방법**: 모든 출력 중 최대값
- **의미**: 가장 많은 금액을 받는 주소의 금액
- **예시**: `44900000000` (449 BTC)

#### `max_input_value`

```python
# 가장 큰 입력 값
max_input_value = 0
for input_tx in transaction['vin']:
    prev_output = get_previous_output(input_tx['txid'], input_tx['vout'])
    max_input_value = max(max_input_value, prev_output['value'])
```

- **생성 방법**: 모든 입력 중 최대값
- **의미**: 가장 많은 금액을 제공한 주소의 금액
- **예시**: `50000000000` (500 BTC)

### 2. **계산된 컬럼들** (연관도: 0.9)

#### `max_output_ratio`

```python
# 최대 출력이 전체 출력에서 차지하는 비율
max_output_ratio = max_output_value / total_output_value
```

- **생성 방법**: 최대 출력값 / 총 출력값
- **의미**: 자금 집중도 (1에 가까울수록 한 주소로 집중)
- **예시**: `0.9987` (99.87% 집중)

### 3. **2차 파생 지표** (연관도: 0.7)

#### `fee_per_max_ratio`

```python
# 수수료를 최대 출력 비율로 나눈 값
fee_per_max_ratio = fee / max_output_ratio
```

- **생성 방법**: 수수료 / 최대출력비율
- **의미**: 집중도 대비 수수료 비율
- **해석**: 값이 클수록 분산된 거래에서 높은 수수료 지불

---

## 🎯 실제 트랜잭션 예시

### 예시 1: 수집형 고래 거래

```
입력 (5개 주소):
├── Address_A: 100 BTC
├── Address_B: 200 BTC
├── Address_C: 150 BTC
├── Address_D: 250 BTC
└── Address_E: 99 BTC

출력 (1개 주소):
└── Address_F: 798 BTC

수수료: 1 BTC
```

**추출되는 데이터**:

- `input_count`: 5
- `output_count`: 1
- `total_input_value`: 79900000000 (799 BTC)
- `total_output_value`: 79800000000 (798 BTC)
- `fee`: 100000000 (1 BTC)
- `max_output_value`: 79800000000 (798 BTC)
- `max_output_ratio`: 1.0 (100% 집중)

### 예시 2: 분산형 고래 거래

```
입력 (1개 주소):
└── Address_A: 1000 BTC

출력 (10개 주소):
├── Address_B: 100 BTC
├── Address_C: 95 BTC
├── Address_D: 90 BTC
├── Address_E: 85 BTC
├── Address_F: 80 BTC
├── Address_G: 75 BTC
├── Address_H: 70 BTC
├── Address_I: 65 BTC
├── Address_J: 60 BTC
└── Address_K: 279 BTC

수수료: 1 BTC
```

**추출되는 데이터**:

- `input_count`: 1
- `output_count`: 10
- `total_input_value`: 100000000000 (1000 BTC)
- `total_output_value`: 99900000000 (999 BTC)
- `fee`: 100000000 (1 BTC)
- `max_output_value`: 27900000000 (279 BTC)
- `max_output_ratio`: 0.279 (27.9% 집중)

### 예시 3: 급행형 고래 거래

```
입력 (1개 주소):
└── Address_A: 1000 BTC

출력 (1개 주소):
└── Address_B: 990 BTC

수수료: 10 BTC (매우 높음!)
```

**추출되는 데이터**:

- `input_count`: 1
- `output_count`: 1
- `total_input_value`: 100000000000 (1000 BTC)
- `total_output_value`: 99000000000 (990 BTC)
- `fee`: 1000000000 (10 BTC) ← 일반적인 수수료의 100배!
- `max_output_value`: 99000000000 (990 BTC)
- `max_output_ratio`: 1.0 (100% 집중)

---

## 🔍 데이터 품질 고려사항

### 1. **데이터 일관성**

- **입력값 검증**: `total_input_value >= total_output_value + fee`
- **비율 검증**: `0 ≤ max_output_ratio ≤ 1`
- **개수 검증**: `input_count ≥ 1`, `output_count ≥ 1`

### 2. **에지 케이스**

```python
# Coinbase 트랜잭션 (블록 보상)
if transaction['vin'][0]['coinbase']:
    # 입력이 없는 특수한 트랜잭션
    total_input_value = total_output_value + fee
    input_count = 0  # 또는 1로 처리
```

### 3. **데이터 정확성**

- **이전 출력 참조**: 각 입력이 참조하는 이전 출력이 존재하고 아직 소비되지 않았는지 확인
- **스크립트 검증**: 입력의 scriptSig가 이전 출력의 scriptPubKey를 만족하는지 확인
- **금액 검증**: 소수점 이하 satoshi 단위가 정확한지 확인

---

## 🛠️ 데이터 추출 도구

### Python을 이용한 비트코인 데이터 추출

```python
import requests
import json

def get_transaction_data(tx_hash):
    """비트코인 API를 통해 트랜잭션 데이터 추출"""
    url = f"https://blockstream.info/api/tx/{tx_hash}"
    response = requests.get(url)
    return response.json()

def extract_features(tx_data):
    """트랜잭션 데이터에서 피처 추출"""
    # 기본 정보
    tx_hash = tx_data['txid']
    input_count = len(tx_data['vin'])
    output_count = len(tx_data['vout'])

    # 출력 값들 계산
    output_values = [vout['value'] for vout in tx_data['vout']]
    total_output_value = sum(output_values)
    max_output_value = max(output_values)

    # 집중도 계산
    max_output_ratio = max_output_value / total_output_value if total_output_value > 0 else 0

    # 수수료 계산 (API에서 제공하는 경우)
    fee = tx_data.get('fee', 0)

    return {
        'tx_hash': tx_hash,
        'input_count': input_count,
        'output_count': output_count,
        'total_output_value': total_output_value,
        'max_output_value': max_output_value,
        'max_output_ratio': max_output_ratio,
        'fee': fee
    }
```

### SQL을 이용한 대량 데이터 처리

```sql
-- 비트코인 노드 데이터베이스에서 추출하는 쿼리 예시
SELECT
    t.txid as tx_hash,
    b.time as block_timestamp,
    (SELECT COUNT(*) FROM tx_in WHERE txid = t.txid) as input_count,
    (SELECT COUNT(*) FROM tx_out WHERE txid = t.txid) as output_count,
    (SELECT SUM(value) FROM tx_out WHERE txid = t.txid) as total_output_value,
    (SELECT MAX(value) FROM tx_out WHERE txid = t.txid) as max_output_value,
    (SELECT MAX(value) / SUM(value) FROM tx_out WHERE txid = t.txid) as max_output_ratio
FROM
    tx t
JOIN
    block b ON t.block_hash = b.hash
WHERE
    (SELECT SUM(value) FROM tx_out WHERE txid = t.txid) >= 10000000000  -- 100 BTC 이상
ORDER BY
    b.height DESC;
```

---

## 📊 데이터 검증 방법

### 1. **기본 검증**

```python
def validate_transaction_data(tx_data):
    """트랜잭션 데이터 검증"""
    assert tx_data['input_count'] >= 1, "입력이 최소 1개 이상이어야 함"
    assert tx_data['output_count'] >= 1, "출력이 최소 1개 이상이어야 함"
    assert tx_data['total_output_value'] > 0, "총 출력값이 0보다 커야 함"
    assert 0 <= tx_data['max_output_ratio'] <= 1, "집중도는 0~1 사이여야 함"
    assert tx_data['fee'] >= 0, "수수료는 0 이상이어야 함"
```

### 2. **논리적 검증**

```python
def validate_business_logic(tx_data):
    """비즈니스 로직 검증"""
    # 수수료가 총 출력값의 50%를 넘으면 의심스러움
    if tx_data['fee'] > tx_data['total_output_value'] * 0.5:
        print("WARNING: 수수료가 비정상적으로 높음")

    # 집중도가 매우 낮은데 출력이 적으면 의심스러움
    if tx_data['max_output_ratio'] < 0.1 and tx_data['output_count'] <= 2:
        print("WARNING: 낮은 집중도에 비해 출력이 적음")
```

---

## 📈 성능 최적화

### 1. **인덱스 활용**

```sql
-- 효율적인 조회를 위한 인덱스
CREATE INDEX idx_tx_output_value ON tx_out(value DESC);
CREATE INDEX idx_block_time ON block(time);
CREATE INDEX idx_tx_block ON tx(block_hash);
```

### 2. **배치 처리**

```python
def process_transactions_batch(tx_hashes, batch_size=1000):
    """대량 트랜잭션 배치 처리"""
    for i in range(0, len(tx_hashes), batch_size):
        batch = tx_hashes[i:i+batch_size]
        results = []

        for tx_hash in batch:
            tx_data = get_transaction_data(tx_hash)
            features = extract_features(tx_data)
            results.append(features)

        yield results
```

---

## 📝 결론

이 문서를 통해 비트코인 트랜잭션의 각 구성 요소가 어떻게 우리 데이터셋의 컬럼으로 변환되는지 명확히 이해할 수 있습니다.

**핵심 포인트**:

- 📊 **직접 추출 데이터**: 블록체인에서 바로 얻을 수 있는 정보 (연관도 1.0)
- 🧮 **계산된 데이터**: 직접 데이터로부터 계산된 지표 (연관도 0.9)
- 📈 **파생 지표**: 여러 계산을 거친 복합 지표 (연관도 0.7)

이러한 구조적 이해를 바탕으로 **더 정확하고 의미 있는 고래 거래 분석**이 가능해집니다.
