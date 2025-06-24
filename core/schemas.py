from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class WhaleTransaction(BaseModel):
    cluster: Optional[int] = Field(None, description="예측된 클러스터 번호")
    btc: float = Field(..., description="총 입력값(BTC)")
    input_count: int = Field(..., description="Input 개수")
    output_count: int = Field(..., description="Output 개수")
    max_output_ratio: float = Field(..., description="최대 Output 비율")
    max_input_ratio: float = Field(..., description="최대 Input 비율")
    fee_per_max_ratio: float = Field(..., description="수수료 / 최대 output 비율")
    timestamp: datetime = Field(..., description="트랜잭션 발생 시간")
    max_input_address: Optional[str] = Field(None, description="가장 큰 input 주소")
    max_output_address: Optional[str] = Field(None, description="가장 큰 output 주소")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cluster": 2,
                "btc": 1340.55,
                "input_count": 3,
                "output_count": 4,
                "max_output_ratio": 0.78,
                "max_input_ratio": 0.95,
                "fee_per_max_ratio": 0.000042,
                "timestamp": "2025-06-21T15:30:00",
                "max_input_address": "1ABCDxyz...",
                "max_output_address": "bc1QWErty..."
            }
        }
    }


class WhaleTransactionList(BaseModel):
    logs: List[WhaleTransaction]

    model_config = {
        "json_schema_extra": {
            "example": {
                "logs": [
                    {
                        "cluster": 2,
                        "btc": 1340.55,
                        "input_count": 3,
                        "output_count": 4,
                        "max_output_ratio": 0.78,
                        "max_input_ratio": 0.95,
                        "fee_per_max_ratio": 0.000042,
                        "timestamp": "2025-06-21T15:30:00",
                        "max_input_address": "1ABCDxyz...",
                        "max_output_address": "bc1QWErty..."
                    }
                ]
            }
        }
    }
    
class AddressInfo(BaseModel):
    address: str = Field(..., description="조회된 비트코인 주소")
    total_received_btc: float = Field(..., description="주소가 받은 총 BTC")
    total_sent_btc: float = Field(..., description="주소가 보낸 총 BTC")
    final_balance_btc: float = Field(..., description="현재 잔고 BTC")
    tx_count: int = Field(..., description="총 트랜잭션 수")
    is_exchange_like: bool = Field(..., description="거래소 지갑으로 추정되는지 여부")

    model_config = {
        "json_schema_extra": {
            "example": {
                "address": "1ABCDxyz...",
                "total_received_btc": 110000.5,
                "total_sent_btc": 109800.3,
                "final_balance_btc": 200.2,
                "tx_count": 3520,
                "is_exchange_like": True
            }
        }
    }

