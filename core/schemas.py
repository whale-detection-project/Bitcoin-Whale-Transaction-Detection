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
