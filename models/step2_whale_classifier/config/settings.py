"""
🐋 고래 거래 분류 시스템 - 설정 파일
실시간 비트코인 거래 분석 및 고래 패턴 탐지
"""

import os
from pathlib import Path

# 🏠 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT / "analysis" / "step1_results" / "class_weight_results"
MODEL_PATH = PROJECT_ROOT / "models" / "step2_whale_classifier" / "trained_models"

# 📊 모델 설정
MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 8,
    'min_samples_split': 50,
    'min_samples_leaf': 20,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
    'class_weight_strategy': 'rarity_based'  # Step 1에서 최적 성능
}

# 🎯 클래스 매핑
WHALE_CLASSES = {
    0: {
        'name': '수집형고래',
        'emoji': '🏦',
        'description': '여러 주소에서 자금을 모아 소수 주소로 집중',
        'behavior': '분산된 자산을 통합하는 집중 매매',
        'market_impact': '중간 (자금 집중으로 인한 유동성 감소)',
        'weight': 5.0
    },
    1: {
        'name': '분산형고래',
        'emoji': '🌊',
        'description': '대량 자금을 여러 주소로 분산',
        'behavior': '리스크 분산 또는 OTC 거래 준비',
        'market_impact': '낮음 (분산 거래로 시장 충격 완화)',
        'weight': 15.0
    },
    2: {
        'name': '급행형고래',
        'emoji': '⚡',
        'description': '높은 수수료로 빠른 거래 처리',
        'behavior': '긴급 거래 또는 시세 대응',
        'market_impact': '높음 (급하게 처리되는 대량 거래)',
        'weight': 15.0
    },
    3: {
        'name': '집중형고래',
        'emoji': '🎯',
        'description': '대부분 자금을 한 주소로 집중',
        'behavior': '일반적인 대량 거래 또는 저장',
        'market_impact': '중간 (집중도에 따라 변동)',
        'weight': 0.8
    },
    4: {
        'name': '거대형고래',
        'emoji': '🐋',
        'description': '압도적 거래량의 초대형 거래',
        'behavior': '기관 거래 또는 대규모 자금 이동',
        'market_impact': '매우 높음 (시장 전체에 큰 영향)',
        'weight': 30.0
    }
}

# 📈 피처 설정
FEATURES = {
    'input_features': [
        'total_volume_btc',
        'input_count',
        'output_count', 
        'concentration',
        'fee_btc'
    ],
    'feature_descriptions': {
        'total_volume_btc': '총 거래량 (BTC)',
        'input_count': '입력 주소 개수',
        'output_count': '출력 주소 개수',
        'concentration': '집중도 (최대출력/총출력)',
        'fee_btc': '거래 수수료 (BTC)'
    },
    'feature_importance_threshold': 0.05  # 5% 이상만 중요 피처로 간주
}

# 🌐 API 설정
API_CONFIG = {
    'bitcoin_api': {
        'base_url': 'https://blockstream.info/api',
        'backup_url': 'https://blockchair.com/bitcoin/api',
        'timeout': 30,
        'retry_count': 3,
        'rate_limit': 1.0  # 초당 요청 수 제한
    },
    'simulation': {
        'enabled': True,  # 개발 중에는 시뮬레이션 모드
        'sample_data_path': DATA_PATH / "optimized_whale_dataset.csv"
    }
}

# 🔍 분석 설정
ANALYSIS_CONFIG = {
    'confidence_threshold': 0.7,  # 70% 이상만 높은 신뢰도로 간주
    'similarity_threshold': 0.85,  # 유사 거래 판단 기준
    'recent_period_hours': 24,     # 최근 거래 분석 기간
    'anomaly_z_score': 2.0,       # 이상치 판단 Z-score 기준
}

# 📊 출력 설정
OUTPUT_CONFIG = {
    'insight_level': 3,  # 1:기본, 2:상세, 3:전문가
    'show_feature_importance': True,
    'show_similar_transactions': True,
    'show_market_impact': True,
    'show_confidence_breakdown': True,
    'max_similar_transactions': 5
}

# 🎨 UI 설정
UI_CONFIG = {
    'colors': {
        'success': '\033[92m',
        'warning': '\033[93m',
        'error': '\033[91m',
        'info': '\033[94m',
        'bold': '\033[1m',
        'end': '\033[0m'
    },
    'console_width': 80,
    'show_progress': True
}

# 🔧 성능 설정
PERFORMANCE_CONFIG = {
    'batch_size': 1000,
    'cache_enabled': True,
    'cache_size': 10000,
    'parallel_processing': True,
    'memory_limit_mb': 2048
} 