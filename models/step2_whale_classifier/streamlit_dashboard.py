"""
📊 Streamlit 실시간 고래 거래 모니터링 대시보드
🐋 AI 기반 실시간 비트코인 고래 거래 분석 및 시각화
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import asyncio
import threading
import json
import requests
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 현재 디렉토리도 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    # 절대 임포트 시도
    from models.step2_whale_classifier.whale_classifier import WhaleClassificationSystem
    from models.step2_whale_classifier.config.settings import WHALE_CLASSES, UI_CONFIG
except ImportError:
    try:
        # 현재 디렉토리에서 임포트 시도
        from whale_classifier import WhaleClassificationSystem
        from config.settings import WHALE_CLASSES, UI_CONFIG
    except ImportError as e:
        st.error(f"모듈 import 오류: {e}")
        st.error("필요한 모듈들이 현재 경로에 없습니다. AI 분류 기능을 시뮬레이션 모드로 실행합니다.")
        
        # 기본 설정 정의
        WHALE_CLASSES = {
            0: {'name': '급행형고래', 'emoji': '🚀', 'market_impact': 'High'},
            1: {'name': '분산형고래', 'emoji': '🌊', 'market_impact': 'Medium'},
            2: {'name': '거대형고래', 'emoji': '🐋', 'market_impact': 'Very High'},
            3: {'name': '수집형고래', 'emoji': '🏪', 'market_impact': 'Low'},
            4: {'name': '집중형고래', 'emoji': '🎯', 'market_impact': 'High'}
        }
        
        UI_CONFIG = {
            'colors': {
                'success': '\033[92m',
                'warning': '\033[93m',
                'error': '\033[91m',
                'info': '\033[94m',
                'end': '\033[0m'
            }
        }
        
        WhaleClassificationSystem = None

# 페이지 설정
st.set_page_config(
    page_title="🐋 고래 거래 모니터링",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="collapsed"  # 사이드바 접기
)

# 세션 상태 초기화
if 'monitor' not in st.session_state:
    st.session_state.monitor = None
if 'whale_data' not in st.session_state:
    st.session_state.whale_data = []
if 'is_monitoring' not in st.session_state:
    st.session_state.is_monitoring = False
if 'processed_count' not in st.session_state:
    st.session_state.processed_count = 0
if 'whale_detected_count' not in st.session_state:
    st.session_state.whale_detected_count = 0

class StreamlitWhaleMonitor:
    """🐋 Streamlit용 고래 모니터링 시스템"""
    
    def __init__(self):
        self.classifier = None
        self.is_running = False
        self.simulation_mode = WhaleClassificationSystem is None
        
    def initialize_system(self):
        """시스템 초기화"""
        try:
            if self.simulation_mode:
                # 시뮬레이션 모드
                with st.spinner("🎭 시뮬레이션 모드로 초기화 중..."):
                    time.sleep(1)  # 초기화 시뮬레이션
                    st.success("✅ 시뮬레이션 모드 준비 완료!")
                    st.info("📝 실제 AI 모델 대신 시뮬레이션 모드로 동작합니다.")
                    return True
            else:
                # 실제 AI 모드
                with st.spinner("🚀 AI 분류 시스템 초기화 중..."):
                    self.classifier = WhaleClassificationSystem(enable_logging=False)
                    success = self.classifier.setup_system(train_new_model=False)
                    
                    if success:
                        st.success("✅ AI 분류 시스템 준비 완료!")
                        return True
                    else:
                        st.error("❌ AI 분류 시스템 초기화 실패")
                        return False
                    
        except Exception as e:
            st.error(f"❌ 시스템 초기화 실패: {e}")
            return False
    
    def generate_sample_transaction(self):
        """샘플 거래 생성"""
        import random
        
        patterns = [
            # 급행형고래
            {
                'total_volume_btc': random.uniform(5000, 15000),
                'input_count': random.randint(1, 3),
                'output_count': random.randint(1, 5),
                'concentration': random.uniform(0.8, 0.98),
                'fee_btc': random.uniform(0.005, 0.02),
                'pattern': '급행형고래'
            },
            # 분산형고래  
            {
                'total_volume_btc': random.uniform(2000, 8000),
                'input_count': random.randint(5, 20),
                'output_count': random.randint(10, 50),
                'concentration': random.uniform(0.2, 0.5),
                'fee_btc': random.uniform(0.001, 0.005),
                'pattern': '분산형고래'
            },
            # 거대형고래
            {
                'total_volume_btc': random.uniform(10000, 50000),
                'input_count': random.randint(1, 5),
                'output_count': random.randint(2, 8),
                'concentration': random.uniform(0.7, 0.95),
                'fee_btc': random.uniform(0.003, 0.01),
                'pattern': '거대형고래'
            },
            # 수집형고래
            {
                'total_volume_btc': random.uniform(3000, 12000),
                'input_count': random.randint(2, 8),
                'output_count': random.randint(5, 15),
                'concentration': random.uniform(0.4, 0.7),
                'fee_btc': random.uniform(0.002, 0.008),
                'pattern': '수집형고래'
            },
            # 집중형고래
            {
                'total_volume_btc': random.uniform(8000, 25000),
                'input_count': random.randint(1, 4),
                'output_count': random.randint(1, 3),
                'concentration': random.uniform(0.85, 0.99),
                'fee_btc': random.uniform(0.008, 0.025),
                'pattern': '집중형고래'
            }
        ]
        
        pattern = random.choice(patterns)
        pattern['timestamp'] = datetime.now()
        pattern['tx_id'] = f"sim_{int(time.time() * 1000000)}"
        
        return pattern
    
    def analyze_transaction(self, transaction):
        """거래 분석"""
        try:
            if self.simulation_mode:
                # 시뮬레이션 모드: 바로 분석 수행
                return self._simulate_analysis(transaction)
            else:
                # 실제 AI 모드
                if not self.classifier:
                    # classifier가 없으면 시뮬레이션 모드로 전환
                    return self._simulate_analysis(transaction)
                
                # AI 분석 수행
                result = self.classifier.analyze_transaction(transaction)
                
                # 결과 정리
                prediction = result['prediction']
                whale_info = prediction['whale_info']
                
                analyzed_result = {
                    'timestamp': transaction['timestamp'],
                    'tx_id': transaction['tx_id'],
                    'whale_type': whale_info['name'],
                    'whale_emoji': whale_info['emoji'],
                    'confidence': prediction['confidence'],
                    'volume_btc': transaction['total_volume_btc'],
                    'input_count': transaction['input_count'],
                    'output_count': transaction['output_count'],
                    'concentration': transaction['concentration'],
                    'fee_btc': transaction['fee_btc'],
                    'fee_rate': (transaction['fee_btc'] / transaction['total_volume_btc']) * 100,
                    'market_impact': whale_info['market_impact'],
                    'complexity': transaction['input_count'] + transaction['output_count'],
                    'expected_pattern': transaction.get('pattern', 'Unknown')
                }
                
                return analyzed_result
                
        except Exception as e:
            # 에러 발생시 시뮬레이션 모드로 fallback
            st.warning(f"AI 모델 오류 발생, 시뮬레이션 모드로 전환: {e}")
            return self._simulate_analysis(transaction)
    
    def _simulate_analysis(self, transaction):
        """시뮬레이션 모드 분석"""
        import random
        
        # 실제 패턴 기반 분류 로직 시뮬레이션
        expected_pattern = transaction.get('pattern', '급행형고래')
        
        # 패턴별 분류 확률 (실제 AI 모델의 성능 시뮬레이션)
        classification_accuracy = 0.7  # 70% 정확도
        
        if random.random() < classification_accuracy:
            # 정확한 분류
            predicted_type = expected_pattern
            confidence = random.uniform(0.6, 0.9)
        else:
            # 오분류
            all_types = ['급행형고래', '분산형고래', '거대형고래', '수집형고래', '집중형고래']
            predicted_type = random.choice([t for t in all_types if t != expected_pattern])
            confidence = random.uniform(0.3, 0.7)
        
        # 고래 정보 매핑
        whale_info_map = {
            '급행형고래': {'emoji': '🚀', 'market_impact': 'High'},
            '분산형고래': {'emoji': '🌊', 'market_impact': 'Medium'},
            '거대형고래': {'emoji': '🐋', 'market_impact': 'Very High'},
            '수집형고래': {'emoji': '🏪', 'market_impact': 'Low'},
            '집중형고래': {'emoji': '🎯', 'market_impact': 'High'}
        }
        
        whale_info = whale_info_map.get(predicted_type, whale_info_map['급행형고래'])
        
        analyzed_result = {
            'timestamp': transaction['timestamp'],
            'tx_id': transaction['tx_id'],
            'whale_type': predicted_type,
            'whale_emoji': whale_info['emoji'],
            'confidence': confidence,
            'volume_btc': transaction['total_volume_btc'],
            'input_count': transaction['input_count'],
            'output_count': transaction['output_count'],
            'concentration': transaction['concentration'],
            'fee_btc': transaction['fee_btc'],
            'fee_rate': (transaction['fee_btc'] / transaction['total_volume_btc']) * 100,
            'market_impact': whale_info['market_impact'],
            'complexity': transaction['input_count'] + transaction['output_count'],
            'expected_pattern': expected_pattern
        }
        
        return analyzed_result

def calculate_accuracy():
    """정확도 계산"""
    if not st.session_state.whale_data or len(st.session_state.whale_data) < 5:
        return 0
    
    df = pd.DataFrame(st.session_state.whale_data)
    
    # 예상 패턴과 실제 분류 결과 비교
    if 'expected_pattern' in df.columns:
        correct_predictions = (df['whale_type'] == df['expected_pattern']).sum()
        total_predictions = len(df)
        return correct_predictions / total_predictions
    
    return 0

def create_whale_distribution_chart():
    """고래 유형 분포 차트"""
    if not st.session_state.whale_data:
        st.info("데이터를 수집 중입니다...")
        return
    
    df = pd.DataFrame(st.session_state.whale_data)
    
    # 고래 유형별 분포
    whale_counts = df['whale_type'].value_counts()
    
    fig = px.pie(
        values=whale_counts.values,
        names=whale_counts.index,
        title="🐋 고래 유형 분포",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def create_volume_timeline():
    """거래량 타임라인"""
    if not st.session_state.whale_data:
        st.info("데이터를 수집 중입니다...")
        return
    
    df = pd.DataFrame(st.session_state.whale_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    # 고래 유형별 색상 매핑
    colors = {
        '급행형고래': '#ff6b6b',
        '분산형고래': '#4ecdc4', 
        '거대형고래': '#45b7d1',
        '수집형고래': '#96ceb4',
        '집중형고래': '#feca57'
    }
    
    for whale_type in df['whale_type'].unique():
        whale_data = df[df['whale_type'] == whale_type]
        
        fig.add_trace(go.Scatter(
            x=whale_data['timestamp'],
            y=whale_data['volume_btc'],
            mode='markers+lines',
            name=whale_type,
            marker=dict(
                size=whale_data['confidence'] * 20,  # 신뢰도에 따른 크기
                color=colors.get(whale_type, '#999999'),
                opacity=0.7
            ),
            hovertemplate=
            '<b>%{fullData.name}</b><br>' +
            '시간: %{x}<br>' +
            '거래량: %{y:,.0f} BTC<br>' +
            '신뢰도: %{marker.size/20:.1%}<br>' +
            '<extra></extra>'
        ))
    
    fig.update_layout(
        title="💰 거래량 타임라인 (마커 크기 = 신뢰도)",
        xaxis_title="시간",
        yaxis_title="거래량 (BTC)",
        height=400,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_confidence_analysis():
    """신뢰도 분석"""
    if not st.session_state.whale_data:
        return
    
    df = pd.DataFrame(st.session_state.whale_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 신뢰도 히스토그램
        fig = px.histogram(
            df, x='confidence', 
            title="🎯 신뢰도 분포",
            nbins=20,
            color_discrete_sequence=['#3498db']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 고래 유형별 평균 신뢰도
        avg_confidence = df.groupby('whale_type')['confidence'].mean().sort_values(ascending=True)
        
        fig = px.bar(
            x=avg_confidence.values,
            y=avg_confidence.index,
            orientation='h',
            title="🏆 고래 유형별 평균 신뢰도",
            color=avg_confidence.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def create_recent_transactions_table():
    """최근 거래 테이블"""
    st.subheader("📋 최근 감지된 고래 거래")
    
    if not st.session_state.whale_data:
        st.info("아직 감지된 거래가 없습니다.")
        return
    
    # 최근 20건만 표시
    recent_data = st.session_state.whale_data[-20:]
    df = pd.DataFrame(recent_data)
    
    # 컬럼 정리
    display_df = df[[
        'timestamp', 'whale_emoji', 'whale_type', 'volume_btc', 
        'confidence', 'fee_rate', 'complexity', 'expected_pattern'
    ]].copy()
    
    display_df.columns = [
        '시간', '이모지', '고래 유형', '거래량(BTC)', 
        '신뢰도', '수수료율(%)', '복잡도', '예상 패턴'
    ]
    
    # 포맷팅
    display_df['시간'] = pd.to_datetime(display_df['시간']).dt.strftime('%H:%M:%S')
    display_df['거래량(BTC)'] = display_df['거래량(BTC)'].apply(lambda x: f"{x:,.0f}")
    display_df['신뢰도'] = display_df['신뢰도'].apply(lambda x: f"{x:.1%}")
    display_df['수수료율(%)'] = display_df['수수료율(%)'].apply(lambda x: f"{x:.4f}")
    
    # 예상 패턴과 실제 결과 비교 컬럼 추가
    display_df['정확도'] = (display_df['고래 유형'] == display_df['예상 패턴']).apply(
        lambda x: "✅ 정확" if x else "❌ 불일치"
    )
    
    st.dataframe(
        display_df.sort_values('시간', ascending=False), 
        use_container_width=True,
        height=400
    )

def display_main_content():
    """메인 컨텐츠 표시"""
    # 차트 행
    col1, col2 = st.columns(2)
    
    with col1:
        create_whale_distribution_chart()
    
    with col2:
        create_volume_timeline()
    
    # 신뢰도 분석
    st.subheader("📊 신뢰도 분석")
    create_confidence_analysis()
    
    # 최근 거래 테이블
    create_recent_transactions_table()

def start_monitoring():
    """모니터링 시작"""
    if 'monitor' not in st.session_state or st.session_state.monitor is None:
        st.session_state.monitor = StreamlitWhaleMonitor()
    
    # 시스템 초기화 시도
    try:
        if st.session_state.monitor.initialize_system():
            st.session_state.is_monitoring = True
            st.sidebar.success("✅ 모니터링 시작!")
        else:
            # 초기화 실패시 시뮬레이션 모드로 전환
            st.session_state.monitor.simulation_mode = True
            st.session_state.is_monitoring = True
            st.sidebar.warning("⚠️ AI 모델 로드 실패, 시뮬레이션 모드로 시작!")
    except Exception as e:
        # 예외 발생시에도 시뮬레이션 모드로 전환
        st.session_state.monitor.simulation_mode = True
        st.session_state.is_monitoring = True
        st.sidebar.warning(f"⚠️ 초기화 오류, 시뮬레이션 모드로 시작: {e}")
    
    if st.session_state.is_monitoring:
        # 백그라운드에서 자동 거래 생성 시뮬레이션
        def background_simulation():
            import time
            import random
            while st.session_state.is_monitoring:
                try:
                    time.sleep(random.uniform(2, 5))  # 2-5초 간격
                    if st.session_state.is_monitoring:
                        generate_sample_transaction()
                except Exception as e:
                    print(f"백그라운드 시뮬레이션 오류: {e}")
        
        # 별도 스레드에서 실행
        thread = threading.Thread(target=background_simulation, daemon=True)
        thread.start()

def stop_monitoring():
    """모니터링 중지"""
    st.session_state.is_monitoring = False
    st.sidebar.warning("⏹️ 모니터링 중지됨")

def generate_sample_transaction():
    """샘플 거래 생성 및 분석"""
    # 모니터 객체가 없으면 생성 (초기화 없이)
    if 'monitor' not in st.session_state or st.session_state.monitor is None:
        st.session_state.monitor = StreamlitWhaleMonitor()
        # 초기화는 스킵하고 바로 시뮬레이션 모드로 설정
        st.session_state.monitor.simulation_mode = True
    
    # 샘플 거래 생성
    transaction = st.session_state.monitor.generate_sample_transaction()
    
    # 분석 수행 (시뮬레이션 모드로)
    result = st.session_state.monitor.analyze_transaction(transaction)
    
    if result:
        # 데이터 추가
        st.session_state.whale_data.append(result)
        st.session_state.processed_count += 1
        st.session_state.whale_detected_count += 1
        
        # 최대 개수 제한
        if len(st.session_state.whale_data) > 1000:
            st.session_state.whale_data = st.session_state.whale_data[-500:]
        
        # 사이드바에 최신 결과 표시
        with st.sidebar:
            st.success(f"🐋 새 거래 감지!")
            st.write(f"**{result['whale_emoji']} {result['whale_type']}**")
            st.write(f"💰 {result['volume_btc']:,.0f} BTC")
            st.write(f"🎯 {result['confidence']:.1%}")
            
        # 성공적으로 생성되었음을 표시
        return True
    else:
        st.error("거래 생성에 실패했습니다.")
        return False

def main():
    """메인 대시보드"""
    # 헤더
    st.title("🐋 실시간 고래 거래 모니터링 시스템")
    st.caption("AI 기반 비트코인 대형 거래 패턴 분석")
    
    # 간단한 제어 버튼들 (메인 화면에)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        if st.button("🚀 시작", type="primary", use_container_width=True):
            start_monitoring()
    
    with col2:
        if st.button("⏹️ 중지", use_container_width=True):
            stop_monitoring()
    
    with col3:
        if st.button("🧪 테스트", use_container_width=True):
            generate_sample_transaction()
    
    with col4:
        if st.button("🔄 초기화", use_container_width=True):
            reset_data()
    
    # 상태 표시
    status_container = st.container()
    with status_container:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.is_monitoring:
                st.success("🟢 모니터링 중")
            else:
                st.error("🔴 중지됨")
        
        with col2:
            st.metric("처리된 거래", st.session_state.processed_count)
        
        with col3:
            st.metric("고래 탐지", st.session_state.whale_detected_count)
        
        with col4:
            accuracy = calculate_accuracy()
            if accuracy > 0:
                st.metric("정확도", f"{accuracy:.1%}")
            else:
                st.metric("정확도", "계산 중...")
    
    st.divider()
    
    # 메인 컨텐츠
    if st.session_state.whale_data:
        display_main_content()
    else:
        st.info("🎯 **시작** 버튼을 눌러 모니터링을 시작하거나 **테스트** 버튼으로 샘플 데이터를 생성하세요!")
    
    # 자동 새로고침
    if st.session_state.is_monitoring:
        time.sleep(0.1)
        st.rerun()

def reset_data():
    """데이터 초기화"""
    st.session_state.whale_data = []
    st.session_state.processed_count = 0
    st.session_state.whale_detected_count = 0
    st.success("🔄 데이터가 초기화되었습니다!")
    st.rerun()

if __name__ == "__main__":
    main() 