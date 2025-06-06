"""
🌐 실시간 고래 거래 모니터링 시스템
WebSocket/SSE 기반 24/7 실시간 분석 스트리밍
"""

import asyncio
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator
from pathlib import Path
import sys

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# FastAPI 및 WebSocket 관련
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️ FastAPI 미설치. pip install fastapi uvicorn websockets 실행 필요")

# Bitcoin API 클라이언트
import requests
import websocket
from websocket import WebSocketApp

from .whale_classifier import WhaleClassificationSystem
from .config.settings import API_CONFIG, UI_CONFIG

class RealTimeWhaleMonitor:
    """🐋 실시간 고래 거래 모니터링 시스템"""
    
    def __init__(self):
        self.classifier = None
        self.is_running = False
        self.connected_clients: List[WebSocket] = []
        self.transaction_queue = asyncio.Queue()
        self.analysis_results = []
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 색상 설정
        self.colors = UI_CONFIG['colors']
        
        # 실시간 스트림 상태
        self.stream_active = False
        self.processed_count = 0
        self.whale_detected_count = 0
        
    async def initialize_system(self):
        """시스템 초기화"""
        try:
            self.logger.info("🚀 실시간 모니터링 시스템 초기화...")
            
            # 분류 시스템 초기화
            self.classifier = WhaleClassificationSystem(enable_logging=False)
            
            # 사전 훈련된 모델 로드 또는 새로 훈련
            success = self.classifier.setup_system(train_new_model=True)
            
            if success:
                self.logger.info("✅ AI 분류 시스템 준비 완료")
                return True
            else:
                self.logger.error("❌ AI 분류 시스템 초기화 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 시스템 초기화 실패: {e}")
            return False
    
    async def start_bitcoin_stream(self):
        """비트코인 실시간 스트림 시작"""
        self.logger.info("🌐 비트코인 실시간 스트림 연결 중...")
        
        # 실제 환경에서는 BlockStream WebSocket 또는 Mempool.space API 사용
        # 개발 환경에서는 시뮬레이션 모드
        
        if API_CONFIG['simulation']['enabled']:
            await self._start_simulation_stream()
        else:
            await self._start_real_bitcoin_stream()
    
    async def _start_simulation_stream(self):
        """시뮬레이션 모드 스트림"""
        self.logger.info("🎭 시뮬레이션 모드로 실시간 스트림 시작")
        self.stream_active = True
        
        while self.stream_active:
            try:
                # 랜덤 거래 생성 (실제 패턴 기반)
                simulated_tx = self._generate_realistic_transaction()
                
                # 큐에 추가
                await self.transaction_queue.put(simulated_tx)
                
                # 1-3초 간격으로 새 거래 생성
                await asyncio.sleep(random.uniform(1, 3))
                
            except Exception as e:
                self.logger.error(f"❌ 시뮬레이션 스트림 오류: {e}")
                await asyncio.sleep(5)
    
    async def _start_real_bitcoin_stream(self):
        """실제 비트코인 네트워크 스트림"""
        self.logger.info("🔗 실제 비트코인 네트워크 연결 중...")
        
        # BlockStream WebSocket API
        ws_url = "wss://blockstream.info/api/ws"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                # 거래 데이터 파싱 및 큐에 추가
                if 'tx' in data:
                    asyncio.create_task(self._process_bitcoin_transaction(data['tx']))
            except Exception as e:
                self.logger.error(f"WebSocket 메시지 처리 오류: {e}")
        
        def on_error(ws, error):
            self.logger.error(f"WebSocket 오류: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.info("WebSocket 연결 종료")
        
        # WebSocket 연결 시도
        ws = WebSocketApp(ws_url,
                         on_message=on_message,
                         on_error=on_error,
                         on_close=on_close)
        
        # 별도 스레드에서 실행
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
    
    def _generate_realistic_transaction(self) -> Dict:
        """현실적인 거래 데이터 생성"""
        import random
        
        # 다양한 고래 패턴 시뮬레이션
        patterns = [
            # 급행형고래
            {
                'total_volume_btc': random.uniform(5000, 15000),
                'input_count': random.randint(1, 3),
                'output_count': random.randint(1, 5),
                'concentration': random.uniform(0.8, 0.98),
                'fee_btc': random.uniform(0.005, 0.02)
            },
            # 분산형고래  
            {
                'total_volume_btc': random.uniform(2000, 8000),
                'input_count': random.randint(5, 20),
                'output_count': random.randint(10, 50),
                'concentration': random.uniform(0.2, 0.5),
                'fee_btc': random.uniform(0.001, 0.005)
            },
            # 거대형고래
            {
                'total_volume_btc': random.uniform(10000, 50000),
                'input_count': random.randint(1, 5),
                'output_count': random.randint(2, 8),
                'concentration': random.uniform(0.7, 0.95),
                'fee_btc': random.uniform(0.003, 0.01)
            }
        ]
        
        pattern = random.choice(patterns)
        pattern['timestamp'] = datetime.now().isoformat()
        pattern['tx_id'] = f"sim_{int(time.time() * 1000000)}"
        
        return pattern
    
    async def process_transactions(self):
        """거래 처리 백그라운드 태스크"""
        self.logger.info("⚡ 거래 분석 엔진 시작")
        
        while self.is_running:
            try:
                # 큐에서 거래 데이터 가져오기 (1초 타임아웃)
                transaction = await asyncio.wait_for(
                    self.transaction_queue.get(), timeout=1.0
                )
                
                # AI 분석 수행
                result = await self._analyze_transaction(transaction)
                
                if result:
                    # 결과를 모든 연결된 클라이언트에게 브로드캐스트
                    await self._broadcast_result(result)
                    
                    # 통계 업데이트
                    self.processed_count += 1
                    
                    # 고래 거래 감지시 카운트 증가
                    if result.get('is_whale_transaction', False):
                        self.whale_detected_count += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"❌ 거래 처리 오류: {e}")
    
    async def _analyze_transaction(self, transaction: Dict) -> Optional[Dict]:
        """단일 거래 AI 분석"""
        try:
            # 고래 거래 여부 사전 필터링 (최소 볼륨 체크)
            if transaction['total_volume_btc'] < 1000:  # 1000 BTC 미만은 스킵
                return None
            
            # AI 분류 수행
            analysis_result = self.classifier.analyze_transaction(transaction)
            
            # 결과 정리
            prediction = analysis_result['prediction']
            whale_info = prediction['whale_info']
            confidence = prediction['confidence']
            
            # 실시간 출력용 간소화된 결과
            simplified_result = {
                'timestamp': transaction.get('timestamp', datetime.now().isoformat()),
                'tx_id': transaction.get('tx_id', 'unknown'),
                'transaction_data': transaction,
                'whale_type': whale_info['name'],
                'whale_emoji': whale_info['emoji'], 
                'confidence': confidence,
                'market_impact': whale_info['market_impact'],
                'is_whale_transaction': True,
                'analysis_summary': {
                    'volume_btc': transaction['total_volume_btc'],
                    'complexity': transaction['input_count'] + transaction['output_count'],
                    'concentration': transaction['concentration'],
                    'fee_rate': (transaction['fee_btc'] / transaction['total_volume_btc']) * 100
                }
            }
            
            return simplified_result
            
        except Exception as e:
            self.logger.error(f"❌ 거래 분석 실패: {e}")
            return None
    
    async def _broadcast_result(self, result: Dict):
        """모든 연결된 클라이언트에게 결과 브로드캐스트"""
        if not self.connected_clients:
            return
        
        message = json.dumps(result, ensure_ascii=False, default=str)
        
        # 연결 끊어진 클라이언트 제거
        disconnected_clients = []
        
        for client in self.connected_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected_clients.append(client)
        
        # 끊어진 클라이언트 제거
        for client in disconnected_clients:
            self.connected_clients.remove(client)
            
        # 콘솔에도 출력
        self._print_real_time_result(result)
    
    def _print_real_time_result(self, result: Dict):
        """실시간 결과 콘솔 출력"""
        timestamp = result['timestamp'][:19]  # 초까지만
        whale_type = result['whale_type']
        emoji = result['whale_emoji']
        confidence = result['confidence']
        volume = result['analysis_summary']['volume_btc']
        
        print(f"{self.colors['info']}[{timestamp}]{self.colors['end']} "
              f"{emoji} {whale_type} | "
              f"💰 {volume:,.0f} BTC | "
              f"🎯 {confidence:.1%} | "
              f"📊 처리: {self.processed_count} | "
              f"🐋 감지: {self.whale_detected_count}")
    
    async def start_monitoring(self):
        """실시간 모니터링 시작"""
        try:
            self.logger.info("🌊 실시간 고래 모니터링 시작...")
            
            # 시스템 초기화
            if not await self.initialize_system():
                return False
            
            self.is_running = True
            
            # 백그라운드 태스크 시작
            tasks = [
                asyncio.create_task(self.start_bitcoin_stream()),
                asyncio.create_task(self.process_transactions())
            ]
            
            self.logger.info("✅ 실시간 모니터링 활성화!")
            self.logger.info("🔍 고래 거래 탐지 중...")
            
            # 태스크 실행
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            self.logger.info("⏹️ 사용자에 의해 모니터링 중단")
        except Exception as e:
            self.logger.error(f"❌ 모니터링 오류: {e}")
        finally:
            await self.stop_monitoring()
    
    async def stop_monitoring(self):
        """모니터링 중단"""
        self.logger.info("🛑 실시간 모니터링 중단 중...")
        self.is_running = False
        self.stream_active = False
        
        # 연결된 모든 클라이언트 정리
        for client in self.connected_clients:
            await client.close()
        self.connected_clients.clear()


# FastAPI WebSocket 서버
if FASTAPI_AVAILABLE:
    app = FastAPI(title="🐋 실시간 고래 거래 모니터링", version="2.0.0")
    monitor = RealTimeWhaleMonitor()
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket 엔드포인트"""
        await websocket.accept()
        monitor.connected_clients.append(websocket)
        
        try:
            while True:
                # 클라이언트로부터 메시지 대기 (연결 유지용)
                await websocket.receive_text()
        except WebSocketDisconnect:
            monitor.connected_clients.remove(websocket)
    
    @app.get("/stream")
    async def sse_stream():
        """Server-Sent Events 스트림"""
        async def event_generator():
            while monitor.is_running:
                try:
                    # 최근 분석 결과 전송
                    if monitor.analysis_results:
                        result = monitor.analysis_results[-1]
                        yield f"data: {json.dumps(result, ensure_ascii=False, default=str)}\n\n"
                    
                    await asyncio.sleep(1)
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/plain")
    
    @app.get("/status")
    async def get_status():
        """모니터링 상태 조회"""
        return {
            "is_running": monitor.is_running,
            "processed_count": monitor.processed_count,
            "whale_detected_count": monitor.whale_detected_count,
            "connected_clients": len(monitor.connected_clients),
            "timestamp": datetime.now().isoformat()
        }


async def run_real_time_monitor():
    """실시간 모니터 실행"""
    monitor = RealTimeWhaleMonitor()
    await monitor.start_monitoring()


def start_web_server():
    """웹 서버 시작"""
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI가 설치되지 않음. pip install fastapi uvicorn websockets 실행")
        return
    
    print("🌐 웹 서버 시작 중...")
    print("📡 WebSocket: ws://localhost:8000/ws")
    print("📊 SSE Stream: http://localhost:8000/stream") 
    print("📈 Status: http://localhost:8000/status")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import sys
    import random
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # 웹 서버 모드
        start_web_server()
    else:
        # 콘솔 모니터링 모드
        print("🐋 실시간 고래 거래 모니터링 시작")
        print("⏹️ Ctrl+C로 중단")
        asyncio.run(run_real_time_monitor()) 