#!/usr/bin/env python3
"""
🚀 Streamlit 고래 거래 모니터링 대시보드 실행기
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """필요한 패키지 설치 확인"""
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'numpy',
        'sklearn'  # scikit-learn은 import할 때 sklearn으로 사용
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 설치됨")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 미설치")
    
    if missing_packages:
        print(f"\n📦 다음 패키지들을 설치해주세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def run_streamlit():
    """Streamlit 대시보드 실행"""
    dashboard_path = Path(__file__).parent / "models" / "step2_whale_classifier" / "streamlit_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ 대시보드 파일을 찾을 수 없습니다: {dashboard_path}")
        return False
    
    print("🚀 Streamlit 대시보드 시작 중...")
    print("🌐 브라우저에서 http://localhost:8501 열림")
    print("⏹️ Ctrl+C로 중단")
    
    try:
        # Streamlit 실행
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n⏹️ 대시보드 중단됨")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")

if __name__ == "__main__":
    print("🐋 실시간 고래 거래 모니터링 대시보드")
    print("=" * 50)
    
    # 의존성 확인
    if not check_dependencies():
        print("\n📥 패키지 설치 후 다시 실행해주세요.")
        sys.exit(1)
    
    # Streamlit 실행
    run_streamlit() 