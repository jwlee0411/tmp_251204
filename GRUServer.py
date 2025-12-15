import os
import time

# [체크 1] 환경 변수 설정
print("--- [1] 환경 변수 설정 중 ---")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Intel CPU 최적화 기능이 충돌날 때가 있어 끄는 옵션 추가
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

print("--- [2] 기본 라이브러리 임포트 중 (numpy, pandas...) ---")
import uvicorn
import numpy as np
import pandas as pd
import ccxt

print("--- [3] TensorFlow 임포트 시작 (여기서 오래 걸릴 수 있음) ---")
# 여기서 멈추면 TensorFlow 설치 문제거나 PC 사양 문제입니다.
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

print("--- [4] FastAPI 앱 초기화 ---")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best_model_GRU_tuned_v14.keras"
WINDOW_SIZE = 48 
PRICE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20']
INDICATOR_FEATURES = ['RSI', 'MACD', 'Signal_Line', 'Log_Return', 'ATR', '%K', '%D']
ALL_FEATURES = PRICE_FEATURES + INDICATOR_FEATURES 

model = None

@app.on_event("startup")
async def startup_event():
    global model
    print("--- [5] 서버 시작 이벤트 트리거됨 ---")
    print(f"--- [6] 모델 파일 로드 시도: {MODEL_PATH} ---")
    
    start_time = time.time()
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ 모델 로드 성공! (소요시간: {time.time() - start_time:.2f}초)")
        
        print("--- [7] 웜업(Warm-up) 예측 시도 ---")
        dummy_input = np.zeros((1, WINDOW_SIZE, len(ALL_FEATURES)))
        model.predict(dummy_input, verbose=0)
        print("✅ 웜업 완료! 서버 준비 끝.")
        
    except Exception as e:
        print(f"❌ 모델 로드 중 에러 발생: {e}")
        model = None

# ... (나머지 유틸리티 함수 등은 그대로 두셔도 됩니다) ...
# 데이터 처리 함수나 API 엔드포인트 코드는 서버 시작 속도에 영향을 안 줍니다.
# 아래 메인 실행 부분만 확인해주세요.

if __name__ == "__main__":
    print("--- [0] 메인 함수 시작 ---")
    # reload=True는 개발할 때 좋지만, 파일 감지 때문에 초기 로딩을 느리게 할 수 있습니다.
    # 일단 꺼보고(False) 실행해보세요.
    uvicorn.run("GRUServer:app", host="0.0.0.0", port=8000, reload=False)
