import os

# [중요] 1. TensorFlow 충돌 방지: CPU만 사용하도록 강제 설정 (GPU 메모리 오류 방지)
# 서버가 터지는 것을 막기 위해 가장 먼저 실행되어야 합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 불필요한 로그 숨김

import uvicorn
import numpy as np
import pandas as pd
import ccxt
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ===== 1. 설정 및 모델 로드 =====

app = FastAPI()

# CORS 설정
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

# 전역 변수로 모델 선언
model = None

# [중요] 2. 시작 시 모델 로드 (Startup Event 활용)
@app.on_event("startup")
async def startup_event():
    global model
    print("⏳ Loading Model...")
    start_time = time.time()
    try:
        # 모델 로드
        model = load_model(MODEL_PATH)
        print(f"✅ GRU Model Loaded Successfully! ({time.time() - start_time:.2f}s)")
        
        # [테스트] 더미 데이터로 예측 한 번 실행 (Warm-up)
        # 처음 요청 시 느린 현상을 방지
        dummy_input = np.zeros((1, WINDOW_SIZE, len(ALL_FEATURES)))
        model.predict(dummy_input, verbose=0)
        print("✅ Model Warm-up Complete!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # 모델 로드 실패 시 서버를 종료하지 않고 None 처리 (디버깅용)
        model = None

# 응답 DTO 정의
class ChartDataDto(BaseModel):
    date: str       
    value: float    

# ===== 2. 데이터 처리 유틸리티 =====

def get_binance_data(limit=1500):
    """CCXT를 이용해 바이낸스 데이터 가져오기 (타임아웃 추가)"""
    exchange = ccxt.binance({
        'timeout': 10000, # 10초 타임아웃 설정
        'enableRateLimit': True,
    })
    try:
        # fetch_ohlcv는 네트워크 상황에 따라 느려질 수 있음
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=limit)
        if not ohlcv:
            raise ValueError("Empty data returned")
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        return df.drop(columns=['timestamp'])
    except Exception as e:
        print(f"❌ Binance Fetch Error: {e}")
        raise HTTPException(status_code=500, detail=f"Binance data fetch failed: {str(e)}")

def calculate_technical_indicators(df):
    """학습 코드와 동일한 보조지표 계산 로직"""
    df = df.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Moving Averages & Log Return
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
    
    # Stochastic
    low_min = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    df['%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-9)
    df['%D'] = df['%K'].rolling(3).mean()
    
    return df

def prepare_inference_data():
    # 1. 데이터 수집
    df = get_binance_data(limit=1000) # limit를 1000으로 줄여서 속도 향상
    
    # 2. 보조지표 계산
    df = calculate_technical_indicators(df)
    
    # 3. Stationarizing
    for col in PRICE_FEATURES:
        if col in df.columns:
            df[col] = df[col].pct_change(1)
            
    for col in INDICATOR_FEATURES:
        if col in df.columns:
            df[col] = df[col].diff(1)
            
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # 4. Feature Selection
    final_df = df[ALL_FEATURES]
    
    # 5. Scaling [주의: 수정 필요]
    # 종우님, 원래는 학습 때 저장한 scaler.pkl을 불러와서 scaler.transform(final_df) 해야 합니다.
    # 현재는 임시방편으로 새 Scaler를 쓰지만, 정확도가 매우 떨어질 수 있습니다.
    scaler = StandardScaler() 
    scaled_values = scaler.fit_transform(final_df) 
    
    # 6. Sequence Generation
    x_input = []
    dates = []
    
    # 최근 200개만 예측 (속도 최적화)
    prediction_limit = 200
    if len(scaled_values) > prediction_limit + WINDOW_SIZE:
        start_idx = len(scaled_values) - prediction_limit - WINDOW_SIZE
    else:
        start_idx = 0

    for i in range(start_idx, len(scaled_values) - WINDOW_SIZE):
        seq = scaled_values[i : i + WINDOW_SIZE]
        x_input.append(seq)
        target_date = final_df.index[i + WINDOW_SIZE - 1]
        dates.append(target_date)
        
    return np.array(x_input), dates

# ===== 3. API 엔드포인트 =====

@app.get("/api/predict/chart")
def get_prediction_chart():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently loading or failed to load.")
        
    try:
        x_input, dates = prepare_inference_data()
        
        # 모델 예측 (Verbose=0으로 로그 숨김)
        predictions = model.predict(x_input, verbose=0).flatten()
        
        results = []
        for date, prob in zip(dates, predictions):
            results.append({
                "date": date.strftime('%Y-%m-%d %H:%M'),
                "predicted": float(prob),
                "actual": 0 
            })
            
        return results
        
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/price/chart")
def get_price_history():
    try:
        df = get_binance_data(limit=1000)
        results = []
        # 최근 500개만 반환 (JSON 응답 크기 축소)
        for date, row in df.tail(500).iterrows():
            results.append({
                "date": date.strftime('%Y-%m-%d %H:%M'),
                "actual": float(row['Close'])
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 실행 시 로그 레벨 info로 설정
    print("🚀 Server Starting...")
    uvicorn.run("GRUServer:app", host="0.0.0.0", port=8000, reload=True)
