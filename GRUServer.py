import uvicorn
import numpy as np
import pandas as pd
import ccxt
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# ===== 1. 설정 및 모델 로드 =====

app = FastAPI()

# CORS 설정 (React 프론트엔드 접속 허용)
# === CORS 설정 수정 (여기부터) ===
# origins = [ #로컬 테스트용
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    # allow_origins=origins,        #로컬 테스트시에는 origins로 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best_model_GRU_tuned_v14.keras"
WINDOW_SIZE = 48  # 학습 코드와 동일 (48시간)

# 학습 시 사용된 Feature 순서 (매우 중요: 순서 틀리면 예측 엉망됨)
# 외부 데이터(S&P500 등)는 학습 실패로 제외되었으므로 BTC 내부 지표 14개만 사용
PRICE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20']
INDICATOR_FEATURES = ['RSI', 'MACD', 'Signal_Line', 'Log_Return', 'ATR', '%K', '%D']
ALL_FEATURES = PRICE_FEATURES + INDICATOR_FEATURES # 총 14개

try:
    model = load_model(MODEL_PATH)
    print("✅ GRU Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# 응답 DTO 정의
class ChartDataDto(BaseModel):
    date: str       # YYYY-MM-DD HH:mm
    value: float    # 예측 확률 or 실제 가격

# ===== 2. 데이터 처리 유틸리티 =====

def get_binance_data(limit=1000):
    """CCXT를 이용해 바이낸스 데이터 가져오기"""
    exchange = ccxt.binance()
    # 현재 시점부터 과거 limit 개수만큼 가져옴
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=limit)
    if not ohlcv:
        raise HTTPException(status_code=500, detail="Binance data fetch failed")
        
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    return df.drop(columns=['timestamp'])

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
    """데이터 수집 -> 지표추가 -> 변환 -> 스케일링 -> 시퀀스생성"""
    # 1. 데이터 수집 (충분히 많이 가져와서 스케일러 안정화)
    df = get_binance_data(limit=1500)
    
    # 2. 보조지표 계산
    df = calculate_technical_indicators(df)
    
    # 3. Stationarizing (학습 데이터와 동일하게 변환)
    # Price 관련 -> pct_change
    for col in PRICE_FEATURES:
        if col in df.columns:
            df[col] = df[col].pct_change(1)
            
    # Indicator 관련 -> diff
    for col in INDICATOR_FEATURES:
        if col in df.columns:
            df[col] = df[col].diff(1)
            
    # inf, nan 제거
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # 4. Feature Selection & Ordering (순서 중요!)
    # 모델은 14개 feature를 기대함
    if not all(col in df.columns for col in ALL_FEATURES):
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        raise ValueError(f"Missing features: {missing}")
        
    final_df = df[ALL_FEATURES]
    
    # 5. Scaling
    # 학습된 scaler가 없으므로 현재 데이터 분포로 fit
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(final_df)
    
    # 6. Sequence Generation (Sliding Window)
    x_input = []
    dates = []
    
    # 최근 데이터부터 과거로 거슬러 올라가며 시퀀스 생성
    # 예측 속도를 위해 최근 500개 정도만 처리
    for i in range(len(scaled_values) - WINDOW_SIZE):
        seq = scaled_values[i : i + WINDOW_SIZE]
        x_input.append(seq)
        
        # 예측 시점의 날짜 (마지막 캔들 시간)
        target_date = final_df.index[i + WINDOW_SIZE - 1]
        dates.append(target_date)
        
    return np.array(x_input), dates

# ===== 3. API 엔드포인트 =====

@app.get("/api/predict/chart")
def get_prediction_chart():
    """AI 모델 예측 결과 (상승 확률) 반환"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        x_input, dates = prepare_inference_data()
        
        # 모델 예측 (0~1 사이 확률)
        predictions = model.predict(x_input, verbose=0).flatten()
        
        results = []
        for date, prob in zip(dates, predictions):
            results.append({
                "date": date.strftime('%Y-%m-%d %H:%M'),
                "predicted": float(prob), # 0.0 ~ 1.0
                "actual": 0 # 프론트에서 병합하므로 비워둠
            })
            
        return results # 리스트 형태 반환 (PredictionChart.js 대응)
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/price/chart")
def get_price_history():
    """실제 비트코인 가격 차트 데이터 반환"""
    try:
        df = get_binance_data(limit=1000)
        results = []
        for date, row in df.iterrows():
            results.append({
                "date": date.strftime('%Y-%m-%d %H:%M'),
                "actual": float(row['Close'])
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("GRUServer:app", host="0.0.0.0", port=2413, reload=True)