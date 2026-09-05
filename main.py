import os
import json
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

app = FastAPI(
    title="Smart Market Watchlist API",
    description="Backend API for AI Shift Analysis and Technical Candlestick Pattern Detection"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ==========================================
# 1. Pydantic Models for API Data Schemas
# ==========================================

class StockState(BaseModel):
    timestamp: str
    price: float
    volume: float
    ema_20: Optional[float] = None
    rsi: Optional[float] = None

class ShiftAnalysisRequest(BaseModel):
    ticker: str
    units_held: int = 1
    currency_symbol: str = "₹"
    last_visit_state: StockState
    current_state: StockState

class NumericalChangeItem(BaseModel):
    label: str = Field(..., description="e.g., 'PRICE GAIN SINCE VISIT'")
    value: str = Field(..., description="e.g., '+₹6.20'")
    sub_value: str = Field(..., description="e.g., '(+3.40%)'")

class EventItem(BaseModel):
    title: str
    timestamp_label: str
    description: str

class ShiftAnalysisResponse(BaseModel):
    events_section_header: str
    numerical_changes: List[NumericalChangeItem]
    events: List[EventItem]

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float

class StockCandleData(BaseModel):
    ticker: str
    candles: List[Candle]

class PatternDetectionResponseItem(BaseModel):
    ticker: str
    pattern: str
    market_sentiment: str 

class PatternDetectionResponse(BaseModel):
    results: List[PatternDetectionResponseItem]

class PatternDetectionRequest(BaseModel):
    stocks: List[StockCandleData]

# ==========================================
# 2. Pure-Python Candlestick & Technical Engine
# ==========================================

def _detect_manual_patterns(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> List[str]:
    """Pure-Python detection for 40 classic Japanese candlestick patterns."""
    patterns = []
    n = len(c)
    if n < 1:
        return patterns

    o_bars = [o[-1 - i] if n > i else 0.0 for i in range(min(5, n))]
    h_bars = [h[-1 - i] if n > i else 0.0 for i in range(min(5, n))]
    l_bars = [l[-1 - i] if n > i else 0.0 for i in range(min(5, n))]
    c_bars = [c[-1 - i] if n > i else 0.0 for i in range(min(5, n))]

    body = [abs(c_bars[i] - o_bars[i]) for i in range(len(c_bars))]
    rng = [max(h_bars[i] - l_bars[i], 1e-6) for i in range(len(c_bars))]
    upper = [h_bars[i] - max(o_bars[i], c_bars[i]) for i in range(len(c_bars))]
    lower = [min(o_bars[i], c_bars[i]) - l_bars[i] for i in range(len(c_bars))]
    is_bull = [c_bars[i] > o_bars[i] for i in range(len(c_bars))]
    is_bear = [c_bars[i] < o_bars[i] for i in range(len(c_bars))]

    is_doji_0 = body[0] <= (0.1 * rng[0])
    is_long_0 = body[0] >= (0.6 * rng[0])
    is_small_0 = body[0] <= (0.3 * rng[0])

    # 1-14: Single Candle Patterns
    if is_doji_0:
        patterns.append("Doji")
    if is_doji_0 and lower[0] >= (0.6 * rng[0]) and upper[0] <= (0.1 * rng[0]):
        patterns.append("Dragonfly Doji")
    if is_doji_0 and upper[0] >= (0.6 * rng[0]) and lower[0] <= (0.1 * rng[0]):
        patterns.append("Gravestone Doji")
    if is_doji_0 and upper[0] >= (0.35 * rng[0]) and lower[0] >= (0.35 * rng[0]):
        patterns.append("Long-Legged Doji")
    if not is_doji_0 and lower[0] >= (2.0 * body[0]) and upper[0] <= (0.15 * rng[0]):
        patterns.append("Hammer")
    if not is_doji_0 and upper[0] >= (2.0 * body[0]) and lower[0] <= (0.15 * rng[0]):
        patterns.append("Inverted Hammer")
    if is_bear[0] and upper[0] >= (2.0 * body[0]) and lower[0] <= (0.15 * rng[0]):
        patterns.append("Shooting Star")
    if is_bear[0] and lower[0] >= (2.0 * body[0]) and upper[0] <= (0.15 * rng[0]):
        patterns.append("Hanging Man")
    if is_small_0 and not is_doji_0 and upper[0] >= (0.25 * rng[0]) and lower[0] >= (0.25 * rng[0]):
        patterns.append("Spinning Top")
    if is_bull[0] and is_long_0 and upper[0] <= (0.05 * rng[0]) and lower[0] <= (0.05 * rng[0]):
        patterns.append("Bullish Marubozu")
    if is_bear[0] and is_long_0 and upper[0] <= (0.05 * rng[0]) and lower[0] <= (0.05 * rng[0]):
        patterns.append("Bearish Marubozu")
    if is_small_0 and upper[0] >= (0.35 * rng[0]) and lower[0] >= (0.35 * rng[0]):
        patterns.append("High Wave Candle")
    if is_small_0 and lower[0] >= (0.75 * rng[0]):
        patterns.append("Takuri Line")
    if is_doji_0 and abs(upper[0] - lower[0]) <= (0.15 * rng[0]) and rng[0] > (0.5 * np.mean(rng)):
        patterns.append("Rickshaw Man")

    # 15-28: Two Candle Patterns
    if n >= 2:
        prev_mid = (o_bars[1] + c_bars[1]) / 2.0
        if is_bear[1] and is_bull[0] and (o_bars[0] <= c_bars[1]) and (c_bars[0] >= o_bars[1]):
            patterns.append("Bullish Engulfing")
        if is_bull[1] and is_bear[0] and (o_bars[0] >= c_bars[1]) and (c_bars[0] <= o_bars[1]):
            patterns.append("Bearish Engulfing")
        if is_bear[1] and is_bull[0] and (o_bars[0] >= c_bars[1]) and (c_bars[0] <= o_bars[1]) and body[0] < body[1]:
            patterns.append("Bullish Harami")
        if is_bull[1] and is_bear[0] and (o_bars[0] <= c_bars[1]) and (c_bars[0] >= o_bars[1]) and body[0] < body[1]:
            patterns.append("Bearish Harami")
        if is_doji_0 and (min(o_bars[1], c_bars[1]) <= c_bars[0] <= max(o_bars[1], c_bars[1])):
            patterns.append("Harami Cross")
        if is_bear[1] and is_bull[0] and (o_bars[0] < l_bars[1]) and (c_bars[0] > prev_mid) and (c_bars[0] < o_bars[1]):
            patterns.append("Piercing Line")
        if is_bull[1] and is_bear[0] and (o_bars[0] > h_bars[1]) and (c_bars[0] < prev_mid) and (c_bars[0] > o_bars[1]):
            patterns.append("Dark Cloud Cover")
        if is_bear[1] and is_bull[0] and (o_bars[0] >= o_bars[1]) and (l_bars[0] > h_bars[1]):
            patterns.append("Bullish Kicker")
        if is_bull[1] and is_bear[0] and (o_bars[0] <= o_bars[1]) and (h_bars[0] < l_bars[1]):
            patterns.append("Bearish Kicker")
        if abs(l_bars[0] - l_bars[1]) <= (0.05 * rng[0]) and is_bear[1] and is_bull[0]:
            patterns.append("Tweezer Bottom")
        if abs(h_bars[0] - h_bars[1]) <= (0.05 * rng[0]) and is_bull[1] and is_bear[0]:
            patterns.append("Tweezer Top")
        if is_bear[1] and is_bear[0] and abs(c_bars[0] - c_bars[1]) <= (0.05 * rng[0]):
            patterns.append("Matching Low")
        if is_bear[1] and is_bull[0] and (o_bars[0] < l_bars[1]) and abs(c_bars[0] - l_bars[1]) <= (0.08 * rng[0]):
            patterns.append("On Neck")
        if is_bear[1] and is_bull[0] and (o_bars[0] < l_bars[1]) and abs(c_bars[0] - c_bars[1]) <= (0.08 * rng[0]):
            patterns.append("In Neck")

    # 29-38: Three Candle Patterns
    if n >= 3:
        if (is_bear[2] and body[2] >= 0.5 * rng[2] and 
            body[1] <= 0.35 * rng[1] and max(o_bars[1], c_bars[1]) < min(o_bars[2], c_bars[2]) and
            is_bull[0] and c_bars[0] > ((o_bars[2] + c_bars[2]) / 2.0)):
            patterns.append("Morning Star")
        if (is_bull[2] and body[2] >= 0.5 * rng[2] and 
            body[1] <= 0.35 * rng[1] and min(o_bars[1], c_bars[1]) > max(o_bars[2], c_bars[2]) and
            is_bear[0] and c_bars[0] < ((o_bars[2] + c_bars[2]) / 2.0)):
            patterns.append("Evening Star")
        if (is_bear[2] and (body[1] <= 0.1 * rng[1]) and 
            max(o_bars[1], c_bars[1]) < min(o_bars[2], c_bars[2]) and
            is_bull[0] and c_bars[0] > ((o_bars[2] + c_bars[2]) / 2.0)):
            patterns.append("Morning Doji Star")
        if (is_bull[2] and (body[1] <= 0.1 * rng[1]) and 
            min(o_bars[1], c_bars[1]) > max(o_bars[2], c_bars[2]) and
            is_bear[0] and c_bars[0] < ((o_bars[2] + c_bars[2]) / 2.0)):
            patterns.append("Evening Doji Star")
        if (is_bull[2] and is_bull[1] and is_bull[0] and
            (c_bars[0] > c_bars[1] > c_bars[2]) and
            (o_bars[0] > o_bars[1] and o_bars[0] < c_bars[1]) and
            (o_bars[1] > o_bars[2] and o_bars[1] < c_bars[2])):
            patterns.append("Three White Soldiers")
        if (is_bear[2] and is_bear[1] and is_bear[0] and
            (c_bars[0] < c_bars[1] < c_bars[2]) and
            (o_bars[0] < o_bars[1] and o_bars[0] > c_bars[1]) and
            (o_bars[1] < o_bars[2] and o_bars[1] > c_bars[2])):
            patterns.append("Three Black Crows")
        if (is_bear[2] and is_bull[1] and (o_bars[1] >= c_bars[2]) and (c_bars[1] <= o_bars[2]) and
            is_bull[0] and c_bars[0] > c_bars[1]):
            patterns.append("Three Inside Up")
        if (is_bull[2] and is_bear[1] and (o_bars[1] <= c_bars[2]) and (c_bars[1] >= o_bars[2]) and
            is_bear[0] and c_bars[0] < c_bars[1]):
            patterns.append("Three Inside Down")
        if (is_bear[2] and is_bull[1] and (o_bars[1] < c_bars[2]) and (c_bars[1] > o_bars[2]) and
            is_bull[0] and c_bars[0] > c_bars[1]):
            patterns.append("Three Outside Up")
        if (is_bull[2] and is_bear[1] and (o_bars[1] > c_bars[2]) and (c_bars[1] < o_bars[2]) and
            is_bear[0] and c_bars[0] < c_bars[1]):
            patterns.append("Three Outside Down")

    # 39-40: Multi-Candle Continuation Patterns
    if n >= 5:
        if (is_bull[4] and body[4] >= 0.5 * rng[4] and
            is_bear[3] and is_bear[2] and is_bear[1] and
            min(l_bars[1], l_bars[2], l_bars[3]) >= l_bars[4] and
            max(h_bars[1], h_bars[2], h_bars[3]) <= h_bars[4] and
            is_bull[0] and c_bars[0] > c_bars[4]):
            patterns.append("Rising Three Methods")
        if (is_bear[4] and body[4] >= 0.5 * rng[4] and
            is_bull[3] and is_bull[2] and is_bull[1] and
            max(h_bars[1], h_bars[2], h_bars[3]) <= h_bars[4] and
            min(l_bars[1], l_bars[2], l_bars[3]) >= l_bars[4] and
            is_bear[0] and c_bars[0] < c_bars[4]):
            patterns.append("Falling Three Methods")

    return patterns

def evaluate_technicals(candles: List[Candle]) -> dict:
    if not candles or len(candles) < 2:
        return {"pattern": "Insufficient Data", "sentiment": "NEUTRAL"}

    df = pd.DataFrame([c.model_dump() for c in candles])
    pattern_str = "No Clear Pattern"
    sentiment = "NEUTRAL"

    try:
        o = np.array(df['open'], dtype=float)
        h = np.array(df['high'], dtype=float)
        l = np.array(df['low'], dtype=float)
        c = np.array(df['close'], dtype=float)

        detected = _detect_manual_patterns(o, h, l, c)
        if detected:
            pattern_str = ", ".join(sorted(list(set(detected))))

        # Calculate Trend / Sentiment via native pandas EWM
        sentiment = "BULLISH" if c[-1] > o[-1] else "BEARISH"
        if len(df) >= 10:
            ema5 = df['close'].ewm(span=5, adjust=False).mean()
            ema10 = df['close'].ewm(span=10, adjust=False).mean()
            if not ema5.empty and not ema10.empty:
                if ema5.iloc[-1] > ema10.iloc[-1]:
                    sentiment = "BULLISH"
                elif ema5.iloc[-1] < ema10.iloc[-1]:
                    sentiment = "BEARISH"

    except Exception:
        pattern_str = "Calculation Error"

    return {"pattern": pattern_str, "sentiment": sentiment}

# ==========================================
# 3. API Endpoints
# ==========================================

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Smart Watchlist Backend"}

@app.post("/api/v1/analyze-shift", response_model=ShiftAnalysisResponse)
def analyze_shift(req: ShiftAnalysisRequest):
    if not client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")

    ema_str = f" | EMA20:{req.last_visit_state.ema_20}->{req.current_state.ema_20}" if req.current_state.ema_20 else ""
    rsi_str = f" | RSI:{req.last_visit_state.rsi}->{req.current_state.rsi}" if req.current_state.rsi else ""
    
    user_prompt = f"""Output valid JSON matching this structure:
{{"events_section_header":"str","numerical_changes":[{{"label":"str","value":"str","sub_value":"str"}}],"events":[{{"title":"str","timestamp_label":"str","description":"str"}}]}}

RULES:
1. 'events': Maximum 3 items. Deduce structural market shifts strictly from price action, volume anomalies, and technical indicators.
2. 'numerical_changes': Maximum 3 items.

DATA:
Tkr:{req.ticker} | P:{req.last_visit_state.price}->{req.current_state.price} | V:{req.last_visit_state.volume}->{req.current_state.volume}{ema_str}{rsi_str}"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a precise quantitative financial parser. Return minified JSON only."},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        raw_json_string = completion.choices[0].message.content
        parsed_data = json.loads(raw_json_string)
        
        return ShiftAnalysisResponse(**parsed_data)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse LLM output into valid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Processing Error: {str(e)}")

@app.post("/api/v1/detect-patterns", response_model=PatternDetectionResponse)
def detect_patterns(req: PatternDetectionRequest):
    results = []
    for item in req.stocks:
        analysis = evaluate_technicals(item.candles)
        results.append(
            PatternDetectionResponseItem(
                ticker=item.ticker, 
                pattern=analysis["pattern"],
                market_sentiment=analysis["sentiment"]
            )
        )
    return PatternDetectionResponse(results=results)
