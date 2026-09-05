# Smart Market Watchlist API

This API powers a proactive stock market watchlist. It provides AI-driven natural language summaries of quantitative stock shifts (using Groq) and algorithmic technical analysis (using `pandas-ta`) to detect candlestick patterns and market sentiment.

**Base URL:** `http://localhost:8000` *(Update to your Render URL once deployed)*

**Content-Type:** `application/json`

---

## 1. Health Check

Verifies that the API server is running successfully.

* **Endpoint:** `GET /`
* **Description:** Returns a simple status payload.

**Example Response:**

```json
{
  "status": "ok",
  "service": "Smart Watchlist Backend"
}

```

---

## 2. Analyze Shift

Compares a stock's previous state (from the user's last visit) to its current state. Using the Groq LLM, it deduces structural market shifts strictly from quantitative data (price, volume anomalies, and technical indicators) and generates a minified, UI-ready JSON breakdown.

* **Endpoint:** `POST /api/v1/analyze-shift`
* **Description:** Accepts historical and current stock states, outputting up to 3 purely quantitative events and up to 3 numerical impact metrics.

### Example Request (`POST`)

```json
{
  "ticker": "TATASTEEL",
  "units_held": 50,
  "currency_symbol": "₹",
  "last_visit_state": {
    "timestamp": "2026-09-03T10:00:00Z",
    "price": 182.50,
    "volume": 1200000,
    "ema_20": 186.50,
    "rsi": 45.2
  },
  "current_state": {
    "timestamp": "2026-09-06T11:30:00Z",
    "price": 188.70,
    "volume": 1704000,
    "ema_20": 186.50,
    "rsi": 62.1
  }
}

```

### Example Response (`200 OK`)

```json
{
  "events_section_header": "EVENTS SINCE YOUR LAST SESSION (2 NEW)",
  "numerical_changes": [
    {
      "label": "PRICE GAIN SINCE VISIT",
      "value": "+₹6.20",
      "sub_value": "(+3.40%)"
    },
    {
      "label": "HOLDINGS P&L IMPACT",
      "value": "+₹310.00",
      "sub_value": "(50 units)"
    }
  ],
  "events": [
    {
      "title": "VOLUME BREAKOUT",
      "timestamp_label": "TODAY",
      "description": "Trading volume spiked to 1.7M, significantly higher than your last visit, supporting the upward price movement."
    },
    {
      "title": "MOMENTUM SURGE",
      "timestamp_label": "TODAY",
      "description": "RSI climbed from 45.2 to 62.1, indicating a rapid shift from neutral to strong bullish momentum."
    }
  ]
}

```

---

## 3. Detect Candlestick Patterns

Analyzes an array of historical OHLC (Open, High, Low, Close) data for multiple stocks simultaneously to detect over 60 traditional candlestick patterns and determine a primary market sentiment.

* **Endpoint:** `POST /api/v1/detect-patterns`
* **Description:** Processes a list of stocks. For best sentiment results (EMA crossovers), provide at least 10 historical candles per stock.

### Example Request (`POST`)

```json
{
  "stocks": [
    {
      "ticker": "RELIANCE",
      "candles": [
        {"open": 2800.0, "high": 2850.0, "low": 2790.0, "close": 2840.0},
        {"open": 2840.0, "high": 2890.0, "low": 2830.0, "close": 2885.0}
      ]
    },
    {
      "ticker": "HDFCBANK",
      "candles": [
        {"open": 1600.0, "high": 1610.0, "low": 1590.0, "close": 1595.0},
        {"open": 1595.0, "high": 1598.0, "low": 1595.0, "close": 1595.0}
      ]
    }
  ]
}

```

### Example Response (`200 OK`)

```json
{
  "results": [
    {
      "ticker": "RELIANCE",
      "pattern": "Bullish Marubozu",
      "market_sentiment": "BULLISH"
    },
    {
      "ticker": "HDFCBANK",
      "pattern": "Doji",
      "market_sentiment": "NEUTRAL"
    }
  ]
}

```
