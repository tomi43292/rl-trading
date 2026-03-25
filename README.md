# RL Trading System

A production-grade **Reinforcement Learning trading system** built with Django, consuming real market data from [Market Data API](https://www.marketdata.app/). The system ingests live stock prices, calculates technical indicators with Pandas, and trains a DQN (Deep Q-Network) agent to learn optimal buy/sell/hold strategies.

## Tech Stack

| Technology | Role |
|-----------|------|
| **Django 4.2** | Web framework |
| **Django REST Framework** | API layer |
| **Redis** | Caching (prices, API responses) + Celery broker |
| **Celery** | Async task execution (training, data ingestion) |
| **Pandas** | Data manipulation + technical indicator calculation |
| **TensorFlow/Keras** | Deep Q-Network neural network |
| **OpenAI Gym** | RL environment abstraction |
| **Docker** | Containerization |
| **Market Data API** | Real-time stock price data source |

## Quick Start

### With Docker (recommended)

```bash
# Clone
git clone https://github.com/yourusername/rl-trading.git
cd rl-trading

# Configure
cp .env.example .env
# Edit .env with your MARKETDATA_API_TOKEN

# Run
docker compose up -d

# The API is available at http://localhost:8000/
```

### Without Docker

```bash
# Clone and setup
git clone https://github.com/yourusername/rl-trading.git
cd rl-trading
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your MARKETDATA_API_TOKEN

# Database
python manage.py migrate

# Run
python manage.py runserver

# In separate terminals:
celery -A config worker -l info
celery -A config beat -l info
```

## API Endpoints

### Market Data (`/api/market-data/`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/market-data/symbols/` | List tracked symbols |
| `POST` | `/api/market-data/symbols/` | Add a symbol to track |
| `GET` | `/api/market-data/prices/?symbols=AAPL,MSFT` | Get live prices from Market Data API |
| `POST` | `/api/market-data/ingest/` | Ingest historical OHLCV data |
| `GET` | `/api/market-data/candles/?symbol=AAPL` | Get stored candle data |
| `GET` | `/api/market-data/snapshots/` | Get latest price snapshots |

### Indicators (`/api/indicators/`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/indicators/summary/?symbol=AAPL` | Get indicator summary (trend, momentum, volatility signals) |
| `GET` | `/api/indicators/data/?symbol=AAPL&limit=500` | Get full indicator dataset (for RL training) |

### Trading (`/api/trading/`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/trading/train/` | Start async RL training (via Celery) |
| `POST` | `/api/trading/train-sync/` | Start sync RL training (blocks until done) |
| `GET` | `/api/trading/sessions/` | List training sessions |
| `GET` | `/api/trading/sessions/{id}/` | Get session details with trades |
| `GET` | `/api/trading/sessions/{id}/summary/` | Get session summary |
| `GET` | `/api/trading/sessions/{id}/trades/` | Get all trades from a session |

## Example Workflow

```bash
# 1. Add a symbol
curl -X POST http://localhost:8000/api/market-data/symbols/ \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "name": "Apple Inc."}'

# 2. Ingest historical data
curl -X POST http://localhost:8000/api/market-data/ingest/ \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "resolution": "D", "countback": 365}'

# 3. Check indicators
curl http://localhost:8000/api/indicators/summary/?symbol=AAPL

# 4. Train RL agent
curl -X POST http://localhost:8000/api/trading/train/ \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "episodes": 100, "batch_size": 32, "initial_cash": 10000}'

# 5. Check results
curl http://localhost:8000/api/trading/sessions/
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed explanation of the project architecture, design decisions, and code walkthrough.

## Running Tests

```bash
python manage.py test
# or
python manage.py test market_data
python manage.py test indicators
python manage.py test trading
```

## Project Structure

```
rl-trading/
├── config/                  # Django project configuration
│   ├── settings/
│   │   ├── base.py          # Shared settings (Redis, Celery, DRF, Market Data API)
│   │   └── development.py   # Dev overrides
│   ├── celery.py            # Celery app + beat schedule
│   ├── urls.py              # Root URL routing
│   └── wsgi.py
├── market_data/             # Data ingestion from Market Data API
│   ├── models.py            # Symbol, OHLCV, PriceSnapshot
│   ├── services.py          # MarketDataService (API client + Redis cache)
│   ├── tasks.py             # Celery: periodic price ingestion
│   ├── views.py             # REST endpoints
│   └── tests.py
├── indicators/              # Technical analysis with Pandas
│   ├── services.py          # IndicatorService (EMA, RSI, MACD, BB, OBV)
│   ├── tasks.py             # Celery: periodic indicator calculation
│   ├── views.py             # REST endpoints
│   └── tests.py
├── trading/                 # RL training + backtesting
│   ├── environment.py       # StockTradingEnv (OpenAI Gym)
│   ├── agent.py             # DQNAgent (TensorFlow/Keras)
│   ├── models.py            # TrainingSession, Trade
│   ├── services.py          # TradingService (orchestrates training)
│   ├── tasks.py             # Celery: async training
│   ├── views.py             # REST endpoints
│   └── tests.py
├── Dockerfile
├── docker-compose.yml       # API + Worker + Beat + Redis
├── requirements.txt
└── README.md
```

## License

This project is for educational and demonstration purposes.

> **Disclaimer**: This system is for learning purposes only and does not constitute financial advice. Do not use it for real trading without extensive validation.
