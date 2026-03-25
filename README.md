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
| **TensorBoard** | Training visualization and monitoring |
| **OpenAI Gym** | RL environment abstraction |
| **Docker** | Containerization |
| **Market Data API** | Real-time stock price data source |

## Quick Start

> **Requisito**: Necesitás un token de la [Market Data API](https://www.marketdata.app/). Registrate gratis y copiá tu token.

### Con Docker (recomendado)

```bash
# Clonar
git clone https://github.com/yourusername/rl-trading.git
cd rl-trading

# Configurar variables de entorno
cp .env.example .env
# Editar .env y completar MARKETDATA_API_TOKEN

# Construir y levantar (primera vez tarda ~10 min por TensorFlow)
docker compose up -d --build

# La API está disponible en http://localhost:8000/
```

### Sin Docker

```bash
# Clonar y configurar entorno
git clone https://github.com/yourusername/rl-trading.git
cd rl-trading
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env y completar MARKETDATA_API_TOKEN

# Crear y aplicar migraciones
python manage.py makemigrations market_data trading
python manage.py migrate

# Levantar servidor
python manage.py runserver

# En terminales separadas:
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

## Flujo de Ejemplo

```bash
# 1. Agregar un símbolo
curl -X POST http://localhost:8000/api/market-data/symbols/ \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "name": "Apple Inc."}'

# 2. Ingestar datos históricos (365 velas diarias)
curl -X POST http://localhost:8000/api/market-data/ingest/ \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "resolution": "D", "countback": 365}'

# 3. Ver indicadores técnicos
curl http://localhost:8000/api/indicators/summary/?symbol=AAPL

# 4. Entrenar el agente RL
curl -X POST http://localhost:8000/api/trading/train/ \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "episodes": 100, "batch_size": 32, "initial_cash": 10000}'

# 5. Ver resultados
curl http://localhost:8000/api/trading/sessions/
```


## Correr Tests

```bash
python manage.py test
# o por app:
python manage.py test market_data
python manage.py test indicators
python manage.py test trading
```

## TensorBoard — Visualización del Entrenamiento

El proyecto integra **TensorBoard** para monitorear el proceso de entrenamiento del DQN en tiempo real. Los logs se generan automáticamente durante cada sesión de entrenamiento.

### Uso

```bash
# 1. Entrenar un agente (los logs se generan automáticamente)
curl -X POST http://localhost:8000/api/trading/train-sync/ \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "episodes": 50}'

# 2. Lanzar TensorBoard (desde la raíz del proyecto)
tensorboard --logdir=logs/tensorboard --port=6006

# 3. Abrir http://localhost:6006
```

> **Nota Windows**: Si obtenés `ModuleNotFoundError: No module named 'pkg_resources'`, ejecutá:
> ```bash
> pip install --upgrade tensorboard
> ```

### Métricas Disponibles

| Métrica | Descripción |
|---------|-------------|
| `entrenamiento/recompensa` | Recompensa total por episodio |
| `entrenamiento/epsilon` | Decaimiento de la tasa de exploración (1.0 → 0.01) |
| `entrenamiento/valor_portfolio` | Valor del portfolio al final de cada episodio |
| `red_neuronal/loss` | Loss promedio del batch de replay de la red neuronal |
| `backtest/*` | Resultados finales del backtest (P/L, trades, portfolio) |
| `pesos/*` | Histogramas de pesos de la red neuronal (cada 10 episodios) |

Cada sesión de entrenamiento genera su propia carpeta en `logs/tensorboard/`. Podés comparar múltiples sesiones directamente en la interfaz de TensorBoard.

## Problemas Conocidos de Dependencias

### 1. TensorBoard — `No module named 'pkg_resources'` (Windows)

En Windows, la versión de TensorBoard instalada por TensorFlow puede fallar al importar `pkg_resources`.

```bash
# Solución: actualizar TensorBoard manualmente
pip install --upgrade tensorboard
```

Esta versión actualizada es **solo para uso local** (TensorBoard se usa fuera de Docker). No agregar al `requirements.txt` porque conflictúa con `tensorflow==2.17.1` que requiere `tensorboard<2.18`.

---

### 2. OpenAI Gym — Warning de NumPy 2.0

Al correr tests o el servidor, puede aparecer este warning:

```
Gym has been unmaintained since 2022 and does not support NumPy 2.0
```

Esto es solo un **warning informativo**, no afecta el funcionamiento. El proyecto usa `numpy==1.26.2` (NumPy 1.x), por lo que la incompatibilidad real no aplica.

En el futuro, reemplazar `gym` por su sucesor:
```python
# En trading/environment.py
import gymnasium as gym  # en vez de: import gym
```

---

### 3. Migraciones — `InconsistentMigrationHistory`

Si al resetear Docker con `docker compose down -v` y volver a levantar aparece un error de migraciones inconsistentes, correr dentro del contenedor:

```bash
docker compose exec api python manage.py migrate
```

## Estructura del Proyecto

```
rl-trading/
├── config/                  # Configuración del proyecto Django
│   ├── settings/
│   │   ├── base.py          # Settings compartidos (Redis, Celery, DRF, Market Data API, TensorBoard)
│   │   └── development.py   # Overrides de desarrollo
│   ├── celery.py            # App Celery + schedule de Beat
│   ├── urls.py              # Routing de URLs
│   └── wsgi.py
├── market_data/             # Ingesta de datos desde Market Data API
│   ├── migrations/          # Migraciones de base de datos
│   ├── models.py            # Symbol, OHLCV, PriceSnapshot
│   ├── services.py          # MarketDataService (cliente API + caché Redis)
│   ├── tasks.py             # Celery: ingesta periódica de precios
│   ├── views.py             # Endpoints REST
│   └── tests.py
├── indicators/              # Análisis técnico con Pandas
│   ├── services.py          # IndicatorService (EMA, RSI, MACD, BB, OBV)
│   ├── tasks.py             # Celery: cálculo periódico de indicadores
│   ├── views.py             # Endpoints REST
│   └── tests.py
├── trading/                 # Entrenamiento RL + backtesting
│   ├── migrations/          # Migraciones de base de datos
│   ├── environment.py       # StockTradingEnv (OpenAI Gym)
│   ├── agent.py             # DQNAgent (TensorFlow/Keras)
│   ├── callbacks.py         # TrainingLogger (métricas TensorBoard)
│   ├── models.py            # TrainingSession, Trade
│   ├── services.py          # TradingService (orquesta el entrenamiento)
│   ├── tasks.py             # Celery: entrenamiento asíncrono
│   ├── views.py             # Endpoints REST
│   └── tests.py
├── logs/                    # Logs de TensorBoard (ignorado por git)
├── Dockerfile
├── docker-compose.yml       # API + Worker + Beat + Redis
├── requirements.txt
└── README.md
```

## Licencia

Este proyecto es para fines educativos y de demostración.

> **Disclaimer**: Este sistema es solo para aprendizaje y no constituye asesoramiento financiero. No usarlo para trading real sin validación exhaustiva.
