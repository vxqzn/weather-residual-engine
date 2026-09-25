# Timișoara Weather Residual Engine (TWRE)

[![Daily Ingestion, Retraining & Docker Audit](https://github.com/vxqzn/weather-residual-engine/actions/workflows/daily_pipeline.yml/badge.svg)](https://github.com/vxqzn/weather-residual-engine/actions/workflows/daily_pipeline.yml)
[![Live Service](https://img.shields.io/badge/Render-Live%20Endpoint-brightgreen?style=flat&logo=render)](https://weather-residual-engine.onrender.com/health)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving%20%3C%202ms-009688?style=flat&logo=fastapi)](src/twre/service/app.py)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20Neon-336791?style=flat&logo=postgresql)](sql/01_ddl.sql)

Autonomous, self-operating MLOps microservice that continuously learns, predicts, and corrects localized microclimate bias in raw Numerical Weather Prediction (NWP) temperature forecasts for Timișoara, Romania ($45.7537^\circ\text{N}, 21.2257^\circ\text{E}$, elevation $\sim 90\text{m}$).

The engine operates 24/7 with zero ongoing cloud infrastructure costs: it ingests historical forecasts and actuals via Open-Meteo, enforces causal temporal integrity in PostgreSQL, executes automated champion-challenger model retraining, and serves bias-corrected forecasts through a sub-2ms FastAPI layer deployed on Render.

---

## 1. Empirical Holdout Benchmark (2025 Chronological Split)

The engine was evaluated on a strictly separated temporal split: trained on the **2024 Leap Year (366 days)** and evaluated out-of-sample on the **full 2025 calendar year (365 days)**.

| Performance Metric | Raw Open-Meteo NWP Baseline | TWRE Residual Ridge Model | Absolute Delta ($\Delta$) | Relative Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **365-Day Holdout MAE (2025)** | `1.1407°C` | `0.7216°C` | `-0.4191°C` | **-36.7%** |
| **Training Set MAE (2024)** | `1.2826°C` | `0.4155°C` | `-0.8671°C` | **-67.6%** |
| **Winter Regime Drift MAE (Holdout Tail)** | `0.5722°C` | `0.5676°C` | `-0.0046°C` | Outperforms during regime shift |
| **Prediction Latency (Cached Hit)** | N/A (External API: 250ms+) | **`1.62ms – 2.14ms`** | $\sim 100\times$ faster | Live verified on Render |
| **Active Champion Provenance** | Uncorrected Baseline | `model_20260925094150` | Git SHA: `fd65123` | Persisted in `ledger.json` |

*Empirical Proof:* Model provenance, training parameters, Git commit SHAs, and holdout benchmarks are persisted immutably in [`artifacts/models/ledger.json`](artifacts/models/ledger.json).

---

## 2. System Architecture & Live Telemetry

```mermaid
flowchart TD
    subgraph Ingestion["Daily Ingestion & Data Contracts (02:00 UTC)"]
        OM["Open-Meteo Historical & Forecast APIs"] --> Worker["Ingestion Worker\n(src/twre/ingestion/worker.py)"]
        Worker --> Pydantic["Pydantic V2 Quarantining Gate\n(ObservationRecord, ForecastRecord)"]
        Pydantic --> DB[("Neon PostgreSQL Store (AWS Frankfurt)\n1,012+ Contiguous Calendar Records\nON CONFLICT DO UPDATE Idempotency")]
    end

    subgraph MLOps["Continuous Retraining & Champion-Challenger Gating"]
        DB --> FeatStore["Causal Feature Pipeline (pipeline.py)\nZero-leakage shifts: error_lag_1, rolling_bias_7"]
        FeatStore --> Challenger["Train Challenger Regressor (train.py)"]
        Challenger --> Gate{"Promotion Gate (gate.py)\nChallenger MAE < Champion MAE - Margin?"}
        Champion["Active Champion Binary\n(artifacts/models/champion.joblib)"] --> Gate
        Gate -- Promoted --> Atomic["Atomic Swap (.tmp -> os.replace)\nAppend to artifacts/models/ledger.json"]
        Gate -- Rejected --> LogRejection["Log Rejection in ledger.json\nRetain Incumbent Champion"]
    end

    subgraph Serving["High-Throughput Serving Layer (FastAPI)"]
        Atomic -. Non-blocking st_mtime poll .-> ModelEngine["In-Memory ModelEngine (engine.py)"]
        ModelEngine --> Cache["Lifespan Feature Pre-Warming Cache\n(< 2.2ms Latency SLA)"]
        Cache --> Endpoints["FastAPI REST Endpoints\nGET /health | GET /metrics | GET /predict"]
        Endpoints --> Render["Render Cloud Microservice (24/7 Free Tier)\nhttps://weather-residual-engine.onrender.com"]
    end
```

### Live Public Endpoints (Render Cloud)

The production service is publicly accessible without authentication at `https://weather-residual-engine.onrender.com`:

#### 1. System Health Probe (`GET /health`)
Verifies live database connectivity and active model memory loading:
```bash
curl -s https://weather-residual-engine.onrender.com/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_champion_model_id": "model_20260925094150",
  "db_connected": true
}
```

#### 2. Model Provenance & Performance Ledger (`GET /metrics`)
Outputs the audit ledger record for the currently serving champion model:
```bash
curl -s https://weather-residual-engine.onrender.com/metrics
```
```json
{
  "active_champion_metadata": {
    "model_id": "model_20260925094150",
    "created_at": "2026-09-25T09:41:50.677582+00:00",
    "git_commit": "fd65123",
    "train_mae": 0.41545375926336825,
    "holdout_baseline_mae": 1.1407478860388063,
    "holdout_model_mae": 0.7216162933162672,
    "is_promoted": true,
    "promotion_reason": "cold start candidate model promoted as there is currently no champion (cold start)"
  },
  "predictions_served": 1,
  "cache_hits": 1
}
```

#### 3. Low-Latency Bias Correction (`GET /predict`)
Executes real-time residual correction on either live cached features or caller-supplied query parameters:
```bash
curl -s https://weather-residual-engine.onrender.com/predict
```
```json
{
  "target_date": "2026-09-24",
  "raw_forecast_temp_max": 21.66,
  "predicted_residual_bias": -1.06,
  "corrected_temp_max": 20.60,
  "model_id": "model_20260925094150",
  "is_fallback": false,
  "latency_ms": 2.14
}
```

---

## 3. Mathematical Formulation & Engineering Deep Dive

### Physical Residual Learning Formulation
Direct end-to-end temperature prediction via statistical machine learning discards Navier-Stokes planetary fluid dynamics equations computed across atmospheric vertical layers by supercomputers (ECMWF, GFS, ICON). TWRE strictly formulates the learning target as the **forecast residual error**:

$$\epsilon_{t+1} = T^{\text{actual}}_{t+1} - T^{\text{forecast}}_{t+1}$$
$$\hat{T}^{\text{corrected}}_{t+1} = T^{\text{forecast}}_{t+1} + \hat{\epsilon}_{t+1}$$

*Staff-Level Mathematical Rationale:*
1. **Preservation of Global Physics:** Fluid dynamic equations predict macro-scale synoptic advection, frontal passages, and solar radiative transfer.
2. **Isolation of Localized Microclimate Bias:** Statistical regression models only the localized residual $\epsilon_{t+1}$ caused by Timișoara's specific topographic trapping in the Banat plain, surface albedo, and urban heat island effects.
3. **Graceful Fallback:** If model artifacts are missing or inputs corrupted, the system sets $\hat{\epsilon} = 0$, safely degrading to the raw numerical forecast ($T^{\text{forecast}}$) with zero service disruption (`is_fallback: True`).

---

### Causal Feature Store (Zero Lookahead Leakage)
In operational weather prediction, true actual temperatures for day $t$ are physically unobserved when issuing day $t$'s forecast on day $t-1$. [`src/twre/features/pipeline.py`](src/twre/features/pipeline.py) enforces strict causal lag separation:

* **Lagged Error ($\epsilon_{t-1}$):** Realized error from the previous day, strictly shifted by 1 index (`.shift(1)`).
* **7-Day Rolling Bias:** Running window capturing persistent atmospheric regime drift, computed *after* applying the lag shift:
  $$\text{rolling\_bias\_7}_t = \frac{1}{7} \sum_{i=1}^7 \epsilon_{t-i}$$
* **Defensive Integrity:** Calendar-contiguous reindexing (`pd.date_range`) ensures missing days are exposed as explicit nulls; pipeline asserts zero NaNs across historical warmup buffers prior to model ingestion.

---

### Automated Promotion Gate & Atomic Registry
To prevent degraded or overfitted models from entering production, [`src/twre/models/gate.py`](src/twre/models/gate.py) enforces automated holdout gating:

1. **Cold-Start Condition:** If no champion exists in the registry, a candidate model is promoted only if its holdout MAE outperforms the raw uncorrected forecast baseline:
   $$\text{MAE}_{\text{candidate}} < \text{MAE}_{\text{baseline}}$$
2. **Challenger vs. Champion Evaluation:** When an active champion exists, a challenger model is promoted only if it beats the incumbent champion on the unseen holdout partition by margin $\delta$:
   $$\text{MAE}_{\text{challenger}} < \text{MAE}_{\text{champion}} - \delta$$
3. **Atomic File Persistence:** Model serialization utilizes temporary `.tmp` file dumps followed by POSIX/NTFS atomic swaps (`os.replace`). This guarantees that concurrent ASGI reader threads never read partial or corrupted binary streams.
4. **Immutable Audit Ledger:** Model metadata (Git commit SHA, training MAE, holdout baseline MAE, holdout candidate MAE, promotion verdict, timestamp) is appended to [`artifacts/models/ledger.json`](artifacts/models/ledger.json).

---

## 4. Key Architectural Decisions & Negative Space

### 1. In-Memory Caching vs. External Redis
* **The Constraint:** Zero-cost cloud tiers enforce a rigid **512MB RAM ceiling**.
* **The Trade-Off:** Introducing Redis, Celery, or RabbitMQ consumes 80–120MB of resident RAM, introduces external network latency, requires connection pooling overhead, and incurs billing liabilities.
* **The Engineering Decision:** `ModelEngine` stores feature vectors and champion model pointers directly in process memory. Model hot-reloading is handled via non-blocking filesystem `os.stat().st_mtime` polling on incoming requests.
* **The Result:** Prediction latency drops to **`1.62ms`** with zero external operational dependencies and a resident container footprint under **150MB**.

### 2. Hermetic CI/CD via Ephemeral PostgreSQL 16 Service Container
* **The Vulnerability:** Running automated unit tests (`pytest`) against remote managed cloud databases (Neon AWS Frankfurt) in GitHub Actions introduces network latency, transient SSL handshake drops, and database state pollution across parallel branches.
* **The Engineering Decision:** [`.github/workflows/daily_pipeline.yml`](.github/workflows/daily_pipeline.yml) defines an ephemeral `services: postgres: image: postgres:16-alpine` container running on `localhost:5432`. Schema DDL is bootstrapped hermetically in the test runner (`python -m twre.db.session`), running the 17-test suite offline in **3.95s**.
* **The Security Boundary:** The production cloud `secrets.DATABASE_URL` is injected strictly into the scheduled production batch runner step (`daily_runner.py`), completely decoupling test validation from cloud state.

### 3. Cloud Deployment Target Selection
* **Hugging Face Spaces (Docker):** Evaluated and rejected. Docker Spaces requires credit card billing verification, allocates an unnecessary 16GB RAM (200x overprovisioned for a 75MB service), and signals a frontend prototype sandbox (Gradio/Streamlit) rather than backend infrastructure.
* **Microsoft Azure (Container Apps):** Evaluated via Azure for Students ($100 credit pool). High keyword value, but auxiliary infrastructure (Azure Container Registry [ACR], Log Analytics workspaces, NAT egress) quietly drains $15–25/month, resulting in dead portfolio links in 4–6 months once credits deplete.
* **Render (Web Service):** Selected for production. 100% free forever with zero credit card liability. 
  * *Operational Reality:* Render spins down free containers after 15 minutes of inactivity, introducing a **~40-second cold-start delay** on the first incoming request. Once awake, cached serving executes in **`< 2.2ms`**.

---

## 5. Local Reproduction & Verification

### Prerequisites
* Python $\ge$ 3.11
* Docker (optional, for containerized verification)
* PostgreSQL $\ge$ 16 (or local test container)

### Step 1: Clone & Bootstrap Virtual Environment
```bash
git clone https://github.com/vxqzn/weather-residual-engine.git
cd weather-residual-engine

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -e ".[dev]"
```

### Step 2: Initialize Database Schema & Run Test Suite
```bash
# Configure local test database or service container
export DATABASE_URL="postgres://postgres:postgres@localhost:5432/postgres"

# Bootstrap DDL schema
python -m twre.db.session

# Execute automated 17-test verification suite
pytest -v
```

### Step 3: Run the Local Serving Layer
```bash
uvicorn twre.service.app:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000/docs` to test interactive Swagger UI documentation.

### Step 4: Multi-Stage Docker Container Build
```bash
docker build -t twre:latest .
docker run -p 8000:7860 -e PORT=7860 -e DATABASE_URL="$DATABASE_URL" twre:latest
```

---

## 6. Repository Layout

```
weather-residual-engine/
├── .github/workflows/
│   └── daily_pipeline.yml         # CI/CD: Postgres 16 service container, pytest, runner, Docker audit
├── artifacts/models/
│   ├── champion.joblib            # Active serialized production champion model
│   └── ledger.json                # Append-only model provenance and holdout MAE audit ledger
├── research/
│   ├── inspect_api.py             # Open-Meteo API exploration and rate-limit diagnostics
│   ├── pair_forecast.py           # Historical forecast pairing and summer bias discovery
│   └── train_baseline.py          # Initial chronological temporal baseline proof (-44.3% MAE)
├── sql/
│   └── 01_ddl.sql                 # Relational schema DDL, composite keys, and CHECK constraints
├── src/twre/
│   ├── db/session.py              # Psycopg 3 connection pool with SSL and schema migrator
│   ├── features/pipeline.py       # Causal lag and rolling bias feature store (zero leakage)
│   ├── ingestion/
│   │   ├── backfill.py            # 745-day historical bootstrap infrastructure
│   │   └── worker.py              # Pydantic quarantine parsing and idempotent upsert logic
│   ├── models/
│   │   ├── evaluate.py            # Holdout evaluation comparator against uncorrected baseline
│   │   ├── gate.py                # Automated champion-challenger promotion gating engine
│   │   ├── registry.py            # Atomic model persistence (.tmp -> os.replace) and ledger
│   │   └── train.py               # Temporal train/holdout split and regressor training
│   ├── pipeline/
│   │   └── daily_runner.py        # 14-day sliding lookback daily runner with completeness checks
│   ├── schemas/weather.py         # Pydantic V2 frozen contracts with temporal validators
│   └── service/
│       ├── app.py                 # FastAPI ASGI app with lifespan pre-warming
│       ├── engine.py              # In-memory ModelEngine with st_mtime hot-reloading
│       └── schemas.py             # Service response contracts (PredictionResponse, etc.)
├── tests/
│   ├── test_gate.py               # Cold start, degraded rejection, and promotion unit tests
│   ├── test_ingestion.py          # Upsert idempotency, quarantine, and error calculation tests
│   ├── test_schemas.py            # Boundary checks, temporal order, and frozen model tests
│   └── test_service.py            # Health, metrics, cached latency SLA, and fallback tests
├── Dockerfile                     # Two-stage lean build with non-root appuser
└── pyproject.toml                 # PEP 621 packaging dependencies and pytest configuration
```
