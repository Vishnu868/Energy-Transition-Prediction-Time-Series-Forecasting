# Energy Trip AI — Early Power Trip Prediction System

AI-powered time-series system for predicting electrical breaker trips and identifying fault appliances.

## Tech Stack
- **Backend:** FastAPI + PyTorch + Pandas + Statsmodels + SHAP
- **Frontend:** HTML + CSS + JavaScript + Chart.js
- **Models:** LSTM Forecaster, Bidirectional LSTM Classifier with Attention

---

## Prerequisites
- Python 3.9+
- pip
- A UK-DALE or compatible CSV file

---

## Quick Start

### 1. Clone / set up project structure
```bash
mkdir energy-trip-ai && cd energy-trip-ai
# Create all directories
mkdir -p backend/{routes,services,models,training,utils}
mkdir -p frontend data saved_models uploads
```

### 2. Create virtual environment
```bash
python -m venv venv

# Activate (Linux/macOS):
source venv/bin/activate

# Activate (Windows):
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r backend/requirements.txt
```

---

## Data Preparation

Place your UK-DALE CSV in `data/`. The expected format:
```
timestamp,power
1371652001,26
1371652005,25
```

For multi-appliance data:
```
timestamp,kettle,fridge,washing_machine,microwave
1371652001,2500,120,0,0
1371652005,2500,118,1200,0
```

---

## Training the Models

### Train Forecaster:
```bash
cd backend
python -m training.train_forecaster \
    --csv ../data/sample_ukdale.csv \
    --epochs 50 \
    --seq_len 60 \
    --pred_len 10 \
    --voltage 230
```

### Train Classifier:
```bash
cd backend
python -m training.train_classifier \
    --csv ../data/sample_ukdale.csv \
    --breaker 15 \
    --voltage 230 \
    --epochs 40 \
    --seq_len 10
```

### Evaluate Both Models:
```bash
cd backend
python -m training.evaluate \
    --csv ../data/sample_ukdale.csv \
    --breaker 15
```

---

## Running the Application

### Start Backend API:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

### Open Frontend:
Open `frontend/index.html` directly in your browser, OR serve it:
```bash
cd frontend
python -m http.server 3000
# Open: http://localhost:3000
```

---

## Usage Flow

1. Open `index.html`
2. Upload your CSV file
3. Set breaker rating (e.g., 15A), voltage (230V), location type
4. Click **"Analyze Power Data"**
5. View dashboard with:
   - Real-time power timeline + forecast
   - Trip probability and class
   - Stability score (0-100)
   - Appliance fault ranking
   - SHAP AI explainability

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload CSV file |
| POST | `/api/predict` | Run full prediction pipeline |
| GET | `/health` | Health check |

---

## Testing with UK-DALE Data

UK-DALE data download: https://ukerc.rl.ac.uk/DC/cgi-bin/edc_search.pl

1. Download House 1 appliance data
2. Export to CSV with columns: `timestamp`, `power` (or multiple appliance columns)
3. Place in `data/` directory
4. Train models with default 230V, 15A settings
5. Upload via frontend and analyze

---

## Extending to IoT

Future extension points:
- Replace CSV upload with MQTT broker subscriber (`paho-mqtt`)
- Stream real-time readings to the same preprocessing pipeline
- Add WebSocket endpoint in FastAPI for live dashboard updates
- Deploy on Raspberry Pi with CT sensor (e.g., SCT-013) → ADS1115 ADC
- Use `asyncio` + `aiomqtt` for non-blocking data ingestion
- Store readings in InfluxDB time-series database
- Use Grafana for real-time monitoring alongside this dashboard

---

## Project Architecture Summary
```
CSV Upload
    ↓
Preprocessor (timestamp parse, resample, interpolate)
    ↓
Feature Engineer (rolling stats, lag, CUSUM, ADF)
    ↓
Label Generator (breaker simulation → 0/1/2)
    ↓
┌─────────────────┬──────────────────┐
│ LSTM Forecaster │ LSTM Classifier  │
│ (next N watts)  │ (Normal/Warn/Trip)│
└─────────────────┴──────────────────┘
    ↓
Fault Detector (appliance contribution ranking)
    ↓
SHAP Explainability
    ↓
Stability Score (0-100)
    ↓
FastAPI JSON Response → Dashboard
```