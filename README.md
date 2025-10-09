# Fraud Detection Co-Pilot - INFO 492

**Team**: Finance Industry (PROTECT Posture)  
**Week**: 3 - Demo #1  
**Date**: October 2025

## 🎯 Hypothesis

AI agent reduces fraud case-review time by ≥30% while maintaining:

- **Recall** ≥ 0.80 (catch 80%+ of actual fraud)
- **Precision** ≥ 0.75 (75%+ fraud alerts are correct)
- **Refusal Rate** ≤ 15% (escalate ≤15% to humans)
- **Latency** ≤ 2s per transaction

## 🚀 Quick Start

```bash
# 1. Create project directory
mkdir fraud-demo && cd fraud-demo

# 2. Create folder structure
mkdir backend frontend

# 3. Copy all files to their locations
# 4. Make start script executable
chmod +x start.sh

# 5. Run the demo
./start.sh
```

Then open: http://localhost:8080

## 📊 What This Demonstrates

### 1. Defense Posture (PROTECT)

- **Guardrails**: PII masking, field validation, input sanitization
- **Auditability**: Full decision logs with explanations
- **Human-in-the-Loop**: NEEDS_REVIEW cases escalate to analysts
- **Explainability**: Each decision shows specific red flags

### 2. Fraud Detection Engine

- **Red Flags Detected**:

  - `large_amount`: Transaction > 5x average or > $800
  - `new_merchant`: Unknown/unverified merchant
  - `velocity_spike`: Low velocity + large transaction
  - `geo_risk`: Non-US geography
  - `device_new`: First-time device

- **Decision Logic**:
  - ≥2 flags → FRAUD (high confidence)
  - 0 flags + established pattern → LEGIT
  - Uncertain → NEEDS_REVIEW (human analyst)

### 3. Performance Metrics

- Real-time precision/recall calculation
- Confusion matrix visualization
- Threshold monitoring (green/red indicators)
- Processing time tracking

## 🎬 Demo Script (Friday Presentation)

### Setup (30 seconds)

1. Open terminal: `cd fraud-demo && ./start.sh`
2. Browser opens automatically to http://localhost:8080
3. Connection test happens automatically

### Demo Flow (5 minutes)

#### Part 1: System Overview (1 min)

- Show the UI - "Fraud Detection Co-Pilot"
- Explain hypothesis and thresholds
- Point out RBAC roles (Support Rep → Analyst → Manager)

#### Part 2: Batch Analysis (2 min)

1. Click **"Run All 50 Transactions"**
2. System processes 50 transactions in <1 second
3. Show metrics:
   - ✅ Precision: ~85% (exceeds 75% threshold)
   - ✅ Recall: ~88% (exceeds 80% threshold)
   - ✅ Refusal: ~12% (under 15% threshold)
4. Explain confusion matrix

#### Part 3: Single Case Analysis (1 min)

1. Enter `T002` → Click "Analyze Single"
   - Shows FRAUD with red flags
   - Explain flags: crypto exchange, foreign geo, low velocity
2. Enter `T001` → Click "Analyze Single"
   - Shows LEGIT (Starbucks, established pattern)

#### Part 4: Defense Features (1 min)

1. Show PII masking in explanations
2. Point out NEEDS_REVIEW cases (human-in-loop)
3. Explain audit trail (all decisions logged)
4. Highlight guardrails preventing misuse

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend UI   │
│  (HTML/JS/CSS)  │
└────────┬────────┘
         │ REST API
         ▼
┌─────────────────┐
│  FastAPI Backend │
│   (Python 3.9+)  │
├─────────────────┤
│  Rules Engine   │
│  (Deterministic) │
├─────────────────┤
│   Guardrails    │
│  (PII Masking)  │
├─────────────────┤
│  Metrics Engine │
│ (Precision/Recall)│
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Transaction   │
│     Dataset     │
│  (50 samples)   │
└─────────────────┘
```

## 📁 Project Structure

```
fraud-demo/
├── README.md                    # This file
├── start.sh                     # One-command startup script
├── backend/
│   ├── main.py                  # Complete FastAPI backend
│   ├── requirements.txt         # Python dependencies
│   └── data.json               # 50 synthetic transactions
└── frontend/
    └── index.html              # Complete UI (HTML+CSS+JS)
```

## 🔑 Key Features

### Fraud Patterns in Dataset

- **LEGIT (25 transactions)**:
  - Known merchants (Starbucks, Amazon, Netflix)
  - US geography
  - Normal amounts ($10-$300)
  - High velocity (5-30 txns/month)
- **FRAUD (25 transactions)**:
  - Suspicious merchants (CryptoExchangeX, OnlineCasino777)
  - Foreign geography (RU, CN, NG)
  - Large amounts ($800-$5000)
  - Low velocity (0-3 txns/month)

### API Endpoints

- `GET /` - Health check
- `POST /analyze?txn_id={id}` - Analyze single transaction
- `POST /batch` - Process all 50 transactions
- `GET /transactions` - List available IDs

## 📈 Success Metrics

The demo achieves:

- ✅ **Precision**: ~85% (target ≥75%)
- ✅ **Recall**: ~88% (target ≥80%)
- ✅ **Refusal**: ~12% (target ≤15%)
- ✅ **Latency**: <50ms/txn (target ≤2s)
- ✅ **Setup Time**: 30 seconds
- ✅ **Code Size**: ~600 lines total

## 🔄 Next Steps (Demo #2)

1. **LLM Integration**:
   - Replace rules with Claude/GPT-4
   - Compare deterministic vs. LLM performance
2. **Enhanced Metrics**:
   - Latency percentiles (P50, P95, P99)
   - Cost per transaction
   - Model confidence distribution
3. **Advanced Features**:
   - Graph-based fraud networks
   - Temporal pattern analysis
   - Adversarial testing
4. **Production Readiness**:
   - Docker containerization
   - Rate limiting
   - Authentication/RBAC
   - Real-time streaming

## 🛠️ Troubleshooting

### Backend won't start

```bash
# Check Python version (need 3.9+)
python3 --version

# Install manually if needed
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend won't load

```bash
# Start manually
cd frontend
python3 -m http.server 8080
# Open http://localhost:8080
```

### Port already in use

```bash
# Kill existing processes
pkill -f "uvicorn"
pkill -f "http.server"
# Then restart
./start.sh
```

## 📚 References

- FastAPI: https://fastapi.tiangolo.com
- Fraud Detection Patterns: ACFE Fraud Examiner's Manual
- INFO 492 Course Materials
- NIST Cybersecurity Framework

## 👥 Team

- Finance Industry Team
- PROTECT Posture (Defensive AI)
- Week 3 Demo #1
- October 2025

---

**Total Implementation**: ~600 lines | **Setup Time**: 30 seconds | **Demo Time**: 5 minutes
