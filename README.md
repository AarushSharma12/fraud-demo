# Fraud Detection Co-Pilot - INFO 492

**Team**: Aarush, Mike, Nausheer  
**Industry**: Finance  
**Posture**: Protect  

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Hypothesis](#-hypothesis)
3. [Quick Start](#-quick-start)
4. [Code Structure](#-code-structure)
5. [Data](#-data)
6. [Dependencies](#-dependencies)
7. [LLM Prompts](#-llm-prompts)
8. [What This Demonstrates](#-what-this-demonstrates)
9. [Demo Script](#-demo-script-friday-presentation)
10. [Architecture](#️-architecture)
11. [Key Features](#-key-features)
12. [API Endpoints](#api-endpoints)
13. [Performance Metrics](#-success-metrics)
14. [Class Feedback & Actions](#-class-feedback--action-items)
15. [Transcripts from stakeholder interviews](#-transcripts-from-stakeholder-interviews)
16. [Next Steps](#-next-steps-demo-2)
17. [Privacy & Security](#-privacy--security)

---

## Overview

### The Dynamic Challenge of Fraud

Financial fraud is a dynamic "cat-and-mouse" game where attackers constantly evolve, rendering static rule-based systems obsolete. Our team's central hypothesis is that an AI agent using Reinforcement Learning (RL) can overcome these static constraints.

Our Week 2 hypothesis proposed an RL agent could continuously improve fraud detection by learning from outcomes, leading to faster, more accurate, and safer decisions. We track KPIs like precision, recall, and F1-scores, alongside latency and stability, all within a strict governance framework requiring human-in-the-loop approval and action simulation.

This analysis examines our hypothesis's validation through two experiments and crucial stakeholder feedback. While results validate our model's adaptability and accuracy, feedback has expanded our definition of a "safe" system. Our hypothesis is partially validated, with clear directives to refine governance, cost-sensitivity, and human-factors engineering.

## 🆕 NEW: Reinforcement Learning Integration

This demo now includes a **Reinforcement Learning (RL) fraud detection system** alongside the original rule-based approach! The RL model uses PPO (Proximal Policy Optimization) to learn optimal fraud detection strategies through trial and error.

## 🔐 NEW: Role-Based Access Control (RBAC)

This demo now includes **comprehensive RBAC with simulated YubiKey authentication**! Different user roles have different permissions:

- **Admin**: Can train models, view all models, and manage the system
- **Analyst**: Can view trained models and analyze transactions (cannot train)
- **Viewer**: Can view trained models and analyze transactions (read-only)

Models trained by admins are stored in `backend/models/` and can be viewed by all authenticated users.

## 🎯 Hypothesis

Our central hypothesis is that an AI agent using Reinforcement Learning (RL) can continuously improve fraud detection by learning from outcomes, leading to:

- **Faster** decisions (reduced case-review time by ≥30%)
- **More accurate** detection (improved precision and recall)
- **Continuously improve** through adaptive learning
- **Safer** decisions through governance and human-in-the-loop collaboration

**Quantitative Targets:**
- **Recall** ≥ 0.80 (catch 80%+ of actual fraud)
- **Precision** ≥ 0.75 (75%+ fraud alerts are correct)
- **Refusal Rate** ≤ 15% (escalate ≤15% to humans)
- **Latency** ≤ 2s per transaction

## 🧪 Experiment 1: Establishing the Static Baseline

Our first experiment's objective was to set a quantitative baseline without adaptive learning, serving as a control. We designed a traditional, rule-based system: triggering ≥2 flags (e.g., large_amount, new_merchant) marked a transaction as "FRAUD," while zero flags meant "LEGIT." All other cases went to "Needs Review."

**Results on 1,000-transaction dataset:**
- **Recall**: 0.692 (caught 69.2% of actual fraud)
- **Precision**: 0.634 (63.4% of fraud alerts were correct)
- **False Positives**: 366 cases
- **Refusal Rate**: 18.7%

**Key Findings:**
- The flag-based logic was "overly simplistic" for a complex landscape
- Misclassified legitimate high-value or international transactions
- Created high operational overhead and customer friction
- Failed the "continuously improve" component by design
- Proved why a new approach was necessary

## 🧪 Experiment 2: Validating the RL-Driven Adaptive Model

Experiment 2 tested our hypothesis by integrating a Proximal Policy Optimization (PPO) agent to see if learning from feedback could improve adaptability. We used an MLP policy and a business-weighted reward function:

**Reward Structure:**
- +10: Correct fraud detection
- -20: Missing fraud (false negative)
- -5: False alarm (false positive)
- +2: Correct "Needs Review" escalations

**Results on same dataset:**
- **Precision**: 0.789 (+24% improvement)
- **Recall**: 0.856 (+24% improvement)
- **F1-Score**: 0.821 (+24% improvement)
- **Accuracy**: +8% boost
- **False Positives**: -28% reduction
- **Manual Review Queue**: -31% smaller

**Key Findings:**
- RL agent learned complex, non-linear feature interactions
- Reward structure effectively trained agent to "learn from outcomes"
- Validated "faster" and "more accurate" components of hypothesis
- **Limitations noted**: Slight latency increase (7ms to 11ms), reliance on feature engineering, MLP's "black box" nature

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

### 🔐 Login Instructions

When you open the application, you'll be prompted to login with simulated YubiKey credentials:

**Available Users:**
- **Admin**: `admin-yubikey-123` / `admin2024!` - Full access, can train models
- **Analyst**: `analyst-yubikey-456` / `analyst2024!` - Can view models and analyze
- **Viewer**: `viewer-yubikey-789` / `viewer2024!` - Read-only access

**Different Roles See:**
- Admin users see a "Train New Model" button and can access all training endpoints
- Analyst/Viewer users see a "Trained Models" section where they can view all models trained by admins

### 🧠 Testing the RL System

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Start the server
python main.py

# In another terminal, test the RL system
python test_rl.py
```

### 📊 Generating New Data

```bash
# Generate a new 1000-transaction dataset
python generate_data.py

# This creates realistic fraud patterns with:
# - 818 legitimate transactions (81.8%)
# - 182 fraud cases (18.2%)
# - Complex merchant categories and geographic patterns
# - Realistic velocity and amount distributions
```

## 📊 What This Demonstrates

### 1. Defense Posture (PROTECT)

- **Guardrails**: PII masking, field validation, input sanitization
- **Auditability**: Full decision logs with explanations
- **Human-in-the-Loop**: NEEDS_REVIEW cases escalate to analysts
- **Explainability**: Each decision shows specific red flags

### 2. Fraud Detection Engine

**Rule-Based System:**
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

**NEW: Reinforcement Learning System:**
- **Environment**: Custom Gymnasium environment with 7 normalized features
- **Agent**: PPO (Proximal Policy Optimization) with MLP policy
- **Actions**: 0=FRAUD, 1=LEGIT, 2=NEEDS_REVIEW
- **Reward Function**: 
  - +10 for correct fraud detection
  - +1 for correct legitimate classification
  - -20 for missing fraud (false negative)
  - -5 for false fraud alerts (false positive)
- **Training**: Learns optimal decision strategies through trial and error

### 3. Performance Metrics

- Real-time precision/recall calculation
- Confusion matrix visualization
- Threshold monitoring (green/red indicators)
- Processing time tracking

## 🎬 Demo 

### Setup 

1. Open terminal: `cd fraud-demo && ./start.sh`
2. Browser opens automatically to http://localhost:8080
3. Connection test happens automatically

### Demo Flow (5 minutes)

#### Part 1: System Overview (1 min)

- Show the UI - "Fraud Detection Co-Pilot"
- Explain hypothesis and thresholds
- Point out RBAC roles (Support Rep → Analyst → Manager)

#### Part 2: Batch Analysis (2 min)

1. Click **"Run Batch Analysis"**
2. System processes all transactions in <1 second
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
│ (1000 samples)  │
└─────────────────┘
```

## 📁 Project Structure

```
fraud-demo/
├── README.md                    # This file
├── start.sh                     # One-command startup script
├── .gitignore                   # Git ignore patterns
├── backend/
│   ├── main.py                  # Complete FastAPI backend
│   ├── requirements.txt         # Python dependencies
│   └── data.json               # 1000 synthetic transactions
└── frontend/
    └── index.html              # Complete UI (HTML+CSS+JS)
```

---

## 💻 Code Structure

### Backend (`backend/main.py`)

The backend is a comprehensive FastAPI application with two fraud detection systems:

#### 1. Rule-Based Fraud Detection System
- **Red Flag Detection Engine** (lines ~50-150): Analyzes transactions for suspicious patterns
  - Large amount detection (>5x average or >$800)
  - New merchant detection
  - Velocity spike analysis (low activity + large transaction)
  - Geographic risk scoring (non-US)
  - Device fingerprinting (new devices)

- **Decision Logic** (lines ~150-250): Three-tier classification
  - `FRAUD`: ≥2 red flags detected
  - `LEGIT`: 0 flags + established merchant pattern
  - `NEEDS_REVIEW`: Ambiguous cases escalated to human analysts

#### 2. Reinforcement Learning System 
- **Custom Gymnasium Environment** (lines ~250-400): Fraud detection as an RL problem
  - State space: 7 normalized features (amount, velocity, merchant status, etc.)
  - Action space: 3 discrete actions (FRAUD, LEGIT, NEEDS_REVIEW)
  - Reward function: Cost-sensitive penalties/rewards
    - +10: Correct fraud detection
    - +1: Correct legitimate classification
    - -20: Missed fraud (false negative)
    - -5: False alarm (false positive)

- **PPO Agent** (lines ~400-500): Proximal Policy Optimization
  - Multi-layer perceptron policy network
  - Training loop with configurable timesteps
  - Model persistence (saves trained models)

#### 3. API Endpoints (lines ~500-790)
- **Rule-Based**: `/analyze`, `/batch`, `/metrics`, `/provenance`
- **RL System**: `/rl/train`, `/rl/analyze`, `/rl/batch`, `/rl/status`
- **Comparison**: `/compare/{txn_id}` - side-by-side analysis

#### 4. Privacy & Security Features
- **PII Masking**: Automatic masking of `device_id` in all responses
- **Input Validation**: Pydantic models enforce type safety
- **Audit Logging**: Decision provenance tracking
- **CORS Configuration**: Secure cross-origin requests

### Frontend (`frontend/index.html`)

**Single-Page Application** with inline HTML, CSS, and JavaScript:
- Real-time KPI dashboard (precision, recall, refusal rate)
- Interactive transaction analysis interface
- Batch processing with live metrics updates
- Confusion matrix visualization
- Provenance modal for audit trails

### Startup Script (`start.sh`)

**113 lines** of bash automation:
- Virtual environment setup
- Dependency installation
- Port availability checks
- Parallel process management (backend + frontend)
- Automatic browser launch
- Graceful shutdown handling

---

## 📊 Data

### Primary Dataset (`backend/data.json`)

**Size**: 1000 synthetic transactions  
**Fraud Rate**: 18.2% (182 fraud, 818 legitimate)  
**Format**: JSON array of transaction objects

#### Transaction Schema

```json
{
  "id": "T001",
  "amount": 45.67,
  "merchant": "Starbucks",
  "device_id": "device_f123abc",
  "geo": "US",
  "velocity_30d": 15,
  "avg_amount_30d": 42.50,
  "merchant_known": true,
  "label": "LEGIT"
}
```

#### Feature Descriptions

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| `id` | string | Unique transaction identifier | T001-T1000 |
| `amount` | float | Transaction amount in USD | $5 - $50,000 |
| `merchant` | string | Merchant name | Known brands or suspicious names |
| `device_id` | string | Hashed device identifier | device_[hash] |
| `geo` | string | Geographic origin | US, RU, CN, NG, BR, MX, etc. |
| `velocity_30d` | int | Transactions in past 30 days | 0-30 |
| `avg_amount_30d` | float | Average transaction amount | $0-$1000 |
| `merchant_known` | bool | Merchant verification status | true/false |
| `label` | string | Ground truth label | LEGIT/FRAUD |

#### Fraud Patterns in Dataset

**LEGIT Transactions (818, 81.8%)**:
- **Merchants**: Starbucks, Amazon, Netflix, Uber, Apple, Target, Whole Foods, etc.
- **Geography**: Predominantly US (98%+)
- **Amounts**: $5-$500 (normal consumer spending)
- **Velocity**: 5-30 transactions/month (active users)
- **Merchant Status**: 95%+ known/verified merchants

**FRAUD Transactions (182, 18.2%)**:
- **Merchants**: CryptoExchangeX, OnlineCasino777, DarkWebMarket, PharmaDirectXYZ, etc.
- **Geography**: Foreign origins (RU 35%, CN 25%, NG 20%, BR 15%, MX 5%)
- **Amounts**: Bimodal distribution
  - Large: $800-$50,000 (80% of fraud)
  - Small test charges: $5-$50 (20% of fraud)
- **Velocity**: Low activity (0-5 transactions/month)
- **Merchant Status**: 90%+ unknown/unverified merchants
- **Complex Patterns**: Mix of legitimate-looking and overtly suspicious features

#### Data Generation

The dataset was synthetically generated with realistic statistical distributions:
- **Merchant Categories**: E-commerce, food, entertainment, gambling, crypto
- **Geographic Distribution**: Follows real-world fraud geography patterns
- **Velocity Patterns**: Normal users (Poisson distribution) vs. dormant accounts
- **Amount Distribution**: Log-normal for legitimate, power-law for fraud
- **Correlation Structure**: Realistic feature interdependencies

---

## 📦 Dependencies

### Experiment #1: Rule-Based Fraud Detection

**Python Version**: 3.9+

```txt
fastapi==0.115.0        # Web framework for API
uvicorn[standard]==0.30.0  # ASGI server
pydantic==2.9.0         # Data validation
```

**Installation**:
```bash
cd backend
pip install fastapi uvicorn pydantic
```

**No external AI/ML libraries required** - pure rule-based logic using Python standard library.

### Experiment #2: Reinforcement Learning Integration

**Python Version**: 3.9+

```txt
# Core Dependencies (from Experiment #1)
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.9.0

# New RL Dependencies
gymnasium==0.29.1          # RL environment framework (OpenAI Gym successor)
stable-baselines3==2.2.1   # PPO algorithm implementation
numpy==1.26.4              # Numerical computing
scikit-learn==1.3.2        # Feature scaling (StandardScaler)
torch>=2.1.0               # PyTorch for neural networks (PPO policy)
```

**Installation**:
```bash
cd backend
pip install -r requirements.txt
```

**System Requirements**:
- **CPU**: 2+ cores recommended for training
- **RAM**: 4GB minimum (8GB recommended for large training runs)
- **Storage**: 100MB for model checkpoints
- **GPU**: Optional (CPU training is fast enough for this dataset)

### Frontend Dependencies

**Zero npm dependencies** - vanilla JavaScript with modern ES6+ features:
- Fetch API for HTTP requests
- Async/await for asynchronous operations
- Template literals for dynamic HTML
- CSS Grid and Flexbox for layout

**Browser Requirements**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

---

## 🤖 LLM Prompts

### Current Status

The current implementation uses:
1. **Rule-based system**: Deterministic logic with hand-crafted fraud detection rules
2. **RL system**: Reinforcement learning with PPO (no language model)

### Planned for Future Experiments (Demo #2+)

The following LLM integration prompts are planned for upcoming experiments:

#### Prompt Template #1: Transaction Classification

```
You are a fraud detection expert. Analyze the following transaction and classify it as FRAUD, LEGIT, or NEEDS_REVIEW.

Transaction Details:
- Amount: ${amount}
- Merchant: {merchant}
- Geography: {geo}
- Account Velocity: {velocity_30d} transactions in past 30 days
- Average Amount: ${avg_amount_30d}
- Merchant Known: {merchant_known}

Consider these red flags:
1. Large unusual amounts (>5x average or >$800)
2. Unknown/unverified merchants
3. Foreign geography (non-US)
4. Low velocity + high amount combinations
5. Suspicious merchant categories (crypto, gambling, dark web)

Response Format:
Decision: [FRAUD|LEGIT|NEEDS_REVIEW]
Confidence: [0.0-1.0]
Reasoning: [Your explanation]
Red Flags: [Comma-separated list]
```

#### Prompt Template #2: Explanation Generation

```
Explain the following fraud detection decision to a non-technical fraud analyst.

Decision: {decision}
Confidence: {confidence}
Red Flags Detected: {flags}

Generate a clear, concise explanation that:
1. States the decision and confidence level
2. Explains which red flags were detected
3. Provides context on why these flags matter
4. Recommends next steps for the analyst

Use simple language and avoid technical jargon.
```

#### Prompt Template #3: Few-Shot Learning

```
You are a fraud detection system. Learn from these examples and classify the new transaction.

Examples:
[LEGIT] $45.67 at Starbucks, US, 15 txns/month → Known merchant, normal pattern
[FRAUD] $8,500 at CryptoExchangeX, RU, 2 txns/month → Large amount, foreign geo, unknown merchant
[LEGIT] $12.99 at Netflix, US, 22 txns/month → Subscription service, established user

New Transaction:
Amount: ${amount}
Merchant: {merchant}
Geography: {geo}
Velocity: {velocity_30d} txns/month

Classification: 
```

#### Evaluation Plan

When LLM integration is implemented, we will compare:
- **Accuracy**: LLM vs. Rules vs. RL on test set
- **Latency**: Response time (<2s target)
- **Explainability**: Reasoning quality (human evaluation)
- **Cost**: API costs per transaction
- **Consistency**: Same transaction classified consistently across runs

**Target Models**: Claude 3.5 Sonnet, GPT-4o

---

## 🔑 Key Features

### API Endpoints

**Original Rule-Based System:**
- `GET /` - Health check
- `POST /analyze?txn_id={id}` - Analyze single transaction
- `POST /batch` - Process all transactions (batch analysis)
- `GET /transactions` - List available IDs
- `GET /metrics` - Returns precision/recall/refusal and confusion matrix
- `GET /provenance/{txn_id}` - Returns explanation + steps (PII masked)

**NEW: Reinforcement Learning System:**
- `POST /rl/train?timesteps={n}` - Train RL model
- `POST /rl/analyze/{txn_id}` - Analyze with RL model
- `POST /rl/batch` - Batch analysis with RL
- `GET /rl/status` - Check RL model status
- `POST /compare/{txn_id}` - Compare rule-based vs RL predictions

## 🔒 Privacy & Security

**Privacy Note:** All serialized responses mask `device_id`; raw data on disk is synthetic and contains no real PII. The system implements comprehensive PII masking for all API responses while preserving data utility for fraud detection analysis.

## 📈 Success Metrics

The demo achieves:

- ✅ **Precision**: ~85% (target ≥75%)
- ✅ **Recall**: ~88% (target ≥80%)
- ✅ **Refusal**: ~12% (target ≤15%)
- ✅ **Latency**: <50ms/txn (target ≤2s)
- ✅ **Setup Time**: 30 seconds
- ✅ **Code Size**: ~1290 lines total (790 backend + 500 frontend)
- ✅ **Dataset Size**: 1000 transactions (18.2% fraud rate)

## 📝 Class Feedback & Action Items

### Feedback Summary

**Key Strengths Identified**:
- ✅ Real-time KPI/metrics dashboard implementation
- ✅ Provenance & audit features with PII masking
- ✅ Clear review workflow for human-in-the-loop cases
- ✅ Comprehensive confusion matrix and performance tracking

### Consolidated Feedback Themes → Actions

#### 1. Real-time KPIs & Measurement Clarity
**Feedback**: Keep live KPI cards; add fixed operating-point confusion matrix and expose refusal rate alongside precision/recall.

**Actions Taken**:
- ✅ Implemented real-time KPI dashboard showing Precision, Recall, and Refusal Rate
- ✅ Added confusion matrix with TP/FP/TN/FN counts
- ✅ Color-coded threshold indicators (green = meeting targets, red = below)
- 🔄 **TODO**: Add fixed operating-point confusion matrix (at 0.5 threshold)
- 🔄 **TODO**: Display refusal rate side-by-side with PR/F1 in metrics view

#### 2. Error Analysis & Drilldowns
**Feedback**: Deepen error analysis, especially around refusals, false positives/negatives, and pattern-specific thresholds.

**Actions Planned**:
- 🔄 **TODO**: Add Error Analysis view with slicing by pattern (e.g., new-merchant × large-amount)
- 🔄 **TODO**: Show FP/FN counts per pattern family with example cases
- 🔄 **TODO**: Implement drilldown into specific error categories:
  - False Negatives: Which fraud cases were missed and why?
  - False Positives: Which legitimate transactions were flagged incorrectly?
  - Refusals: What patterns trigger human escalation most often?
- 🔄 **TODO**: Add pattern-specific threshold tuning interface

#### 3. Cost-Aware Reward Shaping
**Feedback**: Formalize cost-sensitive training with explicit FN loss vs. FP operational cost.

**Actions Taken**:
- ✅ Implemented cost-sensitive reward function in RL system:
  - FN penalty: -20 (missed fraud = high business risk)
  - FP penalty: -5 (false alarm = operational cost)
  - TP reward: +10 (caught fraud = high value)
  - TN reward: +1 (correct clearance = baseline)

**Actions Planned**:
- 🔄 **TODO**: Make cost parameters configurable via UI
- 🔄 **TODO**: Add cost-benefit analysis dashboard showing:
  - Total fraud prevented ($)
  - Operational cost of false positives ($)
  - Net ROI of the system
- 🔄 **TODO**: Implement per-pattern cost tuning (e.g., crypto fraud has higher FN cost)

#### 4. Human-in-Loop Speed & Bias Control
**Feedback**: Speed up human review experience with batch labeling and reduce anchoring bias by showing model score only after human label.

**Actions Planned**:
- 🔄 **TODO**: Implement batch labeling interface:
  - Queue all `NEEDS_REVIEW` cases
  - Allow rapid keyboard navigation (1=FRAUD, 2=LEGIT, 3=SKIP)
  - Track time-to-action (TAT) per case
- 🔄 **TODO**: Add bias reduction features:
  - Hide model confidence score initially
  - Show model prediction only after analyst submits their label
  - A/B test: show score vs. hide score, measure agreement rates
- 🔄 **TODO**: Track and display human review TAT metrics:
  - Average time per case
  - Cases pending review
  - Analyst agreement rate with model

#### 5. Provenance/Audit in the UI
**Feedback**: Keep provenance modal with model version, features/flags used, masked fields, audit log ID; allow export.

**Actions Taken**:
- ✅ Implemented provenance modal showing:
  - Model version (rule-based vs. RL)
  - Decision reasoning and red flags
  - PII-masked device_id
  - Timestamp and transaction details

**Actions Planned**:
- 🔄 **TODO**: Add to provenance modal:
  - Audit log ID (unique trace identifier)
  - Feature values used in decision
  - Model confidence breakdown
  - Decision lineage (if reviewed by human, show original model decision + human override)
- 🔄 **TODO**: Implement per-alert trace export:
  - JSON export for compliance
  - CSV export for batch analysis
  - PDF export for stakeholder reports
- 🔄 **TODO**: Add audit log search/filter interface

### Implementation Priority (Next 2 Weeks)

**High Priority**:
1. Error Analysis view with pattern slicing
2. Cost-aware threshold tuning interface
3. Batch labeling for NEEDS_REVIEW cases
4. Enhanced provenance modal with audit log ID

**Medium Priority**:
5. Fixed operating-point confusion matrix
6. Bias reduction features (hide score initially)
7. TAT tracking dashboard

**Low Priority** (Post-Demo #2):
8. Cost-benefit ROI calculator
9. Per-alert trace export (JSON/CSV/PDF)
10. Audit log search interface

---

## 📝 Reflection on Industry & Class Feedback: Redefining "Safer Decisions"

A model that is "more accurate" is not necessarily "safer" in finance. The "safer decisions" component of our hypothesis was profoundly shaped by external feedback.

### Fidelity Investments Stakeholder Feedback

A stakeholder from Fidelity Investments emphasized practical governance: tracking key metrics ("precision, recall, refusal") and ensuring "auditable decisions and PII protections." This prompted a critical shift. We responded by building a /metrics endpoint, live KPI cards, a provenance modal for auditing, and PII masking. Demonstrating this audit trail validated the importance of traceable governance and encouraged us to get more granular feedback on operationalizing threshold tuning.

**Key Stakeholder Requirements:**
- **Tracking Metrics**: Keep track of precision, recall, and confusion counts. What cost of FP vs. FN is acceptable for this fraud type?
- **Tune Threshold Patterns**: Adjust weights to balance recall and false positives for each fraud type.
- **Make Decisions Auditable**: All alerts and data must be traceable and protected. Every alert should include features used, rule/LLM rationale, and model/version ID.

### Class Feedback Integration

Class feedback also reinforced this focus. Peers praised our real-time KPIs and provenance, but their best suggestions pushed us to refine the learning mechanism. They suggested "deeper error analysis" and "cost-aware reward shaping" (explicitly weighting FN loss vs. FP cost). This feedback offered a path to stronger validation, challenging us to evolve our "business-weighted" reward function into a more advanced "cost-sensitive" training regimen.

**Class Feedback Themes:**
- **Real-time KPIs & Measurement Clarity**: Keep live KPI cards; add fixed operating-point confusion matrix
- **Error Analysis & Drilldowns**: Deepen error analysis, especially around refusals, false positives/negatives
- **Cost-Aware Reward Shaping**: Formalize cost-sensitive training with explicit FN loss vs. FP operational cost
- **Human-in-Loop Speed & Bias Control**: Speed up human review experience with batch labeling
- **Provenance/Audit in the UI**: Keep provenance modal with model version, features/flags used, masked fields

## 🧠 The Psychological Dimension: Governing the Human-AI System

Our analysis of psychological factors provided nuanced validation for "safer decisions." A safe system must account for the human factors of its analyst users. Our design addresses this with clear KPIs and provenance explanations to mitigate "alert fatigue" and high cognitive load.

**Human Factors Considerations:**
- **Alert Fatigue Mitigation**: Clear KPIs and provenance explanations reduce cognitive load
- **Anchoring Bias Prevention**: Plan to enable batch labeling and experiment with hiding model score until after analyst's initial judgment
- **Mindset Evolution**: From "building more ML" to building governed, auditable AI

**Critical Gap Identified:**
Our "primitive" user model lacks "industry realism" by assuming a single user type. This is unrealistic and insecure. A safe system must reflect varied team responsibilities. Therefore, a key future task is implementing role-based access control (RBAC) to differentiate permissions for analysts, seniors, and governance officers, which is essential for a real-world, human-in-the-loop system.

---

## 📊 Conclusion

At mid-quarter, our hypothesis is partially validated with a clear roadmap. The "faster" and "more accurate" tenets were confirmed by Experiment 2, where our RL agent outperformed the static system. The "continuously improve" tenet was also validated, as the agent learned complex patterns via its reward function.

The "safer" component is the most complex. Feedback has taught us that safety is not just accuracy; it is a product of governance, provenance, cost-sensitive learning, and human-factors engineering. We acknowledge gaps remain in refusal rate calibration, precision stability, mature evaluation design for autonomous runs, and realistic user roles.

**Validation Status:**
- ✅ **Faster**: RL agent achieved target performance improvements
- ✅ **More Accurate**: 24% improvement in precision, recall, and F1-score
- ✅ **Continuously Improve**: Agent learned complex patterns through reward function
- 🔄 **Safer**: Partially validated - requires governance, cost-sensitivity, and human-factors engineering

**Remaining Gaps:**
- Refusal rate calibration
- Precision stability
- Mature evaluation design for autonomous runs
- Realistic user roles (RBAC implementation)

## 🔄 Next Steps (Demo #2)

Our plan for the second half directly addresses the identified gaps:

### High Priority (Immediate)
1. **Cost-Sensitive Training**: Implement configurable cost parameters and cost-benefit analysis dashboard
2. **Shift-Robustness Tests**: Conduct comprehensive evaluation design for autonomous runs
3. **Enhanced Labeling & Governance**: Implement batch labeling interface and bias reduction features
4. **Full RBAC System**: Implement role-based access control for industry realism (analysts, seniors, governance officers)

### Medium Priority
5. **Error Analysis View**: Add pattern slicing and drilldown capabilities
6. **Fixed Operating-Point Confusion Matrix**: Display refusal rate side-by-side with precision/recall
7. **TAT Tracking Dashboard**: Monitor human review time-to-action metrics

### Future Experiments
8. **LLM Integration**: Replace rules with Claude/GPT-4, compare RL vs LLM vs rule-based performance
9. **Advanced Features**: Graph-based fraud networks, temporal pattern analysis, adversarial testing
10. **Production Readiness**: Docker containerization, rate limiting, real-time streaming

**Critical Focus**: Moving from simple simulations to robust A/B testing with industry-realistic user roles and cost-aware governance.

## 📚 References

- FastAPI: https://fastapi.tiangolo.com
- Fraud Detection Patterns: ACFE Fraud Examiner's Manual
- NIST Cybersecurity Framework

---