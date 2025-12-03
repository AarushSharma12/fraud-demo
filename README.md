# Fraud Detection Co-Pilot — Finance — Defense

**Course:** INFO 498B — Agentic Cybersecurity with AI & LLMs

**Team:** Team 4 — Aarush Sharma, Nausheer Syed, Michael Ibrahim

**One-line pitch:** An adaptive fraud detection system using Reinforcement Learning that learns from analyst feedback to outperform static rule-based systems, designed for financial institutions needing explainable, auditable AI decisions.

---

## 1) Live Demo

| Component | URL | Status | Notes |
|-----------|-----|--------|-------|
| **Synthetic Industry** | `http://is-info492.ischool.uw.edu:8004` | Up | Backend API |
| **Frontend Dashboard** | [TBD - deploy to homes.cs] | Up | Test creds: `analyst-yubikey-456` / `analyst2024!` |
| **Logs/Observability** | `/backend/audit_logs/audit.jsonl` | — | Immutable JSONL audit trail |

---

## 2) Thesis & Outcome

**Original thesis (week 2):**
> A Proximal Policy Optimization (PPO) agent treating fraud detection as a cost-sensitive game can outperform static rule-based baselines by learning to maximize long-term business value rather than just minimizing immediate error rates.

**Final verdict:** [True / False / Partially true — TBD after experiments]

**Why (top evidence):**
1. [Evidence 1 — TBD]
2. [Evidence 2 — TBD]
3. [Evidence 3 — TBD]

---

## 3) What We Built

### Synthetic Industry
- **FastAPI Backend** (43 endpoints) — Transaction analysis, model training, live feed streaming
- **Transaction Generator** (`prod_generator.py`) — Monte Carlo simulation producing realistic fraud patterns (velocity attacks, geo-anomalies, amount spikes)
- **1000+ synthetic transactions** with ground-truth labels for training/evaluation

### Agentic System
- **PPO Agent** (Stable-Baselines3) — Learns APPROVE/DENY/ESCALATE actions from 12-dimensional state space
- **Custom Gymnasium Environment** — Models fraud detection as sequential decision-making with asymmetric rewards
- **Deterministic Baseline** — Rule-based engine for A/B comparison
- **Model Versioning** — Checkpoint storage with performance metrics

### Key Risks Addressed
- **Adaptive fraud patterns** — RL agent learns to detect evolving attack vectors
- **False negative cost asymmetry** — 4x penalty for missing fraud vs. false positives
- **Human-in-the-loop escalation** — Uncertain cases routed to analysts
- **Audit trail for compliance** — Every decision logged with full provenance

---

## 4) Roles, Auth, Data

### Roles & Permissions

| Role | Train Models | View Models | Analyze Txns | Manage Users |
|------|--------------|-------------|--------------|--------------|
| **Admin** | ✅ | ✅ | ✅ | ✅ |
| **Analyst** | ❌ | ✅ | ✅ | ❌ |
| **Viewer** | ❌ | ✅ | ✅ | ❌ |

### Authentication
- **Simulated YubiKey + Password** — Hardware token simulation with OTP verification
- **Session Tokens** — 8-hour expiry, stored server-side
- **RBAC Middleware** — Permission checks on every protected endpoint

### Test Credentials (Synthetic)
```
Admin:   yubikey_id=admin-yubikey-123   password=admin2024!
Analyst: yubikey_id=analyst-yubikey-456 password=analyst2024!
Viewer:  yubikey_id=viewer-yubikey-789  password=viewer2024!
```

### Data
- **Synthetic only** — No real PII or financial data
- **Generator:** `generate_data.py` — Monte Carlo simulation
- **Schema:** Transaction ID, amount, from/to accounts, type, category, location, channel, velocity metrics, ground-truth label

---

## 5) Experiments Summary (Demos #3 - #5)

### Demo #3
- **Hypothesis:** [TBD]
- **Setup:** [TBD]
- **Result:** [Pass/Fail + one sentence]
- **Evidence:** [link/note]

### Demo #4 (Continuous Run)
- **Uptime:** [xx.x%]
- **Incidents:** [n]
- **Improvement observed:** [Yes/No + brief]

### Demo #5 (Final)
- **What was validated:** [TBD]
- **Result:** [one sentence]
- **Evidence:** [link/note]

---

## 6) Key Results (Plain Text)

| Metric | Value |
|--------|-------|
| **Effectiveness** | RL accuracy: [X%], Rule-based accuracy: [Y%], Improvement: [Z%] |
| **Reliability** | Uptime: [X%], Mean response time: [Y ms] |
| **Safety** | Policy violations blocked: [N], Unauthorized access attempts: [M] |

---

## 7) How to Use / Deploy

### Prerequisites
- Python 3.11+
- `tmux` (for production orchestration)
- Network access to UW iSchool servers (for team deployment)

### Environment Variables
```bash
# Optional - for HTTPS mode
SSL_KEYFILE=/path/to/key.pem
SSL_CERTFILE=/path/to/cert.pem
FORCE_HTTP=true  # Set if behind reverse proxy
```

### Deploy Steps

**Local Development:**
```bash
cd fraud-demo
./start.sh
```

**Production (UW Server):**
```bash
ssh <netid>@is-info492.ischool.uw.edu
cd /srv/teams/team4
./start_tmux.sh
```

### Test Steps
```bash
# Verify backend is running
curl http://localhost:8004/

# Train RL model
curl -X POST "http://localhost:8004/rl/train?timesteps=20000"

# Compare RL vs Rules
curl -X POST "http://localhost:8004/compare/T0001"

# Run full test suite
python test_rl.py
```

---

## 8) Safety, Ethics, Limits

### Data Safety
- **Synthetic data only** — No real credentials, PII, or organizational systems
- **PII Masking** — Automatic redaction in API responses

### Controls
- **Role gating** — RBAC on all sensitive endpoints
- **Session timeout** — 8-hour automatic expiry
- **Audit logging** — Immutable JSONL ledger of all actions
- **Rate limiting** — [TBD if implemented]

### Known Limits / Failure Modes
- Cold start requires initial training (~16k timesteps, ~30 seconds)
- Model performance degrades on distribution shift without retraining
- Single-node deployment (no HA/failover)
- In-memory session storage (lost on restart)

---

## 9) Final Deliverables

| Deliverable | Link |
|-------------|------|
| **1000-word paper** | [TBD] |
| **Slides** | [TBD] |
| **Evidence folder** | `/evidence/` |
| **Demo video** | [TBD] |

---

## 10) Next Steps

1. **Implement continuous learning** — Online model updates from analyst feedback
2. **Add explainability layer** — SHAP/LIME for decision transparency
3. **Deploy HA architecture** — Redis sessions, load balancing, model serving

---

## Project Structure

```
fraud-demo/
├── backend/
│   ├── main.py              # FastAPI application (43 endpoints)
│   ├── models/              # Serialized PPO models & scalers
│   ├── audit_logs/          # Immutable audit trails (JSONL)
│   ├── data.json            # Synthetic training dataset
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # Event-driven UI (Vanilla JS + SSE)
├── generate_data.py         # Monte Carlo transaction generator
├── prod_generator.py        # Live traffic simulator
├── start_tmux.sh            # Production orchestrator
├── start.sh                 # Development startup script
└── test_rl.py               # CLI verification tool
```

---

**Maintainers:** Aarush Sharma • **Contact:** [your-email@uw.edu]
