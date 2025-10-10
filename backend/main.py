"""
Fraud Detection Demo - Complete Backend
INFO 492 - Week 3 Demo #1
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal
import json
import re
from datetime import datetime

# ============ DATA MODELS ============


class Transaction(BaseModel):
    id: str
    amount: float
    merchant: str
    device_id: str
    geo: str
    velocity_30d: int
    avg_amount_30d: float
    merchant_known: bool
    label: Literal["LEGIT", "FRAUD"]


class FraudDecision(BaseModel):
    decision: Literal["FRAUD", "LEGIT", "NEEDS_REVIEW"]
    confidence: float
    flags: List[str]
    explanation: str


class CaseResult(BaseModel):
    txn_id: str
    decision: str
    confidence: float
    flags: List[str]
    true_label: str


class BatchResult(BaseModel):
    total: int
    results: List[CaseResult]
    metrics: dict


# ============ FASTAPI APP ============

app = FastAPI(title="Fraud Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load transaction data
with open("data.json", "r") as f:
    TRANSACTIONS = {t["id"]: Transaction(**t) for t in json.load(f)}

# ============ GUARDRAILS ============


def mask_pii(text: str) -> str:
    """Mask PII in text"""
    # Mask credit card numbers
    text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[REDACTED_PAN]", text)
    # Mask emails
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]", text
    )
    # Mask phone numbers
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[REDACTED_PHONE]", text)
    return text


def validate_transaction(txn: Transaction) -> tuple[bool, str]:
    """Validate transaction has required fields"""
    if txn.amount <= 0:
        return False, "Invalid amount"
    if not txn.merchant:
        return False, "Missing merchant"
    if not txn.device_id:
        return False, "Missing device_id"
    return True, ""


# ============ FRAUD DETECTION RULES ============


def analyze_transaction(txn: Transaction) -> FraudDecision:
    """
    Deterministic fraud detection with red flags:
    - large_amount: amount > max(800, 5 * avg)
    - new_merchant: merchant not seen before
    - velocity_spike: low velocity but large transaction
    - geo_risk: non-US geography
    - device_new: very low velocity (first-time user pattern)
    """
    flags = []

    # Check for large amount
    threshold = max(800, 5 * txn.avg_amount_30d)
    if txn.amount > threshold:
        flags.append("large_amount")

    # Check for unknown merchant
    if not txn.merchant_known:
        flags.append("new_merchant")

    # Check for velocity spike (low velocity + large amount)
    if txn.velocity_30d <= 3 and txn.amount > 500:
        flags.append("velocity_spike")

    # Check for geographic risk
    if not txn.geo.endswith("-US"):
        flags.append("geo_risk")

    # Check for new device pattern
    if txn.velocity_30d == 0:
        flags.append("device_new")

    # Decision logic
    num_flags = len(flags)

    if num_flags >= 2:
        # High confidence fraud
        confidence = min(0.95, 0.70 + (num_flags * 0.08))
        decision = "FRAUD"
        explanation = f"Multiple red flags detected: {', '.join(flags)}"
    elif num_flags == 0 and txn.velocity_30d >= 5:
        # Low risk, established pattern
        confidence = 0.60 + min(0.25, txn.velocity_30d / 100)
        decision = "LEGIT"
        explanation = "No red flags, established transaction pattern"
    else:
        # Uncertain - needs human review
        confidence = 0.50
        decision = "NEEDS_REVIEW"
        explanation = f"Uncertain: {num_flags} flag(s), needs analyst review"

    return FraudDecision(
        decision=decision,
        confidence=round(confidence, 2),
        flags=flags,
        explanation=explanation,
    )


# ============ METRICS CALCULATION ============


def calculate_metrics(results: List[CaseResult]) -> dict:
    """Calculate precision, recall, confusion matrix"""
    tp = fp = fn = tn = 0

    for r in results:
        if r.decision == "FRAUD" and r.true_label == "FRAUD":
            tp += 1
        elif r.decision == "FRAUD" and r.true_label == "LEGIT":
            fp += 1
        elif r.decision == "LEGIT" and r.true_label == "FRAUD":
            fn += 1
        elif r.decision == "LEGIT" and r.true_label == "LEGIT":
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    refusal_count = sum(1 for r in results if r.decision == "NEEDS_REVIEW")
    refusal_rate = refusal_count / len(results) if results else 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "refusal_rate": round(refusal_rate, 3),
        "total": len(results),
    }


# ============ API ENDPOINTS ============


@app.get("/")
def root():
    return {"service": "Fraud Detection API", "version": "1.0.0", "status": "running"}


@app.post("/analyze")
def analyze_single(txn_id: str) -> dict:
    """Analyze a single transaction"""
    if txn_id not in TRANSACTIONS:
        raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not found")

    txn = TRANSACTIONS[txn_id]

    # Validate
    valid, error = validate_transaction(txn)
    if not valid:
        return {
            "txn_id": txn_id,
            "decision": "NEEDS_REVIEW",
            "confidence": 0.5,
            "flags": [],
            "explanation": f"Validation failed: {error}",
            "true_label": txn.label,
        }

    # Analyze
    result = analyze_transaction(txn)

    return {
        "txn_id": txn_id,
        "decision": result.decision,
        "confidence": result.confidence,
        "flags": result.flags,
        "explanation": mask_pii(result.explanation),
        "true_label": txn.label,
    }


@app.post("/batch")
def analyze_batch() -> BatchResult:
    """Analyze all transactions"""
    results = []

    for txn_id, txn in TRANSACTIONS.items():
        valid, error = validate_transaction(txn)
        if not valid:
            results.append(
                CaseResult(
                    txn_id=txn_id,
                    decision="NEEDS_REVIEW",
                    confidence=0.5,
                    flags=[],
                    true_label=txn.label,
                )
            )
            continue

        decision = analyze_transaction(txn)
        results.append(
            CaseResult(
                txn_id=txn_id,
                decision=decision.decision,
                confidence=decision.confidence,
                flags=decision.flags,
                true_label=txn.label,
            )
        )

    metrics = calculate_metrics(results)

    return BatchResult(
        total=len(results), results=results[-20:], metrics=metrics  # Last 20 for UI
    )


@app.get("/transactions")
def list_transactions():
    """List all transaction IDs"""
    return {"total": len(TRANSACTIONS), "ids": list(TRANSACTIONS.keys())}


@app.get("/data")
def get_raw_data():
    """Get raw transaction data as JSON"""
    return {"transactions": [txn.dict() for txn in TRANSACTIONS.values()]}


@app.get("/transaction/{txn_id}")
def get_transaction_details(txn_id: str):
    """Get detailed transaction data"""
    if txn_id not in TRANSACTIONS:
        raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not found")
    
    txn = TRANSACTIONS[txn_id]
    return {
        "transaction": txn.dict(),
        "analysis": analyze_transaction(txn).dict()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
