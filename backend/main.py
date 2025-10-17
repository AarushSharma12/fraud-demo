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
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sklearn.preprocessing import StandardScaler
import pickle
import os

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


class RLTrainingResult(BaseModel):
    model_type: str
    training_steps: int
    final_reward: float
    accuracy: float
    precision: float
    recall: float


# ============ REINFORCEMENT LEARNING ENVIRONMENT ============


class FraudDetectionEnv(gym.Env):
    """
    RL Environment for Fraud Detection
    
    State: Normalized transaction features
    Actions: 0=FRAUD, 1=LEGIT, 2=NEEDS_REVIEW
    Reward: Based on classification accuracy and business impact
    """
    
    def __init__(self, transactions_data, scaler=None):
        super().__init__()
        
        self.transactions = transactions_data
        self.transaction_ids = list(transactions_data.keys())
        self.current_idx = 0
        
        # Action space: 3 possible decisions
        self.action_space = spaces.Discrete(3)
        
        # State space: 7 features (amount, velocity, avg_amount, merchant_known, geo_risk, device_new, velocity_spike)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        # Initialize scaler for feature normalization
        self.scaler = scaler or StandardScaler()
        self._fit_scaler()
        
        # Reward weights
        self.reward_weights = {
            'correct_fraud': 10.0,      # High reward for catching fraud
            'correct_legit': 1.0,        # Lower reward for correct legit
            'false_positive': -5.0,      # Penalty for false fraud
            'false_negative': -20.0,     # High penalty for missing fraud
            'review_correct': 2.0,       # Reward for correct review
            'review_incorrect': -1.0     # Small penalty for incorrect review
        }
    
    def _fit_scaler(self):
        """Fit scaler on all transaction features"""
        features = []
        for txn in self.transactions.values():
            features.append(self._extract_features(txn))
        self.scaler.fit(features)
    
    def _extract_features(self, txn):
        """Extract normalized features from transaction"""
        # Convert categorical features to numerical
        geo_risk = 1.0 if not txn.geo.endswith("-US") else 0.0
        device_new = 1.0 if txn.velocity_30d == 0 else 0.0
        velocity_spike = 1.0 if txn.velocity_30d <= 3 and txn.amount > 500 else 0.0
        
        return np.array([
            txn.amount,
            txn.velocity_30d,
            txn.avg_amount_30d,
            1.0 if txn.merchant_known else 0.0,
            geo_risk,
            device_new,
            velocity_spike
        ], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset environment to start of episode"""
        super().reset(seed=seed)
        self.current_idx = 0
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        if self.current_idx >= len(self.transaction_ids):
            return np.zeros(7), 0, True, {}
        
        txn_id = self.transaction_ids[self.current_idx]
        txn = self.transactions[txn_id]
        
        # Get true label
        true_label = txn.label
        
        # Calculate reward based on action and true label
        reward = self._calculate_reward(action, true_label)
        
        # Move to next transaction
        self.current_idx += 1
        done = self.current_idx >= len(self.transaction_ids)
        
        # Get next observation
        next_obs = self._get_observation() if not done else np.zeros(7)
        
        info = {
            'txn_id': txn_id,
            'true_label': true_label,
            'predicted_action': action,
            'reward': reward
        }
        
        return next_obs, reward, done, False, info
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_idx >= len(self.transaction_ids):
            return np.zeros(7)
        
        txn_id = self.transaction_ids[self.current_idx]
        txn = self.transactions[txn_id]
        features = self._extract_features(txn)
        return self.scaler.transform([features])[0]
    
    def _calculate_reward(self, action, true_label):
        """Calculate reward based on action and true label"""
        action_map = {0: "FRAUD", 1: "LEGIT", 2: "NEEDS_REVIEW"}
        predicted = action_map[action]
        
        if predicted == true_label:
            if predicted == "FRAUD":
                return self.reward_weights['correct_fraud']
            elif predicted == "LEGIT":
                return self.reward_weights['correct_legit']
            else:  # NEEDS_REVIEW
                return self.reward_weights['review_correct']
        else:
            if predicted == "FRAUD" and true_label == "LEGIT":
                return self.reward_weights['false_positive']
            elif predicted == "LEGIT" and true_label == "FRAUD":
                return self.reward_weights['false_negative']
            else:  # NEEDS_REVIEW incorrect
                return self.reward_weights['review_incorrect']


# ============ RL MODEL MANAGEMENT ============


class RLModelManager:
    """Manages RL model training and inference"""
    
    def __init__(self, transactions_data):
        self.transactions = transactions_data
        self.model = None
        self.scaler = None
        self.env = None
        self.model_path = "rl_fraud_model.pkl"
        self.scaler_path = "rl_scaler.pkl"
    
    def create_environment(self):
        """Create RL environment"""
        self.env = FraudDetectionEnv(self.transactions, self.scaler)
        return self.env
    
    def train_model(self, total_timesteps=20000):
        """Train PPO model with larger dataset"""
        if self.env is None:
            self.create_environment()
        
        # Create vectorized environment with more environments for larger dataset
        vec_env = make_vec_env(lambda: FraudDetectionEnv(self.transactions, self.scaler), n_envs=8)
        
        # Initialize PPO model with optimized hyperparameters for larger dataset
        self.model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=1, 
            learning_rate=3e-4,
            n_steps=2048,  # Larger steps for more data
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5
        )
        
        # Train model
        print(f"🎯 Training RL model on {len(self.transactions)} transactions...")
        self.model.learn(total_timesteps=total_timesteps)
        
        # Save model and scaler
        self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.env.scaler, f)
        
        return self.model
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = PPO.load(self.model_path)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            return True
        return False
    
    def predict(self, transaction):
        """Predict fraud decision for a transaction"""
        if self.model is None:
            if not self.load_model():
                raise HTTPException(status_code=500, detail="No trained model available")
        
        if self.env is None:
            self.create_environment()
        
        # Extract features and normalize
        features = self.env._extract_features(transaction)
        normalized_features = self.env.scaler.transform([features])
        
        # Predict action
        action, _ = self.model.predict(normalized_features, deterministic=True)
        
        # Convert action to decision
        action_map = {0: "FRAUD", 1: "LEGIT", 2: "NEEDS_REVIEW"}
        decision = action_map[action]
        
        # Calculate confidence (simplified)
        confidence = 0.8 if decision == "FRAUD" else 0.7 if decision == "LEGIT" else 0.5
        
        return decision, confidence


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

# Initialize RL model manager
rl_manager = RLModelManager(TRANSACTIONS)

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


# ============ RL API ENDPOINTS ============


@app.post("/rl/train")
def train_rl_model(timesteps: int = 20000) -> RLTrainingResult:
    """Train the RL fraud detection model"""
    try:
        # Train the model
        model = rl_manager.train_model(total_timesteps=timesteps)
        
        # Evaluate the model
        env = rl_manager.create_environment()
        obs, _ = env.reset()
        
        total_reward = 0
        correct_predictions = 0
        total_predictions = 0
        tp = fp = fn = tn = 0
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if 'true_label' in info:
                true_label = info['true_label']
                predicted = {0: "FRAUD", 1: "LEGIT", 2: "NEEDS_REVIEW"}[action]
                
                if predicted == true_label:
                    correct_predictions += 1
                
                # Calculate confusion matrix
                if predicted == "FRAUD" and true_label == "FRAUD":
                    tp += 1
                elif predicted == "FRAUD" and true_label == "LEGIT":
                    fp += 1
                elif predicted == "LEGIT" and true_label == "FRAUD":
                    fn += 1
                elif predicted == "LEGIT" and true_label == "LEGIT":
                    tn += 1
                
                total_predictions += 1
            
            done = done or truncated
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return RLTrainingResult(
            model_type="PPO",
            training_steps=timesteps,
            final_reward=total_reward,
            accuracy=round(accuracy, 3),
            precision=round(precision, 3),
            recall=round(recall, 3)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/rl/analyze/{txn_id}")
def analyze_with_rl(txn_id: str) -> dict:
    """Analyze a transaction using the RL model"""
    if txn_id not in TRANSACTIONS:
        raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not found")
    
    txn = TRANSACTIONS[txn_id]
    
    try:
        decision, confidence = rl_manager.predict(txn)
        
        return {
            "txn_id": txn_id,
            "method": "RL",
            "decision": decision,
            "confidence": round(confidence, 2),
            "flags": ["rl_prediction"],
            "explanation": f"RL model prediction with {confidence:.1%} confidence",
            "true_label": txn.label,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RL prediction failed: {str(e)}")


@app.post("/rl/batch")
def analyze_batch_with_rl() -> BatchResult:
    """Analyze all transactions using the RL model"""
    results = []
    
    for txn_id, txn in TRANSACTIONS.items():
        try:
            decision, confidence = rl_manager.predict(txn)
            results.append(
                CaseResult(
                    txn_id=txn_id,
                    decision=decision,
                    confidence=confidence,
                    flags=["rl_prediction"],
                    true_label=txn.label,
                )
            )
        except Exception as e:
            results.append(
                CaseResult(
                    txn_id=txn_id,
                    decision="NEEDS_REVIEW",
                    confidence=0.5,
                    flags=["rl_error"],
                    true_label=txn.label,
                )
            )
    
    metrics = calculate_metrics(results)
    
    return BatchResult(
        total=len(results), 
        results=results[-20:], 
        metrics=metrics
    )


@app.get("/rl/status")
def get_rl_status():
    """Get RL model status"""
    model_loaded = rl_manager.load_model()
    return {
        "model_available": model_loaded,
        "model_type": "PPO" if model_loaded else None,
        "training_data_size": len(TRANSACTIONS)
    }


@app.post("/compare/{txn_id}")
def compare_methods(txn_id: str) -> dict:
    """Compare rule-based vs RL predictions for a transaction"""
    if txn_id not in TRANSACTIONS:
        raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not found")
    
    txn = TRANSACTIONS[txn_id]
    
    # Rule-based analysis
    rule_result = analyze_transaction(txn)
    
    # RL analysis
    try:
        rl_decision, rl_confidence = rl_manager.predict(txn)
        rl_available = True
    except:
        rl_decision = "UNAVAILABLE"
        rl_confidence = 0.0
        rl_available = False
    
    return {
        "txn_id": txn_id,
        "true_label": txn.label,
        "rule_based": {
            "decision": rule_result.decision,
            "confidence": rule_result.confidence,
            "flags": rule_result.flags,
            "explanation": rule_result.explanation
        },
        "rl_model": {
            "decision": rl_decision,
            "confidence": rl_confidence,
            "available": rl_available
        },
        "agreement": rule_result.decision == rl_decision if rl_available else None
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
