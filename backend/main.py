"""
Fraud Detection Demo - Complete Backend
INFO 492 - Week 3 Demo #1
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Literal, Optional
import json
import re
import random
from datetime import datetime, timedelta
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sklearn.preprocessing import StandardScaler
import pickle
import os
import uuid
import hashlib
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None

# ============ RBAC & AUTHENTICATION ============

# Simulated YubiKey database (in production, this would be a secure database)
# Format: {yubikey_id: {password_hash, user_id, role, name, email}}
YUBIKEY_DB = {
    "admin-yubikey-123": {
        "password_hash": hashlib.sha256("admin2024!".encode()).hexdigest(),
        "user_id": "admin-001",
        "role": "admin",
        "name": "Sarah Johnson",
        "email": "sarah.johnson@fraudco.com",
        "created_at": datetime.utcnow()
    },
    "analyst-yubikey-456": {
        "password_hash": hashlib.sha256("analyst2024!".encode()).hexdigest(),
        "user_id": "analyst-001", 
        "role": "analyst",
        "name": "Michael Chen",
        "email": "michael.chen@fraudco.com",
        "created_at": datetime.utcnow()
    },
    "viewer-yubikey-789": {
        "password_hash": hashlib.sha256("viewer2024!".encode()).hexdigest(),
        "user_id": "viewer-001",
        "role": "viewer",
        "name": "David Wilson",
        "email": "david.wilson@fraudco.com",
        "created_at": datetime.utcnow()
    }
}

# OTP storage: {yubikey_id: {otp, expires_at}}
OTP_STORE = {}

# Active sessions (in production, use Redis or similar)
ACTIVE_SESSIONS = {}

security = HTTPBearer()

# Role definitions and permissions
ROLES_PERMISSIONS = {
    "admin": {
        "can_train": True,
        "can_view_models": True,
        "can_analyze": True,
        "can_manage_users": True,
        "description": "Full access - can train models and manage system"
    },
    "analyst": {
        "can_train": False,
        "can_view_models": True,
        "can_analyze": True,
        "can_manage_users": False,
        "description": "Can analyze transactions and view trained models"
    },
    "viewer": {
        "can_train": False,
        "can_view_models": True,
        "can_analyze": True,
        "can_manage_users": False,
        "description": "Can view models and analyze transactions"
    }
}


class OTPRequest(BaseModel):
    yubikey_id: str


class VerifyOTPRequest(BaseModel):
    yubikey_id: str
    otp: str


class YubiKeyLoginRequest(BaseModel):
    yubikey_id: str
    password: str


class LoginResponse(BaseModel):
    session_token: str
    user_id: str
    name: str
    role: str
    permissions: dict
    expires_at: str


class UserInfo(BaseModel):
    user_id: str
    name: str
    role: str
    permissions: dict


class StoredModel(BaseModel):
    model_id: str
    model_type: str
    training_steps: int
    created_at: str
    metrics: dict
    file_path: str
    
    model_config = {"protected_namespaces": ()}


class ReviewCase(BaseModel):
    txn_id: str
    decision: str
    confidence: float
    flags: List[str]
    explanation: str
    true_label: str


class UpdateDecisionRequest(BaseModel):
    txn_id: str
    human_decision: Literal["FRAUD", "LEGIT"]
    reviewer_notes: str = ""


class RewardWeights(BaseModel):
    correct_fraud: float = 10.0
    correct_legit: float = 1.0
    false_positive: float = -5.0
    false_negative: float = -20.0
    review_correct: float = 2.0
    review_incorrect: float = -1.0


def generate_session_token() -> str:
    """Generate a secure session token"""
    return str(uuid.uuid4())


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password


def generate_otp() -> str:
    """Generate a 6-digit OTP"""
    return f"{random.randint(100000, 999999)}"


def send_otp_to_yubikey(yubikey_id: str, user_data: dict, otp: str) -> None:
    """Simulate sending OTP to YubiKey device (in production, use actual YubiKey API)"""
    print(f"🔐 [SIMULATED] OTP displayed on YubiKey {yubikey_id} ({user_data['name']}): {otp}")
    # In production, this would interface with the actual YubiKey device
    pass


def create_session(yubikey_id: str, user_data: dict) -> dict:
    """Create a new session for authenticated user"""
    session_token = generate_session_token()
    expires_at = datetime.utcnow() + timedelta(hours=8)  # 8 hour session
    
    ACTIVE_SESSIONS[session_token] = {
        "yubikey_id": yubikey_id,
        "user_id": user_data["user_id"],
        "name": user_data["name"],
        "role": user_data["role"],
        "expires_at": expires_at.isoformat(),
        "created_at": datetime.utcnow().isoformat()
    }
    
    return {
        "session_token": session_token,
        "user_id": user_data["user_id"],
        "name": user_data["name"],
        "role": user_data["role"],
        "permissions": ROLES_PERMISSIONS[user_data["role"]],
        "expires_at": expires_at.isoformat()
    }


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Verify session token and return current user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials"
        )
    
    session_token = credentials.credentials
    
    if session_token not in ACTIVE_SESSIONS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token"
        )
    
    session = ACTIVE_SESSIONS[session_token]
    
    # Check if session expired
    expires_at = datetime.fromisoformat(session["expires_at"])
    if datetime.utcnow() > expires_at:
        del ACTIVE_SESSIONS[session_token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired"
        )
    
    return session


def require_permission(permission: str):
    """Decorator to check if user has required permission"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get current user from kwargs (injected by Depends)
            user = kwargs.get("current_user") or args[-1] if args else None
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            role = user.get("role")
            if role not in ROLES_PERMISSIONS:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid user role"
                )
            
            if not ROLES_PERMISSIONS[role].get(permission, False):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission} required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


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
    
    model_config = {"protected_namespaces": ()}


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
        # Use absolute path for models directory
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        # Track if weights were changed
        self.weights_changed = False
        self.last_weights = None
    
    def create_environment(self):
        """Create RL environment"""
        self.env = FraudDetectionEnv(self.transactions, self.scaler)
        return self.env
    
    def train_model(self, total_timesteps=20000, user_id: str = None):
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
        
        # Generate unique model ID with timestamp
        model_id = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        if user_id:
            model_id = f"{model_id}_{user_id[:8]}"
        
        # Save to models directory
        model_file = self.models_dir / f"{model_id}.pkl"
        scaler_file = self.models_dir / f"{model_id}_scaler.pkl"
        
        self.model.save(str(model_file))
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.env.scaler, f)
        
        # Also keep latest for quick access
        self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.env.scaler, f)
        
        return self.model, model_id
    
    def load_model(self, model_id: str = None):
        """Load trained model"""
        if model_id:
            model_file = self.models_dir / f"{model_id}.pkl"
            scaler_file = self.models_dir / f"{model_id}_scaler.pkl"
            if model_file.exists() and scaler_file.exists():
                self.model = PPO.load(str(model_file))
                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                return True
        elif os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = PPO.load(self.model_path)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            return True
        return False
    
    def list_stored_models(self) -> List[StoredModel]:
        """List all stored models"""
        models = []
        for model_file in self.models_dir.glob("*.pkl"):
            if "_scaler" in model_file.name:
                continue
            
            model_id = model_file.stem
            try:
                # Try to load metadata if it exists
                metadata_file = self.models_dir / f"{model_id}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    models.append(StoredModel(
                        model_id=metadata.get("model_id", model_id),
                        model_type=metadata.get("model_type", "PPO"),
                        training_steps=metadata.get("training_steps", 0),
                        created_at=metadata.get("created_at", datetime.utcnow().isoformat()),
                        metrics=metadata.get("metrics", {}),
                        file_path=str(model_file)
                    ))
            except Exception as e:
                # If no metadata, create basic entry
                stat = model_file.stat()
                models.append(StoredModel(
                    model_id=model_id,
                    model_type="PPO",
                    training_steps=0,
                    created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    metrics={},
                    file_path=str(model_file)
                ))
        
        # Sort by creation time, newest first
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    def save_model_metadata(self, model_id: str, metadata: dict):
        """Save metadata for a model"""
        metadata_file = self.models_dir / f"{model_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
    
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
        
        # Get observation (flatten to 1D if needed)
        obs = normalized_features[0] if len(normalized_features.shape) > 1 else normalized_features
        
        try:
            # Get action prediction
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Convert action to Python int if it's a numpy array or scalar
            if hasattr(action, 'item'):
                action = action.item()
            elif isinstance(action, (np.ndarray, np.generic)):
                action = int(action)
            else:
                action = int(action)
            
            # Try to get probability distribution for confidence estimate
            if torch is not None:
                try:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        latent_pi, _ = self.model.policy._get_latent(obs_tensor)
                        distribution = self.model.policy._get_action_dist_from_latent(latent_pi)
                        probs = distribution.distribution.probs[0]
                        confidence = float(torch.max(probs).item())
                except Exception as e:
                    confidence = 0.85  # Default RL confidence
            else:
                confidence = 0.85  # Default RL confidence
            
        except Exception as e:
            # Fallback if we can't get probabilities
            action, _ = self.model.predict(obs, deterministic=True)
            # Convert action to Python int
            if hasattr(action, 'item'):
                action = action.item()
            elif isinstance(action, (np.ndarray, np.generic)):
                action = int(action)
            else:
                action = int(action)
            confidence = 0.85
        
        # Convert action to decision
        action_map = {0: "FRAUD", 1: "LEGIT", 2: "NEEDS_REVIEW"}
        decision = action_map.get(action, "NEEDS_REVIEW")  # Default to NEEDS_REVIEW if action not found
        
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

# Persist last batch for KPI refresh
LAST_BATCH: List[CaseResult] | None = None

# Store all historical results for viewers to see
ALL_RESULTS: List[CaseResult] = []

# Human review queue: NEEDS_REVIEW cases awaiting human decision
REVIEW_QUEUE: List[ReviewCase] = []

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


def iso_utc():
    """Get current UTC timestamp in ISO format"""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def mask_token(s: str, keep_tail: int = 2) -> str:
    """Mask token keeping last few characters"""
    if not s or len(s) <= keep_tail:
        return "***"
    return s[:-keep_tail].replace(s[:-keep_tail], "*" * max(3, len(s)-keep_tail)) + s[-keep_tail:]


def masked_transaction_dict(txn: Transaction) -> dict:
    """Return transaction dict with masked PII"""
    d = txn.dict()
    # Mask device_id (keeping last 2 characters)
    d["device_id"] = mask_token(d["device_id"])
    return d


def build_provenance(txn: Transaction, decision: FraudDecision) -> dict:
    """Build provenance and audit trail for a transaction decision"""
    return {
        "txn_id": txn.id,
        "model_version": "rules-v1",
        "flags": decision.flags,
        "recommended_action": decision.decision,
        "pii_masked": True,
        "explanation": {
            "summary": decision.explanation,
            "red_flags": decision.flags
        },
        "steps": [
            {"type": "ingest", "ref": "loader:data.json", "at": iso_utc()},
            {"type": "featurize", "ref": "rules-v1", "at": iso_utc()},
            {"type": "agent", "ref": "fraud-copilot", "at": iso_utc()}
        ]
    }


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


@app.post("/auth/yubikey/otp/request")
def request_yubikey_otp(request: OTPRequest) -> dict:
    """Request OTP from YubiKey for login"""
    yubikey_id = request.yubikey_id
    
    # Check if YubiKey exists
    if yubikey_id not in YUBIKEY_DB:
        # Don't reveal if YubiKey exists (security best practice)
        return {
            "message": "If this YubiKey exists, an OTP has been sent",
            "yubikey_id": yubikey_id
        }
    
    user_data = YUBIKEY_DB[yubikey_id]
    
    # Generate OTP
    otp = generate_otp()
    expires_at = datetime.utcnow() + timedelta(minutes=10)  # 10 minute expiry
    
    OTP_STORE[yubikey_id] = {
        "otp": otp,
        "expires_at": expires_at.isoformat(),
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Simulate sending OTP to YubiKey device
    send_otp_to_yubikey(yubikey_id, user_data, otp)
    
    return {
        "message": "OTP displayed on YubiKey device",
        "yubikey_id": yubikey_id,
        "user_name": user_data["name"],
        # In production, don't return OTP. This is for demo only:
        "otp_demo": otp  # REMOVE IN PRODUCTION
    }


@app.post("/auth/yubikey/otp/verify", response_model=LoginResponse)
def verify_yubikey_otp(request: VerifyOTPRequest) -> LoginResponse:
    """Verify OTP from YubiKey and login"""
    yubikey_id = request.yubikey_id
    otp = request.otp
    
    # Check if YubiKey exists
    if yubikey_id not in YUBIKEY_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid YubiKey ID"
        )
    
    user_data = YUBIKEY_DB[yubikey_id]
    
    # Check if OTP exists and is valid
    if yubikey_id not in OTP_STORE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No OTP found. Please request a new one."
        )
    
    otp_data = OTP_STORE[yubikey_id]
    
    # Check if OTP expired
    expires_at = datetime.fromisoformat(otp_data["expires_at"])
    if datetime.utcnow() > expires_at:
        del OTP_STORE[yubikey_id]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OTP expired. Please request a new one."
        )
    
    # Verify OTP
    if otp_data["otp"] != otp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid OTP"
        )
    
    # OTP verified, create session
    session = create_session(yubikey_id, user_data)
    
    # Clean up OTP
    del OTP_STORE[yubikey_id]
    
    return LoginResponse(
        session_token=session["session_token"],
        user_id=session["user_id"],
        name=session["name"],
        role=session["role"],
        permissions=session["permissions"],
        expires_at=session["expires_at"]
    )


@app.post("/auth/yubikey/auto-login", response_model=LoginResponse)
def auto_login_with_yubikey(request: OTPRequest) -> LoginResponse:
    """Auto-generate OTP and login (simplified for demo)"""
    yubikey_id = request.yubikey_id
    
    # Check if YubiKey exists
    if yubikey_id not in YUBIKEY_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid YubiKey ID"
        )
    
    user_data = YUBIKEY_DB[yubikey_id]
    
    # Generate OTP for this login
    otp = generate_otp()
    print(f"🔐 [AUTO-LOGIN] Generated OTP for {yubikey_id} ({user_data['name']}): {otp}")
    
    # Create session
    session = create_session(yubikey_id, user_data)
    
    return LoginResponse(
        session_token=session["session_token"],
        user_id=session["user_id"],
        name=session["name"],
        role=session["role"],
        permissions=session["permissions"],
        expires_at=session["expires_at"]
    )


@app.post("/auth/logout")
def logout(current_user: dict = Depends(get_current_user)):
    """Logout and invalidate session"""
    # Get session token from request (you'd need to extract it)
    # For simplicity, we'll just return success
    return {"message": "Logged out successfully"}


@app.get("/auth/me", response_model=UserInfo)
def get_current_user_info(current_user: dict = Depends(get_current_user)) -> UserInfo:
    """Get current authenticated user info"""
    return UserInfo(
        user_id=current_user["user_id"],
        name=current_user["name"],
        role=current_user["role"],
        permissions=ROLES_PERMISSIONS[current_user["role"]]
    )


@app.get("/auth/roles")
def list_roles():
    """List all available roles and their permissions"""
    return {
        "roles": ROLES_PERMISSIONS,
        "available_yubikeys": {
            yid: {"role": data["role"], "name": data["name"], "email": data["email"]}
            for yid, data in YUBIKEY_DB.items()
        }
    }


@app.post("/analyze")
def analyze_single(txn_id: str, current_user: dict = Depends(get_current_user)) -> dict:
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
def analyze_batch(current_user: dict = Depends(get_current_user)) -> BatchResult:
    """Analyze all transactions"""
    global LAST_BATCH, ALL_RESULTS
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

    LAST_BATCH = results[:]  # Persist full results for KPI refresh
    ALL_RESULTS = results[:]  # Store all results for viewers
    
    # Add NEEDS_REVIEW cases to review queue
    global REVIEW_QUEUE
    for result in results:
        if result.decision == "NEEDS_REVIEW":
            review_case = ReviewCase(
                txn_id=result.txn_id,
                decision=result.decision,
                confidence=result.confidence,
                flags=result.flags,
                explanation="",
                true_label=result.true_label
            )
            REVIEW_QUEUE.append(review_case)
    
    metrics = calculate_metrics(results)

    return BatchResult(
        total=len(results), 
        results=results,  # Return ALL results (not just last 20)
        metrics=metrics
    )


@app.get("/metrics")
def get_metrics():
    """Get metrics from last batch or compute over all transactions"""
    global LAST_BATCH
    if LAST_BATCH:
        return calculate_metrics(LAST_BATCH)
    
    # Fallback: compute over all transactions
    temp: List[CaseResult] = []
    for txn in TRANSACTIONS.values():
        dec = analyze_transaction(txn)
        temp.append(CaseResult(
            txn_id=txn.id, decision=dec.decision,
            confidence=dec.confidence, flags=dec.flags,
            true_label=txn.label
        ))
    return calculate_metrics(temp)


@app.get("/transactions")
def list_transactions(current_user: dict = Depends(get_current_user)):
    """List all transaction IDs"""
    return {"total": len(TRANSACTIONS), "ids": list(TRANSACTIONS.keys())}


@app.get("/provenance/{txn_id}")
def provenance(txn_id: str):
    """Get provenance and audit trail for a transaction"""
    if txn_id not in TRANSACTIONS:
        raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not found")
    txn = TRANSACTIONS[txn_id]
    dec = analyze_transaction(txn)
    prov = build_provenance(txn, dec)
    # Ensure explanation is masked
    prov["explanation"]["summary"] = mask_pii(prov["explanation"]["summary"])
    return prov


@app.get("/data")
def get_raw_data():
    """Get raw transaction data as JSON"""
    return {"transactions": [masked_transaction_dict(t) for t in TRANSACTIONS.values()]}


@app.get("/transaction/{txn_id}")
def get_transaction_details(txn_id: str):
    """Get detailed transaction data"""
    if txn_id not in TRANSACTIONS:
        raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not found")
    
    txn = TRANSACTIONS[txn_id]
    return {
        "transaction": masked_transaction_dict(txn),
        "analysis": analyze_transaction(txn).dict()
    }


# ============ RL API ENDPOINTS ============


@app.post("/rl/train")
def train_rl_model(
    timesteps: int = 20000,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Train the RL fraud detection model (admin only)"""
    try:
        # Check if user has permission to train
        if current_user["role"] not in ["admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin users can train models"
            )
        
        # Show warning if weights were changed
        weights_changed_info = ""
        if rl_manager.weights_changed and rl_manager.last_weights:
            weights_changed_info = " (Note: Using updated reward weights - previous weights were different)"
            rl_manager.weights_changed = False  # Reset flag after training
        
        # Train the model
        try:
            model, model_id = rl_manager.train_model(
                total_timesteps=timesteps,
                user_id=current_user["user_id"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")
        
        # Evaluate the model - simplified evaluation
        try:
            # Create a simple evaluation environment
            eval_env = rl_manager.create_environment()
            obs, _ = eval_env.reset()
            
            total_reward = 0
            correct_predictions = 0
            total_predictions = 0
            tp = fp = fn = tn = 0
            
            done = False
            step_count = 0
            max_steps = 100  # Limit evaluation to prevent infinite loops
            
            while not done and step_count < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                total_reward += reward
                step_count += 1
                
                if info and 'true_label' in info:
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
        except Exception as e:
            # If evaluation fails, still return success with default metrics
            print(f"Evaluation warning: {str(e)}")
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            total_reward = 0
        else:
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            "model_type": "PPO",
            "training_steps": timesteps,
            "final_reward": round(total_reward, 2),
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3)
        }
        
        # Save metadata
        metadata = {
            "model_id": model_id,
            "model_type": "PPO",
            "training_steps": timesteps,
            "created_at": datetime.utcnow().isoformat(),
            "created_by": current_user["user_id"],
            "trainer_name": current_user["name"],
            "metrics": metrics
        }
        rl_manager.save_model_metadata(model_id, metadata)
        
        result = RLTrainingResult(**metrics)
        return {
            **result.dict(),
            "model_id": model_id,
            "message": f"Model trained and stored successfully{weights_changed_info}",
            "current_reward_weights": rl_manager.env.reward_weights
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/rl/models", response_model=List[StoredModel])
def list_rl_models(current_user: dict = Depends(get_current_user)) -> List[StoredModel]:
    """List all stored RL models (accessible to all authenticated users)"""
    return rl_manager.list_stored_models()


@app.get("/results/all")
def get_all_results(current_user: dict = Depends(get_current_user)):
    """Get all historical analysis results (accessible to all authenticated users)"""
    global ALL_RESULTS
    return {
        "total": len(ALL_RESULTS),
        "results": ALL_RESULTS,  # Return ALL results (all 1000 transactions)
        "has_data": len(ALL_RESULTS) > 0
    }


@app.get("/review/queue", response_model=List[ReviewCase])
def get_review_queue(current_user: dict = Depends(get_current_user)) -> List[ReviewCase]:
    """Get all cases needing human review (admin/analyst only)"""
    # Check if user has permission
    if current_user["role"] not in ["admin", "analyst"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin and analyst users can review cases"
        )
    
    return REVIEW_QUEUE


@app.post("/review/update")
def update_review_decision(request: UpdateDecisionRequest, current_user: dict = Depends(get_current_user)):
    """Update a human decision for a NEEDS_REVIEW case (admin/analyst only)"""
    # Check if user has permission
    if current_user["role"] not in ["admin", "analyst"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin and analyst users can review cases"
        )
    
    global REVIEW_QUEUE
    # Find and remove from queue
    for i, case in enumerate(REVIEW_QUEUE):
        if case.txn_id == request.txn_id:
            REVIEW_QUEUE.pop(i)
            
            # Update in ALL_RESULTS if it exists
            for j, result in enumerate(ALL_RESULTS):
                if result.txn_id == request.txn_id:
                    ALL_RESULTS[j] = CaseResult(
                        txn_id=request.txn_id,
                        decision=request.human_decision,
                        confidence=case.confidence,
                        flags=case.flags,
                        true_label=case.true_label
                    )
                    break
            
            return {
                "message": "Decision updated",
                "txn_id": request.txn_id,
                "decision": request.human_decision,
                "reviewer": current_user["name"],
                "notes": request.reviewer_notes,
                "remaining_in_queue": len(REVIEW_QUEUE)
            }
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Transaction {request.txn_id} not found in review queue"
    )


@app.get("/rl/reward-weights")
def get_reward_weights(current_user: dict = Depends(get_current_user)):
    """Get current reward weights (admin only)"""
    if current_user["role"] not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can view/modify reward weights"
        )
    
    return rl_manager.env.reward_weights if hasattr(rl_manager, 'env') and rl_manager.env else {
        'correct_fraud': 10.0,
        'correct_legit': 1.0,
        'false_positive': -5.0,
        'false_negative': -20.0,
        'review_correct': 2.0,
        'review_incorrect': -1.0
    }


@app.post("/rl/reward-weights")
def update_reward_weights(weights: RewardWeights, current_user: dict = Depends(get_current_user)):
    """Update reward weights for RL training (admin only)"""
    if current_user["role"] not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can modify reward weights"
        )
    
    # Store old weights before updating
    if rl_manager.env:
        rl_manager.last_weights = rl_manager.env.reward_weights.copy()
    else:
        rl_manager.last_weights = None
    
    # Update reward weights in environment
    if not hasattr(rl_manager, 'env') or rl_manager.env is None:
        rl_manager.create_environment()
    
    new_weights = {
        'correct_fraud': weights.correct_fraud,
        'correct_legit': weights.correct_legit,
        'false_positive': weights.false_positive,
        'false_negative': weights.false_negative,
        'review_correct': weights.review_correct,
        'review_incorrect': weights.review_incorrect
    }
    
    rl_manager.env.reward_weights = new_weights
    rl_manager.weights_changed = True
    
    return {
        "message": "Reward weights updated. IMPORTANT: You must retrain the model for changes to take effect.",
        "weights": new_weights,
        "previous_weights": rl_manager.last_weights,
        "requires_retraining": True
    }


@app.get("/rl/models/{model_id}")
def get_model_details(model_id: str, current_user: dict = Depends(get_current_user)):
    """Get details of a specific model"""
    models = rl_manager.list_stored_models()
    
    for model in models:
        if model.model_id == model_id:
            return {
                "model_id": model.model_id,
                "model_type": model.model_type,
                "training_steps": model.training_steps,
                "created_at": model.created_at,
                "metrics": model.metrics,
                "file_path": model.file_path,
                "can_load": True
            }
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Model {model_id} not found"
    )


@app.post("/rl/analyze/{txn_id}")
def analyze_with_rl(txn_id: str, current_user: dict = Depends(get_current_user)) -> dict:
    """Analyze a transaction using the RL model"""
    if txn_id not in TRANSACTIONS:
        raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not found")
    
    txn = TRANSACTIONS[txn_id]
    
    # Check if model is available
    if rl_manager.model is None:
        model_loaded = rl_manager.load_model()
        if not model_loaded:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No RL model available. Please train a model first using the 'Train RL Model' panel."
            )
    
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
def analyze_batch_with_rl(current_user: dict = Depends(get_current_user)) -> BatchResult:
    """Analyze all transactions using the RL model"""
    global ALL_RESULTS
    
    # Check if model is available
    if rl_manager.model is None:
        model_loaded = rl_manager.load_model()
        if not model_loaded:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No RL model available. Please train a model first using the 'Train RL Model' panel."
            )
    
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
            # Log the actual error
            print(f"RL prediction error for {txn_id}: {str(e)}")
            results.append(
                CaseResult(
                    txn_id=txn_id,
                    decision="NEEDS_REVIEW",
                    confidence=0.5,
                    flags=["rl_error", f"error:{str(e)[:20]}"],
                    true_label=txn.label,
                )
            )
    
    # Store results for viewers
    ALL_RESULTS = results[:]
    
    # Add NEEDS_REVIEW cases to review queue
    global REVIEW_QUEUE
    for result in results:
        if result.decision == "NEEDS_REVIEW":
            review_case = ReviewCase(
                txn_id=result.txn_id,
                decision=result.decision,
                confidence=result.confidence,
                flags=result.flags,
                explanation="",
                true_label=result.true_label
            )
            REVIEW_QUEUE.append(review_case)
    
    metrics = calculate_metrics(results)
    
    return BatchResult(
        total=len(results), 
        results=results,  # Return ALL results
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
