"""
Fraud Detection Demo - Complete Backend
INFO 492 - Week 3 Demo #1
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional
import json
import re
import random
from datetime import datetime, timedelta
import numpy as np
import asyncio
import threading
from collections import deque
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
import csv
import io
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


class DetectionConfig(BaseModel):
    """Detection algorithm configuration"""
    # Amount thresholds
    large_amount_threshold: float = 1000.0
    large_transfer_threshold: float = 2000.0
    unusual_deposit_threshold: float = 5000.0
    suspicious_channel_amount: float = 500.0
    
    # Geographic settings
    high_risk_locations: List[str] = ["RU", "CN", "NG", "BR", "MX", "Tokyo", "Toronto", "London", "Sydney", "Berlin", "Dubai", "Singapore"]
    
    # Category settings
    high_risk_categories: List[str] = ["other", "online"]
    
    # Decision thresholds
    fraud_flag_threshold: int = 2  # ≥2 flags = FRAUD
    legit_flag_threshold: int = 0   # 0 flags = LEGIT
    
    # Confidence settings
    high_confidence_base: float = 0.70
    confidence_per_flag: float = 0.08
    review_confidence: float = 0.50
    
    # Channel risk settings
    suspicious_channels: List[str] = ["mobile", "web"]
    
    model_config = {"protected_namespaces": ()}


class SystemConfig(BaseModel):
    """System-wide configuration"""
    detection: DetectionConfig = DetectionConfig()
    
    # Model settings
    active_model_id: Optional[str] = None
    
    # Session settings
    session_timeout_hours: int = 8
    
    model_config = {"protected_namespaces": ()}


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


# ============ AUDIT LOGGING SYSTEM ============


class AuditLog(BaseModel):
    """Audit log entry"""
    log_id: str
    timestamp: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    user_role: Optional[str] = None
    action: str
    resource: Optional[str] = None
    details: Optional[dict] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_token: Optional[str] = None
    status: str  # "success" or "error"
    error_message: Optional[str] = None
    
    model_config = {"protected_namespaces": ()}


class AuditLogService:
    """Service for managing audit logs"""
    
    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or Path(__file__).parent / "audit_logs"
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "audit.jsonl"
    
    def log(
        self,
        action: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        user_role: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_token: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None
    ) -> AuditLog:
        """Create and store an audit log entry"""
        log_entry = AuditLog(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            user_name=user_name,
            user_role=user_role,
            action=action,
            resource=resource,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            session_token=session_token[:8] + "..." if session_token else None,
            status=status,
            error_message=error_message
        )
        
        # Append to JSONL file (one JSON object per line)
        with open(self.log_file, "a") as f:
            f.write(log_entry.model_dump_json() + "\n")
        
        return log_entry
    
    def get_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        search: Optional[str] = None
    ) -> tuple[List[AuditLog], int]:
        """Retrieve audit logs with filtering"""
        if not self.log_file.exists():
            return [], 0
        
        logs = []
        with open(self.log_file, "r") as f:
            for line in f:
                try:
                    log_data = json.loads(line.strip())
                    log = AuditLog(**log_data)
                    
                    # Apply filters
                    if user_id and log.user_id != user_id:
                        continue
                    if action and log.action != action:
                        continue
                    if status and log.status != status:
                        continue
                    if start_date and log.timestamp < start_date:
                        continue
                    if end_date and log.timestamp > end_date:
                        continue
                    if search:
                        search_lower = search.lower()
                        searchable = f"{log.action} {log.resource} {log.user_name} {json.dumps(log.details)}".lower()
                        if search_lower not in searchable:
                            continue
                    
                    logs.append(log)
                except Exception as e:
                    print(f"Error parsing audit log line: {e}")
                    continue
        
        # Sort by timestamp descending (newest first)
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        total = len(logs)
        return logs[offset:offset + limit], total
    
    def export_logs_csv(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> str:
        """Export logs as CSV"""
        logs, _ = self.get_logs(
            limit=10000,
            user_id=user_id,
            action=action,
            start_date=start_date,
            end_date=end_date
        )
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            'timestamp', 'user_name', 'user_role', 'action', 'resource',
            'status', 'ip_address', 'details'
        ])
        writer.writeheader()
        
        for log in logs:
            writer.writerow({
                'timestamp': log.timestamp,
                'user_name': log.user_name or 'Unknown',
                'user_role': log.user_role or 'Unknown',
                'action': log.action,
                'resource': log.resource or '',
                'status': log.status,
                'ip_address': log.ip_address or '',
                'details': json.dumps(log.details) if log.details else ''
            })
        
        return output.getvalue()


# Initialize audit service
audit_service = AuditLogService()


# ============ CONFIGURATION SERVICE ============


class ConfigurationService:
    """Manages system configuration"""
    
    def __init__(self, config_file: Path = None):
        self.config_file = config_file or Path(__file__).parent / "config.json"
        self.config: SystemConfig = self._load_config()
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    return SystemConfig(**data)
            except Exception as e:
                print(f"Failed to load config: {e}, using defaults")
                return SystemConfig()
        else:
            # Return defaults and save
            config = SystemConfig()
            self.save_config(config)
            return config
    
    def save_config(self, config: SystemConfig) -> None:
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(config.model_dump(), f, indent=2)
        self.config = config
    
    def get_detection_config(self) -> DetectionConfig:
        """Get current detection configuration"""
        return self.config.detection
    
    def update_detection_config(self, updates: dict) -> DetectionConfig:
        """Update detection configuration"""
        current = self.config.detection.model_dump()
        current.update(updates)
        self.config.detection = DetectionConfig(**current)
        self.save_config(self.config)
        return self.config.detection
    
    def reset_to_defaults(self) -> SystemConfig:
        """Reset all configuration to defaults"""
        self.config = SystemConfig()
        self.save_config(self.config)
        return self.config
    
    def get_system_config(self) -> SystemConfig:
        """Get full system configuration"""
        return self.config


# Initialize config service
config_service = ConfigurationService()


# ============ DATA MODELS ============


class Transaction(BaseModel):
    id: str
    timestamp: str  # ISO format datetime string
    from_account: str
    to_account: str
    amount: float
    transaction_type: str  # withdrawal, deposit, transfer, payment
    category: str  # utilities, online, other, entertainment, travel, grocery, retail, restaurant
    location: str  # Tokyo, Toronto, London, Sydney, Berlin, Dubai, New York, Singapore, etc.
    channel: str  # mobile, atm, pos, web
    is_fraud: bool  # True for fraud, False for legitimate


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
        
        # State space: 12 features (amount, geo_risk, transaction types, channels, category_risk, large_amount)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
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
        US_LOCATIONS = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", 
                        "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
                        "San Francisco", "Columbus", "Fort Worth", "Charlotte", "Seattle", "Denver",
                        "Washington", "Boston", "El Paso", "Detroit", "Nashville", "Portland"]
        geo_risk = 1.0 if txn.location not in US_LOCATIONS else 0.0
        
        # Transaction type encoding (one-hot like)
        type_withdrawal = 1.0 if txn.transaction_type == "withdrawal" else 0.0
        type_deposit = 1.0 if txn.transaction_type == "deposit" else 0.0
        type_transfer = 1.0 if txn.transaction_type == "transfer" else 0.0
        type_payment = 1.0 if txn.transaction_type == "payment" else 0.0
        
        # Channel encoding
        channel_mobile = 1.0 if txn.channel == "mobile" else 0.0
        channel_atm = 1.0 if txn.channel == "atm" else 0.0
        channel_pos = 1.0 if txn.channel == "pos" else 0.0
        channel_web = 1.0 if txn.channel == "web" else 0.0
        
        # Category risk (high-risk categories)
        HIGH_RISK_CATEGORIES = ["other", "online"]
        category_risk = 1.0 if txn.category in HIGH_RISK_CATEGORIES else 0.0
        
        # Large amount flag
        large_amount = 1.0 if txn.amount > 1000 else 0.0
        
        return np.array([
            txn.amount,
            geo_risk,
            type_withdrawal,
            type_deposit,
            type_transfer,
            type_payment,
            channel_mobile,
            channel_atm,
            channel_pos,
            channel_web,
            category_risk,
            large_amount
        ], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset environment to start of episode"""
        super().reset(seed=seed)
        self.current_idx = 0
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        if self.current_idx >= len(self.transaction_ids):
            return np.zeros(12), 0, True, {}
        
        txn_id = self.transaction_ids[self.current_idx]
        txn = self.transactions[txn_id]
        
        # Get true label
        true_label = "FRAUD" if txn.is_fraud else "LEGIT"
        
        # Calculate reward based on action and true label
        reward = self._calculate_reward(action, true_label)
        
        # Move to next transaction
        self.current_idx += 1
        done = self.current_idx >= len(self.transaction_ids)
        
        # Get next observation
        next_obs = self._get_observation() if not done else np.zeros(12)
        
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
            return np.zeros(12)
        
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


# ============ LIVE DATA FEED SIMULATOR ============

class LiveTransactionGenerator:
    """Generates realistic transactions for live feed simulation"""
    
    TRANSACTION_TYPES = ["withdrawal", "deposit", "transfer", "payment"]
    CATEGORIES = ["utilities", "online", "other", "entertainment", "travel", "grocery", "retail", "restaurant"]
    LOCATIONS = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
        "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
        "San Francisco", "Columbus", "Fort Worth", "Charlotte", "Seattle", "Denver",
        "Washington", "Boston", "El Paso", "Detroit", "Nashville", "Portland",
        "Tokyo", "Toronto", "London", "Sydney", "Berlin", "Dubai", "Singapore", "Paris"
    ]
    CHANNELS = ["mobile", "atm", "pos", "web"]
    
    def __init__(self, base_transactions: dict):
        """Initialize with base transactions for pattern learning"""
        self.base_transactions = base_transactions
        self.transaction_counter = max([int(tid[1:]) for tid in base_transactions.keys() if tid[1:].isdigit()], default=0)
        self.account_pool = self._extract_accounts()
    
    def _extract_accounts(self) -> List[str]:
        """Extract unique accounts from base transactions"""
        accounts = set()
        for txn in self.base_transactions.values():
            accounts.add(txn.from_account)
            accounts.add(txn.to_account)
        return list(accounts)
    
    def generate_transaction(self, fraud_probability: float = 0.1) -> Transaction:
        """Generate a realistic transaction"""
        self.transaction_counter += 1
        txn_id = f"L{self.transaction_counter:06d}"  # Live transaction ID
        
        # Determine if fraud based on probability and patterns
        is_fraud = random.random() < fraud_probability
        
        # Generate realistic amount (skewed distribution)
        if is_fraud:
            # Fraud transactions tend to be larger
            amount = random.uniform(500, 5000) if random.random() < 0.7 else random.uniform(50, 500)
        else:
            # Legitimate transactions are usually smaller
            amount = random.uniform(10, 500) if random.random() < 0.8 else random.uniform(500, 2000)
        
        # Select transaction type
        transaction_type = random.choice(self.TRANSACTION_TYPES)
        
        # Select category (fraud tends to use high-risk categories)
        if is_fraud and random.random() < 0.6:
            category = random.choice(["online", "other"])
        else:
            category = random.choice(self.CATEGORIES)
        
        # Select location (fraud tends to use non-US locations)
        if is_fraud and random.random() < 0.5:
            location = random.choice([loc for loc in self.LOCATIONS if loc not in [
                "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"
            ]])
        else:
            location = random.choice(self.LOCATIONS)
        
        # Select channel
        channel = random.choice(self.CHANNELS)
        
        # Select accounts
        from_account = random.choice(self.account_pool)
        to_account = random.choice([acc for acc in self.account_pool if acc != from_account])
        
        # Generate timestamp (current time)
        timestamp = datetime.utcnow().isoformat()
        
        return Transaction(
            id=txn_id,
            timestamp=timestamp,
            from_account=from_account,
            to_account=to_account,
            amount=round(amount, 2),
            transaction_type=transaction_type,
            category=category,
            location=location,
            channel=channel,
            is_fraud=is_fraud
        )


# ============ LIVE FEED LOGGING SERVICE ============

class LiveFeedLogger:
    """Service for logging live feed transactions and tracking performance over time"""
    
    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or Path(__file__).parent / "live_feed_logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        self.session_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"live_feed_{self.session_id}.jsonl"
        self.metrics_file = self.log_dir / f"metrics_{self.session_id}.jsonl"
        
        # Performance tracking
        self.performance_history = []
        self.transaction_count = 0
        self.rl_correct = 0
        self.rl_total = 0
        self.rule_correct = 0
        self.rule_total = 0
        
        # Confusion matrices for both models
        self.rl_confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        self.rule_confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        
        # Model retraining tracking
        self.retraining_history = []
        self.last_retrain_time = None
        self.transactions_since_retrain = 0
        self.pending_retrain_update = None  # Store retrain info to update after collecting new data
    
    def log_transaction(self, result: dict):
        """Log a single transaction result"""
        self.transaction_count += 1
        self.transactions_since_retrain += 1
        
        log_entry = {
            "transaction_id": result["transaction"].get("transaction_id"),
            "timestamp": datetime.utcnow().isoformat(),
            "true_label": result["true_label"],
            "rule_based_decision": result["rule_based"]["decision"],
            "rl_decision": result["rl_model"]["decision"],
            "rl_available": result["rl_model"]["available"],
            "rl_confidence": result["rl_model"]["confidence"],
            "rule_confidence": result["rule_based"]["confidence"],
            "transaction_count": self.transaction_count
        }
        
        # Update performance metrics
        true_label = result["true_label"]
        
        # Rule-based metrics
        rule_pred = result["rule_based"]["decision"]
        if rule_pred == true_label:
            self.rule_correct += 1
        self._update_confusion(self.rule_confusion, rule_pred, true_label)
        self.rule_total += 1
        
        # RL model metrics (if available)
        if result["rl_model"]["available"] and result["rl_model"]["decision"]:
            rl_pred = result["rl_model"]["decision"]
            if rl_pred == true_label:
                self.rl_correct += 1
            self._update_confusion(self.rl_confusion, rl_pred, true_label)
            self.rl_total += 1
        
        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Periodically log performance metrics (every 100 transactions)
        if self.transaction_count % 100 == 0:
            self._log_performance_metrics()
            
            # Update pending retraining metrics if we have enough new data
            if self.pending_retrain_update and self.rl_total >= 100:
                # Update the last retraining entry with actual improved metrics
                if self.retraining_history:
                    current_metrics = self.get_current_metrics()
                    if current_metrics.get("rl_model"):
                        self.retraining_history[-1]["metrics_after"] = current_metrics["rl_model"].copy()
                        # Recalculate improvement
                        metrics_before = self.retraining_history[-1]["metrics_before"]
                        metrics_after = current_metrics["rl_model"]
                        self.retraining_history[-1]["improvement"] = {
                            "accuracy_delta": round(metrics_after.get("accuracy", 0) - metrics_before.get("accuracy", 0), 4),
                            "precision_delta": round(metrics_after.get("precision", 0) - metrics_before.get("precision", 0), 4),
                            "recall_delta": round(metrics_after.get("recall", 0) - metrics_before.get("recall", 0), 4),
                            "f1_delta": round(metrics_after.get("f1_score", 0) - metrics_before.get("f1_score", 0), 4)
                        }
                        self.pending_retrain_update = None
                        print(f"📈 Updated retraining metrics: Accuracy improved by {self.retraining_history[-1]['improvement']['accuracy_delta']:.4f}")
    
    def _update_confusion(self, confusion: dict, predicted: str, true_label: str):
        """Update confusion matrix"""
        if predicted == "FRAUD" and true_label == "FRAUD":
            confusion["tp"] += 1
        elif predicted == "FRAUD" and true_label == "LEGIT":
            confusion["fp"] += 1
        elif predicted == "LEGIT" and true_label == "FRAUD":
            confusion["fn"] += 1
        elif predicted == "LEGIT" and true_label == "LEGIT":
            confusion["tn"] += 1
    
    def _log_performance_metrics(self):
        """Log current performance metrics"""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "transaction_count": self.transaction_count,
            "rule_based": self._calculate_metrics(self.rule_confusion, self.rule_correct, self.rule_total),
            "rl_model": self._calculate_metrics(self.rl_confusion, self.rl_correct, self.rl_total) if self.rl_total > 0 else None
        }
        
        self.performance_history.append(metrics)
        
        # Write to metrics file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    def _calculate_metrics(self, confusion: dict, correct: int, total: int):
        """Calculate precision, recall, accuracy from confusion matrix"""
        tp, fp, tn, fn = confusion["tp"], confusion["fp"], confusion["tn"], confusion["fn"]
        
        accuracy = correct / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "total": total
        }
    
    def log_retraining(self, model_id: str, metrics_before: dict, metrics_after: dict):
        """Log model retraining event"""
        retrain_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "transaction_count": self.transaction_count,
            "transactions_since_last_retrain": self.transactions_since_retrain,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "improvement": {
                "accuracy_delta": round(metrics_after.get("accuracy", 0) - metrics_before.get("accuracy", 0), 4),
                "precision_delta": round(metrics_after.get("precision", 0) - metrics_before.get("precision", 0), 4),
                "recall_delta": round(metrics_after.get("recall", 0) - metrics_before.get("recall", 0), 4),
                "f1_delta": round(metrics_after.get("f1_score", 0) - metrics_before.get("f1_score", 0), 4)
            }
        }
        
        self.retraining_history.append(retrain_entry)
        self.last_retrain_time = datetime.utcnow()
        self.transactions_since_retrain = 0
        
        # Write to metrics file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps({"type": "retraining", **retrain_entry}) + "\n")
    
    def get_current_metrics(self):
        """Get current performance metrics"""
        return {
            "rule_based": self._calculate_metrics(self.rule_confusion, self.rule_correct, self.rule_total),
            "rl_model": self._calculate_metrics(self.rl_confusion, self.rl_correct, self.rl_total) if self.rl_total > 0 else None,
            "transaction_count": self.transaction_count,
            "retraining_count": len(self.retraining_history)
        }
    
    def get_performance_history(self, limit: int = None):
        """Get performance history"""
        if limit:
            return self.performance_history[-limit:]
        return self.performance_history
    
    def get_retraining_history(self):
        """Get retraining history"""
        return self.retraining_history


# Global state for live feed
LIVE_FEED_ACTIVE = False
LIVE_FEED_QUEUE = deque(maxlen=1000)  # Store last 1000 transactions
LIVE_FEED_STATS = {
    "total_streamed": 0,
    "fraud_detected": 0,
    "legit_approved": 0,
    "needs_review": 0,
    "start_time": None
}
live_feed_index = 0
live_feed_task = None
live_feed_logger = None  # Will be initialized when feed starts


async def live_feed_worker(interval_seconds: float = 2.0, fraud_rate: float = 0.1):
    """Background worker that streams existing transactions and evaluates them with continuous learning"""
    global LIVE_FEED_ACTIVE, LIVE_FEED_QUEUE, LIVE_FEED_STATS, live_feed_index, live_feed_logger
    
    # Initialize logger (make it global so endpoints can access it)
    live_feed_logger = LiveFeedLogger()
    
    # Get all transaction IDs and shuffle them
    transaction_ids = list(TRANSACTIONS.keys())
    random.shuffle(transaction_ids)
    
    LIVE_FEED_STATS["start_time"] = datetime.utcnow().isoformat()
    LIVE_FEED_STATS["total_streamed"] = 0
    LIVE_FEED_STATS["fraud_detected"] = 0
    LIVE_FEED_STATS["legit_approved"] = 0
    LIVE_FEED_STATS["needs_review"] = 0
    
    live_feed_index = 0
    
    # Continuous learning parameters
    RETRAIN_INTERVAL_TRANSACTIONS = 1000  # Retrain every 1000 transactions
    RETRAIN_INTERVAL_HOURS = 1.0  # Or retrain every hour (whichever comes first)
    last_retrain_time = datetime.utcnow()
    
    print(f"🚀 Live feed started with logging. Session ID: {live_feed_logger.session_id}")
    print(f"📊 Log files: {live_feed_logger.log_file.name}, {live_feed_logger.metrics_file.name}")
    
    while LIVE_FEED_ACTIVE:
        try:
            # Get next transaction (loop back to start if we reach the end)
            if live_feed_index >= len(transaction_ids):
                live_feed_index = 0
                random.shuffle(transaction_ids)  # Re-shuffle for variety
            
            txn_id = transaction_ids[live_feed_index]
            txn = TRANSACTIONS[txn_id]
            live_feed_index += 1
            
            # Evaluate with rule-based system
            decision = analyze_transaction(txn)
            
            # Try RL model if available
            rl_decision = None
            rl_confidence = None
            try:
                rl_decision, rl_confidence = rl_manager.predict(txn)
            except:
                pass
            
            # Create result
            result = {
                "transaction": txn.dict(),
                "rule_based": {
                    "decision": decision.decision,
                    "confidence": decision.confidence,
                    "flags": decision.flags,
                    "explanation": decision.explanation
                },
                "rl_model": {
                    "decision": rl_decision,
                    "confidence": rl_confidence,
                    "available": rl_decision is not None
                },
                "true_label": "FRAUD" if txn.is_fraud else "LEGIT",
                "timestamp": txn.timestamp
            }
            
            # Log transaction
            live_feed_logger.log_transaction(result)
            
            # Add to queue
            LIVE_FEED_QUEUE.append(result)
            LIVE_FEED_STATS["total_streamed"] += 1
            
            # Update stats
            if decision.decision == "FRAUD":
                LIVE_FEED_STATS["fraud_detected"] += 1
            elif decision.decision == "LEGIT":
                LIVE_FEED_STATS["legit_approved"] += 1
            elif decision.decision == "NEEDS_REVIEW":
                LIVE_FEED_STATS["needs_review"] += 1
            
            # Continuous learning: Retrain model periodically
            should_retrain = False
            retrain_reason = ""
            
            # Check if we should retrain based on transaction count
            if live_feed_logger.transactions_since_retrain >= RETRAIN_INTERVAL_TRANSACTIONS:
                should_retrain = True
                retrain_reason = f"transaction count ({live_feed_logger.transactions_since_retrain} transactions)"
            
            # Check if we should retrain based on time
            time_since_retrain = (datetime.utcnow() - last_retrain_time).total_seconds() / 3600
            if time_since_retrain >= RETRAIN_INTERVAL_HOURS:
                should_retrain = True
                retrain_reason = f"time interval ({time_since_retrain:.2f} hours)"
            
            # Retrain if needed and RL model is available
            if should_retrain and rl_manager.model is not None:
                try:
                    print(f"🔄 Retraining RL model (reason: {retrain_reason})...")
                    
                    # Get metrics before retraining
                    current_metrics = live_feed_logger.get_current_metrics()
                    metrics_before = current_metrics.get("rl_model")
                    
                    if metrics_before:
                        # Store metrics before retraining
                        metrics_before_copy = metrics_before.copy()
                        
                        # Incremental learning: continue training existing model
                        if rl_manager.model is not None:
                            # Continue training the existing model (incremental learning)
                            print(f"   Continuing training from existing model...")
                            
                            # Ensure environment is set up
                            if rl_manager.env is None:
                                rl_manager.create_environment()
                            
                            # Create vectorized environment for training
                            vec_env = make_vec_env(
                                lambda: FraudDetectionEnv(rl_manager.transactions, rl_manager.scaler), 
                                n_envs=4  # Use fewer envs for faster incremental training
                            )
                            
                            # Set the environment and continue training
                            rl_manager.model.set_env(vec_env)
                            rl_manager.model.learn(total_timesteps=5000)  # Additional training steps
                            
                            # Generate new model ID
                            model_id = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_live_feed_auto"
                            
                            # Save the improved model
                            model_file = rl_manager.models_dir / f"{model_id}.pkl"
                            scaler_file = rl_manager.models_dir / f"{model_id}_scaler.pkl"
                            rl_manager.model.save(str(model_file))
                            with open(scaler_file, 'wb') as f:
                                pickle.dump(rl_manager.env.scaler, f)
                            
                            # Also update the latest model
                            rl_manager.model.save(rl_manager.model_path)
                            with open(rl_manager.scaler_path, 'wb') as f:
                                pickle.dump(rl_manager.env.scaler, f)
                        else:
                            # Train new model if none exists
                            model, model_id = rl_manager.train_model(
                                total_timesteps=5000,
                                user_id="live_feed_auto"
                            )
                        
                        # Reset confusion matrix for RL model to track new performance
                        # This allows us to see improvement in the next batch
                        live_feed_logger.rl_confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                        live_feed_logger.rl_correct = 0
                        live_feed_logger.rl_total = 0
                        
                        # For now, metrics_after will be the same, but will improve as new transactions come in
                        # We'll update it after collecting some new predictions
                        metrics_after = metrics_before_copy.copy()
                        metrics_after["note"] = "Metrics will improve as new transactions are processed"
                        
                        # Log retraining event
                        live_feed_logger.log_retraining(model_id, metrics_before_copy, metrics_after)
                        
                        # Mark that we need to update this retraining entry after collecting new data
                        live_feed_logger.pending_retrain_update = True
                        
                        last_retrain_time = datetime.utcnow()
                        print(f"✅ Model retrained: {model_id}")
                        print(f"   Metrics before: Accuracy={metrics_before_copy['accuracy']:.4f}, "
                              f"Precision={metrics_before_copy['precision']:.4f}, "
                              f"Recall={metrics_before_copy['recall']:.4f}")
                        print(f"   Model will improve as new transactions are processed...")
                    else:
                        print("⚠️  Skipping retrain: Not enough RL model predictions yet")
                        
                except Exception as e:
                    print(f"❌ Error during model retraining: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Wait before next transaction
            await asyncio.sleep(interval_seconds)
            
        except Exception as e:
            print(f"Error in live feed worker: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(interval_seconds)
    
    # Final metrics log when feed stops
    if live_feed_logger:
        final_metrics = live_feed_logger.get_current_metrics()
        print(f"📊 Final metrics - Transactions: {final_metrics['transaction_count']}, "
              f"Retrainings: {final_metrics['retraining_count']}")
        live_feed_logger._log_performance_metrics()


# ============ FASTAPI APP ============

app = FastAPI(title="Fraud Detection API")

# Request logging middleware to track all incoming connections
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Get client IP address
        client_ip = request.client.host if request.client else "unknown"
        # Get X-Forwarded-For header if behind proxy
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        # Get User-Agent to identify device type
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Log the request
        print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {request.method} {request.url.path} from {client_ip} (UA: {user_agent[:50]})")
        
        response = await call_next(request)
        return response


# Audit logging middleware to capture all API actions
class AuditLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip audit logging for health check and static assets
        if request.url.path in ["/", "/health"]:
            return await call_next(request)
        
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Extract user info from authorization header if present
        user_id = None
        user_name = None
        user_role = None
        session_token = None
        
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            session_token = auth_header.split(" ")[1]
            if session_token in ACTIVE_SESSIONS:
                session = ACTIVE_SESSIONS[session_token]
                user_id = session.get("user_id")
                user_name = session.get("name")
                user_role = session.get("role")
        
        # Determine action from path and method
        action = f"{request.method} {request.url.path}"
        resource = None
        
        # Parse resource from path (e.g., /analyze/T001 -> resource=T001)
        path_parts = request.url.path.split("/")
        if len(path_parts) > 2:
            resource = path_parts[-1]
        
        # Execute request
        start_time = datetime.utcnow()
        status = "success"
        error_message = None
        
        try:
            response = await call_next(request)
            if response.status_code >= 400:
                status = "error"
                error_message = f"HTTP {response.status_code}"
            
            # Log the action
            audit_service.log(
                action=action,
                user_id=user_id,
                user_name=user_name,
                user_role=user_role,
                resource=resource,
                details={
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": str(request.query_params),
                    "status_code": response.status_code,
                    "duration_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000)
                },
                ip_address=client_ip,
                user_agent=user_agent[:100],
                session_token=session_token,
                status=status,
                error_message=error_message
            )
            
            return response
            
        except Exception as e:
            # Log the error
            audit_service.log(
                action=action,
                user_id=user_id,
                user_name=user_name,
                user_role=user_role,
                resource=resource,
                details={
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": str(request.query_params)
                },
                ip_address=client_ip,
                user_agent=user_agent[:100],
                session_token=session_token,
                status="error",
                error_message=str(e)
            )
            raise

# Add middlewares (order matters: last added = first executed)
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Enable CORS
# Backend deployed at: http://attu2.cs.washington.edu:8000
# Frontend deployed at: https://homes.cs.washington.edu/~micibr/fraud-demo/frontend/index.html
# 
# NOTE: Mixed Content Issue
# The frontend is served over HTTPS but the backend is HTTP.
# Browsers block mixed content (HTTPS -> HTTP) for security.
# Solution: Configure HTTPS on the backend with SSL certificates.
# 
# To enable HTTPS, set environment variables:
# export SSL_KEYFILE="/path/to/key.pem"
# export SSL_CERTFILE="/path/to/cert.pem"
# Then use: uvicorn main:app --ssl-keyfile=$SSL_KEYFILE --ssl-certfile=$SSL_CERTFILE --port 8000
#
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins when credentials=False
    allow_credentials=False,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
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
    # Mask account numbers (keeping last 2 characters)
    d["from_account"] = mask_token(d["from_account"])
    d["to_account"] = mask_token(d["to_account"])
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
    if not txn.from_account:
        return False, "Missing from_account"
    if not txn.to_account:
        return False, "Missing to_account"
    if not txn.transaction_type:
        return False, "Missing transaction_type"
    return True, ""


# ============ FRAUD DETECTION RULES ============


def analyze_transaction(txn: Transaction) -> FraudDecision:
    """
    Deterministic fraud detection with red flags using dynamic configuration
    """
    flags = []
    
    # Get current configuration
    config = config_service.get_detection_config()
    
    # US locations (non-high-risk)
    US_LOCATIONS = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", 
                    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
                    "San Francisco", "Columbus", "Fort Worth", "Charlotte", "Seattle", "Denver",
                    "Washington", "Boston", "El Paso", "Detroit", "Nashville", "Portland"]
    
    # Check for large amount (using dynamic threshold)
    if txn.amount > config.large_amount_threshold:
        flags.append("large_amount")
    
    # Check for suspicious category (using dynamic list)
    if txn.category in config.high_risk_categories:
        flags.append("suspicious_category")
    
    # Check for geographic risk (using dynamic list)
    if txn.location in config.high_risk_locations or txn.location not in US_LOCATIONS:
        flags.append("geo_risk")
    
    # Check for suspicious channel combinations
    if txn.transaction_type == "withdrawal" and txn.amount > config.suspicious_channel_amount and txn.channel in config.suspicious_channels:
        flags.append("suspicious_channel")
    
    # Check for unusual transaction types with large amounts
    if txn.transaction_type == "transfer" and txn.amount > config.large_transfer_threshold:
        flags.append("large_transfer")
    
    # Check for deposit anomalies (very large deposits)
    if txn.transaction_type == "deposit" and txn.amount > config.unusual_deposit_threshold:
        flags.append("unusual_deposit")

    # Decision logic (using dynamic thresholds)
    num_flags = len(flags)

    if num_flags >= config.fraud_flag_threshold:
        # High confidence fraud
        confidence = min(0.95, config.high_confidence_base + (num_flags * config.confidence_per_flag))
        decision = "FRAUD"
        explanation = f"Multiple red flags detected: {', '.join(flags)}"
    elif num_flags == config.legit_flag_threshold:
        # Low risk
        confidence = 0.75
        decision = "LEGIT"
        explanation = "No red flags detected"
    else:
        # Uncertain - needs human review
        confidence = config.review_confidence
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
    """Health check endpoint"""
    return {"service": "Fraud Detection API", "version": "1.0.0", "status": "running"}


@app.post("/auth/yubikey/otp/request")
async def request_yubikey_otp(request: OTPRequest) -> dict:
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
        # Log failed login attempt
        audit_service.log(
            action="LOGIN_FAILED",
            resource=yubikey_id,
            details={"reason": "Invalid YubiKey ID"},
            status="error",
            error_message="Invalid YubiKey ID"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid YubiKey ID"
        )
    
    user_data = YUBIKEY_DB[yubikey_id]
    
    # Check if OTP exists and is valid
    if yubikey_id not in OTP_STORE:
        audit_service.log(
            action="LOGIN_FAILED",
            user_id=user_data["user_id"],
            user_name=user_data["name"],
            user_role=user_data["role"],
            resource=yubikey_id,
            details={"reason": "No OTP found"},
            status="error",
            error_message="No OTP found"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No OTP found. Please request a new one."
        )
    
    otp_data = OTP_STORE[yubikey_id]
    
    # Check if OTP expired
    expires_at = datetime.fromisoformat(otp_data["expires_at"])
    if datetime.utcnow() > expires_at:
        del OTP_STORE[yubikey_id]
        audit_service.log(
            action="LOGIN_FAILED",
            user_id=user_data["user_id"],
            user_name=user_data["name"],
            user_role=user_data["role"],
            resource=yubikey_id,
            details={"reason": "OTP expired"},
            status="error",
            error_message="OTP expired"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OTP expired. Please request a new one."
        )
    
    # Verify OTP
    if otp_data["otp"] != otp:
        audit_service.log(
            action="LOGIN_FAILED",
            user_id=user_data["user_id"],
            user_name=user_data["name"],
            user_role=user_data["role"],
            resource=yubikey_id,
            details={"reason": "Invalid OTP"},
            status="error",
            error_message="Invalid OTP"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid OTP"
        )
    
    # OTP verified, create session
    session = create_session(yubikey_id, user_data)
    
    # Log successful login
    audit_service.log(
        action="LOGIN_SUCCESS",
        user_id=session["user_id"],
        user_name=session["name"],
        user_role=session["role"],
        resource=yubikey_id,
        details={
            "session_token": session["session_token"][:8] + "...",
            "expires_at": session["expires_at"]
        },
        status="success"
    )
    
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
            "true_label": "FRAUD" if txn.is_fraud else "LEGIT",
        }

    # Analyze
    result = analyze_transaction(txn)

    return {
        "txn_id": txn_id,
        "decision": result.decision,
        "confidence": result.confidence,
        "flags": result.flags,
        "explanation": mask_pii(result.explanation),
        "true_label": "FRAUD" if txn.is_fraud else "LEGIT",
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
                    true_label="FRAUD" if txn.is_fraud else "LEGIT",
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
                true_label="FRAUD" if txn.is_fraud else "LEGIT",
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
            true_label="FRAUD" if txn.is_fraud else "LEGIT"
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
    analysis = analyze_transaction(txn)
    
    # Return transaction with proper schema mapping
    return {
        "transaction": {
            "id": txn.id,
            "timestamp": txn.timestamp,
            "amount": txn.amount,
            "from_account": mask_token(txn.from_account),
            "to_account": mask_token(txn.to_account),
            "transaction_type": txn.transaction_type,
            "category": txn.category,
            "location": txn.location,
            "channel": txn.channel,
            "label": "FRAUD" if txn.is_fraud else "LEGIT",
            # Add backward compatibility fields for frontend
            "merchant": f"{txn.category.title()} - {txn.location}",
            "device_id": f"device_{txn.from_account[-6:]}",
            "geo": txn.location,
            "velocity_30d": 15 if not txn.is_fraud else 2,  # Mock data
            "avg_amount_30d": txn.amount * 0.8,
            "merchant_known": not txn.is_fraud
        },
        "analysis": analysis.dict()
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
        
        # Log model training
        audit_service.log(
            action="MODEL_TRAINED",
            user_id=current_user["user_id"],
            user_name=current_user["name"],
            user_role=current_user["role"],
            resource=model_id,
            details={
                "model_type": "PPO",
                "training_steps": timesteps,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"]
            },
            status="success"
        )
        
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
            
            # Log the review decision
            audit_service.log(
                action="REVIEW_DECISION",
                user_id=current_user["user_id"],
                user_name=current_user["name"],
                user_role=current_user["role"],
                resource=request.txn_id,
                details={
                    "decision": request.human_decision,
                    "original_decision": case.decision,
                    "confidence": case.confidence,
                    "notes": request.reviewer_notes,
                    "flags": case.flags,
                    "true_label": case.true_label
                },
                status="success"
            )
            
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
            "true_label": "FRAUD" if txn.is_fraud else "LEGIT",
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
                    true_label="FRAUD" if txn.is_fraud else "LEGIT",
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
                    true_label="FRAUD" if txn.is_fraud else "LEGIT",
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
        "true_label": "FRAUD" if txn.is_fraud else "LEGIT",
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


# ============ LIVE FEED ENDPOINTS ============

@app.post("/live-feed/start")
async def start_live_feed(
    interval_seconds: float = 0.33,
    current_user: dict = Depends(get_current_user)
):
    """Start the live transaction feed (streams existing transactions)"""
    global LIVE_FEED_ACTIVE, live_feed_task
    
    if LIVE_FEED_ACTIVE:
        return {"status": "already_running", "message": "Live feed is already running"}
    
    LIVE_FEED_ACTIVE = True
    
    # Start background task (fraud_rate parameter not used anymore)
    live_feed_task = asyncio.create_task(live_feed_worker(interval_seconds, 0.1))
    
    return {
        "status": "started",
        "interval_seconds": interval_seconds,
        "message": f"Live feed started: streaming transactions every {interval_seconds}s"
    }


@app.post("/live-feed/stop")
def stop_live_feed(current_user: dict = Depends(get_current_user)):
    """Stop the live transaction feed"""
    global LIVE_FEED_ACTIVE, live_feed_task
    
    if not LIVE_FEED_ACTIVE:
        return {"status": "not_running", "message": "Live feed is not running"}
    
    LIVE_FEED_ACTIVE = False
    
    # Cancel task if running
    if live_feed_task and not live_feed_task.done():
        live_feed_task.cancel()
    
    return {"status": "stopped", "message": "Live feed stopped"}


@app.get("/live-feed/status")
def get_live_feed_status(current_user: dict = Depends(get_current_user)):
    """Get live feed status and statistics"""
    global LIVE_FEED_ACTIVE, LIVE_FEED_STATS, LIVE_FEED_QUEUE, live_feed_logger
    
    response = {
        "active": LIVE_FEED_ACTIVE,
        "stats": LIVE_FEED_STATS.copy(),
        "queue_size": len(LIVE_FEED_QUEUE),
        "latest_transactions": list(LIVE_FEED_QUEUE)[-10:] if LIVE_FEED_QUEUE else []
    }
    
    # Add performance metrics if logger is available
    if live_feed_logger:
        response["performance_metrics"] = live_feed_logger.get_current_metrics()
        response["session_id"] = live_feed_logger.session_id
        response["log_files"] = {
            "transactions": str(live_feed_logger.log_file.name),
            "metrics": str(live_feed_logger.metrics_file.name)
        }
    
    return response


@app.get("/live-feed/stream")
async def stream_live_feed(current_user: dict = Depends(get_current_user)):
    """Stream live transactions using Server-Sent Events (SSE)"""
    global LIVE_FEED_QUEUE
    
    async def event_generator():
        """Generate SSE events"""
        last_index = len(LIVE_FEED_QUEUE)
        
        while True:
            # Check for new transactions
            current_size = len(LIVE_FEED_QUEUE)
            
            if current_size > last_index:
                # Send new transactions
                for i in range(last_index, current_size):
                    transaction = list(LIVE_FEED_QUEUE)[i]
                    yield f"data: {json.dumps(transaction)}\n\n"
                last_index = current_size
            
            # Send heartbeat to keep connection alive
            yield ": heartbeat\n\n"
            
            await asyncio.sleep(0.5)  # Check every 500ms
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )


# ============ LIVE FEED LOGS API ENDPOINTS ============

@app.get("/live-feed/logs/metrics")
def get_live_feed_metrics(
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """Get performance metrics history from live feed"""
    global live_feed_logger
    
    if not live_feed_logger:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Live feed logger not available. Start the live feed first."
        )
    
    history = live_feed_logger.get_performance_history(limit=limit)
    retraining_history = live_feed_logger.get_retraining_history()
    
    return {
        "performance_history": history,
        "retraining_history": retraining_history,
        "current_metrics": live_feed_logger.get_current_metrics(),
        "session_id": live_feed_logger.session_id
    }


@app.get("/live-feed/logs/transactions")
def get_live_feed_transactions(
    limit: int = 1000,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get logged transactions from live feed"""
    global live_feed_logger
    
    if not live_feed_logger:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Live feed logger not available. Start the live feed first."
        )
    
    if not live_feed_logger.log_file.exists():
        return {
            "transactions": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }
    
    # Read transactions from log file
    transactions = []
    with open(live_feed_logger.log_file, "r") as f:
        lines = f.readlines()
        total = len(lines)
        
        # Get requested range (reverse order - most recent first)
        start = max(0, total - offset - limit)
        end = total - offset
        
        for line in lines[start:end]:
            try:
                transactions.append(json.loads(line.strip()))
            except:
                continue
    
    # Reverse to show most recent first
    transactions.reverse()
    
    return {
        "transactions": transactions,
        "total": total,
        "limit": limit,
        "offset": offset,
        "session_id": live_feed_logger.session_id
    }


@app.get("/live-feed/logs/download")
def download_live_feed_logs(
    log_type: str = "all",  # "transactions", "metrics", or "all"
    current_user: dict = Depends(get_current_user)
):
    """Download live feed logs as files"""
    global live_feed_logger
    
    if not live_feed_logger:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Live feed logger not available. Start the live feed first."
        )
    
    from fastapi.responses import FileResponse
    
    if log_type == "transactions":
        if live_feed_logger.log_file.exists():
            return FileResponse(
                str(live_feed_logger.log_file),
                media_type="application/json",
                filename=f"live_feed_transactions_{live_feed_logger.session_id}.jsonl"
            )
    elif log_type == "metrics":
        if live_feed_logger.metrics_file.exists():
            return FileResponse(
                str(live_feed_logger.metrics_file),
                media_type="application/json",
                filename=f"live_feed_metrics_{live_feed_logger.session_id}.jsonl"
            )
    elif log_type == "all":
        # Create a zip file with both logs
        import zipfile
        import tempfile
        
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            if live_feed_logger.log_file.exists():
                zipf.write(live_feed_logger.log_file, f"transactions_{live_feed_logger.session_id}.jsonl")
            if live_feed_logger.metrics_file.exists():
                zipf.write(live_feed_logger.metrics_file, f"metrics_{live_feed_logger.session_id}.jsonl")
        
        return FileResponse(
            temp_zip.name,
            media_type="application/zip",
            filename=f"live_feed_logs_{live_feed_logger.session_id}.zip"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="log_type must be 'transactions', 'metrics', or 'all'"
        )


# ============ AUDIT LOG API ENDPOINTS ============


@app.get("/audit/logs")
def get_audit_logs(
    limit: int = 100,
    offset: int = 0,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Get audit logs with filtering (admin/analyst only)"""
    # Only admin and analyst can view audit logs
    if current_user["role"] not in ["admin", "analyst"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin and analyst users can view audit logs"
        )
    
    try:
        logs, total = audit_service.get_logs(
            limit=limit,
            offset=offset,
            user_id=user_id,
            action=action,
            status=status,
            start_date=start_date,
            end_date=end_date,
            search=search
        )
        
        return {
            "logs": [log.model_dump() for log in logs],
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        # Return empty logs on first run or error, don't fail
        print(f"Audit log error (non-critical): {e}")
        return {
            "logs": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }


@app.get("/audit/stats")
def get_audit_stats(current_user: dict = Depends(get_current_user)) -> dict:
    """Get audit log statistics (admin only)"""
    if current_user["role"] not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can view audit statistics"
        )
    
    # Get all logs
    all_logs, total = audit_service.get_logs(limit=10000)
    
    # Calculate statistics
    actions = {}
    users = {}
    errors = 0
    
    for log in all_logs:
        # Count by action
        actions[log.action] = actions.get(log.action, 0) + 1
        
        # Count by user
        if log.user_name:
            users[log.user_name] = users.get(log.user_name, 0) + 1
        
        # Count errors
        if log.status == "error":
            errors += 1
    
    return {
        "total_logs": total,
        "total_errors": errors,
        "top_actions": sorted(actions.items(), key=lambda x: x[1], reverse=True)[:10],
        "top_users": sorted(users.items(), key=lambda x: x[1], reverse=True)[:10],
        "error_rate": round(errors / total * 100, 2) if total > 0 else 0
    }


@app.get("/audit/export/csv")
def export_audit_logs_csv(
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Export audit logs as CSV (admin only)"""
    if current_user["role"] not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can export audit logs"
        )
    
    csv_data = audit_service.export_logs_csv(
        user_id=user_id,
        action=action,
        start_date=start_date,
        end_date=end_date
    )
    
    from fastapi.responses import Response
    
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=audit_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    )


@app.get("/live-feed/recent")
def get_recent_transactions(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get recent transactions from the live feed"""
    global LIVE_FEED_QUEUE
    
    recent = list(LIVE_FEED_QUEUE)[-limit:] if LIVE_FEED_QUEUE else []
    return {
        "count": len(recent),
        "transactions": recent
    }


@app.get("/audit/export/json")
def export_audit_logs_json(
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Export audit logs as JSON (admin only)"""
    if current_user["role"] not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can export audit logs"
        )
    
    logs, total = audit_service.get_logs(
        limit=10000,
        user_id=user_id,
        action=action,
        start_date=start_date,
        end_date=end_date
    )
    
    from fastapi.responses import Response
    
    json_data = json.dumps({
        "export_date": datetime.utcnow().isoformat(),
        "total_logs": total,
        "logs": [log.model_dump() for log in logs]
    }, indent=2)
    
    return Response(
        content=json_data,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=audit_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        }
    )


# ============ CONFIGURATION API ENDPOINTS ============


@app.get("/config/detection")
def get_detection_config(current_user: dict = Depends(get_current_user)) -> DetectionConfig:
    """Get current detection configuration (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can view configuration"
        )
    
    return config_service.get_detection_config()


@app.put("/config/detection")
def update_detection_config(
    updates: dict,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Update detection configuration (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can update configuration"
        )
    
    try:
        # Apply updates
        updated_config = config_service.update_detection_config(updates)
        
        # Log the change
        audit_service.log(
            action="CONFIG_UPDATED",
            user_id=current_user["user_id"],
            user_name=current_user["name"],
            user_role=current_user["role"],
            resource="detection_config",
            details={"updates": updates},
            status="success"
        )
        
        return {
            "message": "Configuration updated successfully. Changes will apply to new analyses.",
            "config": updated_config.model_dump()
        }
        
    except Exception as e:
        audit_service.log(
            action="CONFIG_UPDATE_FAILED",
            user_id=current_user["user_id"],
            user_name=current_user["name"],
            user_role=current_user["role"],
            resource="detection_config",
            details={"updates": updates, "error": str(e)},
            status="error",
            error_message=str(e)
        )
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/config/detection/reset")
def reset_detection_config(current_user: dict = Depends(get_current_user)) -> dict:
    """Reset detection configuration to defaults (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can reset configuration"
        )
    
    config = config_service.reset_to_defaults()
    
    # Log the reset
    audit_service.log(
        action="CONFIG_RESET",
        user_id=current_user["user_id"],
        user_name=current_user["name"],
        user_role=current_user["role"],
        resource="detection_config",
        details={"reset_to": "defaults"},
        status="success"
    )
    
    return {
        "message": "Configuration reset to defaults",
        "config": config.detection.model_dump()
    }


@app.get("/config/system")
def get_system_config(current_user: dict = Depends(get_current_user)) -> SystemConfig:
    """Get full system configuration (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can view system configuration"
        )
    
    return config_service.get_system_config()


if __name__ == "__main__":
    import uvicorn
    import os

    # Check if we should force HTTP mode (for reverse proxy setups)
    force_http = os.getenv("FORCE_HTTP", "").lower() in ("true", "1", "yes")
    
    if force_http:
        print("🌐 Running with HTTP (FORCE_HTTP enabled - for reverse proxy setups)")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Get SSL certificate paths from environment or check common locations
        ssl_keyfile = os.getenv("SSL_KEYFILE")
        ssl_certfile = os.getenv("SSL_CERTFILE")
        
        # If not set, check common certificate locations
        if not ssl_keyfile or not ssl_certfile:
            common_key_paths = [
                "/homes/iws/micibr/ssl/attu2.cs.washington.edu.key",
                f"{os.path.expanduser('~')}/ssl/attu2.cs.washington.edu.key",
                f"{os.path.expanduser('~')}/.ssl/attu2.cs.washington.edu.key",
                "/etc/ssl/private/attu2.cs.washington.edu.key",
                "/etc/letsencrypt/live/attu2.cs.washington.edu/privkey.pem",
            ]
            common_cert_paths = [
                "/homes/iws/micibr/ssl/attu2.cs.washington.edu.crt",
                f"{os.path.expanduser('~')}/ssl/attu2.cs.washington.edu.crt",
                f"{os.path.expanduser('~')}/.ssl/attu2.cs.washington.edu.crt",
                "/etc/ssl/certs/attu2.cs.washington.edu.crt",
                "/etc/letsencrypt/live/attu2.cs.washington.edu/fullchain.pem",
            ]
            
            for key_path in common_key_paths:
                if os.path.exists(key_path):
                    ssl_keyfile = key_path
                    break
            
            for cert_path in common_cert_paths:
                if os.path.exists(cert_path):
                    ssl_certfile = cert_path
                    break
        
        # Check if certificates exist
        use_https = ssl_keyfile and ssl_certfile and os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile)
        
        if use_https:
            print(f"🔒 Running with HTTPS using certificates:")
            print(f"   Key: {ssl_keyfile}")
            print(f"   Cert: {ssl_certfile}")
            uvicorn.run(
                app, 
                host="0.0.0.0", 
                port=8000,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile
            )
        else:
            print("⚠️  SSL certificates not found. Running with HTTP")
            print("   To use HTTPS, set SSL_KEYFILE and SSL_CERTFILE environment variables")
            print("   Or set FORCE_HTTP=true if using a reverse proxy (nginx/apache)")
            print("   Note: If using reverse proxy, keep backend on HTTP and configure proxy for HTTPS")
            uvicorn.run(app, host="0.0.0.0", port=8000)
