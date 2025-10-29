#!/usr/bin/env python3
"""
Test script for RL fraud detection system
"""

import requests
import json
import time
import sys
import os

# Add backend to path for local testing
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

BASE_URL = "http://localhost:8000"

def test_rl_system():
    """Test the RL fraud detection system"""
    print("🚀 Testing RL Fraud Detection System")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Server is running: {response.json()}")
    except:
        print("❌ Server is not running. Please start it with: python backend/main.py")
        return
    
    # Check RL status
    print("\n📊 Checking RL Model Status...")
    response = requests.get(f"{BASE_URL}/rl/status")
    status = response.json()
    print(f"Model Available: {status['model_available']}")
    print(f"Training Data Size: {status['training_data_size']}")
    
    # Train the RL model
    print("\n🎯 Training RL Model...")
    print("This may take a few minutes with 1000 transactions...")
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/rl/train", params={"timesteps": 10000})
    training_time = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Training completed in {training_time:.1f} seconds")
        print(f"Model Type: {result['model_type']}")
        print(f"Training Steps: {result['training_steps']}")
        print(f"Final Reward: {result['final_reward']:.2f}")
        print(f"Accuracy: {result['accuracy']:.3f}")
        print(f"Precision: {result['precision']:.3f}")
        print(f"Recall: {result['recall']:.3f}")
    else:
        print(f"❌ Training failed: {response.text}")
        return
    
    # Test individual predictions
    print("\n🔍 Testing Individual Predictions...")
    
    # Get some transaction IDs (test a mix of different types)
    response = requests.get(f"{BASE_URL}/transactions")
    all_ids = response.json()["ids"]
    # Test a mix: first few, some from middle, and some from end
    txn_ids = all_ids[:3] + all_ids[500:503] + all_ids[997:1000]
    
    for txn_id in txn_ids:
        print(f"\n--- Transaction {txn_id} ---")
        
        # Compare rule-based vs RL
        response = requests.post(f"{BASE_URL}/compare/{txn_id}")
        comparison = response.json()
        
        print(f"True Label: {comparison['true_label']}")
        print(f"Rule-based: {comparison['rule_based']['decision']} (confidence: {comparison['rule_based']['confidence']:.2f})")
        print(f"RL Model: {comparison['rl_model']['decision']} (confidence: {comparison['rl_model']['confidence']:.2f})")
        print(f"Agreement: {comparison['agreement']}")
    
    # Test batch analysis
    print("\n📈 Testing Batch Analysis...")
    response = requests.post(f"{BASE_URL}/rl/batch")
    batch_result = response.json()
    
    print(f"Total Transactions: {batch_result['total']}")
    print(f"Metrics: {batch_result['metrics']}")
    
    print("\n🎉 RL System Test Complete!")

if __name__ == "__main__":
    test_rl_system()
