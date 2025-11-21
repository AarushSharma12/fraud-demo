import time
import requests
import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from backend.main import LiveTransactionGenerator

# TEAM 4 CONFIGURATION
# Target Localhost on Port 8004
DETECTOR_URL = "http://127.0.0.1:8004/analyze/raw" 
INTERVAL = 5 

def run_industry():
    print(f"🏭 TEAM 4 FINANCE INDUSTRY STARTED - Target: {DETECTOR_URL}")
    # Initialize with empty dict, similar to local generator
    generator = LiveTransactionGenerator({}) 
    # Populate fake accounts
    generator.account_pool = [f"ACC{i:04d}" for i in range(1, 100)]
    
    while True:
        try:
            txn = generator.generate_transaction(fraud_probability=0.15)
            payload = {
                "amount": txn.amount,
                "from_account": txn.from_account,
                "to_account": txn.to_account,
                "transaction_type": txn.transaction_type,
                "category": txn.category,
                "location": txn.location,
                "channel": txn.channel
            }
            # Short timeout to prevent hanging
            requests.post(DETECTOR_URL, json=payload, timeout=2)
        except Exception:
            pass # Silently continue in prod
        time.sleep(INTERVAL)

if __name__ == "__main__":
    run_industry()

