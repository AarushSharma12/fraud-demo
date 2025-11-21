import time
import requests
import random
from backend.main import LiveTransactionGenerator, Transaction 

# Configuration
# Point to your local backend
DETECTOR_URL = "http://127.0.0.1:8000/analyze/raw" 
INTERVAL = 2  # Faster interval for local testing

def run_industry():
    # Initialize generator with empty base to use defaults
    # In a real scenario, you'd pass the existing TRANSACTIONS from main.py
    # For this test script, we'll instantiate a dummy generator
    dummy_txns = {} 
    generator = LiveTransactionGenerator(dummy_txns)
    
    # Manually populate account pool since we passed empty dict
    generator.account_pool = [f"ACC{i:04d}" for i in range(1, 20)]
    
    print(f"🏭 Local Synthetic Industry Started - Target: {DETECTOR_URL}")
    print("Press Ctrl+C to stop")
    
    while True:
        try:
            # 1. Generate synthetic transaction
            txn = generator.generate_transaction(fraud_probability=0.20)
            
            # 2. Prepare Payload
            payload = {
                "amount": txn.amount,
                "from_account": txn.from_account,
                "to_account": txn.to_account,
                "transaction_type": txn.transaction_type,
                "category": txn.category,
                "location": txn.location,
                "channel": txn.channel
            }
            
            # 3. Attack/Transact
            start = time.time()
            response = requests.post(DETECTOR_URL, json=payload)
            latency = (time.time() - start) * 1000
            
            # 4. Log Result
            if response.status_code == 200:
                result = response.json()
                decision = result['decision']
                conf = result['confidence']
                icon = "🛡️" if decision == "FRAUD" else "✅"
                print(f"{icon} [Latency: {latency:.0f}ms] Sent ${txn.amount:.2f} -> {decision} ({conf:.2f})")
            else:
                print(f"❌ Failed: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("⚠️  Connection refused. Is the backend running?")
        except Exception as e:
            print(f"⚠️  Error: {e}")
            
        time.sleep(INTERVAL)

if __name__ == "__main__":
    run_industry()

