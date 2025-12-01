#!/usr/bin/env python3
"""
Convert Kaggle financial fraud dataset CSV to our JSON format
with balanced sampling of fraud and non-fraud cases
"""

import csv
import json
import random
import sys

def convert_csv_to_json_balanced(csv_path, json_path, target_count=1000, fraud_rate=0.15):
    """Convert CSV to JSON format with balanced fraud/non-fraud sampling"""
    
    print(f"📖 Reading CSV from {csv_path}...")
    
    # First pass: collect fraud and non-fraud indices
    fraud_indices = []
    legit_indices = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(reader):
            is_fraud = row['is_fraud'].strip().lower() == 'true'
            
            if is_fraud:
                fraud_indices.append(idx)
            else:
                legit_indices.append(idx)
            
            if (idx + 1) % 500000 == 0:
                print(f"   Scanned {idx + 1} rows...")
    
    print(f"✅ Found {len(fraud_indices)} fraud cases and {len(legit_indices)} legitimate cases")
    
    # Calculate how many of each we need
    target_fraud = int(target_count * fraud_rate)
    target_legit = target_count - target_fraud
    
    # Sample randomly
    random.seed(42)  # For reproducibility
    sampled_fraud_indices = random.sample(fraud_indices, min(target_fraud, len(fraud_indices)))
    sampled_legit_indices = random.sample(legit_indices, min(target_legit, len(legit_indices)))
    
    # Combine and sort by original index to maintain some order
    all_sampled_indices = sorted(sampled_fraud_indices + sampled_legit_indices)
    
    print(f"📊 Sampling {len(sampled_fraud_indices)} fraud and {len(sampled_legit_indices)} legitimate transactions")
    
    # Second pass: read the sampled rows
    transactions = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        current_sampled_idx = 0
        
        for idx, row in enumerate(reader):
            if current_sampled_idx >= len(all_sampled_indices):
                break
            
            if idx == all_sampled_indices[current_sampled_idx]:
                # Convert is_fraud from string "False"/"True" to boolean
                is_fraud = row['is_fraud'].strip().lower() == 'true'
                
                # Map device_used to channel (mobile, atm, pos, web)
                device_used = row['device_used'].strip().lower()
                # Ensure it's one of our expected values
                if device_used not in ['mobile', 'atm', 'pos', 'web']:
                    # Map common variations
                    if 'mobile' in device_used or 'phone' in device_used:
                        device_used = 'mobile'
                    elif 'atm' in device_used:
                        device_used = 'atm'
                    elif 'pos' in device_used or 'point' in device_used:
                        device_used = 'pos'
                    elif 'web' in device_used or 'online' in device_used:
                        device_used = 'web'
                    else:
                        device_used = 'web'  # default
                
                transaction = {
                    "id": row['transaction_id'].strip(),
                    "timestamp": row['timestamp'].strip(),
                    "from_account": row['sender_account'].strip(),
                    "to_account": row['receiver_account'].strip(),
                    "amount": float(row['amount']),
                    "transaction_type": row['transaction_type'].strip().lower(),
                    "category": row['merchant_category'].strip().lower(),
                    "location": row['location'].strip(),
                    "channel": device_used,
                    "is_fraud": is_fraud
                }
                
                transactions.append(transaction)
                current_sampled_idx += 1
                
                if len(transactions) % 100 == 0:
                    print(f"   Processed {len(transactions)} transactions...")
    
    print(f"✅ Processed {len(transactions)} transactions")
    
    # Save to JSON
    print(f"💾 Saving to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(transactions, f, indent=2)
    
    # Calculate statistics
    total = len(transactions)
    fraud_count = sum(1 for t in transactions if t["is_fraud"])
    legit_count = total - fraud_count
    
    print(f"\n📊 Statistics:")
    print(f"   Total transactions: {total}")
    print(f"   Fraud cases: {fraud_count} ({fraud_count/total*100:.1f}%)")
    print(f"   Legitimate cases: {legit_count} ({legit_count/total*100:.1f}%)")
    print(f"   Average amount: ${sum(t['amount'] for t in transactions)/total:.2f}")
    print(f"   Max amount: ${max(t['amount'] for t in transactions):.2f}")
    print(f"   Min amount: ${min(t['amount'] for t in transactions):.2f}")
    
    # Show some examples
    print(f"\n📋 Sample transactions:")
    fraud_samples = [t for t in transactions if t["is_fraud"]][:3]
    legit_samples = [t for t in transactions if not t["is_fraud"]][:3]
    
    print("   Fraud examples:")
    for t in fraud_samples:
        print(f"     {t['id']}: ${t['amount']:.2f} {t['transaction_type']} ({t['category']}) in {t['location']} via {t['channel']}")
    
    print("   Legitimate examples:")
    for t in legit_samples:
        print(f"     {t['id']}: ${t['amount']:.2f} {t['transaction_type']} ({t['category']}) in {t['location']} via {t['channel']}")

if __name__ == "__main__":
    csv_path = "/tmp/financial_fraud_detection_dataset.csv"
    json_path = "backend/data.json"
    
    target_count = 1000
    fraud_rate = 0.15  # 15% fraud rate for demo
    
    if len(sys.argv) > 1:
        target_count = int(sys.argv[1])
    if len(sys.argv) > 2:
        fraud_rate = float(sys.argv[2])
    
    convert_csv_to_json_balanced(csv_path, json_path, target_count, fraud_rate)
    print(f"\n✅ Conversion complete! Data saved to {json_path}")
