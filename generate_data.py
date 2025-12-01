#!/usr/bin/env python3
"""
Generate comprehensive fraud detection dataset in Kaggle format
Creates ~1000 transactions with realistic patterns
"""

import json
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)

# Transaction types
TRANSACTION_TYPES = ["withdrawal", "deposit", "transfer", "payment"]

# Categories
CATEGORIES = [
    "utilities", "online", "other", "entertainment", 
    "travel", "grocery", "retail", "restaurant"
]

# Locations (mix of US and international)
LOCATIONS = [
    # US cities
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
    "San Francisco", "Columbus", "Fort Worth", "Charlotte", "Seattle", "Denver",
    "Washington", "Boston", "El Paso", "Detroit", "Nashville", "Portland",
    # International cities
    "Tokyo", "Toronto", "London", "Sydney", "Berlin", "Dubai", "Singapore",
    "Paris", "Mumbai", "Shanghai", "Mexico City", "São Paulo", "Bangkok",
    "Hong Kong", "Seoul", "Amsterdam", "Rome", "Madrid", "Vancouver"
]

# Channels
CHANNELS = ["mobile", "atm", "pos", "web"]

# US locations for fraud detection
US_LOCATIONS = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
    "San Francisco", "Columbus", "Fort Worth", "Charlotte", "Seattle", "Denver",
    "Washington", "Boston", "El Paso", "Detroit", "Nashville", "Portland"
]

# High-risk categories
HIGH_RISK_CATEGORIES = ["other", "online"]

def generate_account_id():
    """Generate account ID in format ACC######"""
    return f"ACC{random.randint(100000, 999999)}"

def generate_transaction_id(index):
    """Generate transaction IDs starting from T100001"""
    return f"T{100000 + index}"

def generate_timestamp(start_date=None):
    """Generate random timestamp within last year"""
    if start_date is None:
        start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    random_seconds = random.randint(0, 86400)
    
    timestamp = start_date + timedelta(days=random_days, seconds=random_seconds)
    return timestamp.isoformat()

def determine_fraud(transaction_type, category, location, amount, channel):
    """Determine if transaction is fraud based on patterns"""
    fraud_score = 0
    
    # High-risk categories
    if category in HIGH_RISK_CATEGORIES:
        fraud_score += 2
    
    # Non-US locations
    if location not in US_LOCATIONS:
        fraud_score += 2
    
    # Large amounts
    if amount > 1000:
        fraud_score += 1
    if amount > 5000:
        fraud_score += 1
    
    # Suspicious channel combinations
    if transaction_type == "withdrawal" and amount > 500 and channel in ["mobile", "web"]:
        fraud_score += 1
    
    # Large transfers
    if transaction_type == "transfer" and amount > 2000:
        fraud_score += 1
    
    # Very large deposits
    if transaction_type == "deposit" and amount > 5000:
        fraud_score += 1
    
    # Random fraud injection (5% baseline)
    if random.random() < 0.05:
        fraud_score += 2
    
    return fraud_score >= 3

def generate_transaction(index):
    """Generate a single transaction in Kaggle format"""
    
    # Determine if this will be fraud
    is_fraud = random.random() < 0.15  # 15% fraud rate baseline
    
    # Generate transaction type
    transaction_type = random.choice(TRANSACTION_TYPES)
    
    # Generate category (higher chance of high-risk for fraud)
    if is_fraud:
        category = random.choices(
            CATEGORIES,
            weights=[1, 3, 3, 1, 1, 1, 1, 1]  # Higher weight for "online" and "other"
        )[0]
    else:
        category = random.choice(CATEGORIES)
    
    # Generate location (higher chance of non-US for fraud)
    if is_fraud:
        location = random.choices(
            LOCATIONS,
            weights=[1 if loc in US_LOCATIONS else 3 for loc in LOCATIONS]
        )[0]
    else:
        location = random.choice(LOCATIONS)
    
    # Generate channel
    channel = random.choice(CHANNELS)
    
    # Generate amount based on transaction type and fraud status
    if is_fraud:
        if transaction_type == "withdrawal":
            amount = random.uniform(200, 10000)
        elif transaction_type == "deposit":
            amount = random.uniform(1000, 15000)
        elif transaction_type == "transfer":
            amount = random.uniform(500, 8000)
        else:  # payment
            amount = random.uniform(100, 5000)
    else:
        if transaction_type == "withdrawal":
            amount = random.uniform(10, 500)
        elif transaction_type == "deposit":
            amount = random.uniform(50, 2000)
        elif transaction_type == "transfer":
            amount = random.uniform(20, 1000)
        else:  # payment
            amount = random.uniform(5, 300)
    
    # Add some extreme outliers
    if random.random() < 0.02:  # 2% extreme amounts
        amount = random.uniform(5000, 50000)
    
    amount = round(amount, 2)
    
    # Re-evaluate fraud based on actual generated values
    is_fraud = determine_fraud(transaction_type, category, location, amount, channel)
    
    # Generate accounts
    from_account = generate_account_id()
    to_account = generate_account_id()
    
    # Generate timestamp
    timestamp = generate_timestamp()
    
    return {
        "id": generate_transaction_id(index),
        "timestamp": timestamp,
        "from_account": from_account,
        "to_account": to_account,
        "amount": amount,
        "transaction_type": transaction_type,
        "category": category,
        "location": location,
        "channel": channel,
        "is_fraud": is_fraud
    }

def main():
    """Generate the dataset"""
    print("🔄 Generating comprehensive fraud detection dataset in Kaggle format...")
    
    transactions = []
    
    # Generate 1000 transactions
    for i in range(1, 1001):
        if i % 100 == 0:
            print(f"Generated {i} transactions...")
        transactions.append(generate_transaction(i))
    
    # Save to file
    with open("backend/data.json", "w") as f:
        json.dump(transactions, f, indent=2)
    
    # Calculate statistics
    total = len(transactions)
    fraud_count = sum(1 for t in transactions if t["is_fraud"])
    legit_count = total - fraud_count
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"📊 Statistics:")
    print(f"   Total transactions: {total}")
    print(f"   Fraud cases: {fraud_count} ({fraud_count/total*100:.1f}%)")
    print(f"   Legitimate cases: {legit_count} ({legit_count/total*100:.1f}%)")
    print(f"   Average amount: ${sum(t['amount'] for t in transactions)/total:.2f}")
    print(f"   Max amount: ${max(t['amount'] for t in transactions):.2f}")
    print(f"   Min amount: ${min(t['amount'] for t in transactions):.2f}")
    
    # Show some examples
    print(f"\n📋 Sample transactions:")
    for i in [0, 1, 2, 998, 999]:
        t = transactions[i]
        fraud_label = "FRAUD" if t["is_fraud"] else "LEGIT"
        print(f"   {t['id']}: ${t['amount']:.2f} {t['transaction_type']} ({t['category']}) in {t['location']} via {t['channel']} → {fraud_label}")

if __name__ == "__main__":
    main()
