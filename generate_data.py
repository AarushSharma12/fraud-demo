#!/usr/bin/env python3
"""
Generate comprehensive fraud detection dataset
Creates ~1000 transactions with realistic patterns
"""

import json
import random
import uuid
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)

# Merchant categories and risk levels
LEGIT_MERCHANTS = [
    "Starbucks", "McDonald's", "Subway", "Chipotle", "Panera Bread",
    "Amazon", "Netflix", "Spotify", "Apple Store", "Google Play",
    "Target", "Walmart", "Costco", "Home Depot", "Best Buy",
    "Shell", "Exxon", "Chevron", "BP", "7-Eleven",
    "Uber", "Lyft", "DoorDash", "Grubhub", "Postmates",
    "Airbnb", "Booking.com", "Expedia", "Delta", "American Airlines",
    "Bank of America", "Chase", "Wells Fargo", "Citi", "Capital One",
    "PayPal", "Venmo", "Cash App", "Zelle", "Square"
]

SUSPICIOUS_MERCHANTS = [
    "CryptoExchangeX", "BitcoinMax", "CryptoKing", "DigitalGold",
    "OnlineCasino777", "LuckySlots", "PokerStars", "Bet365",
    "DarkWebMarket", "AnonymousShop", "CryptoMiningPool", "TorExchange",
    "OffshoreBank", "TaxHavenCorp", "ShellCompany", "MoneyLaundry",
    "FakeMerchant", "ScamSite", "PhishingStore", "MalwareShop",
    "StolenGoods", "CounterfeitCorp", "FakeID", "IdentityTheft",
    "DrugMarket", "WeaponShop", "IllegalServices", "BlackMarket"
]

# Geographic locations with risk levels
US_LOCATIONS = [
    "NY-US", "CA-US", "TX-US", "FL-US", "IL-US", "PA-US", "OH-US", "GA-US",
    "NC-US", "MI-US", "NJ-US", "VA-US", "WA-US", "AZ-US", "MA-US", "TN-US"
]

HIGH_RISK_LOCATIONS = [
    "RU-EU", "CN-AS", "NG-AF", "BR-SA", "MX-NA", "IN-AS", "PK-AS", "BD-AS",
    "VN-AS", "TH-AS", "ID-AS", "PH-AS", "MY-AS", "SG-AS", "HK-AS", "TW-AS",
    "IR-ME", "IQ-ME", "SY-ME", "LB-ME", "JO-ME", "SA-ME", "AE-ME", "KW-ME"
]

def generate_device_id():
    """Generate realistic device IDs"""
    prefixes = ["d_", "device_", "mobile_", "tablet_", "laptop_"]
    return random.choice(prefixes) + str(random.randint(1000, 9999))

def generate_transaction_id(index):
    """Generate transaction IDs"""
    return f"T{index:04d}"

def calculate_velocity_pattern(merchant_risk, geo_risk, amount):
    """Calculate realistic velocity based on patterns"""
    base_velocity = random.randint(0, 50)
    
    # Legitimate users tend to have higher velocity
    if merchant_risk == "LEGIT" and geo_risk == "US":
        base_velocity = max(base_velocity, random.randint(5, 30))
    
    # Fraud patterns often have low velocity
    if merchant_risk == "FRAUD":
        base_velocity = random.randint(0, 5)
    
    # Large amounts with low velocity are suspicious
    if amount > 1000 and base_velocity < 3:
        base_velocity = random.randint(0, 2)
    
    return base_velocity

def calculate_avg_amount(velocity, merchant_risk):
    """Calculate realistic average amount"""
    if merchant_risk == "LEGIT":
        return random.uniform(20, 200)
    else:
        return random.uniform(50, 500)

def determine_label(merchant_risk, geo_risk, velocity, amount, avg_amount):
    """Determine if transaction is fraud based on patterns"""
    fraud_score = 0
    
    # Merchant risk
    if merchant_risk == "FRAUD":
        fraud_score += 3
    elif merchant_risk == "SUSPICIOUS":
        fraud_score += 1
    
    # Geographic risk
    if geo_risk == "HIGH":
        fraud_score += 2
    
    # Velocity patterns
    if velocity == 0:  # New device
        fraud_score += 2
    elif velocity < 3 and amount > 500:  # Low velocity + large amount
        fraud_score += 2
    
    # Amount patterns
    if amount > 5 * avg_amount:  # Unusual amount
        fraud_score += 1
    if amount > 2000:  # Very large amount
        fraud_score += 1
    
    # Random fraud injection (5% of transactions are random fraud)
    if random.random() < 0.05:
        fraud_score += 2
    
    return "FRAUD" if fraud_score >= 3 else "LEGIT"

def generate_transaction(index):
    """Generate a single transaction"""
    
    # Determine merchant risk
    merchant_risk = random.choices(
        ["LEGIT", "SUSPICIOUS", "FRAUD"], 
        weights=[0.7, 0.2, 0.1]
    )[0]
    
    if merchant_risk == "LEGIT":
        merchant = random.choice(LEGIT_MERCHANTS)
        merchant_known = True
    elif merchant_risk == "SUSPICIOUS":
        merchant = random.choice(SUSPICIOUS_MERCHANTS[:10])  # Less suspicious ones
        merchant_known = random.choice([True, False])
    else:  # FRAUD
        merchant = random.choice(SUSPICIOUS_MERCHANTS)
        merchant_known = False
    
    # Determine geographic risk
    geo_risk = random.choices(
        ["US", "HIGH"], 
        weights=[0.8, 0.2]
    )[0]
    
    if geo_risk == "US":
        geo = random.choice(US_LOCATIONS)
    else:
        geo = random.choice(HIGH_RISK_LOCATIONS)
    
    # Generate amount with realistic distribution
    if merchant_risk == "LEGIT":
        amount = random.uniform(5, 500)
    elif merchant_risk == "SUSPICIOUS":
        amount = random.uniform(50, 2000)
    else:  # FRAUD
        amount = random.uniform(200, 10000)
    
    # Add some extreme outliers
    if random.random() < 0.02:  # 2% extreme amounts
        amount = random.uniform(5000, 50000)
    
    amount = round(amount, 2)
    
    # Calculate velocity and average amount
    velocity = calculate_velocity_pattern(merchant_risk, geo_risk, amount)
    avg_amount = calculate_avg_amount(velocity, merchant_risk)
    avg_amount = round(avg_amount, 2)
    
    # Determine label
    label = determine_label(merchant_risk, geo_risk, velocity, amount, avg_amount)
    
    return {
        "id": generate_transaction_id(index),
        "amount": amount,
        "merchant": merchant,
        "device_id": generate_device_id(),
        "geo": geo,
        "velocity_30d": velocity,
        "avg_amount_30d": avg_amount,
        "merchant_known": merchant_known,
        "label": label
    }

def main():
    """Generate the dataset"""
    print("🔄 Generating comprehensive fraud detection dataset...")
    
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
    fraud_count = sum(1 for t in transactions if t["label"] == "FRAUD")
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
        print(f"   {t['id']}: ${t['amount']:.2f} at {t['merchant']} ({t['geo']}) → {t['label']}")

if __name__ == "__main__":
    main()
