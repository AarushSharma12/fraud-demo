# Live Feed - Quick Usage Guide

## Overview

The live transaction feed generates realistic transactions in real-time and automatically feeds them into your frontend interface. Both rule-based and RL models evaluate each transaction as it arrives.

## How to Use

### 1. Start the Backend

```bash
cd backend
python main.py
```

### 2. Open the Frontend

Navigate to: `https://homes.cs.washington.edu/~micibr/fraud-demo/frontend/index.html`

### 3. Login

Use one of the demo accounts:
- **Admin**: `admin-yubikey-123` / `admin2024!`
- **Analyst**: `analyst-yubikey-456` / `analyst2024!`
- **Viewer**: `viewer-yubikey-789` / `viewer2024!`

### 4. Start the Live Feed

1. Scroll to the **"📡 Live Transaction Feed"** section
2. Configure settings:
   - **Interval**: How often to generate transactions (default: 2 seconds)
   - **Fraud Rate**: Percentage of fraudulent transactions (default: 10%)
3. Click **"▶ Start Live Feed"**

### 5. Watch Transactions Appear

- Transactions will automatically appear in the **"Transaction Results"** table below
- Statistics update in real-time (Total, Fraud Detected, Legit, Needs Review)
- Most recent transactions appear at the top

### 6. Stop the Feed

Click **"⏹ Stop Feed"** when you want to stop generating transactions

## What Happens

1. **Generation**: Backend generates realistic transactions based on your existing data patterns
2. **Evaluation**: Each transaction is evaluated by:
   - Rule-based system
   - RL model (if trained and available)
3. **Display**: Transactions appear in the frontend table with:
   - Transaction ID (starts with "L" for live transactions)
   - Decision (FRAUD, LEGIT, or NEEDS_REVIEW)
   - Confidence score
   - Flags detected
   - True label (whether it's actually fraud)

## Tips

- **Start slow**: Begin with a 2-3 second interval to see how it works
- **Adjust fraud rate**: Change the fraud rate to test different scenarios
  - 5-10% = Normal scenario
  - 20-30% = High fraud scenario
  - 50% = Testing scenario
- **Watch the stats**: The statistics box shows real-time counts
- **Filter results**: Use the search and filter options to analyze specific transactions
- **Export data**: Use the "📥 Export Statistics" button to download results

## Example Scenarios

### Scenario 1: Normal Operations
- **Interval**: 2 seconds
- **Fraud Rate**: 10%
- **Duration**: 2-3 minutes
- **Result**: See how the system handles typical transaction flow

### Scenario 2: High Fraud Attack
- **Interval**: 1 second
- **Fraud Rate**: 30%
- **Duration**: 1-2 minutes
- **Result**: Test system performance under attack

### Scenario 3: Rapid Processing
- **Interval**: 0.5 seconds
- **Fraud Rate**: 15%
- **Duration**: 30 seconds
- **Result**: Stress test the system

## Troubleshooting

**Live feed won't start:**
- Ensure you're logged in
- Check that the backend is running
- Verify your session hasn't expired

**No transactions appearing:**
- Check the "Live Feed Status" - should show "Active"
- Verify the interval isn't too long (try reducing to 1 second)
- Look at browser console for errors (F12 → Console)

**Transactions appearing slowly:**
- The frontend polls every 2 seconds for new transactions
- This is normal - transactions batch together
- For real-time streaming, see `LIVE_FEED_GUIDE.md` for SSE implementation

## Technical Details

- Live transactions have IDs starting with "L" (e.g., L000001, L000002)
- Backend stores last 1000 transactions in memory
- Frontend polls backend every 2 seconds for new transactions
- Statistics update every 2 seconds while feed is active
- Transactions are prepended to the table (newest first)

## Next Steps

- Train an RL model to compare rule-based vs RL performance
- Adjust the fraud detection rules to reduce false positives
- Export statistics to analyze model performance over time
- Use the live feed to demonstrate the system to stakeholders

