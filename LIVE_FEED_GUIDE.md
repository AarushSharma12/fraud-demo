# Live Data Feed Simulator Guide

The live data feed simulator generates realistic transactions in real-time and evaluates them using both the rule-based system and RL model (if available).

## Features

- **Realistic Transaction Generation**: Creates transactions with realistic patterns based on your existing data
- **Real-time Evaluation**: Each transaction is immediately evaluated by both rule-based and RL models
- **Configurable Rate**: Control how often transactions are generated (default: every 2 seconds)
- **Configurable Fraud Rate**: Set the percentage of fraudulent transactions (default: 10%)
- **Server-Sent Events (SSE)**: Stream transactions in real-time to the frontend
- **Statistics Tracking**: Track total generated, fraud detected, legit approved, and needs review

## API Endpoints

### Start Live Feed

```bash
POST /live-feed/start?interval_seconds=2.0&fraud_rate=0.1
```

**Parameters:**
- `interval_seconds` (float): Time between transactions in seconds (default: 2.0)
- `fraud_rate` (float): Probability of fraud (0.0 to 1.0, default: 0.1 = 10%)

**Example:**
```bash
curl -X POST "https://attu2.cs.washington.edu:8000/live-feed/start?interval_seconds=1.5&fraud_rate=0.15" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Stop Live Feed

```bash
POST /live-feed/stop
```

**Example:**
```bash
curl -X POST "https://attu2.cs.washington.edu:8000/live-feed/stop" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Get Status

```bash
GET /live-feed/status
```

Returns:
- `active`: Whether feed is running
- `stats`: Statistics (total_generated, fraud_detected, legit_approved, needs_review, start_time)
- `queue_size`: Number of transactions in queue
- `latest_transactions`: Last 10 transactions

**Example:**
```bash
curl "https://attu2.cs.washington.edu:8000/live-feed/status" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Stream Live Transactions (SSE)

```bash
GET /live-feed/stream
```

Streams transactions in real-time using Server-Sent Events. Each event contains a complete transaction with evaluation results.

**Example (JavaScript):**
```javascript
const eventSource = new EventSource(
  'https://attu2.cs.washington.edu:8000/live-feed/stream',
  { headers: { 'Authorization': `Bearer ${token}` } }
);

eventSource.onmessage = (event) => {
  const transaction = JSON.parse(event.data);
  console.log('New transaction:', transaction);
  // Update UI with new transaction
};

eventSource.onerror = (error) => {
  console.error('SSE error:', error);
  eventSource.close();
};
```

**Example (curl):**
```bash
curl -N "https://attu2.cs.washington.edu:8000/live-feed/stream" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Get Recent Transactions

```bash
GET /live-feed/recent?limit=50
```

**Parameters:**
- `limit` (int): Number of recent transactions to return (default: 50, max: 1000)

**Example:**
```bash
curl "https://attu2.cs.washington.edu:8000/live-feed/recent?limit=20" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Transaction Format

Each transaction in the feed includes:

```json
{
  "transaction": {
    "id": "L000001",
    "timestamp": "2024-01-15T12:34:56.789012",
    "from_account": "ACC123456",
    "to_account": "ACC789012",
    "amount": 1234.56,
    "transaction_type": "transfer",
    "category": "online",
    "location": "Dubai",
    "channel": "web",
    "is_fraud": true
  },
  "rule_based": {
    "decision": "FRAUD",
    "confidence": 0.85,
    "flags": ["large_amount", "geo_risk", "suspicious_category"],
    "explanation": "Multiple red flags detected: large_amount, geo_risk, suspicious_category"
  },
  "rl_model": {
    "decision": "FRAUD",
    "confidence": 0.92,
    "available": true
  },
  "true_label": "FRAUD",
  "timestamp": "2024-01-15T12:34:56.789012"
}
```

## Usage Examples

### Python Example

```python
import requests
import json
import time

API_BASE = "https://attu2.cs.washington.edu:8000"
TOKEN = "your_session_token"

headers = {"Authorization": f"Bearer {TOKEN}"}

# Start live feed (1 transaction per second, 15% fraud rate)
response = requests.post(
    f"{API_BASE}/live-feed/start",
    params={"interval_seconds": 1.0, "fraud_rate": 0.15},
    headers=headers
)
print(response.json())

# Monitor for 30 seconds
time.sleep(30)

# Get status
status = requests.get(f"{API_BASE}/live-feed/status", headers=headers)
print(json.dumps(status.json(), indent=2))

# Get recent transactions
recent = requests.get(f"{API_BASE}/live-feed/recent?limit=10", headers=headers)
print(json.dumps(recent.json(), indent=2))

# Stop feed
response = requests.post(f"{API_BASE}/live-feed/stop", headers=headers)
print(response.json())
```

### JavaScript/TypeScript Example (Frontend)

```javascript
const API = "https://attu2.cs.washington.edu:8000";
const token = sessionStorage.getItem('session_token');

// Start live feed
async function startLiveFeed(intervalSeconds = 2.0, fraudRate = 0.1) {
  const response = await fetch(
    `${API}/live-feed/start?interval_seconds=${intervalSeconds}&fraud_rate=${fraudRate}`,
    {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    }
  );
  return await response.json();
}

// Stream transactions
function streamLiveFeed(onTransaction) {
  const eventSource = new EventSource(
    `${API}/live-feed/stream`,
    { 
      headers: { 'Authorization': `Bearer ${token}` }
    }
  );

  eventSource.onmessage = (event) => {
    const transaction = JSON.parse(event.data);
    onTransaction(transaction);
  };

  eventSource.onerror = (error) => {
    console.error('SSE error:', error);
    eventSource.close();
  };

  return eventSource; // Return to allow closing later
}

// Usage
const eventSource = streamLiveFeed((transaction) => {
  console.log('New transaction:', transaction);
  // Update your UI here
  updateTransactionTable(transaction);
  updateMetrics(transaction);
});

// Stop after 60 seconds
setTimeout(() => {
  eventSource.close();
  fetch(`${API}/live-feed/stop`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` }
  });
}, 60000);
```

## Integration with Frontend

To integrate with your frontend:

1. **Add Live Feed Controls**: Add start/stop buttons and configuration inputs
2. **Display Live Transactions**: Create a table or list that updates in real-time
3. **Show Statistics**: Display running statistics (total, fraud detected, etc.)
4. **Visual Indicators**: Use colors/icons to indicate fraud vs legit transactions

## Notes

- The live feed stores the last 1000 transactions in memory
- Transactions are generated with realistic patterns based on your existing data
- Both rule-based and RL models evaluate each transaction (if RL model is available)
- The feed runs as a background task and doesn't block other API requests
- Statistics are reset when the feed is stopped and restarted

## Troubleshooting

**Feed won't start:**
- Check if already running: `GET /live-feed/status`
- Ensure you're authenticated (valid session token)
- Check server logs for errors

**No transactions appearing:**
- Verify feed is active: `GET /live-feed/status`
- Check `interval_seconds` - might be too long
- Check server logs for errors in transaction generation

**SSE connection issues:**
- Ensure your browser supports EventSource
- Check CORS settings if accessing from different domain
- Verify authentication token is valid


