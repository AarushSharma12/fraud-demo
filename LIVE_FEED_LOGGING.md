# Live Feed Logging & Continuous Learning

## Overview

The live feed now includes comprehensive logging and continuous learning capabilities that allow you to:
- Track all transactions processed during the live feed
- Monitor performance metrics over time
- Automatically retrain the RL model to improve performance
- View and download detailed logs

## Features

### 1. **Comprehensive Transaction Logging**

Every transaction processed during the live feed is logged with:
- Transaction ID and timestamp
- True label (FRAUD/LEGIT)
- Rule-based decision and confidence
- RL model decision and confidence (if available)
- Transaction count

**Log Location**: `backend/live_feed_logs/live_feed_YYYYMMDD_HHMMSS.jsonl`

### 2. **Performance Metrics Tracking**

Performance metrics are logged every 100 transactions, tracking:
- **Accuracy**: Overall correctness
- **Precision**: Fraud detection accuracy (true positives / (true positives + false positives))
- **Recall**: Fraud detection coverage (true positives / (true positives + false negatives))
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: TP, FP, TN, FN counts

**Metrics Location**: `backend/live_feed_logs/metrics_YYYYMMDD_HHMMSS.jsonl`

### 3. **Continuous Learning (Automatic Model Improvement)**

The system automatically retrains the RL model to improve performance:

- **Retraining Triggers**:
  - Every 1000 transactions processed
  - Every 1 hour of runtime
  - Whichever comes first

- **Incremental Learning**:
  - The model continues training from its current state (not from scratch)
  - Uses 5000 additional training steps per retraining cycle
  - Performance metrics are tracked before and after each retraining

- **Improvement Tracking**:
  - Metrics are captured before retraining
  - After retraining, new predictions are collected
  - Improvement deltas are calculated and logged
  - Shows accuracy, precision, recall, and F1 score improvements

### 4. **API Endpoints**

#### Get Performance Metrics
```bash
GET /live-feed/logs/metrics?limit=100
```

Returns:
- Performance history (metrics over time)
- Retraining history (all retraining events with improvements)
- Current metrics (latest performance)
- Session ID

#### Get Logged Transactions
```bash
GET /live-feed/logs/transactions?limit=1000&offset=0
```

Returns paginated list of logged transactions (most recent first).

#### Download Logs
```bash
GET /live-feed/logs/download?log_type=all
```

Download options:
- `log_type=transactions` - Download transaction log only
- `log_type=metrics` - Download metrics log only
- `log_type=all` - Download both as a ZIP file

#### Enhanced Status Endpoint
```bash
GET /live-feed/status
```

Now includes:
- Performance metrics (current and historical)
- Session ID
- Log file names
- Retraining count

## Usage for 48-Hour Run

### Prerequisites

1. **Train an initial RL model** before starting the live feed:
   ```bash
   POST /rl/train?timesteps=20000
   ```
   This ensures continuous learning can work (it needs an existing model).

2. **Start the live feed**:
   ```bash
   POST /live-feed/start?interval_seconds=0.33
   ```
   This will start logging automatically.

### During the Run

- **Logs are automatically created** in `backend/live_feed_logs/`
- **Model retrains automatically** every 1000 transactions or every hour
- **Performance metrics are tracked** and logged every 100 transactions
- **Check status** periodically:
  ```bash
  GET /live-feed/status
  ```

### Monitoring Performance Improvement

The system tracks performance over time. You can see improvement by:

1. **Viewing metrics history**:
   ```bash
   GET /live-feed/logs/metrics
   ```
   Look at the `performance_history` array - metrics should improve over time.

2. **Viewing retraining history**:
   ```bash
   GET /live-feed/logs/metrics
   ```
   Check the `retraining_history` array - each entry shows:
   - Metrics before retraining
   - Metrics after retraining (updated after collecting new predictions)
   - Improvement deltas (accuracy_delta, precision_delta, etc.)

3. **Downloading logs** for analysis:
   ```bash
   GET /live-feed/logs/download?log_type=all
   ```

### Expected Behavior

Over a 48-hour run with `interval_seconds=0.33` (3 transactions/second):
- **Total transactions**: ~518,400 transactions
- **Retraining events**: ~518 retraining cycles (every 1000 transactions)
- **Performance snapshots**: ~5,184 metrics snapshots (every 100 transactions)
- **Model improvement**: The RL model should show gradual improvement in accuracy, precision, and recall as it learns from more data

### Log File Format

#### Transaction Log (JSONL)
Each line is a JSON object:
```json
{
  "transaction_id": "T001",
  "timestamp": "2025-01-15T10:30:45.123456",
  "true_label": "FRAUD",
  "rule_based_decision": "FRAUD",
  "rl_decision": "FRAUD",
  "rl_available": true,
  "rl_confidence": 0.95,
  "rule_confidence": 0.85,
  "transaction_count": 1
}
```

#### Metrics Log (JSONL)
Each line is either a metrics snapshot or a retraining event:

**Metrics Snapshot**:
```json
{
  "timestamp": "2025-01-15T10:35:00.123456",
  "transaction_count": 100,
  "rule_based": {
    "accuracy": 0.8500,
    "precision": 0.7500,
    "recall": 0.8000,
    "f1_score": 0.7746,
    "true_positives": 60,
    "false_positives": 20,
    "true_negatives": 25,
    "false_negatives": 15,
    "total": 100
  },
  "rl_model": {
    "accuracy": 0.9000,
    "precision": 0.8200,
    "recall": 0.8500,
    "f1_score": 0.8348,
    "true_positives": 68,
    "false_positives": 15,
    "true_negatives": 22,
    "false_negatives": 12,
    "total": 100
  }
}
```

**Retraining Event**:
```json
{
  "type": "retraining",
  "timestamp": "2025-01-15T11:00:00.123456",
  "model_id": "model_20250115_110000_live_feed_auto",
  "transaction_count": 1000,
  "transactions_since_last_retrain": 1000,
  "metrics_before": {
    "accuracy": 0.8500,
    "precision": 0.7500,
    "recall": 0.8000,
    "f1_score": 0.7746
  },
  "metrics_after": {
    "accuracy": 0.8800,
    "precision": 0.7800,
    "recall": 0.8200,
    "f1_score": 0.7995
  },
  "improvement": {
    "accuracy_delta": 0.0300,
    "precision_delta": 0.0300,
    "recall_delta": 0.0200,
    "f1_delta": 0.0249
  }
}
```

## Troubleshooting

### Model not improving?
- Ensure an RL model was trained before starting the live feed
- Check that `rl_model.available` is `true` in the logs
- Verify retraining is happening (check retraining_history)

### Logs not being created?
- Check that `backend/live_feed_logs/` directory exists and is writable
- Verify the live feed is actually running (check `/live-feed/status`)

### Performance metrics not updating?
- Metrics are logged every 100 transactions
- RL model metrics require at least some RL predictions to be available
- Check that the RL model is loaded and working

## Notes

- Logs are stored in JSONL format (one JSON object per line) for easy parsing
- Each live feed session gets a unique session ID based on start timestamp
- The system automatically handles log file rotation (new session = new log files)
- Continuous learning uses incremental training (continues from existing model) for efficiency
- Performance improvements may take several retraining cycles to become noticeable

