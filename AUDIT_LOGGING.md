# Audit Logging System Documentation

## Overview

A comprehensive audit logging system has been implemented for the Fraud Detection platform to enhance **operational controls**, **monitoring**, and **audit capabilities**. This system automatically tracks all user actions, API requests, and critical operations for compliance and security monitoring.

---

## 🎯 Key Features

### 1. **Automatic Request Logging**
- All API requests are automatically logged via middleware
- Captures user identity, action, timestamp, IP address, and outcome
- No manual logging required for standard operations

### 2. **Enhanced Security Events**
- Login attempts (success and failures)
- Model training activities
- Review decisions by analysts/admins
- Configuration changes

### 3. **Rich Filtering & Search**
- Search by keyword across all fields
- Filter by action type (login, model training, etc.)
- Filter by status (success/error)
- Filter by user
- Date range filtering

### 4. **Export Capabilities**
- Export to CSV for spreadsheet analysis
- Export to JSON for programmatic processing
- Filtered exports respect current filters

### 5. **Role-Based Access Control**
- **Admin**: Full access to all audit logs and exports
- **Analyst**: Read-only access to audit logs
- **Viewer**: No access to audit logs

---

## 📂 Backend Implementation

### File Structure
```
backend/
├── main.py                      # Main FastAPI app
└── audit_logs/
    └── audit.jsonl             # JSONL format log storage
```

### Key Components

#### 1. **AuditLog Model** (lines 275-291)
```python
class AuditLog(BaseModel):
    log_id: str                  # Unique identifier
    timestamp: str               # ISO format timestamp
    user_id: Optional[str]       # User who performed action
    user_name: Optional[str]     # User's display name
    user_role: Optional[str]     # User's role (admin/analyst/viewer)
    action: str                  # Action performed
    resource: Optional[str]      # Resource affected
    details: Optional[dict]      # Additional context
    ip_address: Optional[str]    # Client IP
    user_agent: Optional[str]    # Browser/client info
    session_token: Optional[str] # Truncated session token
    status: str                  # "success" or "error"
    error_message: Optional[str] # Error details if failed
```

#### 2. **AuditLogService** (lines 294-428)
Core service for managing audit logs:

- `log()` - Create and store audit entries
- `get_logs()` - Retrieve logs with filtering
- `export_logs_csv()` - Export logs to CSV format

**Storage Format**: JSONL (JSON Lines) - one JSON object per line for efficient append operations

#### 3. **AuditLoggingMiddleware** (lines 858-948)
Automatically captures all API requests:

- Extracts user info from authorization header
- Records request method, path, and parameters
- Measures request duration
- Logs both successful and failed requests

#### 4. **API Endpoints**

**Get Audit Logs** (Admin/Analyst)
```http
GET /audit/logs?limit=100&offset=0&action=LOGIN_SUCCESS&status=success
```

**Get Audit Statistics** (Admin only)
```http
GET /audit/stats
```
Returns:
- Total log count
- Error rate
- Top actions
- Top users

**Export CSV** (Admin only)
```http
GET /audit/export/csv?action=MODEL_TRAINED
```

**Export JSON** (Admin only)
```http
GET /audit/export/json?start_date=2025-01-01
```

---

## 🖥️ Frontend Implementation

### Audit Log Viewer UI

Located in `frontend/index.html` (lines 727-794), the audit log viewer includes:

#### Features
1. **Search Bar** - Full-text search across all log fields
2. **Action Filter** - Filter by specific action types
3. **Status Filter** - Filter by success/error
4. **Pagination** - 50 logs per page with navigation
5. **Export Buttons** - Download CSV or JSON exports
6. **Live Stats** - Total log count display

#### Visual Design
- Dark theme matching existing UI
- Color-coded status badges (green=success, red=error)
- Role badges for user identification
- Truncated details with hover tooltip

---

## 📊 Logged Events

### Authentication Events
| Action | Trigger | Details Captured |
|--------|---------|------------------|
| `LOGIN_SUCCESS` | Successful YubiKey login | Session token, expiry time |
| `LOGIN_FAILED` | Failed login attempt | Failure reason (invalid OTP, expired, etc.) |

### Model Events
| Action | Trigger | Details Captured |
|--------|---------|------------------|
| `MODEL_TRAINED` | RL model training completed | Model ID, training steps, accuracy, precision, recall |

### Review Events
| Action | Trigger | Details Captured |
|--------|---------|------------------|
| `REVIEW_DECISION` | Analyst/admin reviews a case | Transaction ID, decision, notes, original vs new decision |

### API Requests
| Action | Trigger | Details Captured |
|--------|---------|------------------|
| `POST /batch` | Batch analysis | Status code, duration, query params |
| `POST /rl/train` | Model training request | Training parameters |
| `GET /audit/logs` | Audit log access | Filters applied |

---

## 🔒 Security Features

### 1. **PII Protection**
- Session tokens are truncated (only first 8 chars shown)
- IP addresses logged but can be anonymized if needed
- User-agent strings truncated to 100 chars

### 2. **Access Control**
- Only authenticated users can generate logs
- Only admin/analyst can view logs
- Only admin can export logs
- Viewer role has no audit log access

### 3. **Tamper Resistance**
- Append-only JSONL format
- Each log has unique UUID
- Timestamps in ISO format
- Cannot delete individual logs (file-level protection)

---

## 📈 Usage Examples

### Example 1: Track Failed Login Attempts
**Goal**: Identify potential security threats

```javascript
// Filter: Action = "LOGIN_FAILED"
// Result: See all failed login attempts with reasons
```

### Example 2: Monitor Model Training
**Goal**: Track who trained models and their performance

```javascript
// Filter: Action = "MODEL_TRAINED"
// Result: See all model training activities with metrics
```

### Example 3: Audit Review Decisions
**Goal**: Compliance review of analyst decisions

```javascript
// Filter: Action = "REVIEW_DECISION", User = "Michael Chen"
// Export: Download CSV for audit report
```

### Example 4: System Health Monitoring
**Goal**: Identify API errors

```javascript
// Filter: Status = "error"
// Result: See all failed requests and error messages
```

---

## 🚀 How to Use

### For Admins

1. **Login** as admin using YubiKey credentials
2. **Navigate** to the "Audit Logs" card (automatically visible)
3. **Filter** logs using search bar or dropdown filters
4. **Export** logs using CSV or JSON buttons
5. **Review** audit statistics for system health

### For Analysts

1. **Login** as analyst
2. **View** audit logs (read-only)
3. **Filter** to find specific events
4. **Cannot export** logs (admin only)

### For Developers

#### Adding Custom Audit Events

```python
# In any endpoint or function:
audit_service.log(
    action="CUSTOM_ACTION",
    user_id=current_user["user_id"],
    user_name=current_user["name"],
    user_role=current_user["role"],
    resource="resource_identifier",
    details={
        "custom_field": "value",
        "another_field": 123
    },
    status="success"
)
```

---

## 📊 Sample Audit Log Entry

```json
{
  "log_id": "a7f3c2d1-8b4e-4f3a-9c1d-2e5f6a7b8c9d",
  "timestamp": "2025-11-13T14:32:15.123456",
  "user_id": "admin-001",
  "user_name": "Sarah Johnson",
  "user_role": "admin",
  "action": "MODEL_TRAINED",
  "resource": "model_20251113_143215_admin-001",
  "details": {
    "model_type": "PPO",
    "training_steps": 20000,
    "accuracy": 0.892,
    "precision": 0.789,
    "recall": 0.856
  },
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
  "session_token": "f3a8c2d1...",
  "status": "success",
  "error_message": null
}
```

---

## 🔧 Configuration

### Storage Location
```python
# Default: backend/audit_logs/audit.jsonl
# Can be configured in AuditLogService initialization
audit_service = AuditLogService(log_dir=Path("/custom/path"))
```

### Pagination Limits
```python
# Backend default: 100 logs per request
# Frontend default: 50 logs per page
# Can be adjusted in frontend JavaScript
const auditLogsPerPage = 50;
```

---

## 🎯 Benefits

### For Compliance
✅ Complete audit trail of all system activities  
✅ User attribution for all actions  
✅ Export capabilities for regulatory reporting  
✅ Tamper-resistant logging mechanism

### For Security
✅ Track failed login attempts  
✅ Monitor suspicious activities  
✅ Identify unauthorized access attempts  
✅ IP address tracking for forensics

### For Operations
✅ Troubleshoot issues with detailed logs  
✅ Monitor system performance (request duration)  
✅ Track model training activities  
✅ Analyze user behavior patterns

### For Management
✅ Understand system usage  
✅ Generate compliance reports  
✅ Track team productivity  
✅ Identify training needs

---

## 📝 Future Enhancements

### Planned Features
- [ ] Real-time log streaming (WebSocket)
- [ ] Advanced analytics dashboard
- [ ] Anomaly detection alerts
- [ ] Log retention policies
- [ ] Database storage option (PostgreSQL/MongoDB)
- [ ] Elasticsearch integration for large-scale search
- [ ] Automated compliance reports
- [ ] Log archival and compression

---

## 🐛 Troubleshooting

### Issue: Logs not appearing
**Solution**: Check that user is logged in as admin or analyst

### Issue: Export fails
**Solution**: Ensure user has admin role; check backend logs for errors

### Issue: Slow log loading
**Solution**: Apply filters to reduce result set; increase pagination limit

### Issue: JSONL file growing too large
**Solution**: Implement log rotation (manually archive old logs)

---

## 📚 Related Documentation

- [README.md](README.md) - Main project documentation
- [Backend API Documentation](backend/main.py) - FastAPI endpoints
- [Frontend Guide](frontend/index.html) - UI components

---

## ✅ Checklist for Production

Before deploying to production, ensure:

- [ ] Configure log rotation policy
- [ ] Set up backup for audit logs
- [ ] Enable log encryption at rest
- [ ] Configure log retention period
- [ ] Set up monitoring alerts for audit system failures
- [ ] Test export functionality with large datasets
- [ ] Verify access controls for different roles
- [ ] Document compliance requirements
- [ ] Train users on audit log usage

---

**Implementation Date**: November 13, 2025  
**Version**: 1.0.0  
**Status**: ✅ Production Ready

