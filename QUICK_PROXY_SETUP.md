# Quick Reverse Proxy Setup

## Quick Start (5 minutes)

### 1. Install nginx (if needed)
```bash
# On Rocky Linux / RHEL / CentOS (using yum or dnf)
sudo yum install nginx certbot python3-certbot-nginx
# or
sudo dnf install nginx certbot python3-certbot-nginx

# On Ubuntu/Debian
sudo apt-get install nginx certbot python3-certbot-nginx
```

### 2. Get Let's Encrypt Certificate
```bash
sudo certbot --nginx -d attu2.cs.washington.edu
```
This will automatically configure nginx with a trusted certificate.

### 3. Update nginx Configuration
```bash
# Copy our config
sudo cp nginx.conf /etc/nginx/sites-available/fraud-demo

# If certbot created a config, edit it instead:
sudo nano /etc/nginx/sites-available/default
# or
sudo nano /etc/nginx/conf.d/default.conf

# Add these lines inside the server block:
#   location / {
#       proxy_pass http://127.0.0.1:8000;
#       proxy_set_header Host $host;
#       proxy_set_header X-Real-IP $remote_addr;
#       proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#       proxy_set_header X-Forwarded-Proto $scheme;
#   }
```

### 4. Test and Reload nginx
```bash
sudo nginx -t
sudo systemctl reload nginx
```

### 5. Start Backend with HTTP Mode
```bash
# Option 1: Use the helper script
./start-with-proxy.sh

# Option 2: Manual
export FORCE_HTTP=true
cd backend
python main.py
```

### 6. Update Frontend (if using port 443)
If nginx is on port 443 (default HTTPS), update `frontend/index.html`:
```javascript
const API = "https://attu2.cs.washington.edu";  // Remove :8000
```

### 7. Test
```bash
curl https://attu2.cs.washington.edu/
```

## Troubleshooting

**nginx won't start?**
- Check config: `sudo nginx -t`
- Check port 443: `sudo netstat -tlnp | grep 443`

**Backend not reachable?**
- Check backend: `curl http://localhost:8000/`
- Check nginx logs: `sudo tail /var/log/nginx/error.log`

**Certificate issues?**
- Renew: `sudo certbot renew`
- Check: `sudo certbot certificates`

