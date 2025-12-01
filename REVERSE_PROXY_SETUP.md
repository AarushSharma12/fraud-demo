# Reverse Proxy Setup Guide

This guide explains how to set up nginx as a reverse proxy to handle HTTPS with a trusted certificate, while the backend runs on HTTP.

## Why Use a Reverse Proxy?

- **Trusted SSL Certificate**: Use Let's Encrypt or a properly signed certificate
- **Better Security**: Centralized SSL/TLS termination
- **Performance**: nginx handles static files efficiently
- **Mobile Device Compatibility**: iPhones and other devices will trust the certificate

## Prerequisites

1. **nginx installed** (check with `nginx -v`)
2. **Root/sudo access** to configure nginx
3. **Domain name** pointing to your server (attu2.cs.washington.edu)

## Installation Steps

### 1. Install nginx (if not installed)

```bash
# On Rocky Linux / RHEL / CentOS
sudo yum install nginx
# or (newer versions)
sudo dnf install nginx

# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install nginx
```

### 2. Get SSL Certificate (Let's Encrypt - Recommended)

```bash
# Install certbot
# On Rocky Linux / RHEL / CentOS
sudo yum install certbot python3-certbot-nginx
# or
sudo dnf install certbot python3-certbot-nginx

# On Ubuntu/Debian
sudo apt-get install certbot python3-certbot-nginx

# Get certificate (nginx will auto-configure)
sudo certbot --nginx -d attu2.cs.washington.edu

# Or get certificate only (manual config)
sudo certbot certonly --standalone -d attu2.cs.washington.edu
```

### 3. Configure nginx

```bash
# Copy the configuration file
sudo cp nginx.conf /etc/nginx/sites-available/fraud-demo
# or
sudo cp nginx.conf /etc/nginx/conf.d/fraud-demo.conf

# Enable the site (if using sites-available)
sudo ln -s /etc/nginx/sites-available/fraud-demo /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# If using Let's Encrypt, update the SSL paths in nginx.conf:
# ssl_certificate /etc/letsencrypt/live/attu2.cs.washington.edu/fullchain.pem;
# ssl_certificate_key /etc/letsencrypt/live/attu2.cs.washington.edu/privkey.pem;

# Reload nginx
sudo systemctl reload nginx
# or
sudo nginx -s reload
```

### 4. Configure Backend to Use HTTP

The backend should run on HTTP (port 8000) when using a reverse proxy:

```bash
# Set environment variable
export FORCE_HTTP=true

# Or update start.sh to set FORCE_HTTP=true
```

### 5. Update Frontend API URL (if needed)

If you're using a different port or domain for the reverse proxy, update `frontend/index.html`:

```javascript
const API = "https://attu2.cs.washington.edu";  // Remove :8000, use default HTTPS port 443
```

### 6. Firewall Configuration

Ensure port 443 (HTTPS) is open:

```bash
# Check firewall
sudo firewall-cmd --list-all

# Open port 443
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
```

## Testing

1. **Start the backend** (with FORCE_HTTP=true):
   ```bash
   cd backend
   export FORCE_HTTP=true
   python main.py
   ```

2. **Test the reverse proxy**:
   ```bash
   curl https://attu2.cs.washington.edu/
   ```

3. **Check nginx logs**:
   ```bash
   sudo tail -f /var/log/nginx/fraud-demo-access.log
   sudo tail -f /var/log/nginx/fraud-demo-error.log
   ```

## Troubleshooting

### nginx won't start
- Check configuration: `sudo nginx -t`
- Check if port 443 is already in use: `sudo netstat -tlnp | grep 443`
- Check nginx error log: `sudo tail /var/log/nginx/error.log`

### Certificate errors
- Verify certificate paths in nginx.conf
- Check certificate expiration: `sudo certbot certificates`
- Renew certificate: `sudo certbot renew`

### Backend not reachable
- Verify backend is running: `curl http://localhost:8000/`
- Check nginx proxy_pass URL matches backend
- Check backend logs for errors

### Permission issues
- Ensure nginx can read certificate files: `sudo chmod 644 /path/to/cert.crt`
- Ensure nginx can read certificate key: `sudo chmod 600 /path/to/cert.key`

## Alternative: User-Space nginx (No Root Access)

If you don't have root access, you can compile and run nginx in user space:

```bash
# Download nginx source
wget http://nginx.org/download/nginx-1.24.0.tar.gz
tar -xzf nginx-1.24.0.tar.gz
cd nginx-1.24.0

# Configure for user-space installation
./configure --prefix=$HOME/nginx --with-http_ssl_module

# Compile and install
make
make install

# Run nginx (on port 8443 or other non-privileged port)
$HOME/nginx/sbin/nginx -c $HOME/fraud-demo/nginx-user.conf
```

Then update nginx-user.conf to use a non-privileged port (8443 instead of 443).

## Maintenance

### Renew Let's Encrypt Certificate
```bash
# Test renewal
sudo certbot renew --dry-run

# Manual renewal
sudo certbot renew
sudo systemctl reload nginx
```

### Update Configuration
After changing nginx.conf:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

