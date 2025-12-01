# Installing nginx on Rocky Linux

## Option 1: Using yum/dnf (Requires sudo)

```bash
# Install nginx
sudo yum install nginx
# or for newer Rocky Linux versions
sudo dnf install nginx

# Install certbot for Let's Encrypt certificates
sudo yum install certbot python3-certbot-nginx
# or
sudo dnf install certbot python3-certbot-nginx

# Start and enable nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Check status
sudo systemctl status nginx
```

## Option 2: Compile nginx from source (No sudo required)

If you don't have sudo access, you can compile and run nginx in user space:

```bash
# Install dependencies (if you have access to development tools)
# Or download pre-compiled binaries

# Download nginx source
cd ~
wget http://nginx.org/download/nginx-1.24.0.tar.gz
tar -xzf nginx-1.24.0.tar.gz
cd nginx-1.24.0

# Configure for user-space installation
./configure \
    --prefix=$HOME/nginx \
    --with-http_ssl_module \
    --with-http_realip_module \
    --with-http_stub_status_module \
    --with-http_gzip_static_module \
    --without-http_rewrite_module

# Compile (requires make and gcc)
make
make install

# Create nginx config directory
mkdir -p $HOME/nginx/conf/conf.d

# Copy configuration
cp /cse/web/homes/micibr/fraud-demo/nginx.conf $HOME/nginx/conf/nginx.conf

# Edit config to use non-privileged port (8443 instead of 443)
sed -i 's/listen 443/listen 8443/g' $HOME/nginx/conf/nginx.conf

# Start nginx
$HOME/nginx/sbin/nginx -c $HOME/nginx/conf/nginx.conf

# Test
curl http://localhost:8443/
```

## Option 3: Use Python-based reverse proxy (Simpler, no compilation)

If you can't install nginx, you can use a Python-based reverse proxy. However, this is less ideal for production:

```bash
# Install required Python packages
pip install gunicorn uvicorn[standard] httpx

# Use uvicorn with SSL (but still need certificates)
# This doesn't solve the certificate trust issue
```

## Option 4: Request nginx installation from admin

If you're on a university server, you might be able to request nginx installation from the system administrator.

## After Installation

Once nginx is installed, follow the main setup guide:
1. Get SSL certificate: `sudo certbot --nginx -d attu2.cs.washington.edu`
2. Configure nginx (use the provided `nginx.conf`)
3. Start backend with `./start-with-proxy.sh`

## Testing Installation

```bash
# Check if nginx is installed
nginx -v

# Check if nginx is running
systemctl status nginx
# or
ps aux | grep nginx

# Test nginx
curl http://localhost/
```



