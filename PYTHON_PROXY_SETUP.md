# Python Reverse Proxy Setup (No sudo required)

This is a simpler alternative to nginx that doesn't require root access, but **it still uses your self-signed certificate**, so it won't solve the iPhone certificate issue completely.

## Installation

```bash
# Install required Python packages
pip install aiohttp aiohttp-cors

# Or if using the backend virtual environment
cd backend
source .venv311/bin/activate
pip install aiohttp aiohttp-cors
```

## Usage

```bash
# Start the backend in HTTP mode
export FORCE_HTTP=true
cd backend
python main.py &

# Start the Python proxy (in another terminal)
cd /cse/web/homes/micibr/fraud-demo
python python_proxy.py
```

The proxy will run on port 8443 (non-privileged port).

## Limitations

⚠️ **This still uses your self-signed certificate**, so:
- iPhones will still show certificate warnings
- Users still need to accept the certificate manually
- Not ideal for production

## Better Solution

For a **trusted certificate** that works on all devices, you need:
1. **nginx with Let's Encrypt** (requires sudo or admin help)
2. **Or request a proper certificate** from your university IT department

## Alternative: Contact IT Department

Since you're on a university server (`attu2.cs.washington.edu`), you might be able to:
1. Request nginx installation from IT
2. Request a proper SSL certificate
3. Use an existing reverse proxy if the university provides one



