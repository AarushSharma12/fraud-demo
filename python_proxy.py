#!/usr/bin/env python3
"""
Simple Python Reverse Proxy with SSL
This runs on port 8443 (non-privileged) and proxies to the backend on port 8000
Requires: pip install aiohttp aiohttp_cors ssl
"""

import asyncio
import ssl
from aiohttp import web, ClientSession, TCPConnector
from aiohttp_cors import setup as cors_setup, ResourceOptions
import sys

# Configuration
BACKEND_URL = "http://127.0.0.1:8000"
PROXY_PORT = 8443  # Non-privileged port (use 443 if you have root)
SSL_CERT = "/homes/iws/micibr/ssl/attu2.cs.washington.edu.crt"
SSL_KEY = "/homes/iws/micibr/ssl/attu2.cs.washington.edu.key"

async def proxy_handler(request):
    """Proxy requests to backend"""
    # Get the path and query string
    path = request.path_qs
    
    # Create backend URL
    backend_url = f"{BACKEND_URL}{path}"
    
    # Get request body if present
    body = await request.read() if request.can_read_body else None
    
    # Prepare headers (exclude host and connection)
    headers = dict(request.headers)
    headers.pop('Host', None)
    headers.pop('Connection', None)
    
    # Create client session
    async with ClientSession() as session:
        try:
            # Forward the request
            async with session.request(
                method=request.method,
                url=backend_url,
                headers=headers,
                data=body,
                allow_redirects=False
            ) as resp:
                # Get response body
                response_body = await resp.read()
                
                # Create response with same status and headers
                response = web.Response(
                    body=response_body,
                    status=resp.status,
                    headers=dict(resp.headers)
                )
                return response
        except Exception as e:
            print(f"Proxy error: {e}")
            return web.Response(
                text=f"Proxy error: {str(e)}",
                status=502
            )

def create_app():
    """Create the aiohttp application"""
    app = web.Application()
    
    # Setup CORS
    cors = cors_setup(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Add route for all paths
    app.router.add_route('*', '/{path:.*}', proxy_handler)
    
    return app

async def init_app(app):
    """Initialize the app"""
    pass

def main():
    """Main function"""
    print(f"🚀 Starting Python Reverse Proxy")
    print(f"   Backend: {BACKEND_URL}")
    print(f"   Proxy Port: {PROXY_PORT}")
    print(f"   SSL Cert: {SSL_CERT}")
    print(f"   SSL Key: {SSL_KEY}")
    
    # Check if SSL files exist
    import os
    if not os.path.exists(SSL_CERT):
        print(f"⚠️  SSL certificate not found: {SSL_CERT}")
        print("   Running without SSL (HTTP only)")
        ssl_context = None
    elif not os.path.exists(SSL_KEY):
        print(f"⚠️  SSL key not found: {SSL_KEY}")
        print("   Running without SSL (HTTP only)")
        ssl_context = None
    else:
        # Create SSL context
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(SSL_CERT, SSL_KEY)
        print("✅ SSL configured")
    
    # Create app
    app = create_app()
    app.on_startup.append(init_app)
    
    # Run the server
    print(f"\n📡 Proxy running on port {PROXY_PORT}")
    print(f"   Access at: https://attu2.cs.washington.edu:{PROXY_PORT}/")
    print(f"   (Note: Still uses self-signed cert - same issue as before)")
    print("\n⚠️  For trusted certificates, you still need nginx with Let's Encrypt")
    print("   This proxy just moves the SSL handling to Python\n")
    
    web.run_app(app, port=PROXY_PORT, ssl_context=ssl_context)

if __name__ == "__main__":
    main()

