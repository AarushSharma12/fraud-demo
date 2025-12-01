#!/bin/bash

# Start script for use with nginx reverse proxy
# This runs the backend in HTTP mode (port 8000) while nginx handles HTTPS (port 443)

echo "🚀 Starting Fraud Detection Demo with Reverse Proxy"
echo "===================================================="
echo "⚠️  Make sure nginx is configured and running!"
echo ""

# Set reverse proxy mode
export USE_REVERSE_PROXY=true
export FORCE_HTTP=true

# Run the main start script
./start.sh



