#!/bin/bash

# Deployment script for UW CSE Servers
# This script deploys the frontend to your homes.cs.washington.edu space

echo "🚀 Deploying Frontend to UW Servers"
echo "====================================="

# Check if frontend exists
if [ ! -f "frontend/index.html" ]; then
    echo "❌ Error: frontend/index.html not found"
    exit 1
fi

# Ask for UW NetID if not provided as argument
if [ -z "$1" ]; then
    read -p "Enter your UW NetID: " NETID
else
    NETID=$1
fi

if [ -z "$NETID" ]; then
    echo "❌ Error: NetID is required"
    exit 1
fi

REMOTE_HOST="attu.cs.washington.edu"
REMOTE_DIR="~/public_html/fraud-demo"
PUBLIC_URL="https://homes.cs.washington.edu/~$NETID/fraud-demo/index.html"

echo "📡 Connecting to $REMOTE_HOST as $NETID..."

# 1. Create directory
echo "📂 Creating directory $REMOTE_DIR..."
ssh "$NETID@$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Failed to connect or create directory. Check your NetID and VPN/Network connection."
    exit 1
fi

# 2. Copy file
echo "📤 Uploading index.html..."
scp "frontend/index.html" "$NETID@$REMOTE_HOST:$REMOTE_DIR/index.html"

# 3. Set permissions (Critical for homes.cs web hosting)
echo "🔒 Setting permissions..."
ssh "$NETID@$REMOTE_HOST" "chmod 755 ~/public_html"
ssh "$NETID@$REMOTE_HOST" "chmod 755 $REMOTE_DIR"
ssh "$NETID@$REMOTE_HOST" "chmod 644 $REMOTE_DIR/index.html"

echo ""
echo "✅ Deployment Complete!"
echo "====================================="
echo "🌍 Your frontend is live at:"
echo "$PUBLIC_URL"
echo ""

