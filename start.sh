#!/bin/bash

echo "🚀 Starting Fraud Detection Co-Pilot"
echo "===================================="

# Check directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Run from fraud-demo directory"
    exit 1
fi

# Kill existing processes
pkill -f "uvicorn main:app" 2>/dev/null
pkill -f "python -m http.server" 2>/dev/null
sleep 1

# Start backend
echo "📡 Starting backend (port 8000)..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate and install dependencies
source .venv/bin/activate
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Start backend server
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend
echo "⏳ Waiting for backend..."
for i in {1..10}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "✅ Backend running"
        break
    fi
    sleep 1
done

# Start frontend
echo "🌐 Starting frontend (port 8080)..."
cd frontend
python3 -m http.server 8080 &
FRONTEND_PID=$!
cd ..
sleep 2

echo ""
echo "🎉 Demo Ready!"
echo "===================================="
echo "🌐 Open: http://localhost:8080"
echo ""
echo "📊 Demo Flow:"
echo "  1. Click 'Test Connection' → Verify backend"
echo "  2. Click 'Run All 50 Transactions' → See metrics"
echo "  3. Try single transaction: T002 (fraud) or T001 (legit)"
echo ""
echo "🛑 Press Ctrl+C to stop"

# Trap Ctrl+C to cleanup
trap "echo ''; echo '🛑 Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

# Open browser (platform-specific)
if command -v open >/dev/null; then
    open http://localhost:8080
elif command -v xdg-open >/dev/null; then
    xdg-open http://localhost:8080
elif command -v start >/dev/null; then
    start http://localhost:8080
fi

# Keep script running
wait