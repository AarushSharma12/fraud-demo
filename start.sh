#!/bin/bash

echo "🚀 Starting Fraud Detection Co-Pilot with RL"
echo "============================================="

# Check directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Run from fraud-demo directory"
    exit 1
fi

# Kill existing processes
pkill -f "uvicorn main:app" 2>/dev/null
pkill -f "python -m http.server" 2>/dev/null
sleep 1

# Check Python version
echo "🐍 Checking Python version..."
if command -v python3.11 >/dev/null; then
    PYTHON_CMD="python3.11"
    echo "✅ Using Python 3.11"
elif command -v python3 >/dev/null; then
    PYTHON_CMD="python3"
    echo "⚠️  Using Python 3 (may have compatibility issues with RL libraries)"
else
    echo "❌ Python 3 not found"
    exit 1
fi

# Start backend
echo "📡 Starting backend (port 8000)..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d ".venv311" ]; then
    echo "📦 Creating Python 3.11 virtual environment..."
    $PYTHON_CMD -m venv .venv311
fi

# Activate and install dependencies
source .venv311/bin/activate
echo "📦 Installing dependencies (this may take a few minutes for RL libraries)..."
pip install -q --upgrade pip setuptools wheel

# Install PyTorch first (CPU version for compatibility)
echo "🔥 Installing PyTorch..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "📚 Installing ML and RL libraries..."
pip install -q -r requirements.txt

# Verify RL libraries work
echo "🧪 Testing RL libraries..."
python -c "import gymnasium as gym; from stable_baselines3 import PPO; print('✅ RL libraries working!')" || {
    echo "❌ RL library test failed"
    exit 1
}

# Start backend server
# Use python main.py to ensure SSL certificate detection runs
echo "🚀 Starting backend server..."

# Check if using reverse proxy (nginx handles HTTPS)
if [ "$USE_REVERSE_PROXY" = "true" ] || [ "$FORCE_HTTP" = "true" ]; then
    echo "🌐 Running backend in HTTP mode (reverse proxy will handle HTTPS)"
    export FORCE_HTTP=true
    export SSL_KEYFILE=""
    export SSL_CERTFILE=""
else
    # Set SSL certificate paths for direct HTTPS
    export SSL_KEYFILE="/homes/iws/micibr/ssl/attu2.cs.washington.edu.key"
    export SSL_CERTFILE="/homes/iws/micibr/ssl/attu2.cs.washington.edu.crt"
fi

python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend
echo "⏳ Waiting for backend..."
for i in {1..10}; do
    # Try both HTTP and HTTPS
    if curl -s http://localhost:8000/ > /dev/null 2>&1 || curl -s -k https://localhost:8000/ > /dev/null 2>&1; then
        echo "✅ Backend running"
        break
    fi
    sleep 1
done

echo ""
echo "🎉 Demo Ready!"
echo "===================================="
echo "🌐 Frontend: https://homes.cs.washington.edu/~micibr/fraud-demo/frontend/index.html"
echo "📡 Backend: https://attu2.cs.washington.edu:8000"
echo ""
echo "📊 Demo Flow:"
echo "  1. Click 'Test Connection' → Verify backend"
echo "  2. Click 'Run All 1000 Transactions' → See metrics"
echo "  3. Try single transaction: T0002 (fraud) or T0001 (legit)"
echo ""
echo "🧠 RL System Testing:"
echo "  • Train RL model: curl -X POST 'https://attu2.cs.washington.edu:8000/rl/train?timesteps=20000'"
echo "  • Compare methods: curl -X POST 'https://attu2.cs.washington.edu:8000/compare/T0001'"
echo "  • Test RL: python test_rl.py"
echo ""
echo "🛑 Press Ctrl+C to stop"

# Trap Ctrl+C to cleanup
trap "echo ''; echo '🛑 Stopping...'; kill $BACKEND_PID 2>/dev/null; exit" INT

# Keep script running
wait