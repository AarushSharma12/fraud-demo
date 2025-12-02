#!/bin/bash

# Configuration
SESSION_NAME="fraud-demo"
WORK_DIR=$(pwd)
BACKEND_DIR="$WORK_DIR/backend"
PYTHON_ENV=".venv311"
VENV_PATH="$BACKEND_DIR/$PYTHON_ENV"

# Kill session if it exists to ensure a fresh start
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "🗑️  Killing old session '$SESSION_NAME'..."
    tmux kill-session -t $SESSION_NAME
fi

echo "🚀 Starting Fraud Detection Demo..."

# Create new session
tmux new-session -d -s $SESSION_NAME -n backend

# Pane 1: Backend Server (Top Left)
# ----------------------------------
tmux send-keys -t $SESSION_NAME:backend "cd $BACKEND_DIR" C-m
# Force cleanup and fresh install
tmux send-keys -t $SESSION_NAME:backend "rm -rf $PYTHON_ENV" C-m
tmux send-keys -t $SESSION_NAME:backend "python3 -m venv $PYTHON_ENV" C-m
tmux send-keys -t $SESSION_NAME:backend "$VENV_PATH/bin/pip install -r requirements.txt" C-m
tmux send-keys -t $SESSION_NAME:backend "$VENV_PATH/bin/python main.py" C-m

# Pane 2: Live Feed Generator (Right)
# -----------------------------------
tmux split-window -h -t $SESSION_NAME:backend
# Use absolute path to be safe
tmux send-keys -t $SESSION_NAME:backend.1 "cd $WORK_DIR" C-m
tmux send-keys -t $SESSION_NAME:backend.1 "sleep 10" C-m
# Run generator using the backend's venv
tmux send-keys -t $SESSION_NAME:backend.1 "$VENV_PATH/bin/python prod_generator.py" C-m

# Pane 3: System Monitor (Bottom Right)
# -------------------------------------
tmux split-window -v -t $SESSION_NAME:backend.1
# Check the root endpoint (/) which is public, instead of /live-feed/status
tmux send-keys -t $SESSION_NAME:backend.2 "watch -n 1 'curl -s http://localhost:8004/ | python3 -m json.tool'" C-m

# Focus and Attach
tmux select-window -t $SESSION_NAME:backend
echo "✅ Session started! Attaching..."
tmux attach-session -t $SESSION_NAME
