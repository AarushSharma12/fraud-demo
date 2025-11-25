#!/bin/bash

# Configuration
SESSION_NAME="fraud-demo"
BACKEND_DIR="backend"
PYTHON_ENV=".venv311"

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    echo "🚀 Starting Fraud Detection Demo in tmux session '$SESSION_NAME'..."
    
    # Create new session and name the first window 'backend'
    tmux new-session -d -s $SESSION_NAME -n backend
    
    # Pane 1: Backend Server (Top Left)
    tmux send-keys -t $SESSION_NAME:backend "cd $BACKEND_DIR" C-m
    
    # FORCE REMOVE existing venv to fix OS compatibility issues (in case it was copied)
    tmux send-keys -t $SESSION_NAME:backend "rm -rf $PYTHON_ENV" C-m
    
    # Create new venv using python3 (default available python)
    tmux send-keys -t $SESSION_NAME:backend "python3 -m venv $PYTHON_ENV" C-m
    
    # Install dependencies using the venv's pip explicitly
    tmux send-keys -t $SESSION_NAME:backend "./$PYTHON_ENV/bin/pip install -r requirements.txt" C-m
    
    # Run main.py using venv's python
    tmux send-keys -t $SESSION_NAME:backend "./$PYTHON_ENV/bin/python main.py" C-m
    
    # Split window horizontally for Live Feed Generator
    tmux split-window -h -t $SESSION_NAME:backend
    # Ensure we are in the project root for the generator
    tmux send-keys -t $SESSION_NAME:backend.1 "cd .." C-m
    tmux send-keys -t $SESSION_NAME:backend.1 "sleep 10" C-m
    # Use the backend's venv python to run the generator
    tmux send-keys -t $SESSION_NAME:backend.1 "$BACKEND_DIR/$PYTHON_ENV/bin/python prod_generator.py" C-m
    
    # Split the right pane vertically for System Monitor / Training
    tmux split-window -v -t $SESSION_NAME:backend.1
    tmux send-keys -t $SESSION_NAME:backend.2 "watch -n 1 'curl -s http://localhost:8000/live-feed/status | python3 -m json.tool'" C-m

    # Select the backend window
    tmux select-window -t $SESSION_NAME:backend
    
    echo "✅ Session started! Attaching..."
    tmux attach-session -t $SESSION_NAME
else
    echo "⚠️  Session '$SESSION_NAME' already exists. Attaching..."
    tmux attach-session -t $SESSION_NAME
fi

