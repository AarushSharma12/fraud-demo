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
    # Check if venv exists, if not create it
    tmux send-keys -t $SESSION_NAME:backend "if [ ! -d $PYTHON_ENV ]; then python3.11 -m venv $PYTHON_ENV; fi" C-m
    tmux send-keys -t $SESSION_NAME:backend "source $PYTHON_ENV/bin/activate" C-m
    tmux send-keys -t $SESSION_NAME:backend "pip install -r requirements.txt" C-m
    tmux send-keys -t $SESSION_NAME:backend "python main.py" C-m
    
    # Split window horizontally for Live Feed Generator
    tmux split-window -h -t $SESSION_NAME:backend
    tmux send-keys -t $SESSION_NAME:backend.1 "source $BACKEND_DIR/$PYTHON_ENV/bin/activate" C-m
    # Wait a bit for backend to start before running generator
    tmux send-keys -t $SESSION_NAME:backend.1 "sleep 5" C-m
    tmux send-keys -t $SESSION_NAME:backend.1 "python prod_generator.py" C-m
    
    # Split the right pane vertically for System Monitor / Training
    tmux split-window -v -t $SESSION_NAME:backend.1
    tmux send-keys -t $SESSION_NAME:backend.2 "source $BACKEND_DIR/$PYTHON_ENV/bin/activate" C-m
    tmux send-keys -t $SESSION_NAME:backend.2 "watch -n 1 'curl -s http://localhost:8000/live-feed/status | python3 -m json.tool'" C-m

    # Select the backend window
    tmux select-window -t $SESSION_NAME:backend
    
    echo "✅ Session started! Attaching..."
    tmux attach-session -t $SESSION_NAME
else
    echo "⚠️  Session '$SESSION_NAME' already exists. Attaching..."
    tmux attach-session -t $SESSION_NAME
fi

