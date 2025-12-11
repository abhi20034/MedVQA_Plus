#!/bin/bash
# Quick start script for MedVQA+ Web Application

echo "=================================="
echo "MedVQA+ Web Application"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "⚠ Virtual environment not found. Creating one..."
    python3 -m venv env
fi

# Activate virtual environment
source env/bin/activate

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    pip install Flask Werkzeug
fi

# Check if models exist
if [ ! -f "checkpoints/best_closed_improved.pt" ]; then
    echo "⚠ Warning: checkpoints/best_closed_improved.pt not found"
fi

if [ ! -f "checkpoints/best_topk50_improved.pt" ]; then
    echo "⚠ Warning: checkpoints/best_topk50_improved.pt not found"
fi

echo ""
echo "🚀 Starting Flask server..."
echo "📱 Open http://localhost:5000 in your browser"
echo ""

python app.py
