#!/bin/bash
# TD SYNNEX RAG Demo - Quick Start Script
# Run: ./scripts/run_demo.sh

set -e

echo "🚀 TD SYNNEX Production RAG Demo"
echo "================================"

# Check Python version
python3 --version

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet streamlit pandas numpy plotly

# Optional: Install full requirements
# pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Starting Streamlit app..."
echo "   URL: http://localhost:8501"
echo ""

# Run Streamlit
streamlit run main.py --server.port=8501 --server.address=0.0.0.0
