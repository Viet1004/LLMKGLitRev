#!/bin/bash

# Run Research Workflow Streamlit Application
# This script activates the virtual environment and launches the Streamlit app

echo "🔬 Starting Research Workflow Application..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   uv pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "✅ Activating virtual environment..."
source .venv/bin/activate

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not installed. Installing now..."
    uv pip install streamlit
fi

# Check environment variables
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: No API keys found in environment"
    echo "   Please set OPENAI_API_KEY or ANTHROPIC_API_KEY"
    echo ""
fi

# Launch Streamlit app
echo "🚀 Launching application..."
echo "   The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run app_research_workflow.py
