#!/bin/bash

# Run both FastAPI and Streamlit
# Usage: ./run_app.sh

echo "🚀 Starting Research Assistant Application..."
echo ""
echo "Starting FastAPI backend on port 8000..."
echo "Starting Streamlit frontend on port 8501..."
echo ""
echo "URLs:"
echo "  - Streamlit UI: http://localhost:8501"
echo "  - FastAPI Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""
source /Users/quocviet.nguyen/paper_idea/LLM_KG_LitRev/LLMKGLitRev/.venv/bin/activate
# Run FastAPI in background
python -m llmkglitrev.api_endpoint.chat &
FASTAPI_PID=$!

# Wait a bit for FastAPI to start
sleep 3

# Run Streamlit in foreground
streamlit run src/llmkglitrev/api_endpoint/streamlit_app.py

# Kill FastAPI when Streamlit stops
kill $FASTAPI_PID
