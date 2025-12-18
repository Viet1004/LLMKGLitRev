# Research Proposal Generator

A FastAPI backend with Streamlit frontend for generating comprehensive research proposals using a multi-agent system.

## Architecture

- **Backend**: FastAPI (`chat.py`) - Multi-agent research system with web search and coordination
- **Frontend**: Streamlit (`streamlit_app.py`) - Interactive interface for research proposal generation

## Quick Start

### Option 1: Run both services together (Recommended)
```bash
./run_app.sh
```

Then open:
- Streamlit UI: http://localhost:8501
- FastAPI Docs: http://localhost:8000/docs

### Option 2: Run services separately

**Terminal 1 - Start FastAPI Backend:**
```bash
python -m llmkglitrev.api_endpoint.chat
```

**Terminal 2 - Start Streamlit Frontend:**
```bash
streamlit run src/llmkglitrev/api_endpoint/streamlit_app.py
```

## Features

### Current Features
- ✅ Multi-agent research coordination
- ✅ Web search integration (Tavily)
- ✅ Parallel research investigation
- ✅ Research synthesis and evaluation
- ✅ API status monitoring
- ✅ Chat history management
- ✅ Example research topics
- ✅ Error handling

### Future Enhancements (Ready for visualization)
- 📊 Agent workflow visualization
- 📈 Research metrics dashboard
- 🔍 Literature analysis charts
- 📉 Progress tracking for long-running research

## API Endpoints

### POST `/research`
Generate comprehensive research proposals using multi-agent system.

**Request Body:**
```json
{
  "query": "How can LLMs improve literature review processes?"
}
```

**Response:**
```json
{
  "final_proposal": "# Research Proposal\n\n...",
  "raw_notes": ["Research findings from agent 1...", "..."],
  "research_proposals": ["Proposal from perspective 1...", "..."]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/research" \
  -H "Content-Type: application/json" \
  -d '{"query": "Using AI for climate change mitigation"}'
```

**Note:** This endpoint may take 3-10 minutes to complete as it:
1. Coordinates multiple research agents
2. Conducts web searches for current literature
3. Evaluates and synthesizes findings
4. Generates a comprehensive proposal

## Environment Variables

Make sure you have `.env` file with:
```
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

The system uses:
- **Anthropic Claude**: Main research agents and supervisor
- **OpenAI GPT-4**: Research evaluation and synthesis
- **Tavily**: Web search for literature discovery

## Dependencies

- `fastapi` - Web framework for API
- `streamlit` - Frontend framework
- `langchain` - LLM orchestration
- `langgraph` - Agent workflow management
- `tavily-python` - Web search API
- `uvicorn` - ASGI server
- `requests` - HTTP client

## Development

### Adding Visualizations

Streamlit makes it easy to add charts and graphs:

```python
import streamlit as st
import pandas as pd
import plotly.express as px

# Example: Add a chart to the sidebar
with st.sidebar:
    st.subheader("📊 Research Metrics")
    data = pd.DataFrame({
        'metric': ['Novelty', 'Impact', 'Feasibility'],
        'score': [8, 7, 9]
    })
    fig = px.bar(data, x='metric', y='score')
    st.plotly_chart(fig)
```

### Customizing the UI

Edit `streamlit_app.py` to:
- Add more sidebar widgets
- Create multi-page apps
- Add file upload functionality
- Display knowledge graphs
- Show analysis results

## Troubleshooting

**API Connection Error:**
- Make sure FastAPI is running on port 8000
- Check that `.env` file contains valid API keys

**Port Already in Use:**
- Change ports in the respective files:
  - FastAPI: `chat.py` line with `port=8000`
  - Streamlit: Use `streamlit run --server.port 8502 ...`

**Module Import Errors:**
- Run `uv sync` to install dependencies
- Make sure you're in the correct Python environment
