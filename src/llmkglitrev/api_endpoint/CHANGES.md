# API Migration: Simple Chat → Research Agent System

## Summary of Changes

This document summarizes the migration from a simple chat endpoint to a sophisticated multi-agent research proposal generator.

## What Changed

### 1. Backend API (`chat.py`)

**Before:**
- Simple GET endpoint `/chat` with query parameter
- Direct LLM call using Claude
- Synchronous execution
- Simple prompt template

**After:**
- POST endpoint `/research` with JSON body
- Multi-agent research system (`proposal_generator_agent`)
- Async execution with `ainvoke`
- Structured request/response models
- Error handling and validation

### 2. Frontend (`streamlit_app.py`)

**Changes:**
- API URL: `http://localhost:8000/chat` → `http://localhost:8000/research`
- HTTP Method: `GET` → `POST`
- Request Format: Query params → JSON body
- Response Field: `response.content` → `result.final_proposal`
- Timeout: 310s → 600s (research takes longer)
- UI Text: Updated to reflect research proposal generation
- Example Prompts: Changed to research topics

### 3. Key Functional Differences

| Feature | Before | After |
|---------|--------|-------|
| **Endpoint** | GET /chat | POST /research |
| **Input** | Simple prompt string | Structured JSON with validation |
| **Processing** | Single LLM call | Multi-agent coordination |
| **Output** | Text response | Structured proposal + notes |
| **Duration** | ~5-30 seconds | 3-10 minutes |
| **Capabilities** | Q&A | Research + Web Search + Synthesis |

## How the Multi-Agent System Works

1. **User Input** → Streamlit receives research topic
2. **API Call** → POST request to `/research` endpoint
3. **Supervisor Agent** → Coordinates research strategy
4. **Research Agents** → Multiple agents investigate in parallel
5. **Web Search** → Tavily searches for current literature
6. **Synthesis** → GPT-4 evaluates and combines findings
7. **Final Proposal** → Claude generates comprehensive proposal
8. **Response** → Returns to Streamlit for display

## API Integration Example

### Before (Simple Chat)
```python
# Frontend
response = requests.get(
    "http://localhost:8000/chat",
    params={"prompt": "What makes a good research idea?"}
)
answer = response.json()["response"]
```

### After (Research Agent)
```python
# Frontend
response = requests.post(
    "http://localhost:8000/research",
    json={"query": "How can LLMs improve literature reviews?"},
    timeout=600
)
result = response.json()
proposal = result["final_proposal"]
notes = result["raw_notes"]
```

## Running the System

### Terminal 1 - Backend
```bash
python -m llmkglitrev.api_endpoint.chat
```

### Terminal 2 - Frontend
```bash
streamlit run src/llmkglitrev/api_endpoint/streamlit_app.py
```

### Or use the launcher script
```bash
./run_app.sh
```

## Environment Requirements

Required API keys in `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

## Performance Considerations

- **Timeout**: Set to 600s (10 minutes) to accommodate research
- **Concurrency**: System runs multiple agents in parallel
- **Cost**: Multiple LLM calls + web searches per request
- **Rate Limits**: May hit API rate limits with concurrent requests

## Future Enhancements

1. **Streaming**: Add SSE for real-time progress updates
2. **Caching**: Cache research results for similar queries
3. **Customization**: Allow users to configure research depth
4. **Visualization**: Show agent workflow in real-time
5. **Export**: Download proposals as PDF/Markdown

## Migration Checklist

- [x] Update FastAPI endpoint from GET to POST
- [x] Integrate `proposal_generator_agent`
- [x] Add request/response models
- [x] Update Streamlit to use POST requests
- [x] Increase timeout for long operations
- [x] Update UI text and examples
- [x] Update documentation (README.md)
- [x] Test error handling
- [ ] Add progress indicators (future)
- [ ] Add streaming support (future)

## Testing

Test the endpoint:
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Using machine learning for climate modeling"}' \
  --max-time 600
```

Expected response structure:
```json
{
  "final_proposal": "# Research Proposal\n\n## Introduction\n...",
  "raw_notes": ["Agent 1 findings...", "Agent 2 findings..."],
  "research_proposals": ["Proposal 1...", "Proposal 2..."]
}
```
