import fastapi
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables BEFORE importing agents
load_dotenv()

# Import agents after environment variables are loaded
from llmkglitrev.agents.research_proposal_generator import proposal_generator_agent


# Request/Response models
class ResearchRequest(BaseModel):
    """Request model for research proposal generation."""
    query: str = Field(
        ..., 
        description="Research topic or question to investigate",
        min_length=10,
        examples=["How can LLMs improve literature review processes?"]
    )


class ResearchResponse(BaseModel):
    """Response model for research proposal."""
    final_proposal: str = Field(description="The complete research proposal")
    raw_notes: Optional[List[str]] = Field(
        default=None, 
        description="Raw research notes from sub-agents"
    )
    research_proposals: Optional[List[str]] = Field(
        default=None,
        description="Individual research proposals from sub-agents"
    )


app = FastAPI(
    title="Research Proposal Generator API",
    version="0.0.1",
    description="An API endpoint for generating research proposals using multi-agent system",
)


@app.post("/research", response_model=ResearchResponse)
async def generate_research_proposal(request: ResearchRequest):
    """
    Generate a comprehensive research proposal using multi-agent system.
    
    This endpoint uses a sophisticated multi-agent system that:
    1. Coordinates multiple research agents to investigate different aspects
    2. Conducts web searches for relevant literature
    3. Evaluates and synthesizes findings
    4. Generates a comprehensive research proposal
    
    Note: This is a long-running operation that may take several minutes.
    """
    try:
        # Invoke the research agent
        result = await proposal_generator_agent.ainvoke({
            "messages": [HumanMessage(content=request.query)]
        })
        
        return ResearchResponse(
            final_proposal=result.get("final_proposal", ""),
            raw_notes=result.get("raw_notes", []),
            research_proposals=result.get("research_proposals", [])
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating research proposal: {str(e)}"
        )

# api_router = APIRouter()

if __name__ == "__main__":
    uvicorn.run("llmkglitrev.api_endpoint.chat:app", host="0.0.0.0", port=8000, reload=True)

