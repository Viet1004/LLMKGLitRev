import fastapi
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import List, Optional
import uuid
from langgraph.types import Command

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
    thread_id: Optional[str] = Field(
        default=None,
        description="Thread ID for resuming interrupted workflows. If not provided, a new one will be generated."
    )

class ResumeRequest(BaseModel):
    """Request model for resuming interrupted research."""
    thread_id: str = Field(
        ...,
        description="Thread ID of the interrupted workflow to resume"
    )
    feedback: str = Field(
        ...,
        description="Human feedback to provide to the interrupted workflow",
        examples=["Approved", "Focus more on practical applications", "Reject this direction"]
    )


class ResearchResponse(BaseModel):
    """Response model for research proposal."""
    status: str = Field(
        description="Status of the research: 'complete', 'interrupted', or 'error'"
    )
    thread_id: str = Field(
        description="Thread ID for this research workflow"
    )
    final_proposal: Optional[str] = Field(
        default=None,
        description="The complete research proposal (only present when status='complete')"
    )
    raw_notes: Optional[List[str]] = Field(
        default=None, 
        description="Raw research notes from sub-agents"
    )
    research_proposals: Optional[List[str]] = Field(
        default=None,
        description="Individual research proposals from sub-agents"
    )
    retrieved_papers: Optional[List[dict]] = Field(
        default=None,
        description="Papers retrieved from Neo4j literature database"
    )
    literature_context: Optional[str] = Field(
        default=None,
        description="Formatted literature context used by the supervisor"
    )
    research_keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords extracted for literature search"
    )
    # Interrupt-specific fields
    interrupt_type: Optional[str] = Field(
        default=None,
        description="Type of interrupt (e.g., 'human_feedback_request')"
    )
    interrupt_question: Optional[str] = Field(
        default=None,
        description="Question or prompt for the user when interrupted"
    )
    interrupt_proposals: Optional[List[str]] = Field(
        default=None,
        description="Research proposals awaiting human feedback"
    )
    interrupt_instructions: Optional[str] = Field(
        default=None,
        description="Instructions for providing feedback"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status='error'"
    )


app = FastAPI(
    title="Research Proposal Generator API",
    version="0.0.1",
    description="An API endpoint for generating research proposals using multi-agent system",
)


@app.post("/research/start", response_model=ResearchResponse)
async def start_research(request: ResearchRequest):
    """
    Start a research proposal generation workflow.
    
    This endpoint initiates a multi-agent research system that may require
    human-in-the-loop feedback. If the workflow needs human input, it will
    return status='interrupted' with details about what input is needed.
    
    Returns:
        - status='interrupted': Workflow paused, waiting for human feedback
        - status='complete': Research proposal completed without interruption
        - status='error': An error occurred
    """
    try:
        # Generate or use provided thread_id
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Configure with thread_id for state persistence
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke the research agent
        result = await proposal_generator_agent.ainvoke(
            {"messages": [HumanMessage(content=request.query)]},
            config=config
        )
        
        # Check if workflow was interrupted
        if "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][0].value
            
            return ResearchResponse(
                status="interrupted",
                thread_id=thread_id,
                interrupt_type=interrupt_data.get("type"),
                interrupt_question=interrupt_data.get("question"),
                interrupt_proposals=interrupt_data.get("proposals", []),
                interrupt_instructions=interrupt_data.get("instructions"),
                # Include any partial results
                retrieved_papers=result.get("retrieved_papers", []),
                literature_context=result.get("literature_context", ""),
                research_keywords=result.get("research_keywords", [])
            )
        
        # Workflow completed without interruption
        return ResearchResponse(
            status="complete",
            thread_id=thread_id,
            final_proposal=result.get("final_proposal", ""),
            raw_notes=result.get("raw_notes", []),
            research_proposals=result.get("research_proposals", []),
            retrieved_papers=result.get("retrieved_papers", []),
            literature_context=result.get("literature_context", ""),
            research_keywords=result.get("research_keywords", [])
        )
        
    except Exception as e:
        return ResearchResponse(
            status="error",
            thread_id=request.thread_id or "unknown",
            error=f"Error generating research proposal: {str(e)}"
        )


@app.post("/research/resume", response_model=ResearchResponse)
async def resume_research(request: ResumeRequest):
    """
    Resume an interrupted research workflow with human feedback.
    
    Provide the thread_id from the interrupted workflow and the human feedback
    to continue the research process.
    
    Returns:
        - status='interrupted': Workflow paused again (if multiple interrupts)
        - status='complete': Research proposal completed
        - status='error': An error occurred
    """
    try:
        # Configure with the same thread_id to resume
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Resume with human feedback using Command
        result = await proposal_generator_agent.ainvoke(
            Command(resume=request.feedback),
            config=config
        )
        
        # Check if workflow was interrupted again
        if "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][0].value
            
            return ResearchResponse(
                status="interrupted",
                thread_id=request.thread_id,
                interrupt_type=interrupt_data.get("type"),
                interrupt_question=interrupt_data.get("question"),
                interrupt_proposals=interrupt_data.get("proposals", []),
                interrupt_instructions=interrupt_data.get("instructions"),
                retrieved_papers=result.get("retrieved_papers", []),
                literature_context=result.get("literature_context", ""),
                research_keywords=result.get("research_keywords", [])
            )
        
        # Workflow completed
        return ResearchResponse(
            status="complete",
            thread_id=request.thread_id,
            final_proposal=result.get("final_proposal", ""),
            raw_notes=result.get("raw_notes", []),
            research_proposals=result.get("research_proposals", []),
            retrieved_papers=result.get("retrieved_papers", []),
            literature_context=result.get("literature_context", ""),
            research_keywords=result.get("research_keywords", [])
        )
        
    except Exception as e:
        return ResearchResponse(
            status="error",
            thread_id=request.thread_id,
            error=f"Error resuming research: {str(e)}"
        )


# Legacy endpoint for backward compatibility
@app.post("/research", response_model=ResearchResponse)
async def generate_research_proposal(request: ResearchRequest):
    """
    Legacy endpoint - redirects to /research/start
    
    Deprecated: Use /research/start and /research/resume for full interrupt support.
    """
    return await start_research(request)

# api_router = APIRouter()

if __name__ == "__main__":
    uvicorn.run("llmkglitrev.api_endpoint.chat:app", host="0.0.0.0", port=8000, reload=True)

