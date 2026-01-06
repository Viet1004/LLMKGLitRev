
"""
State Definitions and Pydantic Schemas for Research Agent

This module defines the state objects and structured schemas used for
the research agent workflow, including researcher state management and output schemas.
"""

import operator
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Literal
from langgraph.graph import MessagesState
# ===== STATE DEFINITIONS =====

class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata.

    This state tracks the researcher's conversation, iteration count for limiting
    tool calls, the research topic being investigated, compressed findings,
    and raw research notes for detailed analysis.
    """
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    maximum_number_of_plan: int
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]

class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent containing final research results.

    This represents the final output of the research process with compressed
    research findings and all raw notes from the research process.
    """
    research_plan: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decisions during scoping phase."""
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class KeyWordsList(BaseModel):
    """
    Schema for keywords to search
    """
    keywords: List[str] = Field(
        description="List of keywords extracted for literature search",
    )
class ResearchQuestion(BaseModel):
    """Schema for research brief generation."""
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class ResearchSummary(BaseModel):
    """Schema for research summary."""
    
    topic: str = Field(
        description="General topic of the research article",
        min_length=3,
        max_length=200,
        examples=["Machine Learning in Healthcare", "Climate Change Mitigation"]
    )
    
    disciplinaries: list[str] = Field(
        description="Domains involved in this research", 
        min_length=1,
        max_length=5,
        examples=[["Computer Science", "Medicine"], ["Environmental Science", "Economics"]]
    )
    
    research_question: str = Field(
        description="The key research question",
        min_length=10,
        max_length=500,
        examples=["How can machine learning improve early disease detection in medical imaging?"]
    )
    
    method: str = Field(
        description="Proposed methodology",
        min_length=10,
        max_length=1000,
        examples=["Systematic literature review followed by comparative analysis"]
    )
    
    result: str = Field(
        description="Primary research findings",
        min_length=10,
        max_length=2000,
        examples=["The study reveals that deep learning models achieve 95% accuracy in tumor detection"]
    )


class SupervisorState(TypedDict):
    """
    State for the multi-agent research supervisor.

    Manages coordination between supervisor and research agents, assessing novelty of each research proposition from each 
    sub-agent.
    """
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Research proposals from sub-agents
    research_proposals: List[ResearchSummary]
    # Processed and structured notes ready for final report generation
    notes: Annotated[list[str], operator.add]
    # Topic of research
    research_topic: Literal[str]

    got_human_feedback: bool
    # Counter tracking the number of research iterations performed
    research_iterations: int
    # Raw unprocessed research notes collected from sub-agent research
    raw_notes: Annotated[list[str], operator.add]
    pending_proposals: List[str]
    # Retrieved papers from Neo4j literature database
    retrieved_papers: List[dict]
    # Formatted literature context for LLM
    literature_context: str


class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass


class AgentState(MessagesState):
    """
    Main state for the full multi-agent research proposition.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """
    research_topic: Literal[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_proposals: List[ResearchSummary]
    research_keywords: list[str]
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[list[str], operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Final formatted research report (empty if workflow interrupted before completion)
    final_proposal: str = ""

