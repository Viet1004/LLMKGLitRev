"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation  
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from llmkglitrev.agents.tools import get_today_str
from llmkglitrev.agents.prompts.research_planning import plan_research_full_agent
from llmkglitrev.agents.prompts.research_summary import research_agent_keyword_extractor
from llmkglitrev.agents.states import AgentState, AgentInputState, SupervisorState, KeyWordsList
from llmkglitrev.agents.research_supervisor import create_interactive_supervisor
from llmkglitrev.retrieval.neo4j_search import get_neo4j_search
from typing import Union
from langchain.chat_models import init_chat_model
# writer_model = init_chat_model(model="openai:gpt-4o", max_tokens=16000) # model="anthropic:claude-sonnet-4-20250514", max_tokens=64000
writer_model = init_chat_model(model="deepseek:deepseek-chat")

summarize_model = init_chat_model(model="deepseek:deepseek-chat").with_structured_output(KeyWordsList)
async def format_question(state:AgentState):
    """
    Generate research keywords
    """
    query = research_agent_keyword_extractor.format(
        research_prompt=state.get('messages', "")
    )

    keywords = await summarize_model.ainvoke(query)

    return {
        "supervisor_messages": [HumanMessage(content=f"{state['messages']}.")],
        "research_keywords": keywords.keywords
    }

def retrieve_literature(state: AgentState):
    """
    Retrieve relevant papers from Neo4j literature database.
    
    This node searches the knowledge base for papers related to the user's query
    and provides them as context for the supervisor.
    """
    query_text = state['research_keywords'] 

    query_text = ", ".join(query_text)

    print(f"\n🔍 Searching literature database for: {query_text}")
    
    try:
        # Get Neo4j search instance
        neo4j_search = get_neo4j_search()
        
        # Search for relevant papers
        papers = neo4j_search.search_similar_papers(
            query_text=str(query_text),
            top_k=10  # Retrieve top 10 most relevant papers
        )
        
        # Format papers for LLM context
        literature_context = neo4j_search.format_papers_for_llm(papers)
        
        print(f"✅ Retrieved {len(papers)} relevant papers from database")
        
        return {
            "retrieved_papers": papers,
            "literature_context": literature_context
        }
        
    except Exception as e:
        print(f"⚠️  Error retrieving literature: {e}")
        return {
            "retrieved_papers": [],
            "literature_context": "Literature database unavailable."
        }

async def final_research_proposal(state:AgentState | SupervisorState):
    """
    Final research proposal.

    Synthesizes all literature and propositions from sub-agents 
    """
    proposals = state.get("research_proposals", [])

    notes = state.get("raw_notes", [])

    findings = "\n".join(proposals)

    final_research_proposal_prompt = plan_research_full_agent.format(
        research_topic=state.get("research_topic", ""),
        findings=findings,
        notes=notes,
        date=get_today_str()
    )
    final_proposal = await writer_model.ainvoke([HumanMessage(content=final_research_proposal_prompt)])

    return {
        "final_proposal": final_proposal.content, 
        "messages": ["Here is the final proposal: " + final_proposal.content],
    }
from langgraph.checkpoint.memory import MemorySaver

supervisor_agent = create_interactive_supervisor()
agent_builder = StateGraph(AgentState, input_schema=AgentInputState)
agent_builder.add_node("format_question", format_question)
agent_builder.add_node("retrieve_literature", retrieve_literature)  # NEW: Literature retrieval
agent_builder.add_node("supervisor_subgraph", supervisor_agent)
agent_builder.add_node("final_research_proposal", final_research_proposal)


agent_builder.add_edge(START, "format_question")
agent_builder.add_edge("format_question", "retrieve_literature")  # NEW: Retrieve before supervisor
agent_builder.add_edge("retrieve_literature", "supervisor_subgraph")  # NEW: Pass context to supervisor
agent_builder.add_edge("supervisor_subgraph", "final_research_proposal")
agent_builder.add_edge("final_research_proposal", END)

# Compile with MemorySaver checkpointer to support interrupts
checkpointer = MemorySaver()
proposal_generator_agent = agent_builder.compile(checkpointer=checkpointer)

# Import interactive runner for standalone usage
from llmkglitrev.agents.interactive_runner import (
    run_research_interactive,
    run_research_interactive_sync,
    resume_research_interactive
)
    