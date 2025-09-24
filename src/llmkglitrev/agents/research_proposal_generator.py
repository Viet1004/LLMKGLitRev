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

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from llmkglitrev.agents.tools import get_today_str
from llmkglitrev.agents.prompts.research_planning import plan_research_full_agent
from llmkglitrev.agents.states import AgentState, AgentInputState, SupervisorState
from llmkglitrev.agents.research_supervisor import supervisor_agent
from typing import Union
from langchain.chat_models import init_chat_model
writer_model = init_chat_model(model="openai:gpt-4o", max_tokens=16000) # model="anthropic:claude-sonnet-4-20250514", max_tokens=64000

def format_question(state:AgentState):
    return {
        "supervisor_messages": [HumanMessage(content=f"{state['messages']}.")]
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

agent_builder = StateGraph(AgentState, input_schema=AgentInputState)
agent_builder.add_node("format_question", format_question)
agent_builder.add_node("supervisor_subgraph", supervisor_agent)
agent_builder.add_node("final_research_proposal", final_research_proposal)


agent_builder.add_edge(START, "format_question")
agent_builder.add_edge("format_question","supervisor_subgraph")
agent_builder.add_edge("supervisor_subgraph", "final_research_proposal")
agent_builder.add_edge("final_research_proposal", END)

proposal_generator_agent = agent_builder.compile()
    