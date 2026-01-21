
from pydantic import BaseModel, Field
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, filter_messages
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from llmkglitrev.agents.states import ResearcherState, ResearcherOutputState, ResearchSummary

from llmkglitrev.agents.tools import tavily_search, get_today_str, evaluation_tool, planning_tool

from llmkglitrev.agents.academic_search_tools import (
    search_academic_papers,
    search_google_scholar,
    search_arxiv,
    search_ieee,
    search_scopus,
)

from llmkglitrev.agents.prompts.research_summary import research_agent_prompt

from llmkglitrev.agents.prompts.research_planning import plan_research_system_message, plan_research_human_message

from llmkglitrev.characters import extract_dialogue_notes

from dotenv import load_dotenv

load_dotenv()

# ===== CONFIGURATION =====
# Maximum number of tool call iterations per research agent
# This prevents infinite loops and controls research depth
MAX_TOOL_CALL_ITERATIONS = 5  # Agent can search/evaluate up to 3 times

# Set up tools and model binding
# tools = [tavily_search, planning_tool, evaluation_tool]
# Academic search tools (primary for academic content)
tools = [
    # search_academic_papers,    # PRIMARY: Multi-source academic search (Google Scholar + arXiv)
    # search_google_scholar,     # Specific: Google Scholar only (IEEE, Springer, ACM, Nature, etc.)
    search_arxiv,              # Specific: arXiv preprints (CS, physics, math, stats)
    tavily_search,             # FALLBACK: General web search for non-academic content
    evaluation_tool
]
tools_by_name = {tool.name: tool for tool in tools}

# researcher = init_chat_model(model="anthropic:claude-sonnet-4-20250514")
# researcher = init_chat_model(model="anthropic:claude-sonnet-4-5-20250929")
researcher = init_chat_model(model="deepseek:deepseek-chat")
# evaluator = init_chat_model(model="openai:gpt-4o", max_tokens=16000)
evaluator = init_chat_model(model="deepseek:deepseek-chat")

researcher_with_tools = researcher.bind_tools(tools)

async def researcher_llm_call(state: ResearcherState):
    """ Analyze the current critics and decide if new plan should be derived.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools upon new direction.
    2. Call search tools build upon current ideas with modified plan.
    3. Provide final research idea and plan.

    CRITICAL: Checks iteration limit BEFORE calling LLM to prevent generating
    tool_calls that would violate API constraints (tool_calls must be followed by ToolMessages).

    Return updated state with the model's response.
    """
    current_iterations = state.get("tool_call_iterations", 0)

    # CRITICAL: Check iteration limit BEFORE calling LLM
    # This prevents generating tool_calls when we're at max iterations
    if current_iterations >= MAX_TOOL_CALL_ITERATIONS:
        print(f"⚠️  Max iterations ({MAX_TOOL_CALL_ITERATIONS}) reached. Finalizing research.")
        # Return an AIMessage without tool_calls to trigger research_formulation
        # Must be AIMessage (not HumanMessage) because should_continue checks .tool_calls attribute
        return {
            "researcher_messages": [
                AIMessage(content="Maximum research depth reached. Proceeding to finalize research summary.")
            ]
        }

    # Use character system prompt if provided, otherwise default
    system_prompt = state.get("character_system_prompt") or research_agent_prompt

    # Use async invoke to properly handle async context
    response = await researcher_with_tools.ainvoke(
        [SystemMessage(content=system_prompt)] + state["researcher_messages"]
    )

    return {
        "researcher_messages": [response]
    }

# ===== AGENT NODES =====

async def tool_node(state: ResearcherState):
    """
    Execute tool calls from previous LLM response.

    Also increments tool_call_iterations to track research depth.
    Uses async to properly handle async tools and prevent "Task destroyed" errors.
    """
    import asyncio

    tool_calls = state["researcher_messages"][-1].tool_calls
    current_iterations = state.get("tool_call_iterations", 0)

    # Execute tools concurrently with proper async handling
    async def invoke_tool(tool_call):
        tool = tools_by_name[tool_call["name"]]
        # Check if tool has async invoke method
        if hasattr(tool, 'ainvoke'):
            return await tool.ainvoke(tool_call["args"])
        else:
            # Fallback to sync invoke wrapped in executor
            return await asyncio.get_event_loop().run_in_executor(
                None, tool.invoke, tool_call["args"]
            )

    observations = await asyncio.gather(*[invoke_tool(tc) for tc in tool_calls])

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    # CRITICAL: Increment iteration counter to prevent infinite loops
    return {
        "researcher_messages": tool_outputs,
        "tool_call_iterations": current_iterations + 1
    }

async def research_formulation(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.

    Also extracts dialogue notes for Socratic dialogue.
    Uses async to properly handle async LLM calls.
    """

    system_message = plan_research_system_message.format(date=get_today_str(), maximum_number_of_plan=state.get("maximum_number_of_plan", 3))
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=plan_research_human_message.format(research_topic=state.get("research_topic", "")))]

    # Use async invoke to properly handle async context
    response = await evaluator.ainvoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"]
        )
    ]

    # NEW: Extract dialogue notes for Socratic dialogue
    dialogue_notes = extract_dialogue_notes(
        messages=state.get("researcher_messages", []),
        research_output=str(response.content),
        max_notes_per_type=3
    )

    # Convert to dicts for JSON serialization
    dialogue_notes_dicts = [note.model_dump() for note in dialogue_notes]

    return {
        "research_plan": str(response.content),
        "raw_notes": ["\n".join(raw_notes)],
        "dialogue_notes": dialogue_notes_dicts  # NEW
    }



# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "research_formulation"]:
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.
    
    NOTE: Iteration limit is checked in researcher_llm_call BEFORE calling LLM,
    not here. This ensures any pending tool_calls are always executed first,
    satisfying the API constraint that tool_calls must be followed by ToolMessages.

    Returns:
        "tool_node": Continue to tool execution
        "research_formulation": Stop and compress research
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, ALWAYS execute it (don't check iterations here)
    # The iteration check happens in researcher_llm_call to prevent new tool_calls
    # Use hasattr to safely check for tool_calls (only AIMessage has this attribute)
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    # Otherwise, we have a final answer
    return "research_formulation"



# =========== GRAPH CONSTRUCTION ============

agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
# agent_builder = StateGraph(ResearcherState, output_schema=ResearchSummary)
# agent_builder = StateGraph(ResearcherState)

agent_builder.add_node("researcher_llm_call", researcher_llm_call)

agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("research_formulation", research_formulation)


agent_builder.add_edge(START, "researcher_llm_call")
agent_builder.add_conditional_edges(
    "researcher_llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        "research_formulation":"research_formulation"
    }
)
agent_builder.add_edge("tool_node", "researcher_llm_call")
agent_builder.add_edge("research_formulation", END)

research_agent = agent_builder.compile()
