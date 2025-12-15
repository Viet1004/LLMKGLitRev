
from pydantic import BaseModel, Field
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from llmkglitrev.agents.states import ResearcherState, ResearcherOutputState, ResearchSummary

from llmkglitrev.agents.tools import tavily_search, get_today_str, evaluation_tool, planning_tool

from llmkglitrev.agents.prompts.research_summary import research_agent_prompt

from llmkglitrev.agents.prompts.research_planning import plan_research_system_message, plan_research_human_message

from dotenv import load_dotenv

load_dotenv()
# Set up tools and model binding
# tools = [tavily_search, planning_tool, evaluation_tool]
tools = [tavily_search, evaluation_tool]
tools_by_name = {tool.name: tool for tool in tools}

# researcher = init_chat_model(model="anthropic:claude-sonnet-4-20250514")
# researcher = init_chat_model(model="anthropic:claude-sonnet-4-5-20250929")
researcher = init_chat_model(model="deepseek:deepseek-chat")
# evaluator = init_chat_model(model="openai:gpt-4o", max_tokens=16000)
evaluator = init_chat_model(model="deepseek:deepseek-chat")

researcher_with_tools = researcher.bind_tools(tools)

def researcher_llm_call(state: ResearcherState):
    """ Analyze the current critics and decide if new plan should be derived.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools upon new direction.
    2. Call search tools build upon current ideas with modified plan.
    3. Provide final research idea and plan.

    Return updated state with the model's response.
    """

    return {
        "researcher_messages": [
            researcher_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]
            )
        ]
    }

# ===== AGENT NODES =====

def tool_node(state: ResearcherState):
    """
    Execute tool calls from previous LLM response.
    """

    tool_calls = state["researcher_messages"][-1].tool_calls

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}

def research_formulation(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
    """

    system_message = plan_research_system_message.format(date=get_today_str(), maximum_number_of_plan=state.get("maximum_number_of_plan", 3))
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=plan_research_human_message.format(research_topic=state.get("research_topic", "")))]
    response = evaluator.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

    return {
        "research_plan": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }



# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "research_formulation"]:
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "research_formulation": Stop and compress research
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
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
