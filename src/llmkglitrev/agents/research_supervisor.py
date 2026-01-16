"""
    Multi-agent supervisor to delegate research job to subagents, which can investigate problems in different aspects,
    such as methods, evaluation, research questions and then propose research ideas
"""

from typing import Annotated, Callable, List, Sequence
from pydantic import BaseModel, Field
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage,ToolMessage, filter_messages
from langchain.chat_models import init_chat_model
from llmkglitrev.agents.tools import (
    ConductResearch,
    ResearchComplete,
    # AskHumanFeedback
)
from llmkglitrev.agents.states import SupervisorState
from llmkglitrev.agents.research_agents import research_agent
from llmkglitrev.agents.character_agent import CharacterBasedResearchAgent  # NEW
from llmkglitrev.characters import (  # NEW
    CharacterManager,
    ResearchCharacter,
    ConversationArtifact,
    ConversationArtifactManager
)
from llmkglitrev.agents.tools import supervisor_think_tool, get_today_str
from llmkglitrev.agents.prompts.supervisor_prompt import lead_researcher_prompt
from langgraph.types import Command
import asyncio
from langgraph.checkpoint.memory import MemorySaver
import uuid


def get_proposals_from_tool_calls(messages: List[BaseMessage]) -> List[str]:
    """
        Collect research findings from sub-agents. 
        When the supervisor delegates research to
        sub-agents via ConductResearch tool calls, each sub-agent returns its
        compressed findings as the content of a ToolMessage. This function
        extracts all such ToolMessage content to compile the final research notes.
        Args:        
            messages: List of messages from supervisor's conversation history
        Returns:
            List of research note strings extracted from ToolMessage objects
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]



# Ensure async compatibility for Jupyter environments
try:
    import nest_asyncio
    # Only apply if running in Jupyter/IPython environment
    try:
        from IPython.core.getipython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not in Jupyter, no need for nest_asyncio
except ImportError:
    pass  # nest_asyncio not available, proceed without it



# ===== CONFIGURATION =====

supervisor_tools = [ConductResearch, ResearchComplete, supervisor_think_tool]
# supervisor_model = init_chat_model(model="anthropic:claude-sonnet-4-20250514")
supervisor_model = init_chat_model(model="deepseek:deepseek-chat")
# supervisor_model = init_chat_model(model="openai:gpt-4o")
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

# System constants
# Maximum number of tool call iterations for individual researcher agents
# This prevents infinite loops and controls research depth per topic
max_researcher_iterations = 2 # Calls to supervisor_think_tool + ConductResearch

# Maximum number of concurrent research agents the supervisor can launch
# This is passed to the lead_researcher_prompt to limit parallel research tasks
max_concurrent_researchers = 2

# ===== SUPERVISOR NODES =====

async def supervisor(state: SupervisorState) -> Command[Literal["get_human_feedback", "supervisor_tools"]]:
    print("🔍 SUPERVISOR node executing")
    supervisor_messages = state.get("supervisor_messages", [])
    got_human_feedback = state.get("got_human_feedback", False)
    research_iterations = state.get("research_iterations", 0)

    print(f"   📊 got_human_feedback: {got_human_feedback}")
    print(f"   📊 research_iterations: {research_iterations}")
    print(f"   📊 supervisor_messages count: {len(supervisor_messages)}")

    # Get literature context if available
    literature_context = state.get("literature_context", "")
    if not literature_context:
        literature_context = "No literature database available."

    # Prepare system message with current date and constraints
    system_message = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations,
        literature_context=literature_context
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    print(f"   🤖 Calling supervisor LLM...")
    response = await supervisor_model_with_tools.ainvoke(input=messages)
    print(f"   ✅ LLM response received")

    pending_proposals = []

    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"   🔧 Response has {len(response.tool_calls)} tool calls")
        for i, tool_call in enumerate(response.tool_calls):
            print(f"      {i+1}. {tool_call['name']}")
            if tool_call["name"] == "supervisor_think_tool":
                pending_proposals.append(tool_call["args"].get("reflection"))  # or "thinking" - check your tool's arg name
    else:
        print(f"   ❌ Response has NO tool calls")

    # Print for CLI/Jupyter
    if not state.get("got_human_feedback") and pending_proposals:
        print("\n" + "="*70)
        print("🔍 PROPOSED RESEARCH DIRECTIONS:")
        print("="*70)
        for i, topic in enumerate(pending_proposals, 1):
            print(f"\n{i}. {topic}")
        print("\n" + "="*70)


    if state.get("got_human_feedback"):
        print(f"   ➡️  Decision: goto supervisor_tools (feedback already received)")
        return Command(
            goto="supervisor_tools",
            update={
                "supervisor_messages": [response],
                "research_iterations": state.get("research_iterations", 0) + 1
            }
        )
    else:
        print(f"   ➡️  Decision: goto get_human_feedback (need feedback)")
        return Command(
            goto="get_human_feedback",
            update={
                "supervisor_messages": [response],
                "pending_proposals": pending_proposals  # Store for Streamlit
            }
        )

async def get_human_feedback(state: SupervisorState) -> Command[Literal["supervisor"]]:
    """Incorporate human feedback into the supervisor's decision-making loop.

    Note: This node is preceded by an interrupt (interrupt_before configuration).
    The graph pauses BEFORE this node executes, allowing Streamlit to detect the interrupt
    and present the UI to the user. When the user provides feedback, the workflow resumes
    with Command(resume={"human_feedback": "..."}), and then this node executes.

    This node allows a human user to review the supervisor's latest decisions
    and provide feedback or corrections before proceeding.

    Args:
        state: Current supervisor state with messages and iteration count

    Returns:
        Command to supervisor with instruction incorporated with human feedback.

    """
    print("💬 GET_HUMAN_FEEDBACK node executing")
    supervisor_messages = state.get("supervisor_messages", [])
    most_recent_message = supervisor_messages[-1]
    tool_calls = most_recent_message.tool_calls if hasattr(most_recent_message, 'tool_calls') else []
    
    # Get pending proposals from state
    pending_proposals = state.get("pending_proposals", [])

    # Get feedback from resume Command (graph pauses before this node)
    # workflow_utils.py will detect interrupt and present UI to user
    # When user provides feedback, it resumes with: Command(resume={"human_feedback": "..."})
    # If no feedback provided (e.g., CLI auto-approve), use default approval
    feedback = state.get("human_feedback", "Looks good, proceed with the proposed research directions.")
    
    # Create tool messages with the human feedback
    feedback_messages = [
        ToolMessage(
            content=f"Human feedback: {feedback}",
            tool_call_id=tool_call["id"],
            name=tool_call["name"]
        )
        for tool_call in tool_calls
    ]
    
    return Command(
        goto="supervisor",
        update={
            "supervisor_messages": feedback_messages,
            "got_human_feedback": True
        }
    )


async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """Execute supervisor decisions - either conduct research or end the process.

    This is the critical node where character-based research agents are instantiated
    and parallel research is conducted. Artifacts are generated here.

    Handles:
    - Executing supervisor_think_tool calls for strategic reflection
    - Launching parallel CHARACTER-BASED research agents for different topics
    - Aggregating research results and conversation artifacts
    - Determining when research is complete

    Args:
        state: Current supervisor state with messages and iteration count

    Returns:
        Command to continue supervision, end process, or handle errors
    """
    print("🔧 SUPERVISOR_TOOLS node executing")
    supervisor_messages = state.get("supervisor_messages", [])
    research_proposals = state.get("research_proposals", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # NEW: Get character and session info
    session_id = state.get("session_id", str(uuid.uuid4()))
    active_characters = state.get("active_characters", [])

    print(f"   📊 research_iterations: {research_iterations}/{max_researcher_iterations}")
    print(f"   👥 active_characters count: {len(active_characters)}")
    print(f"   📝 most_recent_message type: {type(most_recent_message).__name__}")

    # Initialize variables for single return pattern
    tool_messages = []
    all_raw_notes = []
    # CRITICAL: Load existing artifacts from state and append to them (don't reset!)
    conversation_artifacts = state.get("conversation_artifacts", [])
    print(f"   📦 Existing artifacts in state: {len(conversation_artifacts)}")
    next_step = "supervisor"  # Default next step
    should_end = False

    # Check exit criteria first
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    print(f"   ⚠️  Exit criteria check:")
    print(f"      - exceeded_iterations: {exceeded_iterations}")
    print(f"      - no_tool_calls: {no_tool_calls}")
    print(f"      - research_complete: {research_complete}")

    if most_recent_message.tool_calls:
        print(f"   🔧 Tool calls found: {len(most_recent_message.tool_calls)}")
        for i, tc in enumerate(most_recent_message.tool_calls):
            print(f"      {i+1}. {tc['name']}")
    else:
        print(f"   ❌ NO TOOL CALLS in most_recent_message")

    if exceeded_iterations or no_tool_calls or research_complete:
        should_end = True
        next_step = END

    else:
        # Execute ALL tool calls before deciding next step
        try:
            # Separate think_tool calls from ConductResearch calls
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "supervisor_think_tool"
            ]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            # Handle think_tool calls (synchronous)
            for tool_call in think_tool_calls:
                observation = supervisor_think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # Handle ConductResearch calls (asynchronous)
            if conduct_research_calls:
                print(f"   🔬 Found {len(conduct_research_calls)} ConductResearch calls")

                # NEW: Use character-based agents if available, otherwise fall back to generic
                if active_characters and len(active_characters) > 0:
                    print(f"   ✅ Using character-based agents ({len(active_characters)} characters available)")

                    # Instantiate character-based agents
                    character_agents = []

                    for i, tool_call in enumerate(conduct_research_calls):
                        # Get character for this agent (cycle through if more calls than characters)
                        char_dict = active_characters[i % len(active_characters)]

                        # Extract topic and seed papers (NEW)
                        assigned_topic = char_dict.get("assigned_topic", "")
                        seed_papers = char_dict.get("seed_papers", [])

                        # Convert dict back to ResearchCharacter
                        character = ResearchCharacter.model_validate(char_dict)
                        print(f"      📌 Creating agent for {character.name} ({character.domain})")
                        if assigned_topic:
                            print(f"         🎯 Topic: {assigned_topic}")
                            print(f"         📄 Seed papers: {len(seed_papers)}")

                        # Partition literature for this domain
                        retrieved_papers = state.get("retrieved_papers", [])
                        literature_subset = [
                            paper for paper in retrieved_papers
                            if character.domain.lower() in paper.get("title", "").lower()
                            or character.domain.lower() in paper.get("abstract", "").lower()
                        ]

                        # Create character agent with topic and seed papers (NEW)
                        agent = CharacterBasedResearchAgent(
                            character=character,
                            literature_subset=literature_subset,
                            session_id=session_id,
                            assigned_topic=assigned_topic,
                            seed_papers=seed_papers
                        )

                        character_agents.append((agent, tool_call))

                    # Launch parallel research with character agents
                    print(f"   🚀 Launching {len(character_agents)} parallel research tasks...")
                    coros = [
                        agent.conduct_research(tool_call["args"]["research_topic"])
                        for agent, tool_call in character_agents
                    ]

                    # Wait for all research to complete
                    print(f"   ⏳ Waiting for research to complete (this may take several minutes)...")
                    artifacts = await asyncio.gather(*coros)
                    print(f"   ✅ Research complete! Generated {len(artifacts)} artifacts")

                    # Format research results as tool messages
                    research_tool_messages = [
                        ToolMessage(
                            content=artifact.research_output,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        ) for artifact, (_, tool_call) in zip(artifacts, character_agents)
                    ]

                    tool_messages.extend(research_tool_messages)

                    # Store artifacts (as dicts for JSON serialization)
                    # CRITICAL: EXTEND existing artifacts, don't replace!
                    new_artifacts = [artifact.model_dump() for artifact in artifacts]
                    conversation_artifacts.extend(new_artifacts)
                    print(f"   📦 Added {len(new_artifacts)} artifacts. Total now: {len(conversation_artifacts)}")

                    # Aggregate raw notes from artifacts
                    all_raw_notes = [
                        "\n".join(artifact.raw_notes)
                        for artifact in artifacts
                    ]

                else:
                    # FALLBACK: Use generic research agents
                    coros = [
                        research_agent.ainvoke({
                            "researcher_messages": [
                                HumanMessage(content=tool_call["args"]["research_topic"])
                            ],
                            "research_topic": tool_call["args"]["research_topic"],
                            "tool_call_iterations": 0,
                            "maximum_number_of_plan": 3,
                            "compressed_research": "",
                            "raw_notes": [],
                            "dialogue_notes": [],
                            "character_system_prompt": None
                        })
                        for tool_call in conduct_research_calls
                    ]

                    # Wait for all research to complete
                    tool_results = await asyncio.gather(*coros)

                    # Format research results as tool messages
                    research_tool_messages = [
                        ToolMessage(
                            content=result.get("research_plan", "Error synthesizing research proposition"),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        ) for result, tool_call in zip(tool_results, conduct_research_calls)
                    ]

                    tool_messages.extend(research_tool_messages)

                    # Aggregate raw notes from all research
                    all_raw_notes = [
                        "\n".join(result.get("raw_notes", []))
                        for result in tool_results
                    ]

        except Exception as e:
            print(f"Error in supervisor tools: {e}")
            import traceback
            traceback.print_exc()
            should_end = True
            next_step = END

    # Single return point with appropriate state updates
    if should_end:
        print(f"   🏁 ENDING supervisor loop. Generated {len(conversation_artifacts)} artifacts")
        print(f"      goto: {next_step}")
        return Command(
            goto=next_step,
            update={
                "research_proposals": get_proposals_from_tool_calls(supervisor_messages),
                "conversation_artifacts": conversation_artifacts  # NEW
            }
        )
    else:
        print(f"   🔄 Continuing supervisor loop. Generated {len(conversation_artifacts)} artifacts so far")
        print(f"      goto: {next_step}")
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes,
                "conversation_artifacts": conversation_artifacts  # NEW
            }
        )

# ===== GRAPH CONSTRUCTION =====
def create_interactive_supervisor():
    # Build supervisor graph
    supervisor_builder = StateGraph(SupervisorState)
    supervisor_builder.add_node("supervisor", supervisor)
    supervisor_builder.add_node("supervisor_tools", supervisor_tools)
    supervisor_builder.add_node("get_human_feedback", get_human_feedback)
    supervisor_builder.add_edge(START, "supervisor")
    supervisor_agent = supervisor_builder.compile(
        checkpointer=MemorySaver()
        # Note: interrupt_before removed - supervisor feedback now auto-approves
        # This allows research to execute without additional user intervention
    )
    return supervisor_agent



