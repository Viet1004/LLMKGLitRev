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
from llmkglitrev.agents.research_planner import propose_research_plan, process_plan_approval
from llmkglitrev.retrieval.neo4j_search import get_neo4j_search
from typing import Union
from langchain.chat_models import init_chat_model
import uuid
# writer_model = init_chat_model(model="openai:gpt-4o", max_tokens=16000) # model="anthropic:claude-sonnet-4-20250514", max_tokens=64000
writer_model = init_chat_model(model="deepseek:deepseek-chat")

summarize_model = init_chat_model(model="deepseek:deepseek-chat").with_structured_output(KeyWordsList)
async def format_question(state:AgentState):
    """
    Generate research keywords and initialize session.
    """
    query = research_agent_keyword_extractor.format(
        research_prompt=state.get('messages', "")
    )

    keywords = await summarize_model.ainvoke(query)

    # Extract research topic from messages
    messages = state.get('messages', [])
    research_topic = str(messages[0].content) if messages else ""

    # Generate session ID if not present
    session_id = state.get('session_id', '') or str(uuid.uuid4())

    return {
        "supervisor_messages": [HumanMessage(content=f"{state['messages']}.")],
        "research_keywords": keywords.keywords,
        "research_topic": research_topic,
        "session_id": session_id,
        "plan_approved": False,
        "active_characters": [],
        "conversation_artifacts": [],
        "character_configs": []
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

def save_artifacts(state: AgentState) -> dict:
    """
    Save conversation artifacts to disk.

    This node persists artifacts from the supervisor to the session directory
    so they can be accessed later for dialogue or review.
    """
    from llmkglitrev.characters import ConversationArtifactManager, ConversationArtifact

    session_id = state.get("session_id", "")
    conversation_artifacts = state.get("conversation_artifacts", [])

    if not session_id:
        print("\n⚠️  Warning: No session ID found, artifacts not saved")
        return {}

    if not conversation_artifacts:
        print("\n⚠️  Warning: No conversation artifacts to save")
        return {}

    # Initialize artifact manager
    artifact_manager = ConversationArtifactManager()

    print(f"\n💾 Saving {len(conversation_artifacts)} artifacts to sessions/{session_id}/")

    # Save each artifact
    saved_count = 0
    for artifact_dict in conversation_artifacts:
        try:
            # Convert dict back to ConversationArtifact
            artifact = ConversationArtifact.model_validate(artifact_dict)

            # Save to disk
            artifact_manager.save_artifact(session_id, artifact)
            saved_count += 1

            print(f"  ✅ Saved artifact for {artifact.domain} ({artifact.character_id})")

        except Exception as e:
            print(f"  ❌ Error saving artifact: {e}")

    print(f"✅ Saved {saved_count}/{len(conversation_artifacts)} artifacts\n")

    return {}  # No state changes needed


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


def route_after_plan_approval(state: AgentState) -> str:
    """
    Route to supervisor if plan is approved, otherwise loop back to propose_research_plan.
    """
    if state.get("plan_approved", False):
        return "instantiate_agents"
    else:
        # Plan was rejected or needs modification - re-propose
        return "propose_research_plan"


def instantiate_agents(state: AgentState) -> dict:
    """
    Instantiate character-based agents from the approved research plan.

    This converts the approved plan into actual character configurations
    that the supervisor can use to spawn agents.
    """
    from llmkglitrev.characters import CharacterManager, ResearchCharacter

    print("\n" + "="*70)
    print("🎭 INSTANTIATING RESEARCH AGENTS")
    print("="*70)

    character_configs = state.get("character_configs", [])
    char_manager = CharacterManager()

    active_characters = []

    for config in character_configs:
        char_id = config.get("character_id")
        domain = config.get("domain")
        stance = config.get("stance", "neutral")

        print(f"\n📌 Configuring agent for {domain} ({stance})")

        if char_id != "custom":
            # Load existing character template
            try:
                character = char_manager.load_character(char_id)
                print(f"   ✅ Loaded template: {character.name}")
            except Exception as e:
                print(f"   ⚠️  Failed to load {char_id}, using fallback")
                # Create a basic character if template fails
                character = ResearchCharacter(
                    character_id=f"agent_{domain.lower().replace(' ', '_')}",
                    name=f"{domain} Expert",
                    domain=domain,
                    stance=stance,
                    system_prompt_template=f"You are a {domain} expert with a {stance} perspective.",
                    expertise_areas=[domain],
                    research_style=stance
                )
        else:
            # Create custom character from config
            custom_config = config.get("custom_config", {})
            character = ResearchCharacter(
                character_id=custom_config.get("character_id", f"custom_{domain.lower()}"),
                name=custom_config.get("name", f"{domain} Expert"),
                domain=domain,
                stance=stance,
                system_prompt_template=custom_config.get("system_prompt", ""),
                expertise_areas=custom_config.get("expertise_areas", [domain]),
                research_style=custom_config.get("research_style", stance)
            )
            print(f"   ✅ Created custom character: {character.name}")

        # Add to active characters list
        active_characters.append(character.model_dump())

    print(f"\n✅ Instantiated {len(active_characters)} research agents")
    print("="*70)

    return {
        "active_characters": active_characters
    }


supervisor_agent = create_interactive_supervisor()
agent_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add nodes
agent_builder.add_node("format_question", format_question)
agent_builder.add_node("retrieve_literature", retrieve_literature)
agent_builder.add_node("propose_research_plan", propose_research_plan)  # NEW: Propose agents
agent_builder.add_node("process_plan_approval", process_plan_approval)  # NEW: Process approval
agent_builder.add_node("instantiate_agents", instantiate_agents)  # NEW: Create character agents
agent_builder.add_node("supervisor_subgraph", supervisor_agent)
agent_builder.add_node("save_artifacts", save_artifacts)  # NEW: Save artifacts to disk
agent_builder.add_node("final_research_proposal", final_research_proposal)

# Add edges - NEW WORKFLOW:
# 1. Format question and extract keywords
agent_builder.add_edge(START, "format_question")

# 2. Retrieve relevant literature
agent_builder.add_edge("format_question", "retrieve_literature")

# 3. Propose research plan with interrupt for approval
agent_builder.add_edge("retrieve_literature", "propose_research_plan")

# 4. Process approval (runs after interrupt is resumed)
agent_builder.add_edge("propose_research_plan", "process_plan_approval")

# 5. Route based on approval: instantiate agents or re-propose
agent_builder.add_conditional_edges(
    "process_plan_approval",
    route_after_plan_approval,
    {
        "instantiate_agents": "instantiate_agents",
        "propose_research_plan": "propose_research_plan"
    }
)

# 6. Run supervisor with instantiated agents
agent_builder.add_edge("instantiate_agents", "supervisor_subgraph")

# 7. Save conversation artifacts to disk
agent_builder.add_edge("supervisor_subgraph", "save_artifacts")

# 8. Generate final proposal
agent_builder.add_edge("save_artifacts", "final_research_proposal")

# 9. End
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
    