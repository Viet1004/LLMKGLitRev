"""
Workflow Utilities for Streamlit Integration

This module provides helper functions to bridge async agent operations
with Streamlit's synchronous interface, handle LangGraph interrupts,
and manage workflow state.
"""

from typing import Dict, List, Optional, Any, AsyncIterator
from pathlib import Path
import asyncio
import json
from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from llmkglitrev.agents.research_proposal_generator import (
    format_question,
    broad_literature_search,
    proposal_generator_agent
)
from llmkglitrev.characters import (
    CharacterManager,
    ConversationArtifactManager,
    ResearchCharacter
)


# ===== WORKFLOW ORCHESTRATION =====

async def initialize_research_session(
    research_topic: str,
    session_id: str,
    enable_literature_search: bool = True,
    top_k_papers: int = 10
) -> Dict[str, Any]:
    """
    Initialize a research session with user input.

    Args:
        research_topic: User's research question
        session_id: Unique session identifier
        enable_literature_search: Whether to perform broad literature search (arXiv)
        top_k_papers: Number of papers to retrieve (not used anymore, kept for backward compatibility)

    Returns:
        Dictionary with keywords, literature context, topics, and initial state
    """
    # Create initial state
    state: Dict[str, Any] = {
        "messages": [HumanMessage(content=research_topic)],
        "session_id": session_id
    }

    # Extract keywords
    keyword_result = await format_question(state)  # type: ignore

    # Merge keyword results into state
    state.update(keyword_result)

    # Retrieve literature if enabled (using broad search with arXiv)
    if enable_literature_search:
        try:
            lit_result = await broad_literature_search(state)  # type: ignore
            state.update(lit_result)
        except Exception as e:
            print(f"⚠️  Literature search failed: {e}")
            state["retrieved_papers"] = []
            state["literature_context"] = "Literature database unavailable."
            state["broad_papers"] = []
            state["topics"] = []
            state["topic_papers"] = {}

    return state


async def generate_research_plan(
    research_topic: str,
    thread_id: str
) -> Optional[Dict[str, Any]]:
    """
    Generate research plan and return when interrupt is reached.

    Args:
        research_topic: User's research question
        thread_id: LangGraph thread ID for conversation

    Returns:
        Proposed research plan or None if error
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50  # Increased from default 25 to handle nested agent loops
    }

    # Use the pre-compiled agent graph
    agent = proposal_generator_agent

    try:
        # Initial invocation - this will run until interrupt_before process_plan_approval
        await agent.ainvoke(
            {"messages": [HumanMessage(content=research_topic)]},
            config=config
        )

        # Get current state to check if interrupted
        agent_state = agent.get_state(config)

        # Check if interrupted before process_plan_approval
        if agent_state.next and "process_plan_approval" in agent_state.next:
            # Extract proposed plan from state
            proposed_plan = agent_state.values.get("proposed_research_plan")
            if proposed_plan:
                print(f"✅ Interrupted at plan approval. Plan has {len(proposed_plan.get('proposed_agents', []))} agents")
                return proposed_plan
            else:
                print("⚠️  Warning: Interrupted but no plan in state")
                return None

        print("⚠️  Warning: Did not reach interrupt point")
        return None

    except Exception as e:
        print(f"❌ Error generating research plan: {e}")
        import traceback
        traceback.print_exc()
        return None


async def resume_after_plan_approval(
    thread_id: str,
    approved: bool,
    modified_plan: Optional[Dict] = None,
    user_feedback: str = ""
) -> Dict[str, Any]:
    """
    Resume workflow after user approves/modifies research plan.

    Args:
        thread_id: LangGraph thread ID
        approved: Whether plan was approved
        modified_plan: User-modified plan (if any)
        user_feedback: User's feedback text

    Returns:
        Updated state after resuming
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50
    }
    agent = proposal_generator_agent

    # Prepare resume data
    resume_data = {
        "plan_approved": approved,
        "user_feedback": user_feedback
    }

    if modified_plan:
        resume_data["proposed_research_plan"] = modified_plan

    # Resume agent with Command
    result = await agent.ainvoke(Command(resume=resume_data), config)

    return result


async def execute_research_with_monitoring(
    thread_id: str,
    resume_with_approval: bool = False,
    modified_plan: Optional[Dict] = None
) -> AsyncIterator[Dict[str, Any]]:
    """
    Execute research and yield progress events.

    This is an async generator that streams research progress events
    for real-time UI updates.

    Args:
        thread_id: LangGraph thread ID
        resume_with_approval: If True, resume from plan approval interrupt first
        modified_plan: Modified plan to use if resuming

    Yields:
        Progress events with status, character info, and actions
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50  # Increased from default 25 to handle nested agent loops
    }
    agent = proposal_generator_agent

    try:
        # If we need to resume from plan approval, do it via streaming
        if resume_with_approval:
            # Debug: Check state BEFORE resume
            state_before = agent.get_state(config)
            original_plan = state_before.values.get("proposed_research_plan", {})
            print(f"\n🔍 DEBUG: State BEFORE resume has {len(original_plan.get('proposed_agents', []))} agents")
            
            # CRITICAL FIX: Use agent.update_state() to modify the checkpoint BEFORE resuming
            # This ensures the modified plan is in the state when process_plan_approval runs
            if modified_plan:
                num_agents = len(modified_plan.get("proposed_agents", []))
                print(f"🔍 DEBUG: Updating state with modified plan ({num_agents} agents)")
                for idx, agent_config in enumerate(modified_plan.get("proposed_agents", []), 1):
                    print(f"   Agent {idx}: {agent_config['domain']} ({agent_config['recommended_character']})")
                
                # Update the state directly to overwrite the checkpoint
                agent.update_state(config, {
                    "proposed_research_plan": modified_plan,
                    "plan_approved": True
                })
                
                print("✅ State updated successfully")
            
            # Now resume execution - the state already has the modified plan
            resume_data = {
                "plan_approved": True,
                "user_feedback": ""
            }

            # Resume and stream the execution
            async for event in agent.astream(Command(resume=resume_data), config, stream_mode="values"):
                # Check for supervisor interrupt
                agent_state = agent.get_state(config)

                if agent_state.next and "get_human_feedback" in agent_state.next:
                    # Supervisor needs feedback
                    pending_proposals = agent_state.values.get("pending_proposals", [])
                    yield {
                        "type": "supervisor_interrupt",
                        "status": "awaiting_feedback",
                        "pending_proposals": pending_proposals
                    }
                    return

                # Extract progress information
                progress_event = _parse_progress_event(event)
                if progress_event:
                    yield progress_event
        else:
            # Continue from where we left off (e.g., after supervisor feedback)
            async for event in agent.astream(None, config, stream_mode="values"):
                # Check for supervisor interrupt
                agent_state = agent.get_state(config)

                if agent_state.next and "get_human_feedback" in agent_state.next:
                    # Supervisor needs feedback
                    pending_proposals = agent_state.values.get("pending_proposals", [])
                    yield {
                        "type": "supervisor_interrupt",
                        "status": "awaiting_feedback",
                        "pending_proposals": pending_proposals
                    }
                    return

                # Extract progress information
                progress_event = _parse_progress_event(event)
                if progress_event:
                    yield progress_event

        # Research complete
        final_state = agent.get_state(config)
        artifacts = final_state.values.get("conversation_artifacts", [])
        final_proposal = final_state.values.get("final_proposal", "")

        print(f"✅ Research complete! Generated {len(artifacts)} artifacts")

        yield {
            "type": "complete",
            "status": "complete",
            "artifacts": artifacts,
            "final_proposal": final_proposal
        }

    except Exception as e:
        print(f"❌ Error in execute_research_with_monitoring: {e}")
        import traceback
        traceback.print_exc()
        yield {
            "type": "error",
            "status": "error",
            "error": str(e)
        }


async def resume_after_supervisor_feedback(
    thread_id: str,
    feedback: str
) -> Dict[str, Any]:
    """
    Resume research after supervisor feedback.

    Args:
        thread_id: LangGraph thread ID
        feedback: User's feedback on research directions

    Returns:
        Updated state after resuming
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50
    }
    agent = proposal_generator_agent

    # Resume with feedback
    result = await agent.ainvoke(Command(resume=feedback), config)

    return result


def _parse_progress_event(event: Dict) -> Optional[Dict]:
    """
    Parse agent event to extract progress information.

    Args:
        event: Raw event from agent stream

    Returns:
        Parsed progress event or None
    """
    # Extract active characters
    active_chars = event.get("active_characters", [])

    # Extract any status messages
    supervisor_messages = event.get("supervisor_messages", [])

    if not active_chars and not supervisor_messages:
        return None

    return {
        "type": "progress",
        "status": "running",
        "active_characters": [
            {
                "character_id": char.get("character_id", "unknown"),
                "name": char.get("name", "Unknown"),
                "domain": char.get("domain", "Unknown")
            }
            for char in active_chars
        ] if active_chars else [],
        "messages": [
            msg.content if hasattr(msg, 'content') else str(msg)
            for msg in supervisor_messages[-3:]  # Last 3 messages
        ]
    }


# ===== CHARACTER MANAGEMENT =====

def get_available_characters() -> List[Dict[str, Any]]:
    """
    Get all available research characters.

    Returns:
        List of character info dictionaries
    """
    manager = CharacterManager()
    return manager.list_characters()


def load_character_details(character_id: str) -> Optional[ResearchCharacter]:
    """
    Load full details for a specific character.

    Args:
        character_id: Character ID to load

    Returns:
        ResearchCharacter object or None if not found
    """
    manager = CharacterManager()
    try:
        return manager.load_character(character_id)
    except Exception as e:
        print(f"⚠️  Error loading character {character_id}: {e}")
        return None


def save_custom_character(character_data: Dict[str, Any]) -> str:
    """
    Save a custom character permanently.

    Args:
        character_data: Character configuration dictionary

    Returns:
        Character ID of saved character
    """
    manager = CharacterManager()

    # Create ResearchCharacter object
    character = ResearchCharacter(**character_data)

    # Save to user_defined directory (use create_character, not save_character)
    manager.create_character(character, is_system=False)

    return character.character_id


def modify_character_in_plan(
    proposed_plan: Dict[str, Any],
    agent_index: int,
    modifications: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Modify a specific agent in the proposed research plan.

    Args:
        proposed_plan: Current research plan
        agent_index: Index of agent to modify
        modifications: Dictionary of fields to update

    Returns:
        Updated research plan
    """
    if "proposed_agents" not in proposed_plan:
        return proposed_plan

    if agent_index >= len(proposed_plan["proposed_agents"]):
        return proposed_plan

    # Update agent configuration
    agent = proposed_plan["proposed_agents"][agent_index]
    agent.update(modifications)

    return proposed_plan


# ===== ARTIFACT MANAGEMENT =====

def load_session_artifacts(session_id: str) -> List[Dict[str, Any]]:
    """
    Load all artifacts for a session from disk.

    NEW: Uses unified format - loads artifacts from character entries

    Args:
        session_id: Session ID to load

    Returns:
        List of artifact dictionaries
    """
    from llmkglitrev.characters.artifact import UnifiedCharacterManager
    unified_manager = UnifiedCharacterManager()
    artifacts = []

    session_path = Path(f"sessions/{session_id}/conversations/")
    if not session_path.exists():
        return []

    try:
        # Load all unified characters
        loaded_characters = unified_manager.load_all_characters(session_id)

        # Extract artifacts (skip characters without artifacts)
        for character_with_artifact in loaded_characters.values():
            if character_with_artifact.research_artifact:
                artifacts.append(character_with_artifact.research_artifact.model_dump())
    except Exception as e:
        print(f"⚠️  Error loading artifacts: {e}")

    return artifacts


def get_session_summary(session_id: str) -> Dict[str, Any]:
    """
    Get summary statistics for a session.

    Args:
        session_id: Session ID

    Returns:
        Dictionary with summary statistics
    """
    artifacts = load_session_artifacts(session_id)

    if not artifacts:
        return {
            "session_id": session_id,
            "artifact_count": 0,
            "total_dialogue_notes": 0,
            "papers_consulted": [],
            "characters_used": []
        }

    all_papers = []
    total_notes = 0
    characters = []

    for artifact in artifacts:
        total_notes += len(artifact.get("dialogue_notes", []))
        all_papers.extend(artifact.get("papers_consulted", []))
        characters.append(artifact.get("character_id", "Unknown"))

    # Deduplicate papers
    unique_papers = list(set(all_papers))

    return {
        "session_id": session_id,
        "artifact_count": len(artifacts),
        "total_dialogue_notes": total_notes,
        "papers_consulted": unique_papers,
        "characters_used": characters
    }


def export_session_to_json(session_id: str, research_topic: str) -> str:
    """
    Export session to JSON string.

    Args:
        session_id: Session ID to export
        research_topic: Original research topic

    Returns:
        JSON string of session data
    """
    artifacts = load_session_artifacts(session_id)
    summary = get_session_summary(session_id)

    export_data = {
        "session_id": session_id,
        "research_topic": research_topic,
        "exported_at": datetime.now().isoformat(),
        "summary": summary,
        "artifacts": artifacts
    }

    return json.dumps(export_data, indent=2)


# ===== DIALOGUE HELPERS =====

def extract_high_priority_dialogue_notes(
    artifacts: List[Dict[str, Any]],
    min_priority: int = 7
) -> List[Dict[str, Any]]:
    """
    Extract high-priority dialogue notes from artifacts.

    Args:
        artifacts: List of conversation artifacts
        min_priority: Minimum priority threshold

    Returns:
        List of dialogue notes sorted by priority
    """
    all_notes = []

    for artifact in artifacts:
        character_id = artifact.get("character_id", "Unknown")
        domain = artifact.get("domain", "Unknown")

        for note in artifact.get("dialogue_notes", []):
            if note.get("priority", 0) >= min_priority:
                note_with_context = {
                    **note,
                    "character_id": character_id,
                    "character_domain": domain
                }
                all_notes.append(note_with_context)

    # Sort by priority (descending)
    all_notes.sort(key=lambda x: x.get("priority", 0), reverse=True)

    return all_notes


# ===== VALIDATION HELPERS =====

def validate_research_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate research plan structure.

    Args:
        plan: Research plan to validate

    Returns:
        Validation result with errors if any
    """
    errors = []
    warnings = []

    # Check required fields
    if "research_strategy" not in plan:
        errors.append("Missing 'research_strategy'")

    if "proposed_agents" not in plan:
        errors.append("Missing 'proposed_agents'")
    elif not isinstance(plan["proposed_agents"], list):
        errors.append("'proposed_agents' must be a list")
    elif len(plan["proposed_agents"]) == 0:
        errors.append("At least one agent must be proposed")
    else:
        # Validate each agent
        for i, agent in enumerate(plan["proposed_agents"]):
            if "domain" not in agent:
                errors.append(f"Agent {i}: Missing 'domain'")
            if "stance" not in agent:
                warnings.append(f"Agent {i}: Missing 'stance', will default to 'neutral'")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def format_elapsed_time(start_time: datetime) -> str:
    """
    Format elapsed time from start.

    Args:
        start_time: Start datetime

    Returns:
        Formatted time string (e.g., "2m 35s")
    """
    elapsed = datetime.now() - start_time
    total_seconds = int(elapsed.total_seconds())

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"
