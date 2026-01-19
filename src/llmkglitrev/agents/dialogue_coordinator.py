"""Socratic Dialogue Coordinator for iterative Q&A with research characters.

This module implements an iterative Socratic dialogue system where:
1. High-priority dialogue notes are extracted from research artifacts
2. Questions are asked one at a time (not all at once)
3. Character agents answer using their domain expertise
4. User provides feedback after each answer
5. Process continues until all questions answered or user stops
"""

from typing import List, Dict, Optional, Literal
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
import operator

from llmkglitrev.characters.artifact import DialogueNote, ConversationArtifact
from llmkglitrev.characters.schema import ResearchCharacter
from llmkglitrev.agents.character_agent import CharacterBasedResearchAgent


# ===== DIALOGUE STATE =====

class DialogueQA(BaseModel):
    """Single question-answer exchange in the dialogue."""
    question: str = Field(description="The question asked")
    character_id: str = Field(description="Character who answered")
    character_name: str = Field(description="Name of character")
    answer: str = Field(description="Character's answer")
    user_feedback: Optional[str] = Field(default=None, description="User's feedback on the answer")
    note_type: str = Field(description="Type of dialogue note this question came from")
    priority: int = Field(description="Priority of the original note (1-10)")


class DialogueState(TypedDict):
    """State for the Socratic dialogue coordinator."""
    # Session and character info
    session_id: str
    conversation_artifacts: List[dict]  # List of ConversationArtifact dicts

    # Dialogue notes to explore
    pending_notes: List[dict]  # List of DialogueNote dicts (not yet asked)
    current_note: Optional[dict]  # Current DialogueNote being discussed

    # Q&A history
    dialogue_history: Annotated[List[dict], operator.add]  # List of DialogueQA dicts

    # Current question/answer state
    current_question: str
    current_answer: str
    waiting_for_feedback: bool

    # Control flags
    dialogue_complete: bool
    user_requested_stop: bool
    min_priority: int  # Minimum priority threshold for questions


# ===== DIALOGUE COORDINATOR NODES =====

async def prepare_dialogue(state: DialogueState) -> Command[Literal["ask_question"]]:
    """Prepare dialogue by extracting high-priority notes from artifacts.

    This node:
    1. Loads conversation artifacts
    2. Extracts dialogue notes from all characters
    3. Filters by minimum priority
    4. Sorts by priority (highest first)
    5. Initializes dialogue state
    """
    conversation_artifacts = state.get("conversation_artifacts", [])
    min_priority = state.get("min_priority", 7)  # Default: only high-priority notes

    # Extract all dialogue notes from artifacts
    all_notes = []
    for artifact_dict in conversation_artifacts:
        artifact = ConversationArtifact.model_validate(artifact_dict)

        # Get dialogue notes and convert to dicts
        for note in artifact.dialogue_notes:
            if note.priority >= min_priority:
                note_dict = note.model_dump()
                # Add character context for better questions
                note_dict["character_id"] = artifact.character_id
                note_dict["character_name"] = artifact_dict.get("character_id", "Unknown")
                note_dict["domain"] = artifact.domain
                all_notes.append(note_dict)

    # Sort by priority (highest first)
    all_notes.sort(key=lambda x: x["priority"], reverse=True)

    print(f"\n{'='*70}")
    print(f"📚 DIALOGUE PREPARATION")
    print(f"{'='*70}")
    print(f"Found {len(all_notes)} high-priority dialogue notes (min_priority={min_priority})")

    if all_notes:
        print(f"\nTop 3 questions to explore:")
        for i, note in enumerate(all_notes[:3], 1):
            print(f"{i}. [{note['type']}] Priority {note['priority']}")
            print(f"   {note['suggested_question']}")
    else:
        print("No high-priority dialogue notes found. Dialogue will end.")

    print(f"{'='*70}\n")

    # Check if dialogue should end immediately
    if not all_notes:
        return Command(
            goto=END,
            update={
                "pending_notes": [],
                "dialogue_complete": True
            }
        )

    return Command(
        goto="ask_question",
        update={
            "pending_notes": all_notes,
            "dialogue_complete": False,
            "user_requested_stop": False,
            "waiting_for_feedback": False
        }
    )


async def ask_question(state: DialogueState) -> Command[Literal["get_character_answer", "__end__"]]:
    """Ask the next high-priority question.

    This node:
    1. Gets the next pending note
    2. Formats the question
    3. Moves to character answer phase
    """
    pending_notes = state.get("pending_notes", [])

    # Check if dialogue is complete
    if not pending_notes or state.get("user_requested_stop", False):
        print("\n✅ Dialogue complete - no more questions to explore")
        return Command(
            goto=END,
            update={"dialogue_complete": True}
        )

    # Get next note
    current_note = pending_notes[0]
    remaining_notes = pending_notes[1:]

    question = current_note["suggested_question"]

    print(f"\n{'='*70}")
    print(f"❓ QUESTION {len(state.get('dialogue_history', [])) + 1}")
    print(f"{'='*70}")
    print(f"Type: {current_note['type']}")
    print(f"Priority: {current_note['priority']}")
    print(f"Domain: {current_note.get('domain', 'General')}")
    print(f"\nQuestion: {question}")
    print(f"{'='*70}\n")

    return Command(
        goto="get_character_answer",
        update={
            "current_note": current_note,
            "current_question": question,
            "pending_notes": remaining_notes
        }
    )


async def get_character_answer(state: DialogueState) -> Command[Literal["get_user_feedback"]]:
    """Get answer from the appropriate character agent.

    This node:
    1. Identifies which character should answer
    2. Loads the character's artifact
    3. Uses the character agent to answer the question
    4. Stores the answer
    """
    current_note = state.get("current_note")
    current_question = state.get("current_question")
    conversation_artifacts = state.get("conversation_artifacts", [])

    # Find the artifact for this character
    character_id = current_note.get("character_id")
    artifact_dict = next(
        (a for a in conversation_artifacts if a["character_id"] == character_id),
        None
    )

    if not artifact_dict:
        # Fallback: use first artifact
        artifact_dict = conversation_artifacts[0] if conversation_artifacts else None

    if not artifact_dict:
        print("⚠️  No artifacts available for answering")
        answer = "Unable to generate answer - no character artifacts available."
    else:
        # Load character and create agent
        artifact = ConversationArtifact.model_validate(artifact_dict)

        # Get character from artifact
        character_id = artifact.character_id

        # Import here to avoid circular dependency
        from llmkglitrev.characters import CharacterManager
        manager = CharacterManager()
        character = manager.load_character(character_id)

        # Create agent
        agent = CharacterBasedResearchAgent(
            character=character,
            session_id=state.get("session_id"),
            literature_subset=[]
        )

        # Restore artifact to agent
        agent.artifact = artifact

        # Get answer from character
        print(f"🤖 Getting answer from {character.name}...")

        context = {
            "research_output": artifact.research_output,
            "previous_dialogue": state.get("dialogue_history", [])
        }

        answer = await agent.answer_question(current_question, context=context)

        print(f"\n💬 {character.name}'s Answer:")
        print(f"{'='*70}")
        print(answer)
        print(f"{'='*70}\n")

    return Command(
        goto="get_user_feedback",
        update={
            "current_answer": answer,
            "waiting_for_feedback": True
        }
    )


async def get_user_feedback(state: DialogueState) -> Command[Literal["record_dialogue", "__end__"]]:
    """Get user feedback on the character's answer.

    This node:
    1. Presents the question and answer to user
    2. Interrupts for user feedback
    3. Allows user to continue or stop dialogue
    """
    current_question = state.get("current_question")
    current_answer = state.get("current_answer")
    current_note = state.get("current_note")
    pending_notes = state.get("pending_notes", [])

    # Create interrupt payload
    interrupt_payload = {
        "type": "dialogue_feedback_request",
        "question": current_question,
        "answer": current_answer,
        "note_type": current_note.get("type"),
        "priority": current_note.get("priority"),
        "remaining_questions": len(pending_notes),
        "instructions": (
            "Please provide your feedback on this answer. "
            "You can:\n"
            "- Provide specific feedback or follow-up thoughts\n"
            "- Type 'continue' to move to the next question\n"
            "- Type 'stop' to end the dialogue"
        )
    }

    # Interrupt and wait for user feedback
    feedback = interrupt(interrupt_payload)

    # Check if user wants to stop
    user_wants_to_stop = False
    if isinstance(feedback, str):
        feedback_lower = feedback.lower().strip()
        if feedback_lower in ["stop", "end", "quit", "exit", "done"]:
            user_wants_to_stop = True
            print("\n🛑 User requested to stop dialogue")

    if user_wants_to_stop:
        return Command(
            goto=END,
            update={
                "user_requested_stop": True,
                "dialogue_complete": True,
                "waiting_for_feedback": False
            }
        )

    return Command(
        goto="record_dialogue",
        update={
            "current_feedback": feedback,
            "waiting_for_feedback": False
        }
    )


async def record_dialogue(state: DialogueState) -> Command[Literal["ask_question"]]:
    """Record the question-answer exchange in dialogue history.

    This node:
    1. Creates a DialogueQA object
    2. Adds it to dialogue history
    3. Returns to ask next question
    """
    current_note = state.get("current_note")
    current_question = state.get("current_question")
    current_answer = state.get("current_answer")
    current_feedback = state.get("current_feedback", "")

    # Create dialogue QA record
    qa_record = DialogueQA(
        question=current_question,
        character_id=current_note.get("character_id"),
        character_name=current_note.get("character_name", "Unknown"),
        answer=current_answer,
        user_feedback=current_feedback if current_feedback else None,
        note_type=current_note.get("type"),
        priority=current_note.get("priority")
    )

    print(f"✅ Recorded dialogue exchange (Question {len(state.get('dialogue_history', [])) + 1})\n")

    return Command(
        goto="ask_question",
        update={
            "dialogue_history": [qa_record.model_dump()]
        }
    )


# ===== DIALOGUE GRAPH CONSTRUCTION =====

def create_dialogue_coordinator(min_priority: int = 7) -> StateGraph:
    """Create the Socratic dialogue coordinator graph.

    Args:
        min_priority: Minimum priority threshold for dialogue notes (1-10)

    Returns:
        Compiled StateGraph for dialogue coordination
    """
    builder = StateGraph(DialogueState)

    # Add nodes
    builder.add_node("prepare_dialogue", prepare_dialogue)
    builder.add_node("ask_question", ask_question)
    builder.add_node("get_character_answer", get_character_answer)
    builder.add_node("get_user_feedback", get_user_feedback)
    builder.add_node("record_dialogue", record_dialogue)

    # Add edges
    builder.add_edge(START, "prepare_dialogue")
    # Edges are defined via Command returns in nodes

    # Compile with checkpointer for interrupt support
    return builder.compile(checkpointer=MemorySaver())


# ===== CONVENIENCE FUNCTIONS =====

async def run_dialogue(
    session_id: str,
    conversation_artifacts: List[ConversationArtifact],
    min_priority: int = 7,
    thread_id: Optional[str] = None
) -> List[DialogueQA]:
    """Run Socratic dialogue with character agents.

    Args:
        session_id: Session identifier
        conversation_artifacts: List of conversation artifacts from research
        min_priority: Minimum priority for questions (default: 7)
        thread_id: Optional thread ID for resuming

    Returns:
        List of DialogueQA records
    """
    # Create dialogue coordinator
    coordinator = create_dialogue_coordinator(min_priority=min_priority)

    # Initial state
    initial_state = {
        "session_id": session_id,
        "conversation_artifacts": [a.model_dump() for a in conversation_artifacts],
        "pending_notes": [],
        "current_note": None,
        "dialogue_history": [],
        "current_question": "",
        "current_answer": "",
        "waiting_for_feedback": False,
        "dialogue_complete": False,
        "user_requested_stop": False,
        "min_priority": min_priority
    }

    # Configuration
    config = {"configurable": {"thread_id": thread_id or f"dialogue-{session_id}"}}

    # Run dialogue
    result = await coordinator.ainvoke(initial_state, config=config)

    # Extract dialogue history
    dialogue_history = result.get("dialogue_history", [])
    return [DialogueQA.model_validate(qa) for qa in dialogue_history]
