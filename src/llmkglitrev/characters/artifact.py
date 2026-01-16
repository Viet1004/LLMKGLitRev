"""Conversation artifact schemas and management.

This module defines ephemeral conversation artifacts that are separate from
reusable character configurations. Artifacts store research outputs, dialogue
notes, and conversation history for a specific research session.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional
from datetime import datetime
from pathlib import Path
import json


class DialogueNote(BaseModel):
    """
    A note that warrants follow-up in Socratic dialogue.

    Dialogue notes are automatically extracted during research and represent
    topics that deserve deeper exploration through questioning.
    """

    type: Literal[
        "assumption",
        "methodological_choice",
        "limitation",
        "alternative_approach",
        "gap_identified"
    ] = Field(
        description="Type of dialogue note"
    )

    content: str = Field(
        description="The specific content worth discussing"
    )

    suggested_question: str = Field(
        description="Suggested follow-up question for Socratic dialogue"
    )

    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Question priority (1=low, 10=high)"
    )

    context: str = Field(
        description="Context from research where this arose"
    )

    def __hash__(self):
        """Make DialogueNote hashable for set operations."""
        return hash((self.type, self.content[:50], self.priority))

    def __eq__(self, other):
        """Check equality based on type and content."""
        if not isinstance(other, DialogueNote):
            return False
        return (self.type == other.type and
                self.content[:50] == other.content[:50])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "assumption",
                    "content": "We assume that the training data is representative",
                    "suggested_question": "What evidence supports this assumption about data representativeness?",
                    "priority": 7,
                    "context": "During discussion of dataset selection"
                }
            ]
        }
    }


class ConversationArtifact(BaseModel):
    """
    Ephemeral conversation state for a research agent in a session.

    Unlike ResearchCharacter which is reusable, ConversationArtifact is
    session-specific and contains the actual research output and dialogue
    preparation from a particular research session.
    """

    # Identity
    session_id: str = Field(
        description="Unique session identifier"
    )

    character_id: str = Field(
        description="ID of the character that produced this artifact"
    )

    domain: str = Field(
        description="Research domain for this artifact"
    )

    # Research Output (for supervisor synthesis)
    research_output: str = Field(
        default="",
        description="Compressed research findings for supervisor"
    )

    raw_notes: List[str] = Field(
        default_factory=list,
        description="Detailed research notes"
    )

    # Dialogue Preparation
    dialogue_notes: List[DialogueNote] = Field(
        default_factory=list,
        description="Notes that warrant follow-up questions in Socratic dialogue"
    )

    # Literature
    papers_consulted: List[str] = Field(
        default_factory=list,
        description="Papers/sources used during research"
    )

    # Socratic Dialogue State
    questions_asked: List[str] = Field(
        default_factory=list,
        description="Questions that have been asked to this character"
    )

    questions_answered: List[Dict] = Field(
        default_factory=list,
        description="Question-answer pairs from Socratic dialogue"
    )

    dialogue_active: bool = Field(
        default=False,
        description="Whether dialogue is currently active for this artifact"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the artifact was created"
    )

    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="When the artifact was last updated"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "session-123",
                    "character_id": "ml_expert_critical",
                    "domain": "Machine Learning",
                    "research_output": "Research findings on deep learning for medical imaging...",
                    "raw_notes": ["Found 5 relevant papers", "Key finding: transfer learning helps"],
                    "dialogue_notes": []
                }
            ]
        }
    }


class ConversationArtifactManager:
    """
    Manage conversation artifacts for research sessions.

    Handles creating, loading, and querying conversation artifacts which are
    stored separately from reusable character configurations.
    """

    def __init__(self, storage_path: str = "sessions/"):
        """
        Initialize artifact manager.

        Args:
            storage_path: Base directory for session storage
        """
        self.storage_path = Path(storage_path)

    def save_artifact(
        self,
        session_id: str,
        artifact: ConversationArtifact
    ):
        """
        Save conversation artifact for a session.

        Args:
            session_id: The session ID
            artifact: The artifact to save
        """
        session_path = self.storage_path / session_id / "conversations"
        session_path.mkdir(parents=True, exist_ok=True)

        artifact.last_updated = datetime.now()

        file_path = session_path / f"{artifact.character_id}.json"
        with open(file_path, 'w') as f:
            f.write(artifact.model_dump_json(indent=2))

    def load_artifact(
        self,
        session_id: str,
        character_id: str
    ) -> Optional[ConversationArtifact]:
        """
        Load a specific artifact from a session.

        Args:
            session_id: The session ID
            character_id: The character ID

        Returns:
            The artifact, or None if not found
        """
        file_path = self.storage_path / session_id / "conversations" / f"{character_id}.json"

        if not file_path.exists():
            return None

        with open(file_path) as f:
            return ConversationArtifact.model_validate_json(f.read())

    def load_artifacts(
        self,
        session_id: str
    ) -> Dict[str, ConversationArtifact]:
        """
        Load all conversation artifacts for a session.

        Args:
            session_id: The session ID

        Returns:
            Dictionary mapping character_id to artifact
        """
        session_path = self.storage_path / session_id / "conversations"

        if not session_path.exists():
            return {}

        artifacts = {}
        for file_path in session_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    artifact = ConversationArtifact.model_validate_json(f.read())
                    artifacts[artifact.character_id] = artifact
            except Exception as e:
                print(f"Warning: Could not load artifact from {file_path}: {e}")

        return artifacts

    def get_dialogue_notes(
        self,
        session_id: str,
        character_id: Optional[str] = None,
        min_priority: int = 1
    ) -> List[DialogueNote]:
        """
        Get all dialogue notes from session, sorted by priority.

        Args:
            session_id: The session ID
            character_id: Optional filter by character ID
            min_priority: Minimum priority threshold

        Returns:
            List of dialogue notes sorted by priority (highest first)
        """
        artifacts = self.load_artifacts(session_id)

        all_notes = []
        for char_id, artifact in artifacts.items():
            if character_id is None or char_id == character_id:
                all_notes.extend(artifact.dialogue_notes)

        # Filter by priority
        filtered_notes = [note for note in all_notes if note.priority >= min_priority]

        # Sort by priority (highest first)
        filtered_notes.sort(key=lambda x: x.priority, reverse=True)

        return filtered_notes

    def session_exists(self, session_id: str) -> bool:
        """
        Check if session exists.

        Args:
            session_id: The session ID

        Returns:
            True if session directory exists
        """
        session_path = self.storage_path / session_id
        return session_path.exists()

    def list_sessions(self) -> List[Dict]:
        """
        List all research sessions.

        Returns:
            List of session metadata
        """
        sessions = []

        if not self.storage_path.exists():
            return sessions

        for session_path in self.storage_path.iterdir():
            if session_path.is_dir():
                # Count artifacts
                conversations_path = session_path / "conversations"
                artifact_count = 0
                if conversations_path.exists():
                    artifact_count = len(list(conversations_path.glob("*.json")))

                # Get session metadata
                metadata_path = session_path / "metadata.json"
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                sessions.append({
                    "session_id": session_path.name,
                    "artifact_count": artifact_count,
                    "created_at": metadata.get("created_at"),
                    "topic": metadata.get("topic", "Unknown")
                })

        return sessions

    def save_session_metadata(
        self,
        session_id: str,
        metadata: Dict
    ):
        """
        Save session metadata.

        Args:
            session_id: The session ID
            metadata: Metadata dictionary
        """
        session_path = self.storage_path / session_id
        session_path.mkdir(parents=True, exist_ok=True)

        metadata_path = session_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def delete_session(self, session_id: str):
        """
        Delete entire session and all its artifacts.

        Args:
            session_id: The session ID
        """
        import shutil
        session_path = self.storage_path / session_id

        if session_path.exists():
            shutil.rmtree(session_path)
