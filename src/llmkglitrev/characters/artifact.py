"""Unified character and artifact management.

This module defines the unified CharacterWithArtifact schema that combines
character configuration with optional research artifacts. All characters and
their research outputs are stored together in session directories.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional
from datetime import datetime
from pathlib import Path
import json

# Import ResearchCharacter from schema
from .schema import ResearchCharacter


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


class CharacterWithArtifact(BaseModel):
    """
    Unified schema combining character configuration and optional research artifact.

    This is the single source of truth for a character in a session. It contains:
    - character_config: The character's identity, domain expertise, and behavior (always present)
    - research_artifact: The research output from this character (optional - only after research)

    Usage:
    - Before research: Save character_config only (research_artifact = None)
    - After research: Update with research_artifact containing research output
    - For dialogue: Load both character_config and research_artifact together

    All files are stored in sessions/{session_id}/conversations/{character_id}.json
    """

    # Character configuration (always present)
    character_config: ResearchCharacter = Field(
        description="Character identity, domain knowledge, and behavior configuration"
    )

    # Research artifact (optional - only present after research is complete)
    research_artifact: Optional[ConversationArtifact] = Field(
        default=None,
        description="Research output and dialogue preparation (None before research)"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this entry was created"
    )

    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="When this entry was last updated"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "character_config": {
                        "character_id": "ml_expert_critical",
                        "name": "Dr. Machine Learning",
                        "domain": "Machine Learning",
                        "sub_domains": ["deep learning", "NLP"],
                        "stance": "critical",
                        "communication_style": "academic"
                    },
                    "research_artifact": {
                        "session_id": "session-123",
                        "character_id": "ml_expert_critical",
                        "domain": "Machine Learning",
                        "research_output": "Research findings...",
                        "dialogue_notes": []
                    }
                }
            ]
        }
    }


class UnifiedCharacterManager:
    """
    Manage unified character configurations and artifacts.

    This is the new, simplified manager that stores everything in one place:
    sessions/{session_id}/conversations/{character_id}.json

    Each file contains:
    - character_config: The character's identity and configuration
    - research_artifact: The research output (optional, None before research)

    Usage:
    1. Create character: save_character(session_id, character) -> saves with artifact=None
    2. After research: update_artifact(session_id, character_id, artifact) -> adds research output
    3. Load for dialogue: load_character(session_id, character_id) -> returns CharacterWithArtifact

    System templates are defined in code (see system_templates.py).
    """

    def __init__(self, storage_path: str = "sessions/"):
        """
        Initialize unified character manager.

        Args:
            storage_path: Base directory for session storage (default: sessions/)
        """
        self.storage_path = Path(storage_path)

    def save_character(
        self,
        session_id: str,
        character: ResearchCharacter,
        artifact: Optional[ConversationArtifact] = None
    ) -> str:
        """
        Save character configuration (with optional artifact).

        Use this when:
        - Creating a new character for a session (artifact=None)
        - Saving character with completed research (artifact provided)

        Args:
            session_id: The session ID
            character: The character configuration
            artifact: Optional research artifact (None before research)

        Returns:
            The character ID
        """
        session_path = self.storage_path / session_id / "conversations"
        session_path.mkdir(parents=True, exist_ok=True)

        # Create unified entry
        unified = CharacterWithArtifact(
            character_config=character,
            research_artifact=artifact,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        file_path = session_path / f"{character.character_id}.json"
        with open(file_path, 'w') as f:
            f.write(unified.model_dump_json(indent=2))

        return character.character_id

    def load_character(
        self,
        session_id: str,
        character_id: str
    ) -> CharacterWithArtifact:
        """
        Load character with artifact from session.

        Args:
            session_id: The session ID
            character_id: The character ID

        Returns:
            CharacterWithArtifact containing both config and optional artifact

        Raises:
            ValueError: If character not found
        """
        file_path = self.storage_path / session_id / "conversations" / f"{character_id}.json"

        if not file_path.exists():
            raise ValueError(f"Character {character_id} not found in session {session_id}")

        with open(file_path) as f:
            return CharacterWithArtifact.model_validate_json(f.read())

    def update_artifact(
        self,
        session_id: str,
        character_id: str,
        artifact: ConversationArtifact
    ):
        """
        Update character's research artifact after research completes.

        Args:
            session_id: The session ID
            character_id: The character ID
            artifact: The research artifact to add
        """
        # Load existing character
        unified = self.load_character(session_id, character_id)

        # Update artifact
        unified.research_artifact = artifact
        unified.last_updated = datetime.now()

        # Save back
        file_path = self.storage_path / session_id / "conversations" / f"{character_id}.json"
        with open(file_path, 'w') as f:
            f.write(unified.model_dump_json(indent=2))

    def load_all_characters(
        self,
        session_id: str
    ) -> Dict[str, CharacterWithArtifact]:
        """
        Load all characters for a session.

        Args:
            session_id: The session ID

        Returns:
            Dictionary mapping character_id to CharacterWithArtifact
        """
        session_path = self.storage_path / session_id / "conversations"

        if not session_path.exists():
            return {}

        characters = {}
        for file_path in session_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    unified = CharacterWithArtifact.model_validate_json(f.read())
                    characters[unified.character_config.character_id] = unified
            except Exception as e:
                print(f"Warning: Could not load character from {file_path}: {e}")

        return characters

    def character_exists(
        self,
        session_id: str,
        character_id: str
    ) -> bool:
        """
        Check if character exists in session.

        Args:
            session_id: The session ID
            character_id: The character ID

        Returns:
            True if character exists
        """
        file_path = self.storage_path / session_id / "conversations" / f"{character_id}.json"
        return file_path.exists()

    def list_characters(
        self,
        session_id: str
    ) -> List[Dict]:
        """
        List all characters in a session with metadata.

        Args:
            session_id: The session ID

        Returns:
            List of character metadata dictionaries
        """
        characters = self.load_all_characters(session_id)

        return [
            {
                "character_id": unified.character_config.character_id,
                "name": unified.character_config.name,
                "domain": unified.character_config.domain,
                "stance": unified.character_config.stance,
                "has_artifact": unified.research_artifact is not None,
                "created_at": unified.created_at,
                "last_updated": unified.last_updated
            }
            for unified in characters.values()
        ]


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
