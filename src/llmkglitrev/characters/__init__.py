"""Research Character System for LLMKGLitRev.

This module provides the character-based research agent system that separates
reusable character configurations from ephemeral conversation artifacts.
"""

from .schema import ResearchCharacter
from .manager import CharacterManager
from .artifact import DialogueNote, ConversationArtifact, ConversationArtifactManager
from .note_extractor import DialogueNoteExtractor, extract_dialogue_notes

__all__ = [
    "ResearchCharacter",
    "CharacterManager",
    "DialogueNote",
    "ConversationArtifact",
    "ConversationArtifactManager",
    "DialogueNoteExtractor",
    "extract_dialogue_notes",
]
