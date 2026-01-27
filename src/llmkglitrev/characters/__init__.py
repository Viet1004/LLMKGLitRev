"""Research Character System for LLMKGLitRev.

This module provides the character-based research agent system that separates
reusable character configurations from ephemeral conversation artifacts.
"""

from .schema import ResearchCharacter
from .manager import CharacterManager
from .artifact import ConversationArtifact, ConversationArtifactManager

__all__ = [
    "ResearchCharacter",
    "CharacterManager",
    "ConversationArtifact",
    "ConversationArtifactManager",
]
