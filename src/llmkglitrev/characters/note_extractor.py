"""Dialogue note extraction from research conversations.

This module provides utilities to automatically extract dialogue-worthy topics
from research agent conversations, identifying assumptions, methodological choices,
limitations, and other items that warrant follow-up questions.
"""

from typing import List
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from .artifact import DialogueNote


class DialogueNoteExtractor:
    """
    Extract dialogue notes from research conversations.

    Uses heuristic patterns and optional LLM analysis to identify topics
    that warrant follow-up in Socratic dialogue.
    """

    # Heuristic patterns for different note types
    ASSUMPTION_PHRASES = [
        "assuming",
        "assume that",
        "given that",
        "if we assume",
        "presuming",
        "presume that",
        "supposing",
        "under the assumption"
    ]

    METHOD_PHRASES = [
        "we will use",
        "we propose",
        "our approach",
        "methodology",
        "we employ",
        "we apply",
        "using the method",
        "the technique involves"
    ]

    LIMITATION_PHRASES = [
        "limitation",
        "caveat",
        "however",
        "challenge",
        "drawback",
        "constraint",
        "restricted to",
        "cannot handle",
        "does not address"
    ]

    ALTERNATIVE_PHRASES = [
        "alternatively",
        "another approach",
        "could also",
        "instead of",
        "different method",
        "alternative solution"
    ]

    GAP_PHRASES = [
        "no research on",
        "gap in",
        "missing",
        "not yet explored",
        "future work",
        "needs further investigation",
        "remains unclear"
    ]

    def __init__(self, max_notes_per_type: int = 3):
        """
        Initialize note extractor.

        Args:
            max_notes_per_type: Maximum notes to extract per type
        """
        self.max_notes_per_type = max_notes_per_type

    def extract_from_messages(
        self,
        messages: List[BaseMessage],
        research_output: str = ""
    ) -> List[DialogueNote]:
        """
        Extract dialogue notes from conversation messages.

        Args:
            messages: List of conversation messages
            research_output: Final research output text

        Returns:
            List of dialogue notes sorted by priority
        """
        notes = []

        # Extract from individual messages
        for msg in messages:
            if isinstance(msg, (AIMessage, ToolMessage)):
                content = str(msg.content).lower()
                original_content = str(msg.content)

                # Check for assumptions
                notes.extend(self._extract_assumptions(original_content, content))

                # Check for methodological choices
                notes.extend(self._extract_methods(original_content, content))

                # Check for limitations
                notes.extend(self._extract_limitations(original_content, content))

                # Check for alternatives
                notes.extend(self._extract_alternatives(original_content, content))

                # Check for gaps
                notes.extend(self._extract_gaps(original_content, content))

        # Extract from final research output
        if research_output:
            output_lower = research_output.lower()
            notes.extend(self._extract_assumptions(research_output, output_lower))
            notes.extend(self._extract_methods(research_output, output_lower))
            notes.extend(self._extract_limitations(research_output, output_lower))
            notes.extend(self._extract_alternatives(research_output, output_lower))
            notes.extend(self._extract_gaps(research_output, output_lower))

        # Deduplicate
        unique_notes = self._deduplicate_notes(notes)

        # Limit per type
        limited_notes = self._limit_notes_per_type(unique_notes)

        # Sort by priority
        limited_notes.sort(key=lambda x: x.priority, reverse=True)

        return limited_notes

    def _extract_assumptions(
        self,
        original_content: str,
        lower_content: str
    ) -> List[DialogueNote]:
        """Extract assumptions from text."""
        notes = []

        for phrase in self.ASSUMPTION_PHRASES:
            if phrase in lower_content:
                # Find sentence containing phrase
                sentence = self._find_sentence_containing(original_content, phrase)
                if sentence:
                    notes.append(DialogueNote(
                        type="assumption",
                        content=sentence,
                        suggested_question=f"What evidence supports the assumption: '{sentence[:80]}...'?",
                        priority=7,
                        context="Identified during research phase"
                    ))

        return notes

    def _extract_methods(
        self,
        original_content: str,
        lower_content: str
    ) -> List[DialogueNote]:
        """Extract methodological choices from text."""
        notes = []

        for phrase in self.METHOD_PHRASES:
            if phrase in lower_content:
                sentence = self._find_sentence_containing(original_content, phrase)
                if sentence:
                    notes.append(DialogueNote(
                        type="methodological_choice",
                        content=sentence,
                        suggested_question="Why is this method preferred over alternatives?",
                        priority=6,
                        context="Methodological decision in research"
                    ))

        return notes

    def _extract_limitations(
        self,
        original_content: str,
        lower_content: str
    ) -> List[DialogueNote]:
        """Extract limitations from text."""
        notes = []

        for phrase in self.LIMITATION_PHRASES:
            if phrase in lower_content:
                sentence = self._find_sentence_containing(original_content, phrase)
                if sentence:
                    notes.append(DialogueNote(
                        type="limitation",
                        content=sentence,
                        suggested_question="How can this limitation be addressed or mitigated?",
                        priority=8,
                        context="Identified limitation"
                    ))

        return notes

    def _extract_alternatives(
        self,
        original_content: str,
        lower_content: str
    ) -> List[DialogueNote]:
        """Extract alternative approaches from text."""
        notes = []

        for phrase in self.ALTERNATIVE_PHRASES:
            if phrase in lower_content:
                sentence = self._find_sentence_containing(original_content, phrase)
                if sentence:
                    notes.append(DialogueNote(
                        type="alternative_approach",
                        content=sentence,
                        suggested_question="What are the trade-offs of this alternative approach?",
                        priority=5,
                        context="Alternative approach mentioned"
                    ))

        return notes

    def _extract_gaps(
        self,
        original_content: str,
        lower_content: str
    ) -> List[DialogueNote]:
        """Extract research gaps from text."""
        notes = []

        for phrase in self.GAP_PHRASES:
            if phrase in lower_content:
                sentence = self._find_sentence_containing(original_content, phrase)
                if sentence:
                    notes.append(DialogueNote(
                        type="gap_identified",
                        content=sentence,
                        suggested_question="How significant is this gap for the proposed research?",
                        priority=9,
                        context="Research gap identified"
                    ))

        return notes

    def _find_sentence_containing(
        self,
        text: str,
        phrase: str,
        max_length: int = 200
    ) -> str:
        """
        Find sentence containing phrase.

        Args:
            text: Text to search
            phrase: Phrase to find
            max_length: Maximum sentence length

        Returns:
            Sentence containing phrase, or empty string
        """
        # Simple sentence extraction (can be improved with NLP)
        sentences = text.replace('!', '.').replace('?', '.').split('.')

        for sentence in sentences:
            if phrase.lower() in sentence.lower():
                sentence = sentence.strip()
                if len(sentence) > max_length:
                    sentence = sentence[:max_length] + "..."
                return sentence

        return ""

    def _deduplicate_notes(self, notes: List[DialogueNote]) -> List[DialogueNote]:
        """
        Remove duplicate notes based on content similarity.

        Args:
            notes: List of notes

        Returns:
            Deduplicated list
        """
        unique_notes = []
        seen_contents = set()

        for note in notes:
            # Use first 50 chars as key
            key = note.content[:50].lower()

            if key not in seen_contents:
                unique_notes.append(note)
                seen_contents.add(key)

        return unique_notes

    def _limit_notes_per_type(
        self,
        notes: List[DialogueNote]
    ) -> List[DialogueNote]:
        """
        Limit number of notes per type.

        Args:
            notes: List of notes

        Returns:
            Limited list
        """
        type_counts = {}
        limited_notes = []

        # Sort by priority first
        notes_sorted = sorted(notes, key=lambda x: x.priority, reverse=True)

        for note in notes_sorted:
            count = type_counts.get(note.type, 0)

            if count < self.max_notes_per_type:
                limited_notes.append(note)
                type_counts[note.type] = count + 1

        return limited_notes


def extract_dialogue_notes(
    messages: List[BaseMessage],
    research_output: str = "",
    max_notes_per_type: int = 3
) -> List[DialogueNote]:
    """
    Convenience function to extract dialogue notes.

    Args:
        messages: Conversation messages
        research_output: Final research output
        max_notes_per_type: Maximum notes per type

    Returns:
        List of dialogue notes
    """
    extractor = DialogueNoteExtractor(max_notes_per_type=max_notes_per_type)
    return extractor.extract_from_messages(messages, research_output)
