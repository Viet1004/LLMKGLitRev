"""Character-based research agent.

This module implements research agents that are instantiated from character
configurations and produce conversation artifacts.
"""

from typing import List, Dict, Optional
from datetime import datetime
import uuid

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model

from llmkglitrev.characters.schema import ResearchCharacter
from llmkglitrev.characters.artifact import ConversationArtifact, DialogueNote
from llmkglitrev.agents.research_agents import research_agent
from llmkglitrev.agents.states import ResearcherState


class CharacterBasedResearchAgent:
    """
    Research agent instantiated from a character configuration.

    This agent:
    1. Uses a ResearchCharacter's domain knowledge and traits
    2. Leverages the existing research_agent workflow
    3. Produces ConversationArtifact outputs with dialogue notes
    4. Can answer Socratic dialogue questions
    """

    def __init__(
        self,
        character: ResearchCharacter,
        literature_subset: Optional[List[dict]] = None,
        session_id: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize character-based research agent.

        Args:
            character: The research character configuration
            literature_subset: Optional subset of literature for this domain
            session_id: Optional session ID (generated if not provided)
            model_name: Optional LLM model name (defaults to character's preference or deepseek)
        """
        self.character = character
        self.literature = literature_subset or []
        self.session_id = session_id or str(uuid.uuid4())

        # Build system prompt from character template
        self.system_prompt = self._build_system_prompt()

        # Initialize LLM for question answering
        self.model_name = model_name or "deepseek:deepseek-chat"
        self.llm = init_chat_model(model=self.model_name)

        # Initialize conversation artifact
        self.artifact = ConversationArtifact(
            session_id=self.session_id,
            character_id=character.character_id,
            domain=character.domain
        )

    def _build_system_prompt(self) -> str:
        """
        Build system prompt from character template.

        Returns:
            Formatted system prompt
        """
        if not self.character.system_prompt_template:
            # Fallback to basic prompt if no template provided
            return f"""You are {self.character.name}, a research expert in {self.character.domain}.

**IMPORTANT: ALL research outputs, responses, and communications MUST be written in English, regardless of the input language.**

Your role is to conduct focused research and provide {self.character.stance} feedback on research proposals.

When conducting research:
- Use tavily_search to find relevant literature
- Use evaluation_tool to reflect on findings
- Note assumptions, methodological choices, and limitations

When answering questions:
- Ground responses in domain knowledge
- Be specific and cite sources
- Acknowledge uncertainty when appropriate

Remember: Always write in English."""

        # Format template with character attributes
        try:
            return self.character.system_prompt_template.format(
                name=self.character.name,
                domain=self.character.domain,
                sub_domains=", ".join(self.character.sub_domains) if self.character.sub_domains else "N/A",
                typical_venues=", ".join(self.character.typical_venues) if self.character.typical_venues else "N/A",
                typical_methods=", ".join(self.character.typical_methods) if self.character.typical_methods else "N/A",
                typical_datasets=", ".join(self.character.typical_datasets) if self.character.typical_datasets else "N/A",
                theoretical_foundations=", ".join(self.character.theoretical_foundations) if self.character.theoretical_foundations else "N/A",
                stance=self.character.stance,
                communication_style=self.character.communication_style,
                focus_areas=", ".join(self.character.focus_areas) if self.character.focus_areas else "N/A"
            )
        except KeyError as e:
            print(f"Warning: Missing placeholder in template: {e}")
            # Fallback if template formatting fails
            return f"You are {self.character.name}, a research expert in {self.character.domain}."

    async def conduct_research(
        self,
        research_topic: str,
        max_iterations: int = 3
    ) -> ConversationArtifact:
        """
        Conduct research using existing research_agent workflow.

        Args:
            research_topic: The research topic to investigate
            max_iterations: Maximum number of tool call iterations

        Returns:
            ConversationArtifact with research outputs and dialogue notes
        """
        # Prepare initial state for research agent
        initial_state: ResearcherState = {
            "researcher_messages": [HumanMessage(content=research_topic)],
            "tool_call_iterations": 0,
            "research_topic": research_topic,
            "maximum_number_of_plan": 3,
            "compressed_research": "",
            "raw_notes": [],
            "dialogue_notes": [],
            "character_system_prompt": self.system_prompt  # Inject character prompt
        }

        # Run research agent workflow
        result = await research_agent.ainvoke(initial_state)

        # Populate artifact from result
        self.artifact.research_output = result.get("research_plan", "")
        self.artifact.raw_notes = result.get("raw_notes", [])
        self.artifact.last_updated = datetime.now()

        # Convert dialogue_notes from dicts to DialogueNote objects
        dialogue_notes_dicts = result.get("dialogue_notes", [])
        self.artifact.dialogue_notes = [
            DialogueNote.model_validate(note_dict)
            for note_dict in dialogue_notes_dicts
        ]

        # Extract papers consulted from messages (if available)
        # This is a simple heuristic - can be improved
        self.artifact.papers_consulted = self._extract_papers_from_notes()

        return self.artifact

    def _extract_papers_from_notes(self) -> List[str]:
        """
        Extract paper references from raw notes.

        Returns:
            List of paper identifiers/URLs
        """
        papers = []
        for note in self.artifact.raw_notes:
            # Simple heuristic: look for URLs
            if "http" in note or "arxiv" in note.lower() or "doi" in note.lower():
                # Extract first URL-like string
                words = note.split()
                for word in words:
                    if "http" in word or "arxiv" in word.lower():
                        papers.append(word.strip("(),.:;"))

        return list(set(papers))  # Deduplicate

    async def answer_question(
        self,
        question: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Answer a Socratic dialogue question.

        Uses character's knowledge and previous research to provide
        domain-specific answers.

        Args:
            question: The question to answer
            context: Optional context (research_output, previous_dialogue, etc.)

        Returns:
            Answer string
        """
        context = context or {}

        # Build prompt with character context
        prompt_parts = [self.system_prompt]

        # Add research context if available
        if context.get("research_output"):
            prompt_parts.append(f"\nYou previously conducted research on: {context['research_output']}")

        # Add previous dialogue if available
        if context.get("previous_dialogue"):
            prev_dialogue = context["previous_dialogue"]
            if prev_dialogue and len(prev_dialogue) > 0:
                prompt_parts.append("\nPrevious dialogue:")
                for qa in prev_dialogue[-3:]:  # Last 3 exchanges
                    if isinstance(qa, dict):
                        prompt_parts.append(f"Q: {qa.get('question', '')}")
                        prompt_parts.append(f"A: {qa.get('user_feedback', '')}")

        # Add the current question
        prompt_parts.append(f"\nNow, you are asked the following question:\n{question}")
        prompt_parts.append("\nProvide a thoughtful response grounded in your domain expertise and previous research.")

        full_prompt = "\n".join(prompt_parts)

        # Get response from LLM
        response = await self.llm.ainvoke([HumanMessage(content=full_prompt)])

        # Record question and answer
        self.artifact.questions_asked.append(question)
        self.artifact.questions_answered.append({
            "question": question,
            "answer": response.content,
            "timestamp": datetime.now().isoformat()
        })
        self.artifact.last_updated = datetime.now()

        return str(response.content)

    def get_artifact(self) -> ConversationArtifact:
        """
        Get the current conversation artifact.

        Returns:
            The conversation artifact
        """
        return self.artifact

    def get_dialogue_notes(self, min_priority: int = 1) -> List[DialogueNote]:
        """
        Get dialogue notes filtered by priority.

        Args:
            min_priority: Minimum priority threshold

        Returns:
            List of dialogue notes sorted by priority (highest first)
        """
        filtered = [
            note for note in self.artifact.dialogue_notes
            if note.priority >= min_priority
        ]
        # Sort by priority (highest first)
        filtered.sort(key=lambda x: x.priority, reverse=True)
        return filtered

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CharacterBasedResearchAgent("
            f"character={self.character.name}, "
            f"domain={self.character.domain}, "
            f"session={self.session_id[:8]}...)"
        )


def create_character_agent(
    character_id: str,
    literature_subset: Optional[List[dict]] = None,
    session_id: Optional[str] = None,
    model_name: Optional[str] = None
) -> CharacterBasedResearchAgent:
    """
    Convenience function to create character agent from character ID.

    Args:
        character_id: ID of the character to load
        literature_subset: Optional literature subset
        session_id: Optional session ID
        model_name: Optional LLM model name

    Returns:
        CharacterBasedResearchAgent instance

    Raises:
        ValueError: If character not found
    """
    from llmkglitrev.characters import CharacterManager

    manager = CharacterManager()
    character = manager.load_character(character_id)

    return CharacterBasedResearchAgent(
        character=character,
        literature_subset=literature_subset,
        session_id=session_id,
        model_name=model_name
    )
