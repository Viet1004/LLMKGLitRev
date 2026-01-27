"""Simple Socratic Dialogue System.

User selects a character, character reads the final proposal and other artifacts,
then asks clarifying questions from their perspective.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from llmkglitrev.characters.artifact import UnifiedCharacterManager, ConversationArtifact
from llmkglitrev.agents.character_agent import CharacterBasedResearchAgent


class DialogueMessage(BaseModel):
    """Single message in the dialogue."""
    speaker: str = Field(description="Who said this: 'user' or character name")
    message: str = Field(description="The message content")
    timestamp: str = Field(description="When this was said")


class SocraticDialogue:
    """Simple dialogue interface for character-based Q&A."""

    def __init__(self, session_id: str):
        """Initialize dialogue for a session.

        Args:
            session_id: The session ID containing artifacts
        """
        self.session_id = session_id
        self.manager = UnifiedCharacterManager()

        # Load all characters with artifacts
        all_characters = self.manager.load_all_characters(session_id)

        # Extract just the artifacts (research_artifact from each CharacterWithArtifact)
        self.artifacts = {}
        for char_id, char_with_artifact in all_characters.items():
            if char_with_artifact.research_artifact:
                self.artifacts[char_id] = char_with_artifact.research_artifact

        # Load final proposal
        from pathlib import Path
        proposal_path = Path(f"sessions/{session_id}/final_proposal.md")
        if proposal_path.exists():
            self.final_proposal = proposal_path.read_text()
        else:
            self.final_proposal = "No final proposal found."

        print(f"✓ Loaded {len(self.artifacts)} character artifacts")
        print(f"✓ Final proposal loaded ({len(self.final_proposal)} chars)")

    def get_available_characters(self) -> List[Dict[str, str]]:
        """Get list of available characters for dialogue.

        Returns:
            List of dicts with character_id, name, domain
        """
        characters = []
        for char_id, artifact in self.artifacts.items():
            char_with_artifact = self.manager.load_character(self.session_id, char_id)
            character = char_with_artifact.character_config

            characters.append({
                "character_id": char_id,
                "name": character.name,
                "domain": character.domain
            })

        return characters

    async def generate_questions(
        self,
        character_id: str,
        num_questions: int = 5
    ) -> List[str]:
        """Generate questions from a character's perspective.

        The character reads:
        - Final proposal
        - Other characters' research artifacts

        And identifies unclear points from THEIR perspective.

        Args:
            character_id: Which character to use
            num_questions: How many questions to generate

        Returns:
            List of questions
        """
        # Load selected character
        char_with_artifact = self.manager.load_character(self.session_id, character_id)
        character = char_with_artifact.character_config
        artifact = char_with_artifact.research_artifact

        # Get other artifacts (for context)
        other_artifacts_text = []
        for char_id, other_artifact in self.artifacts.items():
            if char_id != character_id:
                other_artifacts_text.append(
                    f"**{char_id}** ({other_artifact.domain}):\n{other_artifact.research_output[:500]}..."
                )

        other_context = "\n\n".join(other_artifacts_text)

        # Create agent
        agent = CharacterBasedResearchAgent(
            character=character,
            session_id=self.session_id,
            literature_subset=[]
        )

        # Restore artifact
        if artifact:
            agent.artifact = artifact

        # Calculate depth distribution
        # First 1/3: General questions about fit and context
        # Middle 1/3: Intermediate questions about methods
        # Last 1/3: Deep domain-specific questions
        num_general = max(1, num_questions // 3)
        num_intermediate = max(1, (num_questions - num_general) // 2)
        num_deep = num_questions - num_general - num_intermediate

        # Prompt to generate questions with depth progression
        prompt = f"""You are {character.name}, a {character.domain} expert.

You have reviewed this research proposal:

{self.final_proposal}

You have also seen research from other perspectives:

{other_context}

Your task: Generate {num_questions} questions about unclear points in the proposal from YOUR {character.domain} perspective.

IMPORTANT: Structure your questions with PROGRESSIVE DEPTH:

**First {num_general} question(s) - GENERAL LEVEL:**
- How does this proposal fit within the broader context of {character.domain}?
- What is the overall alignment with current trends/paradigms in {character.domain}?
- General clarity about the research direction and its relevance to {character.domain}

**Next {num_intermediate} question(s) - INTERMEDIATE LEVEL:**
- Methodological choices and their justification from {character.domain} perspective
- Specific techniques, frameworks, or approaches mentioned in the proposal
- Implementation considerations at the method level
- Assumptions that need explanation

**Final {num_deep} question(s) - DEEP DOMAIN-SPECIFIC LEVEL:**
- Highly technical aspects specific to {character.domain}
- Nuanced theoretical implications within your specialized domain
- Advanced implementation details that experts in {character.domain} would scrutinize
- Edge cases, limitations, or challenges specific to your domain expertise

Format your response as a numbered list:
1. [General question about fit/context]
2. [General question continued if needed]
3. [Intermediate question about methods]
4. [Intermediate question continued if needed]
5. [Deep domain-specific question]
...

Generate exactly {num_questions} questions following this general → intermediate → deep progression."""

        # Get response from LLM
        response = await agent.llm.ainvoke([HumanMessage(content=prompt)])

        # Extract questions from response
        answer = response.content

        # Parse numbered list
        questions = []
        for line in answer.split("\n"):
            line = line.strip()
            # Match "1. " or "1) " format
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove number prefix
                question = line
                for i, char in enumerate(line):
                    if char in ['.', ')', '-']:
                        question = line[i+1:].strip()
                        break

                if question:
                    questions.append(question)

        print(f"\n✓ Generated {len(questions)} questions from {character.name}")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q[:80]}...")

        return questions

    async def answer_question(
        self,
        character_id: str,
        user_question: str,
        conversation_history: Optional[List[DialogueMessage]] = None
    ) -> str:
        """Character answers user's question.

        Args:
            character_id: Which character is answering
            user_question: The user's question
            conversation_history: Previous messages in this conversation

        Returns:
            Character's answer
        """
        # Load character
        char_with_artifact = self.manager.load_character(self.session_id, character_id)
        character = char_with_artifact.character_config
        artifact = char_with_artifact.research_artifact

        # Create agent
        agent = CharacterBasedResearchAgent(
            character=character,
            session_id=self.session_id,
            literature_subset=[]
        )

        # Restore artifact
        if artifact:
            agent.artifact = artifact

        # Build context from history
        history_text = ""
        if conversation_history:
            history_text = "\n\nPrevious conversation:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages
                # Handle both dict and DialogueMessage objects
                if isinstance(msg, dict):
                    speaker = msg.get("speaker", "Unknown")
                    message = msg.get("message", "")
                else:
                    speaker = msg.speaker
                    message = msg.message
                history_text += f"{speaker}: {message}\n"

        # Answer the question
        context = {
            "research_output": artifact.research_output if artifact else "",
            "final_proposal": self.final_proposal,
            "conversation_history": history_text
        }

        answer = await agent.answer_question(user_question, context=context)

        return answer


# ===== CONVENIENCE FUNCTIONS =====

async def create_dialogue_session(session_id: str) -> SocraticDialogue:
    """Create a new dialogue session.

    Args:
        session_id: Session ID with research artifacts

    Returns:
        SocraticDialogue instance
    """
    return SocraticDialogue(session_id)


async def get_character_questions(
    dialogue: SocraticDialogue,
    character_id: str,
    num_questions: int = 5
) -> List[str]:
    """Generate questions from a character.

    Args:
        dialogue: The dialogue session
        character_id: Which character
        num_questions: How many questions

    Returns:
        List of questions
    """
    return await dialogue.generate_questions(character_id, num_questions)
