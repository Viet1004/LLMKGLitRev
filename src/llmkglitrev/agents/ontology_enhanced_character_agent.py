"""
Ontology-Enhanced Character Agent.

This module extends CharacterBasedResearchAgent with domain ontology knowledge
to generate better dialogue notes and Socratic questions.
"""

from typing import List, Dict, Optional
from datetime import datetime

from llmkglitrev.agents.character_agent import CharacterBasedResearchAgent
from llmkglitrev.characters.schema import ResearchCharacter
from llmkglitrev.characters.artifact import ConversationArtifact, DialogueNote
from llmkglitrev.ontology import DomainOntologyManager


class OntologyEnhancedCharacterAgent(CharacterBasedResearchAgent):
    """
    Research agent enhanced with domain ontology knowledge.

    This agent extends the base CharacterBasedResearchAgent by:
    1. Validating terminology in research outputs against ontology
    2. Identifying missing related concepts
    3. Generating ontology-grounded dialogue notes
    4. Providing richer context for Socratic questions
    """

    def __init__(
        self,
        character: ResearchCharacter,
        ontology_manager: Optional[DomainOntologyManager] = None,
        ontology_source: str = "edam",
        literature_subset: Optional[List[dict]] = None,
        session_id: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize ontology-enhanced character agent.

        Args:
            character: The research character configuration
            ontology_manager: Pre-loaded ontology manager (optional)
            ontology_source: Ontology to load if manager not provided
            literature_subset: Optional subset of literature for this domain
            session_id: Optional session ID
            model_name: Optional LLM model name
        """
        # Initialize base agent
        super().__init__(
            character=character,
            literature_subset=literature_subset,
            session_id=session_id,
            model_name=model_name
        )

        # Initialize or use provided ontology manager
        if ontology_manager:
            self.ontology = ontology_manager
            print(f"✅ Using provided ontology manager: {self.ontology}")
        else:
            print(f"📥 Loading ontology: {ontology_source}")
            self.ontology = DomainOntologyManager(ontology_source=ontology_source)
            success = self.ontology.load()
            if not success:
                print(f"⚠️  Failed to load ontology, continuing without ontology support")
                self.ontology = None

    async def conduct_research(
        self,
        research_topic: str,
        max_iterations: int = 3,
        enable_ontology_analysis: bool = True
    ) -> ConversationArtifact:
        """
        Conduct research with ontology enhancement.

        This extends the base conduct_research by:
        1. Running standard research
        2. Analyzing research output against ontology
        3. Generating additional ontology-based dialogue notes

        Args:
            research_topic: The research topic to investigate
            max_iterations: Maximum number of tool call iterations
            enable_ontology_analysis: Whether to use ontology for enhancement

        Returns:
            ConversationArtifact with research outputs and enhanced dialogue notes
        """
        # Run base research
        print(f"\n🔬 {self.character.name} conducting research...")
        artifact = await super().conduct_research(research_topic, max_iterations)

        # Enhance with ontology if available and enabled
        if self.ontology and enable_ontology_analysis:
            print(f"\n🧠 Enhancing with ontology analysis...")
            enhanced_notes = self._generate_ontology_dialogue_notes(
                artifact.research_output
            )

            if enhanced_notes:
                # Add to existing dialogue notes (avoid duplicates)
                existing_questions = {note.suggested_question for note in artifact.dialogue_notes}
                new_notes = [
                    note for note in enhanced_notes
                    if note.suggested_question not in existing_questions
                ]

                artifact.dialogue_notes.extend(new_notes)
                print(f"   ✅ Added {len(new_notes)} ontology-based dialogue notes")
                print(f"   📊 Total dialogue notes: {len(artifact.dialogue_notes)}")

        return artifact

    def _generate_ontology_dialogue_notes(
        self,
        research_output: str,
        min_priority: int = 6
    ) -> List[DialogueNote]:
        """
        Generate dialogue notes based on ontology analysis.

        Args:
            research_output: The research output to analyze
            min_priority: Minimum priority threshold

        Returns:
            List of DialogueNote objects
        """
        if not self.ontology:
            return []

        # Generate competency questions from ontology
        competency_questions = self.ontology.generate_competency_questions(
            research_output,
            priority_threshold=min_priority
        )

        # Map question types to DialogueNote types
        type_mapping = {
            "broader_context": "gap_identified",
            "alternative_approach": "alternative_approach",
            "comparison": "methodological_choice",
            "relationship_clarification": "assumption"
        }

        dialogue_notes = []

        for q in competency_questions:
            note_type = type_mapping.get(q['type'], 'gap_identified')

            # Create DialogueNote
            note = DialogueNote(
                type=note_type,
                content=f"Ontology concept: {q['concept']}",
                suggested_question=q['question'],
                priority=q['priority'],
                context=f"Ontology analysis: {q['definition'][:100]}"
            )

            dialogue_notes.append(note)

        return dialogue_notes

    async def answer_question(
        self,
        question: str,
        context: Optional[Dict] = None,
        use_ontology_context: bool = True
    ) -> str:
        """
        Answer a question with ontology-enhanced context.

        This extends the base answer_question by adding ontology
        definitions and relationships to the prompt context.

        Args:
            question: The question to answer
            context: Optional context (research_output, previous_dialogue, etc.)
            use_ontology_context: Whether to add ontology context

        Returns:
            Answer string
        """
        context = context or {}

        # Add ontology context if available
        if self.ontology and use_ontology_context:
            ontology_context = self._build_ontology_context(question)
            if ontology_context:
                context['ontology_context'] = ontology_context

        # Call base answer_question with enhanced context
        return await super().answer_question(question, context)

    def _build_ontology_context(self, question: str) -> str:
        """
        Build ontology context for a question.

        Extracts concepts from the question and provides
        their definitions and relationships from the ontology.

        Args:
            question: The question text

        Returns:
            Formatted ontology context string
        """
        if not self.ontology:
            return ""

        # Extract potential concepts from question
        words = question.split()
        concepts_found = []

        # Try to find concepts for multi-word phrases and single words
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                phrase = ' '.join(words[i:j])
                related = self.ontology.find_related_concepts(phrase, max_results=2)
                if related:
                    concepts_found.extend(related[:1])  # Add top match

        # Build context string
        if not concepts_found:
            return ""

        context_parts = ["\n**Ontology Context:**\n"]

        seen_concepts = set()
        for concept_info in concepts_found[:5]:  # Limit to 5 concepts
            concept_name = concept_info['name']
            if concept_name in seen_concepts:
                continue
            seen_concepts.add(concept_name)

            context_parts.append(f"- **{concept_name}**: {concept_info['definition']}")

        return '\n'.join(context_parts)

    def validate_research_terminology(self) -> Dict:
        """
        Validate the terminology used in the research output.

        Returns:
            Validation report with found/missing terms
        """
        if not self.ontology or not self.artifact.research_output:
            return {"error": "No ontology or research output available"}

        validation_results = self.ontology.validate_terminology(
            self.artifact.research_output
        )

        report = {
            "total_terms": len(validation_results),
            "found_in_ontology": len([r for r in validation_results if r["status"] == "found"]),
            "not_found": len([r for r in validation_results if r["status"] == "not_found"]),
            "details": validation_results
        }

        return report

    def get_ontology_statistics(self) -> Dict:
        """
        Get statistics about the loaded ontology.

        Returns:
            Dictionary with ontology statistics
        """
        if not self.ontology:
            return {"error": "No ontology loaded"}

        return self.ontology.get_statistics()

    def __repr__(self) -> str:
        """String representation."""
        ontology_status = "with ontology" if self.ontology else "no ontology"
        return (
            f"OntologyEnhancedCharacterAgent("
            f"{self.character.name}, "
            f"{self.character.domain}, "
            f"{ontology_status})"
        )


def create_ontology_enhanced_agent(
    character_id: str,
    ontology_source: str = "edam",
    literature_subset: Optional[List[dict]] = None,
    session_id: Optional[str] = None,
    model_name: Optional[str] = None
) -> OntologyEnhancedCharacterAgent:
    """
    Convenience function to create ontology-enhanced agent from character ID.

    Args:
        character_id: ID of the character to load
        ontology_source: Ontology to use ("edam" or other)
        literature_subset: Optional literature subset
        session_id: Optional session ID
        model_name: Optional LLM model name

    Returns:
        OntologyEnhancedCharacterAgent instance

    Raises:
        ValueError: If character not found
    """
    from llmkglitrev.characters import CharacterManager

    manager = CharacterManager()
    character = manager.load_character(character_id)

    return OntologyEnhancedCharacterAgent(
        character=character,
        ontology_source=ontology_source,
        literature_subset=literature_subset,
        session_id=session_id,
        model_name=model_name
    )
