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
from llmkglitrev.agents.prompts.ontology_grounded_prompt import build_ontology_grounded_prompt
from llmkglitrev.ontology.ontology_manager import DomainOntologyManager
from llmkglitrev.agents.paper_reading import (
    PaperReader,
    ReferenceFollower,
    PaperSearchRefiner,
    select_papers_for_deep_dive
)


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
        model_name: Optional[str] = None,
        assigned_topic: Optional[str] = None,
        seed_papers: Optional[List[Dict]] = None,
        preferred_databases: Optional[List[str]] = None,
        preferred_venues: Optional[List[str]] = None
    ):
        """
        Initialize character-based research agent.

        Args:
            character: The research character configuration
            literature_subset: Optional subset of literature for this domain
            session_id: Optional session ID (generated if not provided)
            model_name: Optional LLM model name (defaults to character's preference or deepseek)
            assigned_topic: Optional specific topic from broad search (NEW)
            seed_papers: Optional seed papers from broad search for deep research (NEW)
            preferred_databases: Optional list of databases to prioritize (NEW)
            preferred_venues: Optional list of venues to prioritize (NEW)
        """
        self.character = character
        self.literature = literature_subset or []
        self.session_id = session_id or str(uuid.uuid4())
        self.assigned_topic = assigned_topic or ""
        self.seed_papers = seed_papers or []
        self.papers_read = []  # Track papers read during deep research
        self.preferred_databases = preferred_databases or character.preferred_databases
        self.preferred_venues = preferred_venues or character.typical_venues

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
            # Build database preferences string
            db_names = {"arxiv": "arXiv", "scopus": "Scopus", "ieee": "IEEE Xplore"}
            db_list = ", ".join([db_names.get(db, db) for db in self.preferred_databases])

            # Build venue preferences string
            venue_str = ""
            if self.preferred_venues:
                venue_str = f"\n- **Preferred Venues**: Focus on papers from: {', '.join(self.preferred_venues[:5])}"

            # Fallback to basic prompt if no template provided
            return f"""You are {self.character.name}, a research expert in {self.character.domain}.

**IMPORTANT: ALL research outputs, responses, and communications MUST be written in English, regardless of the input language.**

Your role is to conduct focused research and provide {self.character.stance} feedback on research proposals.

When conducting research:
- **PRIORITIZE ACADEMIC SOURCES**: Focus on peer-reviewed papers from scholarly publishers
- **Preferred Databases**: Prioritize searching {db_list} for papers in your domain{venue_str}
- Use search_arxiv, search_scopus, or search_ieee tools based on your database preferences
- Use tavily_search to find additional academic literature and context
- Cite specific papers, methodologies, and results from academic publications
- Use evaluation_tool to reflect on findings
- Note assumptions, methodological choices, and limitations based on published research

When answering questions:
- Ground responses in domain knowledge from peer-reviewed sources
- Be specific and cite academic sources (journal names, conference proceedings, paper titles)
- Acknowledge uncertainty when appropriate
- Reference theoretical foundations from academic literature

Academic Source Quality:
- Prefer papers from top-tier venues in your domain
- Consider recency of publications (prioritize recent work)
- Look for highly-cited foundational papers
- Cross-reference multiple sources for validation

Remember: Always write in English and prioritize academic rigor."""

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
        Conduct research using existing research_agent workflow + deep research.

        Workflow:
        1. Review seed papers from broad search
        2. Refine search terms based on seed papers
        3. Conduct focused search (via research_agent)
        4. READ PDFs (if available)
        5. FOLLOW REFERENCES (DFS, max 3 hops, max 5 papers)
        6. Build ontology-grounded knowledge
        7. Identify research gaps

        Args:
            research_topic: The research topic to investigate
            max_iterations: Maximum number of tool call iterations

        Returns:
            ConversationArtifact with research outputs and dialogue notes
        """
        print(f"\n🔬 {self.character.name} conducting deep research...")

        # Step 1: Review seed papers (if provided)
        if self.seed_papers:
            print(f"   📚 Reviewing {len(self.seed_papers)} seed papers from broad search")
            seed_summaries = [
                f"- {p.get('title', 'Untitled')} ({p.get('year', 'N/A')})"
                for p in self.seed_papers[:3]
            ]
            print(f"   {chr(10).join(seed_summaries)}")

        # Step 2: Refine search terms (if seed papers available)
        refined_topic = research_topic
        if self.seed_papers and self.assigned_topic:
            print(f"   🎯 Assigned topic: {self.assigned_topic}")
            try:
                refiner = PaperSearchRefiner()
                refined_topic = await refiner.refine_search_terms(
                    original_topic=self.assigned_topic,
                    seed_papers=self.seed_papers,
                    llm_model=self.llm
                )
            except Exception as e:
                print(f"   ⚠️  Could not refine search: {e}")
                refined_topic = self.assigned_topic

        # Step 3: Conduct focused search (via research_agent)
        print(f"   🔍 Running focused research on: {refined_topic}")
        initial_state: ResearcherState = {
            "researcher_messages": [HumanMessage(content=refined_topic)],
            "tool_call_iterations": 0,
            "research_topic": refined_topic,
            "maximum_number_of_plan": 3,
            "compressed_research": "",
            "raw_notes": [],
            "dialogue_notes": [],
            "character_system_prompt": self.system_prompt
        }

        result = await research_agent.ainvoke(initial_state)

        # Step 4 & 5: Deep dive on selected papers (PDF reading + reference following)
        if self.seed_papers:
            print(f"   📖 Starting deep dive on seed papers...")
            selected_papers = select_papers_for_deep_dive(
                seed_papers=self.seed_papers,
                max_papers=5,
                criteria="citations"
            )

            # Read PDFs and follow references
            reader = PaperReader()
            follower = ReferenceFollower(max_depth=3, max_papers=5)

            for paper in selected_papers[:2]:  # Deep dive on top 2 papers
                pdf_url = paper.get("pdf_url")
                if pdf_url:
                    try:
                        # Read PDF
                        pdf_content = await reader.read_pdf(pdf_url)
                        if pdf_content and pdf_content.get("success"):
                            paper["full_text"] = pdf_content["full_text"]
                            paper["sections"] = pdf_content["sections"]
                            self.papers_read.append(paper)

                            # Follow references (DFS)
                            print(f"   🔗 Following references from: {paper.get('title', '')[:50]}...")
                            reference_papers = await follower.follow_dfs(
                                seed_paper=paper,
                                current_depth=0
                            )
                            self.papers_read.extend(reference_papers)

                            # Stop if reached max papers
                            if len(self.papers_read) >= 5:
                                break
                    except Exception as e:
                        print(f"   ⚠️  Error reading PDF: {e}")

            print(f"   ✅ Deep research complete: {len(self.papers_read)} papers read")

        # Step 6: Populate artifact from result
        self.artifact.research_output = result.get("research_plan", "")
        self.artifact.raw_notes = result.get("raw_notes", [])
        self.artifact.last_updated = datetime.now()

        # Convert dialogue_notes from dicts to DialogueNote objects
        dialogue_notes_dicts = result.get("dialogue_notes", [])
        self.artifact.dialogue_notes = [
            DialogueNote.model_validate(note_dict)
            for note_dict in dialogue_notes_dicts
        ]

        # Step 7: Update papers_consulted with deep research results
        if self.papers_read:
            # Include papers with full text
            self.artifact.papers_consulted = [
                p.get("title", p.get("url", "Unknown"))
                for p in self.papers_read
            ]
        else:
            # Fallback to extraction from notes
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

    async def answer_question_with_ontology(
        self,
        question: str,
        ontology_manager: Optional[DomainOntologyManager] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Answer a Socratic dialogue question with ontology grounding.

        Uses character's knowledge and domain ontology to provide
        well-grounded answers with explicit concept references.

        Args:
            question: The question to answer
            ontology_manager: Optional ontology manager for concept grounding
            context: Optional context (research_output, previous_dialogue, etc.)

        Returns:
            Dict with answer, grounded_concepts, relationships, and reasoning_trace
        """
        context = context or {}

        # If no ontology manager provided, fall back to regular answer
        if not ontology_manager:
            answer = await self.answer_question(question, context)
            return {
                "answer": answer,
                "grounded_concepts": [],
                "concept_relationships": [],
                "new_terms": [],
                "reasoning_trace": []
            }

        # Extract concepts from question
        question_concepts = ontology_manager.extract_concepts(question)

        # Get related concepts from ontology
        related_concepts = ontology_manager.get_related_concepts(
            question_concepts,
            max_per_concept=3
        )

        # Combine for full context
        all_concepts = list(set(question_concepts + related_concepts))

        # Get concept definitions
        concept_definitions = {
            concept: ontology_manager.get_concept_definition(concept)
            for concept in all_concepts
        }

        # Get relationships between concepts
        concept_relationships = ontology_manager.get_concept_relationships(all_concepts)

        # Build research context from previous work
        research_context = context.get("research_output", "")
        if self.artifact.research_output and not research_context:
            research_context = self.artifact.research_output

        # Build ontology-grounded prompt
        ontology_prompt = build_ontology_grounded_prompt(
            question=question,
            research_context=research_context,
            ontology_name=ontology_manager.ontology_name,
            ontology_concepts=all_concepts,
            concept_definitions=concept_definitions,
            concept_relationships=concept_relationships
        )

        # Get response from LLM
        response = await self.llm.ainvoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=ontology_prompt)
        ])

        answer_text = str(response.content)

        # Parse the response to extract structured information
        parsed_response = self._parse_ontology_grounded_answer(
            answer_text,
            all_concepts,
            concept_relationships
        )

        # Validate answer against ontology
        validation_result = self._validate_ontology_usage(
            answer_text,
            ontology_manager,
            all_concepts
        )

        # Combine results
        result = {
            "answer": answer_text,
            "grounded_concepts": validation_result.get("used_concepts", all_concepts),
            "concept_relationships": concept_relationships,
            "new_terms": validation_result.get("new_terms", []),
            "reasoning_trace": parsed_response.get("reasoning_trace", []),
            "ontology_name": ontology_manager.ontology_name
        }

        # Record in artifact
        self.artifact.questions_asked.append(question)
        self.artifact.questions_answered.append({
            "question": question,
            "answer": answer_text,
            "grounded_concepts": result["grounded_concepts"],
            "concept_relationships": result["concept_relationships"],
            "timestamp": datetime.now().isoformat()
        })
        self.artifact.last_updated = datetime.now()

        return result

    def _parse_ontology_grounded_answer(
        self,
        answer_text: str,
        expected_concepts: List[str],
        expected_relationships: List[Dict]
    ) -> Dict:
        """Parse structured information from ontology-grounded answer.

        Args:
            answer_text: The answer text
            expected_concepts: Concepts that should be referenced
            expected_relationships: Expected relationships

        Returns:
            Dict with parsed information
        """
        # Extract reasoning trace if present
        reasoning_trace = []

        # Look for "Reasoning Trace:" section
        if "Reasoning Trace:" in answer_text or "### Reasoning Trace:" in answer_text:
            # Extract lines after reasoning trace header
            lines = answer_text.split("\n")
            in_reasoning = False

            for line in lines:
                if "Reasoning Trace:" in line:
                    in_reasoning = True
                    continue
                elif in_reasoning:
                    # Stop at next section header
                    if line.startswith("###") or line.startswith("**"):
                        break
                    # Add non-empty lines
                    line_clean = line.strip()
                    if line_clean and not line_clean.startswith("#"):
                        # Remove numbering if present
                        line_clean = line_clean.lstrip("0123456789.- ")
                        reasoning_trace.append(line_clean)

        return {
            "reasoning_trace": reasoning_trace
        }

    def _validate_ontology_usage(
        self,
        answer_text: str,
        ontology_manager: DomainOntologyManager,
        expected_concepts: List[str]
    ) -> Dict:
        """Validate which ontology concepts were actually used in answer.

        Args:
            answer_text: The answer text
            ontology_manager: Ontology manager
            expected_concepts: Concepts that were provided

        Returns:
            Dict with validation results
        """
        # Extract concepts actually mentioned in answer
        used_concepts = ontology_manager.extract_concepts(answer_text)

        # Find new terms (words in bold or quotes that aren't in ontology)
        new_terms = []

        # Simple heuristic: look for **term** or "term" patterns
        import re
        bold_terms = re.findall(r'\*\*([^*]+)\*\*', answer_text)
        quoted_terms = re.findall(r'"([^"]+)"', answer_text)

        all_candidate_terms = set(bold_terms + quoted_terms)

        for term in all_candidate_terms:
            # Check if it's in ontology
            if ontology_manager.find_concept(term) is None:
                # Not in ontology - potential new term
                if term not in [c.lower() for c in used_concepts]:
                    new_terms.append(term)

        return {
            "used_concepts": used_concepts,
            "new_terms": new_terms[:10]  # Limit to avoid noise
        }

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
