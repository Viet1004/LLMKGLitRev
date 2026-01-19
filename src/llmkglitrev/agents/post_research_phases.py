"""
Post-Research Phases (5-7) - Independent Conversations

These phases run AFTER Phase 4 completes and are separate from the main workflow.
They load saved artifacts and characters to start new conversations.

Phase 5: Research Gap Identification - Characters explore their own ontologies
Phase 6: Socratic Dialogue - Characters ask critical questions
Phase 7: Industry Partner Review - Partners review using other ontologies
"""

from typing import List, Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from llmkglitrev.characters.artifact import UnifiedCharacterManager, ConversationArtifact
from llmkglitrev.agents.character_agent import CharacterBasedResearchAgent


# ===== PHASE 5: RESEARCH GAP IDENTIFICATION =====

async def identify_research_gaps_standalone(
    session_id: str,
    final_proposal: str
) -> List[Dict[str, Any]]:
    """Phase 5: Identify research gaps from ontology exploration.

    This is a STANDALONE function, not part of the main workflow.
    Characters load their saved artifacts and explore their own ontologies.

    NEW: Uses unified character format - loads character_config and research_artifact together

    Args:
        session_id: Session ID to load artifacts from
        final_proposal: The final research proposal from Phase 4

    Returns:
        List of gap identification results from each character
    """
    print(f"\n{'='*70}")
    print(f"PHASE 5: RESEARCH GAP IDENTIFICATION (STANDALONE)")
    print(f"{'='*70}")
    print(f"Session ID: {session_id}")
    print(f"Loading saved characters with artifacts...")
    print(f"{'='*70}\n")

    # Load unified characters from disk
    unified_manager = UnifiedCharacterManager()
    loaded_characters = unified_manager.load_all_characters(session_id)

    if not loaded_characters:
        raise ValueError(f"No characters found for session {session_id}")

    print(f"✓ Loaded {len(loaded_characters)} characters")

    # For each character, identify gaps using THEIR OWN ontology
    identified_gaps = []

    for character_with_artifact in loaded_characters.values():
        character = character_with_artifact.character_config
        artifact = character_with_artifact.research_artifact

        if not artifact:
            print(f"\n⚠️  Skipping {character.character_id} - no research artifact")
            continue

        print(f"\n🔍 Processing {character.character_id}...")

        # Create agent with artifact
        agent = CharacterBasedResearchAgent(
            character=character,
            session_id=session_id,
            literature_subset=[]
        )
        agent.artifact = artifact

        print(f"   Character: {character.name}")
        print(f"   Domain: {artifact.domain}")
        print(f"   Using: {character.name}'s OWN ontology")

        # Prompt: Use YOUR ontology to find gaps
        gap_identification_prompt = f"""
Review the unified research proposal and your research findings.

Using YOUR domain ontology, identify:
1. Research gaps - what's missing or unexplored
2. Alternative perspectives - different ways to approach this
3. New research ideas - extensions or related questions

Unified Proposal:
{final_proposal}

Your Research Output:
{artifact.research_output[:2000]}...

For each gap, provide:
- Gap description
- Why it's important (from YOUR ontology perspective)
- Suggested research direction
- Feasibility assessment
"""

        # Get gap identification from agent using answer_question
        print(f"   Asking agent to identify gaps...")
        gaps_text = await agent.answer_question(
            question=gap_identification_prompt,
            context={"research_output": artifact.research_output}
        )
        print(f"   ✓ Identified gaps ({len(gaps_text)} chars)")

        identified_gaps.append({
            "character_id": artifact.character_id,
            "character_name": character.name,
            "domain": artifact.domain,
            "gaps_identified": gaps_text,
            "ontology_used": "own"
        })

    print(f"\n✓ Phase 5 complete: {len(identified_gaps)} characters identified gaps\n")
    return identified_gaps


# ===== PHASE 6: SOCRATIC DIALOGUE =====

async def run_socratic_dialogue_standalone(
    session_id: str,
    final_proposal: str,
    identified_gaps: List[Dict[str, Any]],
    num_questions_per_character: int = 2
) -> List[Dict[str, Any]]:
    """Phase 6: Socratic dialogue using own ontology perspectives and gap identification.

    This is a STANDALONE function, not part of the main workflow.
    Characters generate critical questions based on their ontology gaps from Phase 5.

    Args:
        session_id: Session ID to load artifacts from
        final_proposal: The final research proposal from Phase 4
        identified_gaps: Gap identification results from Phase 5
        num_questions_per_character: Number of questions each character should ask (default: 2)

    Returns:
        List of dialogue Q&A records with hidden reasoning
    """
    print(f"\n{'='*70}")
    print(f"PHASE 6: SOCRATIC DIALOGUE (STANDALONE)")
    print(f"{'='*70}")
    print(f"Session ID: {session_id}")
    print(f"Generating questions based on ontology gaps from Phase 5...")
    print(f"{'='*70}\n")

    # Load unified characters from disk
    unified_manager = UnifiedCharacterManager()
    loaded_characters = unified_manager.load_all_characters(session_id)

    if not loaded_characters:
        raise ValueError(f"No characters found for session {session_id}")

    print(f"✓ Loaded {len(loaded_characters)} characters")

    # For each character, generate questions based on their gaps
    dialogue_questions = []

    for gap_result in identified_gaps:
        character_id = gap_result["character_id"]
        character_name = gap_result["character_name"]
        domain = gap_result["domain"]
        gaps_identified = gap_result["gaps_identified"]

        print(f"\n🎭 {character_name} generating questions...")

        # Load unified character
        character_with_artifact = loaded_characters.get(character_id)

        if not character_with_artifact:
            print(f"   ⚠️  Character not found for {character_id}, skipping")
            continue

        character = character_with_artifact.character_config
        artifact = character_with_artifact.research_artifact

        if not artifact:
            print(f"   ⚠️  Artifact not found for {character_id}, skipping")
            continue

        # Create agent
        agent = CharacterBasedResearchAgent(
            character=character,
            session_id=session_id,
            literature_subset=[]
        )
        agent.artifact = artifact

        # Prompt: Generate Socratic questions based on gaps
        question_generation_prompt = f"""
Based on the research gaps you identified from YOUR domain ontology, generate {num_questions_per_character} critical Socratic questions to ask the research team.

Your Identified Gaps:
{gaps_identified}

Final Research Proposal:
{final_proposal[:1500]}...

TASK: Generate {num_questions_per_character} thought-provoking questions that:
1. Help clarify assumptions or uncertainties in the proposal
2. Challenge the team to think deeper about the gaps you identified
3. Prepare the team for presentations by anticipating critical questions
4. Are grounded in YOUR domain ontology perspective

For EACH question, provide:
- **question**: The actual question to ask (clear, specific, thought-provoking)
- **hidden_reasoning**: Why you're asking this question from your ontology perspective (2-3 sentences explaining the gap/concern that motivated it)
- **expected_insight**: What kind of answer would address the gap (1 sentence)

Format as a numbered list:

1. Question: [Your first question]
   Hidden Reasoning: [Why you're asking this based on your ontology]
   Expected Insight: [What you hope to learn]

2. Question: [Your second question]
   Hidden Reasoning: [Why you're asking this based on your ontology]
   Expected Insight: [What you hope to learn]
"""

        # Get questions from agent
        print(f"   Asking {character_name} to generate {num_questions_per_character} questions...")
        questions_text = await agent.answer_question(
            question=question_generation_prompt,
            context={
                "research_output": artifact.research_output,
                "identified_gaps": gaps_identified
            }
        )

        print(f"   ✓ Generated questions ({len(questions_text)} chars)")

        # Parse questions (simple parsing - can be enhanced)
        dialogue_questions.append({
            "character_id": character_id,
            "character_name": character_name,
            "domain": domain,
            "questions_text": questions_text,
            "ontology_used": "own",
            "num_questions": num_questions_per_character
        })

    print(f"\n✓ Phase 6 complete: {len(dialogue_questions)} characters generated questions\n")
    return dialogue_questions


# ===== PHASE 7: INDUSTRY PARTNER REVIEW =====

async def industry_partner_review_standalone(
    session_id: str,
    final_proposal: str
) -> List[Dict[str, Any]]:
    """Phase 7: Industry partners review using OTHER researchers' ontologies.

    This is a STANDALONE function, not part of the main workflow.
    Industry partners load OTHER researchers' ontologies and provide feedback.

    Args:
        session_id: Session ID to load artifacts from
        final_proposal: The final research proposal from Phase 4

    Returns:
        List of industry partner feedback
    """
    print(f"\n{'='*70}")
    print(f"PHASE 7: INDUSTRY PARTNER REVIEW (STANDALONE)")
    print(f"{'='*70}")
    print(f"Session ID: {session_id}")
    print(f"Partners will review using OTHER researchers' ontologies...")
    print(f"{'='*70}\n")

    # Load unified characters from disk
    unified_manager = UnifiedCharacterManager()
    loaded_characters = unified_manager.load_all_characters(session_id)

    if not loaded_characters:
        raise ValueError(f"No characters found for session {session_id}")

    # Extract artifacts with research output
    loaded_artifacts = {
        char_id: char_with_artifact.research_artifact
        for char_id, char_with_artifact in loaded_characters.items()
        if char_with_artifact.research_artifact is not None
    }

    print(f"✓ Loaded {len(loaded_artifacts)} artifacts")
    print(f"   Ontologies available: {list(loaded_artifacts.keys())}")

    # Create industry partner personas
    industry_partners = [
        {
            "name": "Project Manager",
            "focus": "Project alignment, timelines, deliverables",
            "ontology_perspective": "operational"
        },
        {
            "name": "Technical Lead",
            "focus": "Implementation steps, technical feasibility",
            "ontology_perspective": "technical"
        },
        {
            "name": "Stakeholder",
            "focus": "ROI, impact, resource allocation",
            "ontology_perspective": "business"
        }
    ]

    industry_feedback = []

    # Use LLM to generate feedback from each partner
    review_model = init_chat_model(model="deepseek:deepseek-chat")

    for partner in industry_partners:
        print(f"\n👔 {partner['name']} reviewing proposal...")
        print(f"   Focus: {partner['focus']}")
        print(f"   Using ontologies: {[a.character_id for a in loaded_artifacts.values()]}")

        # Partner reviews using OTHER researchers' ontologies
        ontology_list = "\n".join([
            f"- {a.character_id} ({a.domain}): {a.research_output[:500]}..."
            for a in loaded_artifacts.values()
        ])

        review_prompt = f"""
You are a {partner['name']} reviewing this research proposal.

IMPORTANT: Review using the ontologies and perspectives of these researchers:
{ontology_list}

Research Proposal:
{final_proposal[:2000]}...

From your {partner['focus']} perspective, provide operational feedback:

1. Project Alignment:
   - How does this align with project objectives?
   - What are potential misalignments?

2. Implementation:
   - What are the concrete implementation steps?
   - What technical challenges do you foresee?

3. Resources:
   - What resources are needed (people, time, budget)?
   - What dependencies exist?

4. Risks & Mitigation:
   - What are the main risks?
   - What mitigation strategies do you recommend?

5. Timeline:
   - What is a realistic timeline?
   - What are the key milestones?

Focus on practical feasibility and actionable recommendations.
"""

        print(f"   Generating feedback...")
        response = await review_model.ainvoke([HumanMessage(content=review_prompt)])
        feedback_text = response.content

        print(f"   ✓ Feedback generated ({len(feedback_text)} chars)")

        industry_feedback.append({
            "partner_name": partner['name'],
            "focus_area": partner['focus'],
            "ontologies_reviewed": [a.character_id for a in loaded_artifacts.values()],
            "feedback": feedback_text
        })

    print(f"\n✓ Phase 7 complete: {len(industry_feedback)} partners provided feedback\n")
    return industry_feedback


# ===== CONVENIENCE FUNCTIONS =====

def get_session_artifacts(session_id: str) -> List[ConversationArtifact]:
    """Load artifacts for a session.

    Args:
        session_id: Session ID to load

    Returns:
        List of conversation artifacts
    """
    artifact_manager = ConversationArtifactManager()
    return artifact_manager.load_artifacts(session_id)


def get_session_summary(session_id: str) -> Dict[str, Any]:
    """Get summary information about a session.

    Args:
        session_id: Session ID to summarize

    Returns:
        Dictionary with session summary
    """
    artifacts = get_session_artifacts(session_id)

    return {
        "session_id": session_id,
        "artifact_count": len(artifacts),
        "characters": [
            {
                "character_id": a.character_id,
                "domain": a.domain,
                "research_output_length": len(a.research_output)
            }
            for a in artifacts
        ]
    }
