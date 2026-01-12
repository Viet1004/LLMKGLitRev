"""
Complete Example: Ontology-Enhanced Research Agent

This script demonstrates the full workflow of using an ontology-enhanced
character agent for research, showing:
1. Creating an agent with ontology support
2. Conducting research with automatic ontology analysis
3. Viewing enhanced dialogue notes
4. Using ontology context for better questions
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmkglitrev.characters import CharacterManager
from llmkglitrev.agents.ontology_enhanced_character_agent import (
    OntologyEnhancedCharacterAgent,
    create_ontology_enhanced_agent
)
from llmkglitrev.ontology import DomainOntologyManager


async def example_basic_workflow():
    """
    Example 1: Basic workflow with ontology-enhanced agent.
    """
    print("="*80)
    print("EXAMPLE 1: Basic Ontology-Enhanced Research Workflow")
    print("="*80)

    # Load a character
    print("\n📚 Step 1: Loading ML Expert character...")
    char_manager = CharacterManager()
    ml_expert = char_manager.load_character("ml_expert_critical")
    print(f"   ✅ Loaded: {ml_expert.name}")
    print(f"   Domain: {ml_expert.domain}")
    print(f"   Stance: {ml_expert.stance}")

    # Create ontology-enhanced agent
    print("\n🧠 Step 2: Creating ontology-enhanced agent...")
    agent = OntologyEnhancedCharacterAgent(
        character=ml_expert,
        ontology_source="edam",  # EDAM includes ML concepts
        session_id="example-session-001"
    )

    print(f"\n   ✅ Created: {agent}")

    if agent.ontology:
        stats = agent.get_ontology_statistics()
        print(f"\n   📊 Ontology: {stats['name']}")
        print(f"      Classes: {stats['total_classes']}")
        print(f"      Indexed terms: {stats['indexed_terms']}")

    # Mock research (simulating what research_agent would return)
    print("\n🔬 Step 3: Conducting research (MOCK - no real LLM calls)...")

    # Create mock artifact manually
    from llmkglitrev.characters.artifact import ConversationArtifact, DialogueNote

    mock_research_output = """
    Transfer Learning for Medical Imaging with Small Datasets

    Our research proposes a novel approach using Convolutional Neural Networks
    pre-trained on ImageNet for Medical Image Classification tasks. We address
    the challenge of Small Datasets through Data Augmentation and Domain Adaptation
    techniques.

    Key findings:
    1. Transfer Learning significantly improves Accuracy on medical imaging tasks
    2. Data Augmentation helps mitigate Overfitting on small datasets
    3. Domain Adaptation is critical for bridging natural image and medical image domains
    4. Cross-Validation confirms the robustness of our approach

    Limitations:
    - ImageNet pre-training may not capture medical-specific features
    - Small dataset size limits model complexity
    - Domain gap between natural and medical images remains a challenge
    """

    # Create base dialogue notes (what research_agent would generate)
    base_notes = [
        DialogueNote(
            type="assumption",
            content="Assumption that ImageNet pre-training transfers to medical domains",
            suggested_question="What evidence supports that ImageNet features transfer well to medical imaging?",
            priority=8,
            context="Transfer learning methodology"
        ),
        DialogueNote(
            type="limitation",
            content="Domain gap between natural and medical images",
            suggested_question="How can we better address the domain gap in transfer learning?",
            priority=7,
            context="Identified limitation"
        )
    ]

    # Create artifact
    agent.artifact = ConversationArtifact(
        session_id="example-session-001",
        character_id=ml_expert.character_id,
        domain=ml_expert.domain,
        research_output=mock_research_output,
        dialogue_notes=base_notes,
        raw_notes=[mock_research_output]
    )

    print(f"\n   ✅ Mock research complete")
    print(f"   📄 Research output: {len(mock_research_output)} characters")
    print(f"   💬 Base dialogue notes: {len(base_notes)}")

    # Enhance with ontology
    print("\n🧠 Step 4: Enhancing with ontology analysis...")

    if agent.ontology:
        enhanced_notes = agent._generate_ontology_dialogue_notes(
            mock_research_output,
            min_priority=6
        )

        # Add new notes (avoiding duplicates)
        existing_questions = {note.suggested_question for note in agent.artifact.dialogue_notes}
        new_notes = [
            note for note in enhanced_notes
            if note.suggested_question not in existing_questions
        ]

        agent.artifact.dialogue_notes.extend(new_notes)

        print(f"   ✅ Added {len(new_notes)} ontology-based notes")
        print(f"   📊 Total dialogue notes: {len(agent.artifact.dialogue_notes)}")

    # Display dialogue notes
    print("\n💬 Step 5: Reviewing Dialogue Notes...")
    print("\n   High-Priority Questions for Socratic Dialogue:\n")

    # Sort by priority
    sorted_notes = sorted(agent.artifact.dialogue_notes, key=lambda x: x.priority, reverse=True)

    for i, note in enumerate(sorted_notes[:8], 1):  # Show top 8
        print(f"   {i}. [{note.type}] Priority: {note.priority}")
        print(f"      Q: {note.suggested_question}")
        print(f"      Context: {note.context[:60]}...")
        print()

    # Terminology validation
    print("\n📋 Step 6: Validating Terminology...")
    if agent.ontology:
        validation = agent.validate_research_terminology()
        print(f"   Terms analyzed: {validation['total_terms']}")
        print(f"   ✅ Found in ontology: {validation['found_in_ontology']}")
        print(f"   ⚠️  Not found: {validation['not_found']}")

        # Show a few examples
        print(f"\n   Examples of validated terms:")
        found_terms = [r for r in validation['details'] if r['status'] == 'found'][:3]
        for term_info in found_terms:
            print(f"      • {term_info['term']} → {term_info['ontology_concept']}")

    return agent


async def example_convenience_function():
    """
    Example 2: Using the convenience function.
    """
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Using Convenience Function")
    print("="*80)

    print("\n✨ Creating agent with one function call...")

    agent = create_ontology_enhanced_agent(
        character_id="medical_imaging_constructive",
        ontology_source="edam",
        session_id="example-session-002"
    )

    print(f"\n✅ Created: {agent}")
    print(f"   Character: {agent.character.name}")
    print(f"   Domain: {agent.character.domain}")
    print(f"   Stance: {agent.character.stance}")

    if agent.ontology:
        stats = agent.get_ontology_statistics()
        print(f"   Ontology: {stats['name']} ({stats['total_classes']} classes)")

    return agent


async def example_ontology_context_for_questions():
    """
    Example 3: Using ontology context when answering questions.
    """
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Ontology Context for Question Answering")
    print("="*80)

    # Create agent
    agent = create_ontology_enhanced_agent(
        character_id="ml_expert_critical",
        session_id="example-session-003"
    )

    # Simulate asking a question
    question = "Why is Transfer Learning effective for Medical Image Classification with limited data?"

    print(f"\n❓ Question: {question}")

    # Build ontology context (without actually calling LLM)
    if agent.ontology:
        print(f"\n🧠 Building ontology context...")
        ontology_context = agent._build_ontology_context(question)

        if ontology_context:
            print(f"\n📚 Ontology context that would be added to prompt:")
            print(ontology_context)
        else:
            print(f"\n⚠️  No ontology concepts found for this question")

        print(f"\n💡 This context would help the LLM provide:")
        print("   • More precise terminology")
        print("   • Grounded definitions")
        print("   • Domain-aware explanations")


async def example_comparison():
    """
    Example 4: Comparing base vs ontology-enhanced agents.
    """
    print("\n\n" + "="*80)
    print("EXAMPLE 4: Comparison - Base vs Ontology-Enhanced")
    print("="*80)

    from llmkglitrev.agents.character_agent import CharacterBasedResearchAgent

    # Load character
    char_manager = CharacterManager()
    character = char_manager.load_character("ml_expert_critical")

    print("\n📊 Creating two agents for comparison...\n")

    # Create base agent
    print("1. Base CharacterBasedResearchAgent")
    base_agent = CharacterBasedResearchAgent(
        character=character,
        session_id="comparison-base"
    )
    print(f"   ✅ Created: {base_agent}")
    print(f"   Features: Standard research + dialogue notes")

    # Create ontology-enhanced agent
    print("\n2. OntologyEnhancedCharacterAgent")
    enhanced_agent = OntologyEnhancedCharacterAgent(
        character=character,
        ontology_source="edam",
        session_id="comparison-enhanced"
    )
    print(f"   ✅ Created: {enhanced_agent}")
    print(f"   Features: Standard research + dialogue notes + ontology analysis")

    if enhanced_agent.ontology:
        stats = enhanced_agent.get_ontology_statistics()
        print(f"   Ontology: {stats['name']}")
        print(f"   Knowledge base: {stats['total_classes']} concepts")

    print("\n💡 Key Differences:")
    print("   Base Agent:")
    print("   • Generates dialogue notes from research process only")
    print("   • Questions based on what's mentioned in findings")
    print("   • No external knowledge validation")

    print("\n   Ontology-Enhanced Agent:")
    print("   • All base agent capabilities PLUS:")
    print("   • Validates terminology against domain ontology")
    print("   • Identifies missing related concepts")
    print("   • Generates ontology-grounded questions")
    print("   • Provides richer context for Socratic dialogue")
    print("   • Catches terminology mismatches and gaps")


async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("🎓 ONTOLOGY-ENHANCED CHARACTER AGENT - COMPLETE EXAMPLES")
    print("="*80)

    print("\nThese examples demonstrate how domain ontologies enhance research agents.")
    print("\nNOTE: These examples use MOCK data (no real LLM calls) to demonstrate")
    print("the ontology integration. In production, the research would be real.\n")

    # Run examples
    agent1 = await example_basic_workflow()
    agent2 = await example_convenience_function()
    await example_ontology_context_for_questions()
    await example_comparison()

    # Final summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)

    print("\n✅ Successfully demonstrated:")
    print("  1. Creating ontology-enhanced agents")
    print("  2. Automatic ontology analysis during research")
    print("  3. Enhanced dialogue note generation")
    print("  4. Terminology validation against domain knowledge")
    print("  5. Ontology context for better question answering")
    print("  6. Comparison with base agent")

    print("\n🎯 Key Benefits of Ontology Integration:")
    print("  • ✅ Better questions grounded in domain knowledge")
    print("  • ✅ Catches terminology mismatches and unclear concepts")
    print("  • ✅ Identifies missing relationships and alternative approaches")
    print("  • ✅ Provides structured knowledge to enhance LLM responses")
    print("  • ✅ Enables systematic exploration of domain concepts")

    print("\n📚 Next Steps to Use This in Production:")
    print("  1. Replace mock research with real conduct_research() calls")
    print("  2. Integrate into research_supervisor.py workflow")
    print("  3. Use OntologyEnhancedCharacterAgent instead of CharacterBasedResearchAgent")
    print("  4. Configure domain-specific ontologies for each character")
    print("  5. Adjust priority thresholds based on research needs")

    print("\n💡 Advanced Options:")
    print("  • Load custom ontologies for specific domains")
    print("  • Combine multiple ontologies (ML + Medical + Ethics)")
    print("  • Fine-tune question generation priorities")
    print("  • Create domain-specific ontology mappings")

    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())
