"""
Demonstration: Ontology-Enhanced Research Agent

This script shows how domain ontologies can enhance research agents by:
1. Validating terminology in research papers
2. Identifying related concepts not mentioned
3. Generating ontology-grounded Socratic questions
4. Detecting missing relationships between concepts
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmkglitrev.ontology import DomainOntologyManager


def demo_basic_usage():
    """Demonstrate basic ontology loading and querying."""
    print("="*80)
    print("DEMO 1: Basic Ontology Usage")
    print("="*80)

    # Create ontology manager for machine learning domain
    print("\n📚 Creating ontology manager for EDAM (includes ML concepts)...")
    ontology = DomainOntologyManager(ontology_source="edam")

    # Load ontology
    success = ontology.load()

    if not success:
        print("❌ Failed to load ontology")
        return None

    # Get statistics
    stats = ontology.get_statistics()
    print(f"\n📊 Ontology Statistics:")
    print(f"   Name: {stats['name']}")
    print(f"   Total classes: {stats['total_classes']}")
    print(f"   Object properties: {stats['object_properties']}")
    print(f"   Indexed terms: {stats['indexed_terms']}")

    return ontology


def demo_concept_search(ontology: DomainOntologyManager):
    """Demonstrate searching for concepts."""
    print("\n" + "="*80)
    print("DEMO 2: Concept Search")
    print("="*80)

    search_terms = [
        "neural network",
        "machine learning",
        "classification",
        "deep learning",
        "supervised learning"
    ]

    print("\n🔍 Searching for ML-related concepts...")

    for term in search_terms:
        print(f"\n  Term: '{term}'")
        concept = ontology.find_concept(term)

        if concept:
            print(f"  ✅ Found: {concept.name}")
            definition = ontology._get_definition(concept)
            if definition:
                print(f"  📖 Definition: {definition[:100]}...")
        else:
            print(f"  ❌ Not found in ontology")


def demo_related_concepts(ontology: DomainOntologyManager):
    """Demonstrate finding related concepts."""
    print("\n" + "="*80)
    print("DEMO 3: Finding Related Concepts")
    print("="*80)

    test_term = "classification"

    print(f"\n🔗 Finding concepts related to '{test_term}'...")

    related = ontology.find_related_concepts(test_term, max_results=10)

    if related:
        print(f"\n✅ Found {len(related)} related concepts:")

        for i, concept_info in enumerate(related, 1):
            print(f"\n  {i}. {concept_info['name']}")
            print(f"     Relationship: {concept_info['type']}")
            definition = concept_info['definition']
            if definition != "No definition available":
                print(f"     Definition: {definition[:80]}...")
    else:
        print(f"❌ No related concepts found")


def demo_terminology_validation(ontology: DomainOntologyManager):
    """Demonstrate validating terminology in research text."""
    print("\n" + "="*80)
    print("DEMO 4: Terminology Validation")
    print("="*80)

    # Sample research abstract about transfer learning
    research_text = """
    We propose a novel Transfer Learning approach for Medical Imaging with Small Datasets.
    Our method uses Convolutional Neural Networks pre-trained on ImageNet and applies
    Domain Adaptation techniques. We evaluate using Cross-Validation and achieve
    high Accuracy on chest X-ray Classification tasks. Data Augmentation was used
    to address Overfitting issues.
    """

    print("\n📝 Research Abstract:")
    print(research_text)

    print("\n🔍 Validating terminology against ontology...")

    validation_results = ontology.validate_terminology(research_text)

    found = [r for r in validation_results if r["status"] == "found"]
    not_found = [r for r in validation_results if r["status"] == "not_found"]

    print(f"\n✅ Terms found in ontology ({len(found)}):")
    for result in found:
        print(f"  • {result['term']}")
        print(f"    → Ontology concept: {result['ontology_concept']}")

    if not_found:
        print(f"\n⚠️  Terms not found in ontology ({len(not_found)}):")
        for result in not_found:
            print(f"  • {result['term']}")
            print(f"    (May need clarification or be domain-specific)")


def demo_question_generation(ontology: DomainOntologyManager):
    """Demonstrate generating Socratic questions."""
    print("\n" + "="*80)
    print("DEMO 5: Ontology-Based Question Generation")
    print("="*80)

    # Sample research output
    research_output = """
    Our research explores Transfer Learning for Medical Imaging tasks.
    We use Convolutional Neural Networks for image Classification.
    The model achieves high Accuracy on our test dataset.
    We address Overfitting through Data Augmentation.
    """

    print("\n📝 Research Output:")
    print(research_output)

    print("\n🤔 Generating Socratic questions based on ontology analysis...")

    questions = ontology.generate_competency_questions(
        research_output,
        priority_threshold=6  # Lower threshold to see more questions
    )

    if questions:
        print(f"\n✅ Generated {len(questions)} questions:\n")

        for i, q in enumerate(questions, 1):
            print(f"{i}. [{q['type']}] Priority: {q['priority']}")
            print(f"   Q: {q['question']}")
            print(f"   Concept: {q['concept']}")
            if q['definition'] != "No definition available":
                print(f"   Context: {q['definition'][:80]}...")
            print()
    else:
        print("⚠️  No questions generated (all concepts may be covered)")


def demo_missing_relationships(ontology: DomainOntologyManager):
    """Demonstrate finding missing relationships."""
    print("\n" + "="*80)
    print("DEMO 6: Identifying Missing Relationships")
    print("="*80)

    # Terms mentioned in the research
    mentioned_terms = [
        "classification",
        "neural network",
        "supervised learning",
        "machine learning"
    ]

    print(f"\n📋 Terms mentioned in research:")
    for term in mentioned_terms:
        print(f"  • {term}")

    print(f"\n🔗 Checking for relationships between these concepts...")

    missing_rels = ontology.get_missing_relationships(mentioned_terms)

    if missing_rels:
        print(f"\n✅ Found {len(missing_rels)} relationships to explore:\n")

        for i, rel in enumerate(missing_rels, 1):
            print(f"{i}. {rel['relationship']}")
            print(f"   Concepts: {rel['concept1']} ↔ {rel['concept2']}")
            print(f"   💡 Question: How does this relationship affect your approach?")
            print()
    else:
        print("⚠️  No implicit relationships found")


def demo_enhanced_dialogue_notes(ontology: DomainOntologyManager):
    """Show how ontology enhances dialogue note creation."""
    print("\n" + "="*80)
    print("DEMO 7: Creating Enhanced Dialogue Notes")
    print("="*80)

    research_output = """
    We developed a Deep Learning model for Medical Image Classification.
    The model uses Transfer Learning from ImageNet pretrained weights.
    Performance was evaluated using standard Accuracy metrics.
    """

    print("\n📝 Research Output:")
    print(research_output)

    # Generate questions
    questions = ontology.generate_competency_questions(
        research_output,
        priority_threshold=5
    )

    print(f"\n📋 Creating Dialogue Notes from Ontology Analysis...")
    print(f"   (These would be integrated into ConversationArtifact)")

    if questions:
        print(f"\n✅ Generated {len(questions)} potential dialogue notes:\n")

        # Map question types to dialogue note types
        type_mapping = {
            "broader_context": "gap_identified",
            "alternative_approach": "alternative_approach",
            "comparison": "methodological_choice",
            "relationship_clarification": "assumption"
        }

        for i, q in enumerate(questions[:5], 1):  # Show first 5
            note_type = type_mapping.get(q['type'], 'gap_identified')

            print(f"{i}. DialogueNote(")
            print(f"     type=\"{note_type}\",")
            print(f"     content=\"{q['concept']}\",")
            print(f"     suggested_question=\"{q['question']}\",")
            print(f"     priority={q['priority']},")
            print(f"     context=\"Ontology: {q['definition'][:50]}...\"")
            print(f"   )\n")
    else:
        print("⚠️  No questions generated")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("🎓 ONTOLOGY-ENHANCED RESEARCH AGENT DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows how domain ontologies can improve research agents")
    print("by providing structured domain knowledge for:")
    print("  • Terminology validation")
    print("  • Concept relationship discovery")
    print("  • Socratic question generation")
    print("  • Knowledge gap identification\n")

    # Run demos
    ontology = demo_basic_usage()

    if not ontology:
        print("\n❌ Cannot continue without ontology")
        return

    demo_concept_search(ontology)
    demo_related_concepts(ontology)
    demo_terminology_validation(ontology)
    demo_question_generation(ontology)
    demo_missing_relationships(ontology)
    demo_enhanced_dialogue_notes(ontology)

    # Summary
    print("\n" + "="*80)
    print("📊 DEMONSTRATION SUMMARY")
    print("="*80)

    print("\n✅ Successfully demonstrated:")
    print("  1. Loading domain ontology (EDAM)")
    print("  2. Searching for concepts")
    print("  3. Finding related concepts")
    print("  4. Validating research terminology")
    print("  5. Generating Socratic questions")
    print("  6. Identifying missing relationships")
    print("  7. Creating enhanced dialogue notes")

    print("\n💡 Next Steps:")
    print("  • Integrate DomainOntologyManager into CharacterBasedResearchAgent")
    print("  • Enhance dialogue note generation with ontology insights")
    print("  • Add ontology context to LLM prompts for better questions")
    print("  • Create domain-specific ontology managers for each character")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
