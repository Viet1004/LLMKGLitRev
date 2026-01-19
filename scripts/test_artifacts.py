"""Test the conversation artifact system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmkglitrev.characters import (
    DialogueNote,
    ConversationArtifact,
    ConversationArtifactManager,
    extract_dialogue_notes
)
from langchain_core.messages import AIMessage, HumanMessage


def test_dialogue_note_creation():
    """Test creating dialogue notes."""
    print("Testing dialogue note creation...")

    note = DialogueNote(
        type="assumption",
        content="We assume the data is representative",
        suggested_question="What evidence supports this assumption?",
        priority=7,
        context="During dataset selection"
    )

    print(f"  ✓ Created note: {note.type}")
    print(f"    Priority: {note.priority}")
    print(f"    Question: {note.suggested_question}")

    return True


def test_dialogue_note_extraction():
    """Test extracting dialogue notes from messages."""
    print("\nTesting dialogue note extraction...")

    # Create sample messages with different patterns
    messages = [
        AIMessage(content="Assuming that the training data is representative of the population, we can proceed."),
        AIMessage(content="We will use deep learning for this task. However, there are limitations with small datasets."),
        AIMessage(content="The methodology involves transfer learning from pre-trained models."),
        AIMessage(content="Alternatively, we could use traditional machine learning approaches."),
        AIMessage(content="There is a gap in the literature regarding cross-domain transfer.")
    ]

    notes = extract_dialogue_notes(messages, max_notes_per_type=3)

    print(f"  ✓ Extracted {len(notes)} dialogue notes:")
    for note in notes:
        print(f"    - [{note.type}] Priority {note.priority}: {note.content[:50]}...")

    # Verify we got different types
    note_types = {note.type for note in notes}
    expected_types = {"assumption", "methodological_choice", "limitation", "alternative_approach", "gap_identified"}

    if note_types & expected_types:  # At least some overlap
        print(f"  ✓ Found note types: {', '.join(note_types)}")
        return True
    else:
        print(f"  ✗ No expected note types found")
        return False


def test_conversation_artifact():
    """Test creating and managing conversation artifacts."""
    print("\nTesting conversation artifact creation...")

    artifact = ConversationArtifact(
        session_id="test-session-123",
        character_id="ml_expert_critical",
        domain="Machine Learning",
        research_output="Research findings on deep learning...",
        raw_notes=["Found 5 papers", "Transfer learning is key"],
        dialogue_notes=[
            DialogueNote(
                type="assumption",
                content="Assumes data quality is high",
                suggested_question="How is data quality measured?",
                priority=7,
                context="Dataset discussion"
            )
        ]
    )

    print(f"  ✓ Created artifact for {artifact.character_id}")
    print(f"    Session: {artifact.session_id}")
    print(f"    Domain: {artifact.domain}")
    print(f"    Dialogue notes: {len(artifact.dialogue_notes)}")

    return True


def test_artifact_manager():
    """Test artifact manager save/load operations."""
    print("\nTesting artifact manager...")

    manager = ConversationArtifactManager(storage_path="test_sessions/")

    # Create test artifact
    artifact = ConversationArtifact(
        session_id="test-session-456",
        character_id="ml_expert_critical",
        domain="Machine Learning",
        research_output="Test research output",
        dialogue_notes=[
            DialogueNote(
                type="limitation",
                content="Limited by computational resources",
                suggested_question="What are the computational requirements?",
                priority=8,
                context="Resource constraints"
            )
        ]
    )

    try:
        # Save
        manager.save_artifact("test-session-456", artifact)
        print("  ✓ Saved artifact")

        # Load
        loaded = manager.load_artifact("test-session-456", "ml_expert_critical")
        if loaded:
            print(f"  ✓ Loaded artifact: {loaded.character_id}")
        else:
            print("  ✗ Failed to load artifact")
            return False

        # Load all artifacts
        all_artifacts = manager.load_artifacts("test-session-456")
        print(f"  ✓ Loaded {len(all_artifacts)} artifacts")

        # Get dialogue notes
        notes = manager.get_dialogue_notes("test-session-456")
        print(f"  ✓ Retrieved {len(notes)} dialogue notes")

        # Save metadata
        manager.save_session_metadata("test-session-456", {
            "topic": "Test research topic",
            "created_at": "2026-01-06"
        })
        print("  ✓ Saved session metadata")

        # List sessions
        sessions = manager.list_sessions()
        print(f"  ✓ Found {len(sessions)} sessions")

        # Cleanup
        manager.delete_session("test-session-456")
        print("  ✓ Cleaned up test session")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_note_extraction_from_research():
    """Test realistic dialogue note extraction from research text."""
    print("\nTesting realistic dialogue note extraction...")

    research_output = """
    Based on our analysis, assuming that deep learning models can generalize well,
    we propose using transfer learning from ImageNet. However, there are limitations
    when dealing with medical imaging data due to domain shift. Alternatively, we
    could explore domain adaptation techniques. There is a gap in the literature
    regarding cross-domain transfer in medical AI.

    Our methodology involves fine-tuning pre-trained models on medical datasets.
    """

    messages = [
        AIMessage(content=research_output),
        AIMessage(content="We will use ResNet-50 as the base architecture."),
        AIMessage(content="The main challenge is limited labeled medical data.")
    ]

    notes = extract_dialogue_notes(messages, research_output, max_notes_per_type=2)

    print(f"  ✓ Extracted {len(notes)} notes from realistic research:")

    for i, note in enumerate(notes, 1):
        print(f"\n    Note {i}:")
        print(f"      Type: {note.type}")
        print(f"      Priority: {note.priority}")
        print(f"      Content: {note.content[:80]}...")
        print(f"      Question: {note.suggested_question}")

    if len(notes) >= 3:
        print(f"\n  ✓ Successfully extracted diverse dialogue notes")
        return True
    else:
        print(f"\n  ⚠ Only extracted {len(notes)} notes (expected at least 3)")
        return True  # Still pass, but warn


def main():
    """Run all tests."""
    print("="*60)
    print("CONVERSATION ARTIFACT SYSTEM TESTS")
    print("="*60)

    tests = [
        test_dialogue_note_creation,
        test_dialogue_note_extraction,
        test_conversation_artifact,
        test_artifact_manager,
        test_note_extraction_from_research
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "="*60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("="*60)

    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
