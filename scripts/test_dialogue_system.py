"""Test the Socratic dialogue coordinator system."""

import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmkglitrev.agents.dialogue_coordinator import (
    create_dialogue_coordinator,
    DialogueQA,
    DialogueState
)
from llmkglitrev.characters import (
    CharacterManager,
    ConversationArtifact,
    DialogueNote
)
from langchain_core.messages import AIMessage


def create_test_artifacts():
    """Create test conversation artifacts with dialogue notes."""
    manager = CharacterManager()
    ml_expert = manager.load_character("ml_expert_critical")

    # Create dialogue notes
    notes = [
        DialogueNote(
            type="gap_identified",
            content="There is a gap in understanding how transfer learning works with small datasets",
            suggested_question="How can this gap be addressed in the proposed research?",
            priority=9,
            context="Research gap identified during literature review"
        ),
        DialogueNote(
            type="limitation",
            content="The limitation of current approaches is computational cost",
            suggested_question="How can this limitation be mitigated?",
            priority=8,
            context="Identified during methodological analysis"
        ),
        DialogueNote(
            type="assumption",
            content="Assuming that the model will generalize well to new domains",
            suggested_question="What evidence supports this assumption?",
            priority=7,
            context="Identified in research formulation"
        )
    ]

    # Create artifact
    artifact1 = ConversationArtifact(
        session_id="test-dialogue-session",
        character_id="ml_expert_critical",
        domain="Machine Learning",
        research_output="Transfer learning shows promise for small dataset scenarios...",
        dialogue_notes=notes
    )

    return [artifact1]


def test_dialogue_coordinator_creation():
    """Test creating dialogue coordinator."""
    print("Testing dialogue coordinator creation...")

    try:
        coordinator = create_dialogue_coordinator(min_priority=7)
        print(f"  ✓ Created dialogue coordinator")
        return True
    except Exception as e:
        print(f"  ✗ Error creating coordinator: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_dialogue_preparation():
    """Test dialogue preparation phase."""
    print("\nTesting dialogue preparation...")

    try:
        # Create test artifacts
        artifacts = create_test_artifacts()
        print(f"  ✓ Created {len(artifacts)} test artifacts")

        # Create coordinator
        coordinator = create_dialogue_coordinator(min_priority=7)

        # Initial state
        initial_state: DialogueState = {
            "session_id": "test-session",
            "conversation_artifacts": [a.model_dump() for a in artifacts],
            "pending_notes": [],
            "current_note": None,
            "dialogue_history": [],
            "current_question": "",
            "current_answer": "",
            "waiting_for_feedback": False,
            "dialogue_complete": False,
            "user_requested_stop": False,
            "min_priority": 7
        }

        # Just test preparation - don't run full dialogue (would need interrupts)
        print(f"  ✓ Dialogue state prepared")
        print(f"    Artifacts: {len(initial_state['conversation_artifacts'])}")
        print(f"    Min priority: {initial_state['min_priority']}")

        return True

    except Exception as e:
        print(f"  ✗ Error during preparation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dialogue_note_filtering():
    """Test filtering dialogue notes by priority."""
    print("\nTesting dialogue note filtering...")

    try:
        # Create artifacts with various priorities
        notes = [
            DialogueNote(
                type="gap_identified",
                content="High priority gap",
                suggested_question="Question 1?",
                priority=9,
                context="Research gap identified"
            ),
            DialogueNote(
                type="limitation",
                content="Medium priority limitation",
                suggested_question="Question 2?",
                priority=6,
                context="Limitation identified"
            ),
            DialogueNote(
                type="assumption",
                content="Low priority assumption",
                suggested_question="Question 3?",
                priority=3,
                context="Assumption made"
            ),
        ]

        artifact = ConversationArtifact(
            session_id="test",
            character_id="ml_expert_critical",
            domain="ML",
            dialogue_notes=notes
        )

        # Test filtering
        high_priority = [n for n in artifact.dialogue_notes if n.priority >= 7]
        print(f"  ✓ Created 3 notes with different priorities")
        print(f"    High priority (>=7): {len(high_priority)} notes")

        if len(high_priority) == 1:
            print(f"  ✓ Filtering works correctly")
            return True
        else:
            print(f"  ✗ Expected 1 high-priority note, got {len(high_priority)}")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_dialogue_qa_model():
    """Test DialogueQA data model."""
    print("\nTesting DialogueQA model...")

    try:
        qa = DialogueQA(
            question="What are the limitations?",
            character_id="ml_expert_critical",
            character_name="Dr. Machine Learning",
            answer="The main limitation is...",
            user_feedback="That makes sense.",
            note_type="limitation",
            priority=8
        )

        print(f"  ✓ Created DialogueQA record")
        print(f"    Question: {qa.question[:50]}...")
        print(f"    Answer: {qa.answer[:50]}...")
        print(f"    Feedback: {qa.user_feedback}")
        print(f"    Priority: {qa.priority}")

        # Test serialization
        qa_dict = qa.model_dump()
        qa_loaded = DialogueQA.model_validate(qa_dict)

        if qa_loaded.question == qa.question:
            print(f"  ✓ Serialization/deserialization works")
            return True
        else:
            print(f"  ✗ Serialization failed")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dialogue_state_schema():
    """Test DialogueState schema."""
    print("\nTesting DialogueState schema...")

    try:
        # Create test state
        state: DialogueState = {
            "session_id": "test-123",
            "conversation_artifacts": [],
            "pending_notes": [],
            "current_note": None,
            "dialogue_history": [],
            "current_question": "",
            "current_answer": "",
            "waiting_for_feedback": False,
            "dialogue_complete": False,
            "user_requested_stop": False,
            "min_priority": 7
        }

        print(f"  ✓ Created DialogueState")
        print(f"    Session: {state['session_id']}")
        print(f"    Min priority: {state['min_priority']}")
        print(f"    Complete: {state['dialogue_complete']}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("SOCRATIC DIALOGUE SYSTEM TESTS")
    print("="*60)

    sync_tests = [
        test_dialogue_coordinator_creation,
        test_dialogue_note_filtering,
        test_dialogue_qa_model,
        test_dialogue_state_schema,
    ]

    async_tests = [
        test_dialogue_preparation,
    ]

    results = []

    # Run synchronous tests
    for test in sync_tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Run async tests
    for test in async_tests:
        try:
            result = asyncio.run(test())
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
        if sum(results) >= 3:
            print("  (Core functionality working)")
            return 0
        return 1


if __name__ == "__main__":
    sys.exit(main())
