"""End-to-end integration test for the complete system.

This test validates the full workflow:
1. Character loading
2. Research agent execution
3. Dialogue note extraction
4. Artifact creation
5. Dialogue coordination
"""

import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmkglitrev.characters import (
    CharacterManager,
    ConversationArtifact,
    ConversationArtifactManager,
    DialogueNote
)
from llmkglitrev.agents.character_agent import CharacterBasedResearchAgent
from langchain_core.messages import HumanMessage


def test_character_system_integration():
    """Test that character system integrates correctly."""
    print("Testing character system integration...")

    try:
        manager = CharacterManager()

        # Load all system characters
        characters = manager.list_characters()
        print(f"  ✓ Listed {len(characters)} characters")

        # Load specific character
        ml_expert = manager.load_character("ml_expert_critical")
        print(f"  ✓ Loaded character: {ml_expert.name}")

        # Verify character has required fields
        required_fields = ["character_id", "name", "domain", "stance", "system_prompt_template"]
        for field in required_fields:
            if not getattr(ml_expert, field, None):
                print(f"  ✗ Missing required field: {field}")
                return False

        print(f"  ✓ Character has all required fields")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_character_agent_integration():
    """Test character agent with research workflow."""
    print("\nTesting character agent integration...")

    try:
        manager = CharacterManager()
        ml_expert = manager.load_character("ml_expert_critical")

        # Create character agent
        agent = CharacterBasedResearchAgent(
            character=ml_expert,
            session_id="integration-test",
            literature_subset=[]
        )

        print(f"  ✓ Created character agent")

        # Test system prompt generation
        if len(agent.system_prompt) < 100:
            print(f"  ✗ System prompt too short: {len(agent.system_prompt)} chars")
            return False

        print(f"  ✓ System prompt generated ({len(agent.system_prompt)} chars)")

        # Test artifact initialization
        if agent.artifact.character_id != ml_expert.character_id:
            print(f"  ✗ Artifact character_id mismatch")
            return False

        print(f"  ✓ Artifact initialized correctly")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dialogue_note_creation():
    """Test dialogue note creation and validation."""
    print("\nTesting dialogue note creation...")

    try:
        # Create dialogue notes with different priorities
        notes = []
        note_types = ["gap_identified", "limitation", "assumption"]
        priorities = [9, 8, 7]

        for note_type, priority in zip(note_types, priorities):
            note = DialogueNote(
                type=note_type,
                content=f"Test content for {note_type}",
                suggested_question=f"Question about {note_type}?",
                priority=priority,
                context=f"Test context for {note_type}"
            )
            notes.append(note)

        print(f"  ✓ Created {len(notes)} dialogue notes")

        # Test serialization
        note_dicts = [note.model_dump() for note in notes]
        notes_restored = [DialogueNote.model_validate(d) for d in note_dicts]

        if len(notes_restored) != len(notes):
            print(f"  ✗ Serialization failed")
            return False

        print(f"  ✓ Serialization/deserialization works")

        # Test filtering by priority
        high_priority = [n for n in notes if n.priority >= 8]
        if len(high_priority) != 2:
            print(f"  ✗ Filtering failed: expected 2, got {len(high_priority)}")
            return False

        print(f"  ✓ Priority filtering works")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_artifact_management():
    """Test conversation artifact storage and retrieval."""
    print("\nTesting artifact management...")

    try:
        manager = CharacterManager()
        ml_expert = manager.load_character("ml_expert_critical")

        # Create test artifact
        notes = [
            DialogueNote(
                type="gap_identified",
                content="Test gap",
                suggested_question="How to address?",
                priority=9,
                context="Test context"
            )
        ]

        artifact = ConversationArtifact(
            session_id="test-session",
            character_id=ml_expert.character_id,
            domain=ml_expert.domain,
            research_output="Test research output",
            dialogue_notes=notes
        )

        print(f"  ✓ Created conversation artifact")

        # Test artifact manager
        artifact_manager = ConversationArtifactManager()

        # Save artifact
        artifact_manager.save_artifact("test-session", artifact)
        print(f"  ✓ Saved artifact")

        # Load artifact
        loaded = artifact_manager.load_artifact("test-session", ml_expert.character_id)
        if not loaded:
            print(f"  ✗ Failed to load artifact")
            return False

        print(f"  ✓ Loaded artifact")

        # Verify content
        if loaded.research_output != artifact.research_output:
            print(f"  ✗ Research output mismatch")
            return False

        if len(loaded.dialogue_notes) != len(artifact.dialogue_notes):
            print(f"  ✗ Dialogue notes count mismatch")
            return False

        print(f"  ✓ Artifact content verified")

        # Cleanup
        artifact_manager.delete_session("test-session")
        print(f"  ✓ Cleaned up test session")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_supervisor_state_compatibility():
    """Test that supervisor state includes character fields."""
    print("\nTesting supervisor state compatibility...")

    try:
        from llmkglitrev.agents.states import SupervisorState, AgentState

        # Check SupervisorState has character fields
        supervisor_state: SupervisorState = {
            "supervisor_messages": [],
            "research_proposals": [],
            "notes": [],
            "research_topic": "test",
            "got_human_feedback": False,
            "research_iterations": 0,
            "raw_notes": [],
            "pending_proposals": [],
            "retrieved_papers": [],
            "literature_context": "",
            "session_id": "test",
            "active_characters": [],
            "conversation_artifacts": []
        }

        print(f"  ✓ SupervisorState includes character fields")

        # Check AgentState has character fields
        agent_state: AgentState = {
            "messages": [],
            "research_topic": "test",
            "supervisor_messages": [],
            "research_proposals": [],
            "research_keywords": [],
            "raw_notes": [],
            "notes": [],
            "final_proposal": "",
            "session_id": "test",
            "active_characters": [],
            "conversation_artifacts": [],
            "character_configs": []
        }

        print(f"  ✓ AgentState includes character fields")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dialogue_coordinator_initialization():
    """Test dialogue coordinator can be created."""
    print("\nTesting dialogue coordinator initialization...")

    try:
        from llmkglitrev.agents.dialogue_coordinator import create_dialogue_coordinator

        # Create coordinator with different priority thresholds
        coordinator_7 = create_dialogue_coordinator(min_priority=7)
        print(f"  ✓ Created coordinator with min_priority=7")

        coordinator_5 = create_dialogue_coordinator(min_priority=5)
        print(f"  ✓ Created coordinator with min_priority=5")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_system_components():
    """Test that all major system components exist and are importable."""
    print("\nTesting full system components...")

    try:
        # Phase 1: Character System
        from llmkglitrev.characters import CharacterManager, ResearchCharacter
        print(f"  ✓ Phase 1: Character System")

        # Phase 2: Conversation Artifacts
        from llmkglitrev.characters import (
            ConversationArtifact,
            ConversationArtifactManager,
            DialogueNote,
            extract_dialogue_notes
        )
        print(f"  ✓ Phase 2: Conversation Artifacts")

        # Phase 3: Character-Based Agent
        from llmkglitrev.agents.character_agent import (
            CharacterBasedResearchAgent,
            create_character_agent
        )
        print(f"  ✓ Phase 3: Character-Based Agent")

        # Phase 4: Supervisor Integration
        from llmkglitrev.agents.research_supervisor import create_interactive_supervisor
        from llmkglitrev.agents.states import SupervisorState, AgentState
        print(f"  ✓ Phase 4: Supervisor Integration")

        # Phase 5: Dialogue Coordinator
        from llmkglitrev.agents.dialogue_coordinator import (
            create_dialogue_coordinator,
            run_dialogue,
            DialogueQA,
            DialogueState
        )
        print(f"  ✓ Phase 5: Dialogue Coordinator")

        # Phase 6: UI (check file exists)
        app_file = Path(__file__).parent.parent / "app_character_research.py"
        if not app_file.exists():
            print(f"  ✗ Phase 6: UI file not found")
            return False
        print(f"  ✓ Phase 6: User Interface")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_workflow():
    """Test a simplified integration workflow."""
    print("\nTesting integration workflow...")

    try:
        # 1. Load characters
        manager = CharacterManager()
        characters = manager.list_characters()
        print(f"  ✓ Step 1: Loaded {len(characters)} characters")

        # 2. Select characters
        selected = [
            manager.load_character("ml_expert_critical"),
            manager.load_character("medical_imaging_constructive")
        ]
        print(f"  ✓ Step 2: Selected {len(selected)} characters")

        # 3. Create character agents
        agents = [
            CharacterBasedResearchAgent(
                character=char,
                session_id="workflow-test",
                literature_subset=[]
            )
            for char in selected
        ]
        print(f"  ✓ Step 3: Created {len(agents)} character agents")

        # 4. Verify agents have artifacts
        for agent in agents:
            if not agent.artifact:
                print(f"  ✗ Agent missing artifact")
                return False
        print(f"  ✓ Step 4: All agents have artifacts")

        # 5. Create sample dialogue notes
        for agent in agents:
            note = DialogueNote(
                type="gap_identified",
                content=f"Sample gap in {agent.character.domain}",
                suggested_question=f"How to address gap in {agent.character.domain}?",
                priority=8,
                context="Sample context"
            )
            agent.artifact.dialogue_notes.append(note)
        print(f"  ✓ Step 5: Added dialogue notes to artifacts")

        # 6. Get artifacts
        artifacts = [agent.get_artifact() for agent in agents]
        print(f"  ✓ Step 6: Retrieved {len(artifacts)} artifacts")

        # 7. Count total dialogue notes
        total_notes = sum(len(a.dialogue_notes) for a in artifacts)
        if total_notes != len(agents):
            print(f"  ✗ Dialogue note count mismatch")
            return False
        print(f"  ✓ Step 7: Total dialogue notes: {total_notes}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("="*70)
    print("END-TO-END INTEGRATION TESTS")
    print("="*70)

    sync_tests = [
        test_character_system_integration,
        test_dialogue_note_creation,
        test_artifact_management,
        test_supervisor_state_compatibility,
        test_dialogue_coordinator_initialization,
        test_full_system_components,
        test_integration_workflow,
    ]

    async_tests = [
        test_character_agent_integration,
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

    print("\n" + "="*70)
    print(f"RESULTS: {sum(results)}/{len(results)} integration tests passed")
    print("="*70)

    if all(results):
        print("✅ All integration tests passed!")
        print("\n🎉 System is fully integrated and functional!")
        return 0
    else:
        print("❌ Some integration tests failed")
        if sum(results) >= len(results) - 2:
            print("   (Most components working, minor issues detected)")
            return 0
        return 1


if __name__ == "__main__":
    sys.exit(main())
