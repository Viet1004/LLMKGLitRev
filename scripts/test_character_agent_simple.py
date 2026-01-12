"""Simple tests for character-based research agent (no LLM calls)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmkglitrev.agents.character_agent import CharacterBasedResearchAgent, create_character_agent
from llmkglitrev.characters import CharacterManager, DialogueNote


def test_agent_creation():
    """Test creating agents from characters."""
    print("Testing agent creation...")

    manager = CharacterManager()

    # Test creating agent from each system character
    character_ids = [
        "ml_expert_critical",
        "medical_imaging_constructive",
        "ethics_neutral",
        "computer_science_researcher"
    ]

    agents = []
    for char_id in character_ids:
        try:
            agent = create_character_agent(char_id)
            agents.append(agent)
            print(f"  ✓ Created {agent.character.name}")
            print(f"    Domain: {agent.character.domain}")
            print(f"    Stance: {agent.character.stance}")
            print(f"    Session: {agent.session_id[:16]}...")
        except Exception as e:
            print(f"  ✗ Failed to create agent for {char_id}: {e}")
            return False

    if len(agents) == len(character_ids):
        print(f"\n  ✓ Successfully created {len(agents)} agents")
        return True
    else:
        return False


def test_system_prompt():
    """Test system prompt generation."""
    print("\nTesting system prompt generation...")

    agent = create_character_agent("ml_expert_critical")

    print(f"  System prompt length: {len(agent.system_prompt)} characters")

    # Check prompt contains key elements
    checks = [
        ("Dr. Machine Learning" in agent.system_prompt, "Character name"),
        ("Machine Learning" in agent.system_prompt, "Domain"),
        ("critical" in agent.system_prompt.lower(), "Stance"),
        ("deep learning" in agent.system_prompt.lower(), "Sub-domain"),
        ("NeurIPS" in agent.system_prompt or "neurips" in agent.system_prompt.lower(), "Venue"),
    ]

    passed = 0
    for check, desc in checks:
        if check:
            print(f"  ✓ Contains {desc}")
            passed += 1
        else:
            print(f"  ✗ Missing {desc}")

    print(f"\n  Passed {passed}/{len(checks)} checks")

    # Show preview
    print(f"\n  System prompt preview:")
    print(f"  " + "="*56)
    for line in agent.system_prompt.split('\n')[:10]:
        print(f"  {line}")
    print(f"  " + "="*56)

    return passed >= 4  # Require at least 4/5 checks


def test_artifact_initialization():
    """Test that artifacts are properly initialized."""
    print("\nTesting artifact initialization...")

    agent = create_character_agent("ml_expert_critical", session_id="test-123")

    # Check artifact exists
    artifact = agent.get_artifact()

    checks = [
        (artifact.session_id == "test-123", "Session ID"),
        (artifact.character_id == "ml_expert_critical", "Character ID"),
        (artifact.domain == "Machine Learning", "Domain"),
        (isinstance(artifact.dialogue_notes, list), "Dialogue notes list"),
        (isinstance(artifact.raw_notes, list), "Raw notes list"),
        (isinstance(artifact.questions_asked, list), "Questions asked list"),
    ]

    passed = 0
    for check, desc in checks:
        if check:
            print(f"  ✓ {desc} correct")
            passed += 1
        else:
            print(f"  ✗ {desc} incorrect")

    return passed == len(checks)


def test_dialogue_notes_handling():
    """Test handling of dialogue notes."""
    print("\nTesting dialogue notes handling...")

    agent = create_character_agent("ml_expert_critical")

    # Manually add some dialogue notes to artifact
    test_notes = [
        DialogueNote(
            type="assumption",
            content="Test assumption",
            suggested_question="Why?",
            priority=8,
            context="test"
        ),
        DialogueNote(
            type="limitation",
            content="Test limitation",
            suggested_question="How to fix?",
            priority=5,
            context="test"
        ),
        DialogueNote(
            type="gap_identified",
            content="Test gap",
            suggested_question="How significant?",
            priority=9,
            context="test"
        ),
    ]

    agent.artifact.dialogue_notes = test_notes

    # Test filtering
    high_priority = agent.get_dialogue_notes(min_priority=7)

    print(f"  Total notes: {len(test_notes)}")
    print(f"  High priority notes (>=7): {len(high_priority)}")

    if len(high_priority) == 2:  # Should get assumption (8) and gap (9)
        print(f"  ✓ Priority filtering works correctly")

        # Check they're sorted by priority
        if high_priority[0].priority >= high_priority[1].priority:
            print(f"  ✓ Notes sorted by priority")
            return True
        else:
            print(f"  ✗ Notes not sorted correctly")
            return False
    else:
        print(f"  ✗ Priority filtering incorrect (got {len(high_priority)}, expected 2)")
        return False


def test_multiple_agents_same_session():
    """Test multiple agents in same session."""
    print("\nTesting multiple agents in same session...")

    session_id = "shared-session-123"

    # Create different character agents for same session
    agent1 = create_character_agent("ml_expert_critical", session_id=session_id)
    agent2 = create_character_agent("medical_imaging_constructive", session_id=session_id)
    agent3 = create_character_agent("ethics_neutral", session_id=session_id)

    # Check they share session ID but have different character IDs
    checks = [
        (agent1.session_id == session_id, "Agent 1 session ID"),
        (agent2.session_id == session_id, "Agent 2 session ID"),
        (agent3.session_id == session_id, "Agent 3 session ID"),
        (agent1.character.character_id != agent2.character.character_id, "Different characters"),
        (agent2.character.character_id != agent3.character.character_id, "Different characters"),
    ]

    passed = 0
    for check, desc in checks:
        if check:
            print(f"  ✓ {desc}")
            passed += 1
        else:
            print(f"  ✗ {desc}")

    print(f"\n  Agents created:")
    print(f"    - {agent1.character.name} ({agent1.character.domain})")
    print(f"    - {agent2.character.name} ({agent2.character.domain})")
    print(f"    - {agent3.character.name} ({agent3.character.domain})")

    return passed == len(checks)


def test_repr():
    """Test string representation."""
    print("\nTesting __repr__...")

    agent = create_character_agent("ml_expert_critical", session_id="test-repr-123")

    repr_str = repr(agent)
    print(f"  {repr_str}")

    if "CharacterBasedResearchAgent" in repr_str and "Machine Learning" in repr_str:
        print(f"  ✓ String representation looks good")
        return True
    else:
        print(f"  ✗ String representation missing key info")
        return False


def main():
    """Run all simple tests."""
    print("="*60)
    print("CHARACTER-BASED RESEARCH AGENT TESTS (Simple)")
    print("="*60)
    print("Note: These tests don't make LLM calls")
    print("="*60)

    tests = [
        test_agent_creation,
        test_system_prompt,
        test_artifact_initialization,
        test_dialogue_notes_handling,
        test_multiple_agents_same_session,
        test_repr,
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
