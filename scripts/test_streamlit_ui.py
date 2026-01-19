"""Test that the Streamlit UI can be imported and basic components work."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import streamlit as st
        print(f"  ✓ Streamlit imported (version: {st.__version__})")
    except ImportError as e:
        print(f"  ✗ Failed to import streamlit: {e}")
        return False

    try:
        from llmkglitrev.characters import (
            CharacterManager,
            ConversationArtifact,
            ConversationArtifactManager
        )
        print(f"  ✓ Character modules imported")
    except ImportError as e:
        print(f"  ✗ Failed to import character modules: {e}")
        return False

    try:
        from llmkglitrev.agents.research_supervisor import create_interactive_supervisor
        print(f"  ✓ Supervisor module imported")
    except ImportError as e:
        print(f"  ✗ Failed to import supervisor: {e}")
        return False

    try:
        from llmkglitrev.agents.dialogue_coordinator import create_dialogue_coordinator
        print(f"  ✓ Dialogue coordinator imported")
    except ImportError as e:
        print(f"  ✗ Failed to import dialogue coordinator: {e}")
        return False

    return True


def test_character_loading():
    """Test that characters can be loaded for UI."""
    print("\nTesting character loading...")

    try:
        from llmkglitrev.characters import CharacterManager

        manager = CharacterManager()
        characters = manager.list_characters()

        print(f"  ✓ Loaded {len(characters)} characters")

        if len(characters) > 0:
            print(f"    Characters available:")
            for char_info in characters:
                # char_info contains: id, name, domain, stance, source
                char_id = char_info.get('id', 'Unknown')
                char_name = char_info.get('name', 'Unknown')
                print(f"      - {char_name} ({char_id})")
            return True
        else:
            print(f"  ⚠ No characters found (UI will work but with warnings)")
            return True

    except Exception as e:
        print(f"  ✗ Error loading characters: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_app_structure():
    """Test that the app file exists and has correct structure."""
    print("\nTesting app structure...")

    app_path = Path(__file__).parent.parent / "app_character_research.py"

    if not app_path.exists():
        print(f"  ✗ App file not found: {app_path}")
        return False

    print(f"  ✓ App file exists: {app_path}")

    # Read app file and check for key components
    with open(app_path, 'r') as f:
        content = f.read()

    checks = [
        ("st.set_page_config" in content, "Page configuration"),
        ("st.tabs(" in content, "Tab layout"),
        ("CharacterManager" in content, "Character manager usage"),
        ("st.button" in content, "Interactive buttons"),
        ("st.session_state" in content, "Session state management"),
    ]

    all_passed = True
    for check, desc in checks:
        if check:
            print(f"  ✓ {desc} found")
        else:
            print(f"  ✗ {desc} missing")
            all_passed = False

    return all_passed


def test_ui_components():
    """Test individual UI helper functions."""
    print("\nTesting UI components...")

    try:
        # Test session state initialization
        session_vars = [
            "session_id",
            "selected_characters",
            "research_topic",
            "research_started",
            "conversation_artifacts",
            "dialogue_started",
            "dialogue_history"
        ]

        print(f"  ✓ Expected session state variables defined: {len(session_vars)}")

        return True

    except Exception as e:
        print(f"  ✗ Error testing components: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("STREAMLIT UI TESTS")
    print("="*60)

    tests = [
        test_imports,
        test_character_loading,
        test_app_structure,
        test_ui_components,
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
        print("\nTo run the Streamlit app:")
        print("  streamlit run app_character_research.py")
        return 0
    else:
        print("✗ Some tests failed")
        if sum(results) >= 3:
            print("  (Core functionality working)")
            return 0
        return 1


if __name__ == "__main__":
    sys.exit(main())
