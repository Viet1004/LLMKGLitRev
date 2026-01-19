"""Test the character system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmkglitrev.characters.schema import ResearchCharacter
from llmkglitrev.characters.manager import CharacterManager


def test_character_loading():
    """Test loading system characters."""
    print("Testing character loading...")
    manager = CharacterManager()

    # Test loading each system character
    character_ids = [
        "ml_expert_critical",
        "medical_imaging_constructive",
        "ethics_neutral",
        "computer_science_researcher"
    ]

    for char_id in character_ids:
        try:
            char = manager.load_character(char_id)
            print(f"  ✓ Loaded: {char.name}")
            print(f"    - Domain: {char.domain}")
            print(f"    - Stance: {char.stance}")
            print(f"    - Sub-domains: {', '.join(char.sub_domains[:3])}...")
        except Exception as e:
            print(f"  ✗ Failed to load {char_id}: {e}")
            return False

    return True


def test_character_listing():
    """Test listing characters."""
    print("\nTesting character listing...")
    manager = CharacterManager()

    characters = manager.list_characters()
    print(f"  Found {len(characters)} characters:")
    for char in characters:
        print(f"    - {char['name']} ({char['id']}) [{char['source']}]")

    return len(characters) == 4


def test_user_character_creation():
    """Test creating a user character."""
    print("\nTesting user character creation...")
    manager = CharacterManager()

    test_char = ResearchCharacter(
        character_id="test_char",
        name="Test Character",
        domain="Test Domain",
        sub_domains=["test1", "test2"],
        stance="neutral",
        communication_style="academic",
        description="This is a test character",
        system_prompt_template="Test prompt"
    )

    try:
        # Create
        char_id = manager.create_character(test_char)
        print(f"  ✓ Created: {char_id}")

        # Load
        loaded = manager.load_character(char_id)
        print(f"  ✓ Loaded: {loaded.name}")

        # Update
        updated = manager.update_character(char_id, {"description": "Updated description"})
        print(f"  ✓ Updated: {updated.description}")

        # Delete
        manager.delete_character(char_id)
        print(f"  ✓ Deleted: {char_id}")

        # Verify deletion
        try:
            manager.load_character(char_id)
            print(f"  ✗ Character still exists after deletion")
            return False
        except ValueError:
            print(f"  ✓ Confirmed deletion")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_system_prompt_formatting():
    """Test system prompt template formatting."""
    print("\nTesting system prompt formatting...")
    manager = CharacterManager()

    char = manager.load_character("ml_expert_critical")

    # Format the prompt
    formatted = char.system_prompt_template.format(
        name=char.name,
        domain=char.domain,
        sub_domains=", ".join(char.sub_domains),
        typical_venues=", ".join(char.typical_venues),
        typical_methods=", ".join(char.typical_methods),
        typical_datasets=", ".join(char.typical_datasets),
        theoretical_foundations=", ".join(char.theoretical_foundations),
        stance=char.stance,
        communication_style=char.communication_style,
        focus_areas=", ".join(char.focus_areas)
    )

    if "Dr. Machine Learning" in formatted and "deep learning" in formatted:
        print("  ✓ System prompt formatted correctly")
        print(f"    Prompt length: {len(formatted)} characters")
        return True
    else:
        print("  ✗ System prompt formatting failed")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("CHARACTER SYSTEM TESTS")
    print("="*60)

    tests = [
        test_character_loading,
        test_character_listing,
        test_user_character_creation,
        test_system_prompt_formatting
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
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
