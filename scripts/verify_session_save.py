"""
Quick verification script to check if session saving works.

This script tests the save_artifacts node in isolation to verify
that artifacts are properly persisted to disk.
"""

import sys
from pathlib import Path
import json
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmkglitrev.characters import (
    ConversationArtifact,
    ConversationArtifactManager,
    DialogueNote
)
from llmkglitrev.agents.research_proposal_generator import save_artifacts
from llmkglitrev.agents.states import AgentState


def test_save_artifacts_node():
    """Test the save_artifacts node directly."""
    print("="*80)
    print("TESTING save_artifacts NODE")
    print("="*80)

    # Create test session ID
    test_session_id = "test-session-verification"

    # Clean up any existing test session
    test_session_path = Path(f"sessions/{test_session_id}")
    if test_session_path.exists():
        print(f"\n🧹 Cleaning up existing test session: {test_session_id}")
        shutil.rmtree(test_session_path)

    # Create mock artifacts (as dicts, like they appear in state)
    mock_artifacts = [
        {
            "session_id": test_session_id,
            "character_id": "test_ml_expert",
            "domain": "Machine Learning",
            "research_output": "Test research output about ML",
            "raw_notes": ["Note 1", "Note 2"],
            "dialogue_notes": [
                {
                    "type": "assumption",
                    "content": "Test assumption",
                    "suggested_question": "Why this assumption?",
                    "priority": 8,
                    "context": "Test context"
                }
            ],
            "papers_consulted": [],
            "questions_asked": [],
            "questions_answered": [],
            "dialogue_active": False,
            "created_at": "2026-01-08T00:00:00",
            "last_updated": "2026-01-08T00:00:00"
        },
        {
            "session_id": test_session_id,
            "character_id": "test_medical_expert",
            "domain": "Medical Imaging",
            "research_output": "Test research output about medical imaging",
            "raw_notes": ["Medical note 1", "Medical note 2"],
            "dialogue_notes": [],
            "papers_consulted": [],
            "questions_asked": [],
            "questions_answered": [],
            "dialogue_active": False,
            "created_at": "2026-01-08T00:00:00",
            "last_updated": "2026-01-08T00:00:00"
        }
    ]

    # Create mock state
    mock_state: AgentState = {
        "messages": [],
        "research_topic": "Test topic",
        "supervisor_messages": [],
        "research_proposals": [],
        "research_keywords": [],
        "raw_notes": [],
        "notes": [],
        "final_proposal": "",
        "session_id": test_session_id,
        "active_characters": [],
        "conversation_artifacts": mock_artifacts,
        "character_configs": [],
        "proposed_research_plan": None,
        "plan_approved": False,
        "retrieved_papers": [],
        "literature_context": ""
    }

    print(f"\n📋 Test Configuration:")
    print(f"  Session ID: {test_session_id}")
    print(f"  Number of artifacts: {len(mock_artifacts)}")
    print(f"  Artifact domains: {[a['domain'] for a in mock_artifacts]}")

    # Call save_artifacts node
    print("\n🚀 Calling save_artifacts node...")
    result = save_artifacts(mock_state)

    print(f"\n✅ save_artifacts completed (returned: {result})")

    # Verify files were created
    print("\n🔍 Verifying saved files...")

    conversations_dir = test_session_path / "conversations"

    if not conversations_dir.exists():
        print(f"❌ FAIL: Directory not created: {conversations_dir}")
        return False

    print(f"✅ Directory created: {conversations_dir}")

    # Check each artifact file
    expected_files = ["test_ml_expert.json", "test_medical_expert.json"]
    all_files_exist = True

    for expected_file in expected_files:
        file_path = conversations_dir / expected_file
        if file_path.exists():
            print(f"✅ File created: {expected_file}")

            # Verify file content
            with open(file_path) as f:
                data = json.load(f)

            print(f"   - Domain: {data.get('domain', 'N/A')}")
            print(f"   - Character ID: {data.get('character_id', 'N/A')}")
            print(f"   - Dialogue notes: {len(data.get('dialogue_notes', []))}")
            print(f"   - Raw notes: {len(data.get('raw_notes', []))}")

        else:
            print(f"❌ FAIL: File not created: {expected_file}")
            all_files_exist = False

    # Test loading artifacts back
    print("\n🔄 Testing artifact loading...")
    manager = ConversationArtifactManager()

    loaded_artifacts = manager.load_artifacts(test_session_id)
    print(f"✅ Loaded {len(loaded_artifacts)} artifacts")

    for char_id, artifact in loaded_artifacts.items():
        print(f"  - {char_id}: {artifact.domain}")

    # Clean up
    print(f"\n🧹 Cleaning up test session...")
    if test_session_path.exists():
        shutil.rmtree(test_session_path)
        print(f"✅ Removed {test_session_path}")

    # Final result
    print("\n" + "="*80)
    if all_files_exist and len(loaded_artifacts) == 2:
        print("✅ TEST PASSED: save_artifacts works correctly!")
        print("="*80)
        return True
    else:
        print("❌ TEST FAILED: Some artifacts were not saved correctly")
        print("="*80)
        return False


if __name__ == "__main__":
    print("\n")
    success = test_save_artifacts_node()
    print("\n")

    if success:
        print("✅ The save_artifacts node is working correctly.")
        print("   Session directories will now be created during the full workflow.")
        sys.exit(0)
    else:
        print("❌ The save_artifacts node is not working as expected.")
        print("   Please check the implementation.")
        sys.exit(1)
