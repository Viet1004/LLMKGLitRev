"""
Test Script for Dynamic Research Planning

This script demonstrates the new dynamic planning workflow where:
1. User inputs research topic
2. Supervisor proposes research plan with recommended agents
3. User approves/edits/rejects the plan
4. Agents are dynamically instantiated based on approved plan
5. Research is conducted and final proposal is generated
"""

import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmkglitrev.agents.research_proposal_generator import proposal_generator_agent
from llmkglitrev.agents.interactive_runner import run_research_interactive


async def test_dynamic_planning():
    """
    Test the complete dynamic planning workflow.
    """
    print("="*80)
    print("DYNAMIC RESEARCH PLANNING TEST")
    print("="*80)
    print("""
This test demonstrates the new workflow:
1. ✅ User provides research topic
2. ✅ System retrieves relevant literature
3. ✅ LLM proposes research plan with 2-4 agents
4. ⏸️  INTERRUPT: User approves/edits/rejects plan
5. ✅ Agents are instantiated based on approved plan
6. ✅ Supervisor coordinates research
7. ✅ Final proposal is generated

Press Ctrl+C at any time to exit.
""")

    # Example research topics to test with
    example_topics = [
        "How can transfer learning improve medical image analysis with small datasets?",
        "What are the ethical implications of large language models in education?",
        "How can blockchain technology enhance supply chain transparency?",
    ]

    print("Example research topics:")
    for i, topic in enumerate(example_topics, 1):
        print(f"  {i}. {topic}")

    print("\n" + "-"*80)

    try:
        topic_choice = input("\nSelect a topic (1-3) or type your own: ").strip()

        if topic_choice.isdigit() and 1 <= int(topic_choice) <= 3:
            research_topic = example_topics[int(topic_choice) - 1]
        else:
            research_topic = topic_choice

        if not research_topic:
            research_topic = example_topics[0]
            print(f"\nUsing default topic: {research_topic}")

        print("\n" + "="*80)
        print("🚀 Starting Dynamic Research Planning")
        print("="*80)

        # Run the interactive workflow
        result = await run_research_interactive(
            query=research_topic,
            agent=proposal_generator_agent,
            max_iterations=10
        )

        print("\n" + "="*80)
        print("📊 RESULTS")
        print("="*80)

        # Display session ID prominently
        session_id = result.get("session_id", "unknown")
        print(f"\n🆔 SESSION ID: {session_id}")
        print(f"   Use this ID to access artifacts and run dialogue")
        print(f"   Location: sessions/{session_id}/")

        # Display final proposal
        final_proposal = result.get("final_proposal", "")
        if final_proposal:
            print("\n📄 Final Research Proposal:")
            print("-"*80)
            print(final_proposal[:1000])  # Show first 1000 characters
            if len(final_proposal) > 1000:
                print(f"\n... (truncated, total {len(final_proposal)} characters)")
        else:
            print("\n⚠️  No final proposal generated (workflow may have been interrupted)")

        # Display active characters
        active_characters = result.get("active_characters", [])
        if active_characters:
            print(f"\n👥 Active Characters ({len(active_characters)}):")
            for char in active_characters:
                print(f"  - {char.get('name', 'Unknown')} ({char.get('domain', 'Unknown')})")

        # Display conversation artifacts
        artifacts = result.get("conversation_artifacts", [])
        if artifacts:
            print(f"\n📦 Conversation Artifacts: {len(artifacts)}")
            for artifact in artifacts:
                domain = artifact.get("domain", "Unknown")
                notes_count = len(artifact.get("dialogue_notes", []))
                print(f"  - {domain}: {notes_count} dialogue notes")

        print("\n" + "="*80)
        print("✅ TEST COMPLETE")
        print("="*80)

        return result

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return None
    except Exception as e:
        print(f"\n\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_plan_proposal_only():
    """
    Test only the plan proposal step (no actual research execution).

    This is useful for quickly testing the planning LLM without waiting
    for full research execution.
    """
    from llmkglitrev.agents.research_planner import propose_research_plan
    from llmkglitrev.agents.states import AgentState

    print("="*80)
    print("PLAN PROPOSAL ONLY TEST")
    print("="*80)

    research_topic = "How can few-shot learning improve medical diagnosis systems?"

    # Simulate state after literature retrieval
    mock_state: AgentState = {
        "messages": [],
        "research_topic": research_topic,
        "supervisor_messages": [],
        "research_proposals": [],
        "research_keywords": ["few-shot learning", "medical diagnosis", "AI"],
        "raw_notes": [],
        "notes": [],
        "final_proposal": "",
        "session_id": "test-session",
        "active_characters": [],
        "conversation_artifacts": [],
        "character_configs": [],
        "proposed_research_plan": None,
        "plan_approved": False,
        "retrieved_papers": [],
        "literature_context": """
Recent papers show that few-shot learning approaches can adapt to new medical
conditions with limited training data. Meta-learning and transfer learning
techniques have shown promise in medical imaging tasks.
"""
    }

    print(f"\n📝 Research Topic: {research_topic}")
    print("\n🤖 Calling propose_research_plan node...")

    try:
        result = await propose_research_plan(mock_state)

        print("\n✅ Plan proposal generated!")
        print("\nNote: In real workflow, this would trigger an interrupt for user approval.")

        proposed_plan = result.get("proposed_research_plan", {})
        if proposed_plan:
            print(f"\n📋 Proposed {len(proposed_plan.get('proposed_agents', []))} agents:")
            for i, agent in enumerate(proposed_plan.get('proposed_agents', []), 1):
                print(f"\n  {i}. {agent.get('domain', 'Unknown')} ({agent.get('stance', 'neutral')})")
                print(f"     Template: {agent.get('recommended_character', 'custom')}")
                print(f"     Focus: {', '.join(agent.get('search_scope', [])[:3])}")

        return result

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main entry point for testing.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test dynamic research planning")
    parser.add_argument(
        "--mode",
        choices=["full", "plan-only"],
        default="full",
        help="Test mode: 'full' for complete workflow, 'plan-only' for just planning"
    )

    args = parser.parse_args()

    if args.mode == "full":
        asyncio.run(test_dynamic_planning())
    else:
        asyncio.run(test_plan_proposal_only())


if __name__ == "__main__":
    main()
