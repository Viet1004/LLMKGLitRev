"""
Complete Workflow Test: Research + Dialogue

This script demonstrates the FULL end-to-end workflow:
1. User inputs research topic
2. System proposes research plan → User approves
3. Agents conduct research → Generate artifacts
4. Dialogue coordinator asks questions → User responds
5. Final summary with all results
"""

import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmkglitrev.agents.research_proposal_generator import proposal_generator_agent
from llmkglitrev.agents.interactive_runner import run_research_interactive
from llmkglitrev.agents.dialogue_coordinator import create_dialogue_coordinator
from langgraph.types import Command


async def run_complete_workflow(research_topic: str):
    """
    Run complete workflow from research topic to Socratic dialogue.

    Args:
        research_topic: The research question to investigate
    """
    print("="*80)
    print("COMPLETE WORKFLOW: DYNAMIC PLANNING + RESEARCH + DIALOGUE")
    print("="*80)
    print(f"\n📝 Research Topic: {research_topic}\n")

    # =========================================================================
    # PHASE 1: DYNAMIC PLANNING + RESEARCH
    # =========================================================================

    print("\n" + "="*80)
    print("PHASE 1: DYNAMIC PLANNING + RESEARCH")
    print("="*80)

    try:
        research_result = await run_research_interactive(
            query=research_topic,
            agent=proposal_generator_agent,
            max_iterations=10
        )

        print("\n✅ Research phase complete!")

    except Exception as e:
        print(f"\n❌ Research failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # PHASE 2: EXTRACT ARTIFACTS FOR DIALOGUE
    # =========================================================================

    print("\n" + "="*80)
    print("PHASE 2: EXTRACTING CONVERSATION ARTIFACTS")
    print("="*80)

    # Get conversation artifacts from research
    artifacts = research_result.get("conversation_artifacts", [])
    session_id = research_result.get("session_id", "unknown")

    print(f"\n🆔 SESSION ID: {session_id}")
    print(f"   Artifacts location: sessions/{session_id}/")

    if not artifacts:
        print("\n⚠️  No conversation artifacts found.")
        print("This means research didn't generate dialogue notes.")
        print("Dialogue phase requires artifacts with dialogue notes.\n")

        # Check if we have active_characters at least
        active_characters = research_result.get("active_characters", [])
        if active_characters:
            print(f"✅ Found {len(active_characters)} active characters:")
            for char in active_characters:
                print(f"   - {char.get('name', 'Unknown')} ({char.get('domain', 'Unknown')})")

        # Create mock artifacts for demonstration
        print("\n💡 Creating mock dialogue notes for demonstration...")

        mock_artifacts = []
        for char in active_characters[:2]:  # Use first 2 characters
            from llmkglitrev.characters import ConversationArtifact, DialogueNote

            mock_notes = [
                DialogueNote(
                    type="gap_identified",
                    content=f"Research gap in {char.get('domain', 'this domain')}",
                    suggested_question=f"How can we address this gap in {char.get('domain', 'research')}?",
                    priority=9,
                    context="Identified during research"
                ),
                DialogueNote(
                    type="assumption",
                    content=f"Assumption about {char.get('domain', 'methodology')}",
                    suggested_question=f"What evidence supports this assumption in {char.get('domain', 'research')}?",
                    priority=7,
                    context="Methodological assumption"
                )
            ]

            mock_artifact = ConversationArtifact(
                session_id=session_id,
                character_id=char.get('character_id', 'unknown'),
                domain=char.get('domain', 'Unknown'),
                research_output=f"Mock research output from {char.get('name', 'agent')}",
                dialogue_notes=mock_notes,
                raw_notes=[],
                papers_consulted=[]
            )

            mock_artifacts.append(mock_artifact.model_dump())

        artifacts = mock_artifacts
        print(f"✅ Created {len(artifacts)} mock artifacts for dialogue")

    print(f"\n📦 Found {len(artifacts)} conversation artifacts")

    total_notes = sum(len(a.get('dialogue_notes', [])) for a in artifacts)
    print(f"💬 Total dialogue notes: {total_notes}")

    if total_notes == 0:
        print("\n⚠️  No dialogue notes found in artifacts.")
        print("Dialogue coordinator needs dialogue notes to generate questions.")
        print("Skipping dialogue phase.\n")
        return research_result

    # Count high-priority notes
    high_priority_count = 0
    for artifact in artifacts:
        for note in artifact.get('dialogue_notes', []):
            if note.get('priority', 0) >= 7:
                high_priority_count += 1

    print(f"🔝 High-priority notes (≥7): {high_priority_count}")

    # =========================================================================
    # PHASE 3: SOCRATIC DIALOGUE
    # =========================================================================

    print("\n" + "="*80)
    print("PHASE 3: SOCRATIC DIALOGUE")
    print("="*80)
    print("""
The dialogue coordinator will now:
1. Extract high-priority questions from research artifacts
2. Ask questions ONE AT A TIME
3. Wait for your feedback after each answer
4. Continue until all questions explored or you stop

You can type 'stop' at any feedback prompt to end the dialogue.
""")

    input("\nPress Enter to start Socratic dialogue...")

    try:
        # Create dialogue coordinator
        dialogue_coordinator = create_dialogue_coordinator(min_priority=7)

        # Initial dialogue state
        dialogue_state = {
            "session_id": session_id,
            "conversation_artifacts": artifacts,
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

        # Run dialogue with interrupts
        config = {"configurable": {"thread_id": f"dialogue-{session_id}"}}

        print("\n🎙️ Starting Socratic dialogue...\n")

        iteration = 0
        max_iterations = 20  # Prevent infinite loops

        while iteration < max_iterations:
            iteration += 1

            # Invoke dialogue coordinator
            result = await dialogue_coordinator.ainvoke(dialogue_state, config=config)

            # Check if dialogue is complete
            if result.get("dialogue_complete", False):
                print("\n✅ Dialogue complete!")
                break

            # Check for interrupt (waiting for user feedback)
            if "__interrupt__" in result:
                interrupt_data = result["__interrupt__"][0].value

                print("\n" + "="*70)
                print("💬 DIALOGUE INTERACTION")
                print("="*70)

                # Display current question and answer
                current_q = result.get("current_question", "")
                current_a = result.get("current_answer", "")

                if current_q:
                    print(f"\n❓ Question:")
                    print(f"   {current_q}")

                if current_a:
                    print(f"\n💡 Answer:")
                    print(f"   {current_a[:500]}...")  # Show first 500 chars

                print("\n" + "-"*70)

                # Get user feedback
                try:
                    feedback = input("\n👤 Your feedback (or 'stop' to end dialogue): ").strip()

                    if feedback.lower() == 'stop':
                        print("\n⏹️  Stopping dialogue as requested...")
                        dialogue_state["user_requested_stop"] = True
                        # Resume with stop signal
                        result = await dialogue_coordinator.ainvoke(
                            Command(resume={"user_requested_stop": True}),
                            config=config
                        )
                        break

                    if not feedback:
                        feedback = "Continue to next question."

                    print(f"\n✅ Feedback: {feedback}")

                    # Resume dialogue with feedback
                    result = await dialogue_coordinator.ainvoke(
                        Command(resume=feedback),
                        config=config
                    )

                    # Update state for next iteration
                    dialogue_state = result

                except (EOFError, KeyboardInterrupt):
                    print("\n\n⚠️  Dialogue interrupted by user")
                    break
            else:
                # No interrupt, update state and continue
                dialogue_state = result

        if iteration >= max_iterations:
            print(f"\n⚠️  Reached maximum iterations ({max_iterations})")

        # Display dialogue summary
        dialogue_history = dialogue_state.get("dialogue_history", [])

        print("\n" + "="*80)
        print("DIALOGUE SUMMARY")
        print("="*80)
        print(f"\nQuestions explored: {len(dialogue_history)}")

        for i, qa in enumerate(dialogue_history, 1):
            print(f"\n{i}. {qa.get('question', 'N/A')[:100]}...")
            print(f"   Feedback: {qa.get('user_feedback', 'N/A')[:100]}...")

    except Exception as e:
        print(f"\n❌ Dialogue failed: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print("\n" + "="*80)
    print("COMPLETE WORKFLOW SUMMARY")
    print("="*80)

    print(f"\n🆔 Session ID: {session_id}")
    print(f"\n📝 Research Topic:")
    print(f"   {research_topic}")

    active_characters = research_result.get("active_characters", [])
    print(f"\n👥 Active Characters: {len(active_characters)}")
    for char in active_characters:
        print(f"   - {char.get('name', 'Unknown')} ({char.get('domain', 'Unknown')})")

    print(f"\n📦 Conversation Artifacts: {len(artifacts)}")
    print(f"💬 Total Dialogue Notes: {total_notes}")
    print(f"🎙️ Dialogue Questions: {len(dialogue_history)}")

    final_proposal = research_result.get("final_proposal", "")
    if final_proposal:
        print(f"\n📄 Final Proposal: {len(final_proposal)} characters")
        print(f"   Preview: {final_proposal[:200]}...")

    print("\n" + "="*80)
    print("✅ COMPLETE WORKFLOW FINISHED!")
    print("="*80)

    return {
        "research_result": research_result,
        "dialogue_history": dialogue_history,
        "artifacts": artifacts
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test complete workflow: dynamic planning + research + dialogue"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="How can transfer learning improve medical imaging with small datasets?",
        help="Research topic to investigate"
    )

    args = parser.parse_args()

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                     COMPLETE WORKFLOW TEST                                 ║
║                                                                            ║
║  This test runs the ENTIRE system:                                        ║
║    1. Dynamic agent planning with user approval                           ║
║    2. Character-based research execution                                  ║
║    3. Socratic dialogue with interactive Q&A                              ║
║                                                                            ║
║  Expected duration: 5-15 minutes (depending on iterations)                ║
║  Interrupts: Multiple (plan approval, dialogue feedback)                  ║
║                                                                            ║
║  Press Ctrl+C at any time to exit                                         ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    asyncio.run(run_complete_workflow(args.topic))


if __name__ == "__main__":
    main()
