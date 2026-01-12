"""
Interactive Runner for Research Proposal Agent

This module provides functions to run the research proposal agent interactively,
handling interrupts and collecting human feedback in CLI/Jupyter environments.
"""

from typing import Optional, Dict, Any
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import uuid
import json


def _display_research_plan(plan: Dict[str, Any]) -> None:
    """
    Display a research plan in a readable format.

    Args:
        plan: Research plan dictionary with proposed_agents
    """
    print("\n" + "="*70)
    print("📋 PROPOSED RESEARCH PLAN")
    print("="*70)

    print(f"\n🎯 Strategy: {plan.get('research_strategy', 'N/A')}")

    if plan.get('interdisciplinary_connections'):
        print(f"\n🔗 Interdisciplinary Connections:")
        print(f"   {plan['interdisciplinary_connections']}")

    print(f"\n👥 Proposed Agents ({len(plan.get('proposed_agents', []))}):")
    print("="*70)

    for i, agent in enumerate(plan.get('proposed_agents', []), 1):
        print(f"\n{i}. {agent.get('domain', 'Unknown Domain')} ({agent.get('stance', 'neutral')})")
        print(f"   Template: {agent.get('recommended_character', 'custom')}")
        print(f"   Search Scope: {', '.join(agent.get('search_scope', []))}")
        print(f"   Rationale: {agent.get('rationale', 'N/A')}")


async def _handle_plan_approval_interrupt(interrupt_data: Dict[str, Any]) -> str:
    """
    Handle research plan approval interrupt with user interaction.

    Allows user to:
    - Approve the plan
    - Reject and request new plan
    - Modify agent configurations
    - Add/remove agents

    Args:
        interrupt_data: Interrupt data containing the proposed plan

    Returns:
        Feedback string to resume the workflow
    """
    plan = interrupt_data.get('plan', {})

    # Display the plan
    _display_research_plan(plan)

    print("\n" + "="*70)
    print("🤔 YOUR OPTIONS")
    print("="*70)
    print("""
1. Type 'approve' or 'yes' - Approve and proceed with this plan
2. Type 'reject' or 'no' - Reject and request a new plan
3. Type 'edit' - Modify the plan (interactive JSON editor)
4. Type 'show json' - View full plan as JSON
5. Provide custom feedback - Give specific instructions to refine the plan
""")

    try:
        feedback = input("\n👤 Your choice: ").strip().lower()

        if not feedback or feedback in ['approve', 'yes', 'y']:
            print("\n✅ Plan approved!")
            return json.dumps({
                "action": "approve",
                "message": "Plan approved by user"
            })

        elif feedback in ['reject', 'no', 'n']:
            rejection_reason = input("   Why reject? (optional): ").strip()
            print("\n❌ Plan rejected. Will generate a new proposal...")
            return json.dumps({
                "action": "reject",
                "reason": rejection_reason or "User requested new plan"
            })

        elif feedback == 'edit':
            print("\n📝 Interactive Plan Editor")
            print("-"*70)
            print("Current agents:")
            for i, agent in enumerate(plan.get('proposed_agents', []), 1):
                print(f"  {i}. {agent.get('domain')} ({agent.get('stance')})")

            print("\nOptions:")
            print("  - Type a number to edit that agent")
            print("  - Type 'add' to add a new agent")
            print("  - Type 'remove N' to remove agent N")
            print("  - Type 'done' when finished")

            modified_plan = plan.copy()

            while True:
                action = input("\n  Edit action: ").strip().lower()

                if action == 'done':
                    break
                elif action == 'add':
                    print("\n  Adding new agent:")
                    domain = input("    Domain (e.g., 'Ethics'): ").strip()
                    stance = input("    Stance (critical/constructive/neutral): ").strip()
                    template = input("    Template ID (or 'custom'): ").strip()
                    scope = input("    Search keywords (comma-separated): ").strip()

                    new_agent = {
                        "domain": domain,
                        "stance": stance,
                        "recommended_character": template,
                        "search_scope": [s.strip() for s in scope.split(',')],
                        "rationale": f"User-added agent for {domain}"
                    }
                    modified_plan['proposed_agents'].append(new_agent)
                    print(f"    ✅ Added {domain} agent")

                elif action.startswith('remove '):
                    try:
                        idx = int(action.split()[1]) - 1
                        if 0 <= idx < len(modified_plan['proposed_agents']):
                            removed = modified_plan['proposed_agents'].pop(idx)
                            print(f"    ✅ Removed {removed.get('domain')} agent")
                        else:
                            print("    ❌ Invalid agent number")
                    except (ValueError, IndexError):
                        print("    ❌ Invalid format. Use 'remove N'")

                elif action.isdigit():
                    idx = int(action) - 1
                    if 0 <= idx < len(modified_plan['proposed_agents']):
                        agent = modified_plan['proposed_agents'][idx]
                        print(f"\n  Editing: {agent.get('domain')}")
                        print(f"    Current stance: {agent.get('stance')}")
                        new_stance = input("    New stance (or Enter to keep): ").strip()
                        if new_stance:
                            agent['stance'] = new_stance
                        print(f"    Current template: {agent.get('recommended_character')}")
                        new_template = input("    New template (or Enter to keep): ").strip()
                        if new_template:
                            agent['recommended_character'] = new_template
                        print("    ✅ Updated")
                    else:
                        print("    ❌ Invalid agent number")
                else:
                    print("    ❌ Unknown action")

            print("\n✅ Plan modifications complete!")
            return json.dumps({
                "action": "approve",
                "modified_plan": modified_plan,
                "message": "User modified and approved plan"
            })

        elif feedback == 'show json':
            print("\n" + "="*70)
            print("📄 PLAN JSON")
            print("="*70)
            print(json.dumps(plan, indent=2))
            print("="*70)
            # Recursive call to get actual decision
            return await _handle_plan_approval_interrupt(interrupt_data)

        else:
            # Custom feedback
            print(f"\n✅ Custom feedback provided: {feedback}")
            return json.dumps({
                "action": "refine",
                "feedback": feedback,
                "message": f"User feedback: {feedback}"
            })

    except (EOFError, KeyboardInterrupt):
        print("\n\n⚠️  Interrupted. Defaulting to approval...")
        return json.dumps({
            "action": "approve",
            "message": "Defaulting to approval due to interrupt"
        })


async def run_research_interactive(
    query: str,
    agent,
    thread_id: Optional[str] = None,
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    Run research agent interactively, handling interrupts and collecting feedback.
    
    This function wraps the agent invocation to handle the interrupt pattern:
    1. Invoke agent with query
    2. If interrupted, prompt user for feedback
    3. Resume with feedback using Command(resume=...)
    4. Repeat until completion or max iterations
    
    Args:
        query: The research query/topic
        agent: The compiled LangGraph agent with checkpointer
        thread_id: Optional thread ID for resuming existing sessions
        max_iterations: Maximum number of interrupt-resume cycles
        
    Returns:
        Final result dictionary with the complete research proposal
        
    Example:
        >>> from llmkglitrev.agents.research_proposal_generator import proposal_generator_agent
        >>> result = await run_research_interactive(
        ...     "How can LLMs improve literature review?",
        ...     proposal_generator_agent
        ... )
    """
    # Generate thread_id if not provided
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"🔬 Starting research with thread_id: {thread_id}")
    print(f"📝 Query: {query}\n")
    
    # Initial invocation
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )
    
    iteration = 0
    
    # Handle interrupts in a loop
    while "__interrupt__" in result and iteration < max_iterations:
        iteration += 1
        
        # Extract interrupt data
        interrupt_data = result["__interrupt__"][0].value

        print("\n" + "="*70)
        print("🔔 INTERRUPT: Agent is requesting human feedback")
        print("="*70)

        # Display interrupt information
        if isinstance(interrupt_data, dict):
            interrupt_type = interrupt_data.get('type', 'general')

            if interrupt_type == 'research_plan_approval':
                # Special handling for research plan approval
                feedback = await _handle_plan_approval_interrupt(interrupt_data)
            else:
                # Generic interrupt handling
                print(f"\n❓ {interrupt_data.get('question', 'Please provide feedback')}")

                if interrupt_data.get('instructions'):
                    print(f"\n💡 {interrupt_data['instructions']}")

                if interrupt_data.get('proposals'):
                    print("\n📋 Proposed Research Directions:")
                    for i, proposal in enumerate(interrupt_data['proposals'], 1):
                        print(f"   {i}. {proposal}")

                print("\n" + "-"*70)

                # Get user feedback
                try:
                    feedback = input("👤 Your feedback (or press Enter to approve): ").strip()
                    if not feedback:
                        feedback = "Approved. Proceed with all proposed research directions."
                except (EOFError, KeyboardInterrupt):
                    print("\n\n⚠️  Interrupted by user. Exiting...")
                    return result
        else:
            print(f"\n❓ {interrupt_data}")

            print("\n" + "-"*70)

            # Get user feedback
            try:
                feedback = input("👤 Your feedback (or press Enter to approve): ").strip()
                if not feedback:
                    feedback = "Approved."
            except (EOFError, KeyboardInterrupt):
                print("\n\n⚠️  Interrupted by user. Exiting...")
                return result

        print(f"\n✅ Feedback received")
        print(f"🔄 Resuming research (iteration {iteration})...\n")
        
        # Resume with feedback
        result = await agent.ainvoke(
            Command(resume=feedback),
            config=config
        )
    
    # Check if we exceeded max iterations
    if iteration >= max_iterations and "__interrupt__" in result:
        print(f"\n⚠️  Warning: Reached maximum iterations ({max_iterations})")
        print("The research may be incomplete. Consider increasing max_iterations.")
    
    print("\n" + "="*70)
    print("✅ Research Complete!")
    print("="*70)
    
    return result


def run_research_interactive_sync(
    query: str,
    agent,
    thread_id: Optional[str] = None,
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    Synchronous wrapper for run_research_interactive.
    
    Useful for environments where async/await is not convenient.
    """
    import asyncio
    
    try:
        # Check if there's already a running event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in a Jupyter notebook or similar
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(
                run_research_interactive(query, agent, thread_id, max_iterations)
            )
    except RuntimeError:
        pass
    
    # No running loop, create a new one
    return asyncio.run(
        run_research_interactive(query, agent, thread_id, max_iterations)
    )


async def resume_research_interactive(
    agent,
    thread_id: str,
    feedback: str
) -> Dict[str, Any]:
    """
    Resume a specific interrupted research session.
    
    Use this when you want to resume an existing session by its thread_id.
    
    Args:
        agent: The compiled LangGraph agent with checkpointer
        thread_id: Thread ID of the interrupted session
        feedback: Feedback to provide to the interrupted workflow
        
    Returns:
        Result dictionary (may contain another interrupt or completion)
        
    Example:
        >>> result = await resume_research_interactive(
        ...     proposal_generator_agent,
        ...     "thread-123",
        ...     "Focus more on practical applications"
        ... )
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"🔄 Resuming research session: {thread_id}")
    print(f"💬 Feedback: {feedback}\n")
    
    result = await agent.ainvoke(
        Command(resume=feedback),
        config=config
    )
    
    if "__interrupt__" in result:
        print("⚠️  Another interrupt detected. Call this function again with new feedback.")
    else:
        print("✅ Research completed!")
    
    return result
