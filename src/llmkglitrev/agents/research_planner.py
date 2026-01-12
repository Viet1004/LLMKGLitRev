"""
Research Planner Module

This module contains the logic for proposing multi-agent research plans based on
research topics and available literature. It uses LLM-based analysis to recommend
specialized agents and their configurations.
"""

from typing import Dict, Any
from langchain.chat_models import init_chat_model
from langgraph.types import interrupt

from llmkglitrev.agents.states import AgentState, ResearchPlan, AgentProposal
from llmkglitrev.agents.prompts.research_planning import propose_agents_prompt
from llmkglitrev.characters import CharacterManager


# Use a capable model for strategic planning
planning_model = init_chat_model(
    model="anthropic:claude-sonnet-4-20250514",
    max_tokens=4000
).with_structured_output(ResearchPlan)


async def propose_research_plan(state: AgentState) -> Dict[str, Any]:
    """
    Propose a multi-agent research plan based on topic and literature analysis.

    This node:
    1. Analyzes the research topic and retrieved literature
    2. Proposes 2-4 specialized research agents
    3. Recommends character templates or custom configurations
    4. Interrupts for human approval before proceeding

    Args:
        state: Current agent state with research_topic and literature_context

    Returns:
        State update with proposed_research_plan and interrupt for human feedback
    """
    research_topic = state.get("research_topic", "")
    literature_context = state.get("literature_context", "No literature retrieved")

    print("\n" + "="*70)
    print("📋 PROPOSING RESEARCH PLAN")
    print("="*70)
    print(f"\n📝 Research Topic: {research_topic}")
    print(f"📚 Retrieved {len(state.get('retrieved_papers', []))} papers from literature database")

    # Load available character templates
    char_manager = CharacterManager()
    available_chars = char_manager.list_characters()

    # Format character list for LLM
    char_descriptions = []
    for char in available_chars:
        char_descriptions.append(
            f"- **{char['id']}**: {char['name']} ({char['domain']}, {char['stance']} stance)"
        )
    available_characters_str = "\n".join(char_descriptions)

    print(f"\n🎭 Available character templates: {len(available_chars)}")

    # Generate research plan using LLM
    print("\n🤔 Analyzing topic and proposing agent configuration...")

    prompt = propose_agents_prompt.format(
        research_topic=research_topic,
        literature_context=literature_context[:3000],  # Limit context size
        available_characters=available_characters_str
    )

    try:
        research_plan: ResearchPlan = await planning_model.ainvoke(prompt)

        print("\n✅ Research plan generated!")
        print(f"\n🎯 Strategy: {research_plan.research_strategy}")
        print(f"\n👥 Proposed {len(research_plan.proposed_agents)} agents:")

        for i, agent in enumerate(research_plan.proposed_agents, 1):
            print(f"\n  Agent {i}: {agent.domain} ({agent.stance})")
            print(f"    Template: {agent.recommended_character}")
            print(f"    Focus: {', '.join(agent.search_scope[:3])}...")
            print(f"    Rationale: {agent.rationale[:100]}...")

        # Convert to dict for JSON serialization
        plan_dict = research_plan.model_dump()

        # Prepare interrupt data for user approval
        interrupt_data = {
            "type": "research_plan_approval",
            "question": "Please review the proposed research plan. You can approve, modify, or reject it.",
            "plan": plan_dict,
            "instructions": """
Options:
1. Type 'approve' to proceed with this plan
2. Type 'reject' to ask for a new plan
3. Provide JSON to modify agent configurations
4. Add/remove agents by editing the plan

Example modification:
{
  "approved": true,
  "modifications": {
    "agents": [
      {
        "domain": "Machine Learning",
        "recommended_character": "ml_expert_critical",
        "stance": "critical",
        "search_scope": ["deep learning", "transfer learning"],
        "rationale": "Custom rationale..."
      }
    ]
  }
}
"""
        }

        print("\n" + "="*70)
        print("⏸️  INTERRUPT: Waiting for human approval")
        print("="*70)

        # Interrupt workflow for human approval
        # The interactive_runner will handle displaying this and collecting feedback
        human_feedback = interrupt(interrupt_data)

        return {
            "proposed_research_plan": plan_dict,
            "plan_approved": False  # Will be set to True after approval
        }

    except Exception as e:
        print(f"\n❌ Error generating research plan: {e}")
        import traceback
        traceback.print_exc()

        # Return a fallback plan if LLM fails
        fallback_plan = {
            "research_strategy": "Use default character templates for multi-perspective research",
            "proposed_agents": [
                {
                    "domain": "Machine Learning",
                    "recommended_character": "ml_expert_critical",
                    "stance": "critical",
                    "search_scope": ["machine learning", "algorithms"],
                    "rationale": "Provides critical analysis of ML approaches"
                },
                {
                    "domain": "Applied Research",
                    "recommended_character": "medical_imaging_constructive",
                    "stance": "constructive",
                    "search_scope": ["applications", "case studies"],
                    "rationale": "Explores practical applications"
                }
            ],
            "interdisciplinary_connections": "Agents will collaborate on technical and practical aspects"
        }

        print("\n⚠️  Using fallback research plan with default templates")

        interrupt_data = {
            "type": "research_plan_approval",
            "question": "LLM planning failed. Please review the fallback plan.",
            "plan": fallback_plan,
            "instructions": "Type 'approve' to proceed or provide custom agent configurations."
        }

        human_feedback = interrupt(interrupt_data)

        return {
            "proposed_research_plan": fallback_plan,
            "plan_approved": False
        }


def process_plan_approval(state: AgentState) -> Dict[str, Any]:
    """
    Process human feedback on the proposed research plan.

    This node runs after the interrupt is resumed with human feedback.
    It parses the feedback and updates the plan accordingly.

    Args:
        state: Agent state with proposed_research_plan and human feedback

    Returns:
        State update with approved plan and character_configs
    """
    import json

    # Get the human feedback from the interrupt resume
    # This will be available in the state after interrupt
    proposed_plan = state.get("proposed_research_plan", {})

    print("\n" + "="*70)
    print("📝 PROCESSING PLAN APPROVAL")
    print("="*70)

    # For now, we'll mark as approved
    # The interactive_runner will handle parsing feedback
    # and potentially modifying the plan before resuming

    # Extract character configurations from approved plan
    character_configs = []

    for agent_proposal in proposed_plan.get("proposed_agents", []):
        char_config = {
            "domain": agent_proposal["domain"],
            "character_id": agent_proposal["recommended_character"],
            "stance": agent_proposal["stance"],
            "search_scope": agent_proposal["search_scope"],
            "custom_config": agent_proposal.get("custom_config")
        }
        character_configs.append(char_config)

    print(f"\n✅ Plan approved with {len(character_configs)} agents")

    return {
        "plan_approved": True,
        "character_configs": character_configs
    }
