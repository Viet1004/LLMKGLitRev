"""
Research Planner Module

This module contains the logic for proposing multi-agent research plans based on
research topics and available literature. It uses LLM-based analysis to recommend
specialized agents and their configurations.
"""

from typing import Dict, Any
from langchain.chat_models import init_chat_model

from llmkglitrev.agents.states import AgentState, ResearchPlan, AgentProposal
from llmkglitrev.agents.prompts.research_planning import propose_agents_prompt
from llmkglitrev.characters import CharacterManager


# Use a capable model for strategic planning
# planning_model = init_chat_model(
#     model="anthropic:claude-sonnet-4-20250514",
#     max_tokens=4000
# ).with_structured_output(ResearchPlan)

planning_model = init_chat_model(model="deepseek:deepseek-chat").with_structured_output(ResearchPlan)

async def propose_research_plan(state: AgentState) -> Dict[str, Any]:
    """
    Propose a multi-agent research plan based on topic and literature analysis.

    This node:
    1. Analyzes the research topic and retrieved literature
    2. Uses identified topics from broad search
    3. Proposes 2-4 specialized research agents (one per topic)
    4. Assigns seed papers to each agent
    5. Interrupts for human approval before proceeding

    Args:
        state: Current agent state with research_topic, topics, topic_papers, and literature_context

    Returns:
        State update with proposed_research_plan and interrupt for human feedback
    """
    research_topic = state.get("research_topic", "")
    literature_context = state.get("literature_context", "No literature retrieved")
    topics = state.get("topics", [])
    topic_papers = state.get("topic_papers", {})
    broad_papers = state.get("broad_papers", [])

    print("\n" + "="*70)
    print("📋 PROPOSING RESEARCH PLAN")
    print("="*70)
    print(f"\n📝 Research Topic: {research_topic}")
    print(f"📚 Retrieved {len(broad_papers)} papers from arXiv")
    print(f"🏷️  Identified {len(topics)} topics: {', '.join(topics)}")

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
    print("\n🤔 Analyzing topics and proposing agent configuration...")

    # Format topics for prompt
    topics_str = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics)])

    # Add topic information to prompt
    prompt_with_topics = f"""
Based on broad literature search, we identified these research topics:

{topics_str}

Each topic should be assigned to a different domain expert character.

{propose_agents_prompt.format(
    research_topic=research_topic,
    literature_context=literature_context[:3000],
    available_characters=available_characters_str
)}
"""

    prompt = prompt_with_topics

    try:
        research_plan: ResearchPlan = await planning_model.ainvoke(prompt)

        print("\n✅ Research plan generated!")
        print(f"\n🎯 Strategy: {research_plan.research_strategy}")
        print(f"\n👥 Proposed {len(research_plan.proposed_agents)} agents:")

        for i, agent in enumerate(research_plan.proposed_agents, 1):
            print(f"\n  Agent {i}: {agent.domain} ({agent.stance})")
            print(f"    Template: {agent.recommended_character}")
            print(f"    Focus: {', '.join(agent.search_scope[:3])}...")
            print(f"    Databases: {', '.join(agent.preferred_databases)}")
            print(f"    Venues: {', '.join(agent.preferred_venues[:3]) if agent.preferred_venues else 'N/A'}")
            print(f"    Rationale: {agent.rationale[:100]}...")

        # Convert to dict for JSON serialization
        plan_dict = research_plan.model_dump()

        # Assign topics and seed papers to each agent
        # Match agents to topics (1-to-1 mapping)
        for i, agent in enumerate(plan_dict["proposed_agents"]):
            # Assign topic (round-robin if more agents than topics)
            topic_idx = i % len(topics) if topics else 0
            assigned_topic = topics[topic_idx] if topics else research_topic

            # Get seed papers for this topic
            seed_papers = topic_papers.get(assigned_topic, [])[:5]  # Max 5 seed papers per agent

            # Add to agent config
            agent["assigned_topic"] = assigned_topic
            agent["seed_papers"] = seed_papers

            print(f"\n  📌 Agent {i+1} assigned to topic: {assigned_topic}")
            print(f"     📄 Seed papers: {len(seed_papers)}")

        print("\n" + "="*70)
        print("⏸️  Plan ready for approval - graph will interrupt before process_plan_approval")
        print("="*70)

        # Return the plan - graph will automatically interrupt before process_plan_approval
        # No need to call interrupt() - it's handled by interrupt_before configuration
        return {
            "proposed_research_plan": plan_dict,
            "plan_approved": False  # Will be set to True after approval
        }

    except Exception as e:
        print(f"\n❌ Error generating research plan: {e}")
        print(f"⚠️  ERROR TYPE: {type(e).__name__}")

        # Check if it's a validation error
        if "validation error" in str(e).lower() or "string_too_long" in str(e).lower():
            print("⚠️  CAUSE: LLM response exceeded schema constraints (likely research_strategy > 500 chars)")

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

        print("\n" + "="*70)
        print("⚠️  WARNING: Using fallback research plan with default templates")
        print("⚠️  This plan may NOT match your research topic!")
        print("⚠️  You can modify the plan during approval or re-run the workflow.")
        print("="*70)
        print("⏸️  Fallback plan ready - graph will interrupt before process_plan_approval")

        # Return fallback plan - graph will automatically interrupt
        return {
            "proposed_research_plan": fallback_plan,
            "plan_approved": False
        }


def process_plan_approval(state: AgentState) -> Dict[str, Any]:
    """
    Process human feedback on the proposed research plan.

    This node runs after the interrupt is resumed with human feedback.
    It parses the feedback and updates the plan accordingly.

    CRITICAL: When resuming via Command(resume={...}), the resume data is NOT automatically
    merged into the state. We must explicitly return the updated plan to override the 
    checkpoint state.

    Args:
        state: Agent state with proposed_research_plan and human feedback

    Returns:
        State update with approved plan and character_configs
    """
    import json

    # Get the plan from state (this might be the OLD plan from checkpoint)
    proposed_plan = state.get("proposed_research_plan", {})

    print("\n" + "="*70)
    print("📝 PROCESSING PLAN APPROVAL")
    print("="*70)
    
    # Debug: Show what plan we received
    num_agents_in_plan = len(proposed_plan.get("proposed_agents", []))
    print(f"\n🔍 DEBUG: Received plan with {num_agents_in_plan} agents in state")
    for idx, agent in enumerate(proposed_plan.get("proposed_agents", []), 1):
        print(f"   Agent {idx}: {agent.get('domain', 'Unknown')} ({agent.get('recommended_character', 'Unknown')})")

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
            "assigned_topic": agent_proposal.get("assigned_topic", ""),  # NEW: Topic from broad search
            "seed_papers": agent_proposal.get("seed_papers", []),  # NEW: Seed papers for this topic
            "custom_config": agent_proposal.get("custom_config")
        }
        character_configs.append(char_config)

    print(f"\n✅ Plan approved with {len(character_configs)} agents")
    print("\n🔍 DEBUG: Final character_configs:")
    for idx, config in enumerate(character_configs, 1):
        print(f"   Config {idx}: {config['domain']} ({config['character_id']})")

    # CRITICAL FIX: Return the proposed_research_plan in the update to ensure
    # it overwrites the checkpoint state. LangGraph's Command(resume={...}) doesn't
    # automatically merge nested dicts - we must explicitly return it.
    return {
        "plan_approved": True,
        "character_configs": character_configs,
        "proposed_research_plan": proposed_plan  # Explicitly return to overwrite checkpoint
    }
