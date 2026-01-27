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
from llmkglitrev.characters.schema import ResearchCharacter


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
    print(f"📚 Retrieved {len(broad_papers)} papers from literature search")
    print(f"🏷️  Identified {len(topics)} topics: {', '.join(topics)}")

    # Generate research plan using LLM - characters will be generated dynamically
    print("\n🤔 Analyzing topics and generating specialized research characters...")

    # Format topics for prompt
    topics_str = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics)])

    # Build prompt with topic information
    prompt = f"""
Based on broad literature search, we identified these research topics:

{topics_str}

Each topic should be covered by a specialized research character with appropriate expertise.

{propose_agents_prompt.format(
    research_topic=research_topic,
    literature_context=literature_context[:3000]
)}
"""

    try:
        research_plan: ResearchPlan = await planning_model.ainvoke(prompt)

        print("\n✅ Research plan generated!")
        print(f"\n🎯 Strategy: {research_plan.research_strategy}")
        print(f"\n👥 Generated {len(research_plan.proposed_agents)} specialized characters:")

        for i, agent in enumerate(research_plan.proposed_agents, 1):
            char = agent.character
            print(f"\n  Agent {i}: {char.name}")
            print(f"    Domain: {char.domain} ({char.stance} stance)")
            print(f"    Expertise: {', '.join(char.expertise_areas[:3])}...")
            print(f"    Venues: {', '.join(char.typical_venues[:3])}...")
            print(f"    Databases: {', '.join(char.preferred_databases)}")
            print(f"    Search: {', '.join(agent.search_scope[:3])}...")
            print(f"    Rationale: {agent.rationale[:100]}...")

        # Convert to dict for JSON serialization
        plan_dict = research_plan.model_dump()

        # Assign topics and seed papers to each agent
        # Match agents to topics (1-to-1 mapping)
        for i, agent in enumerate(plan_dict["proposed_agents"]):
            # Get character info for better topic assignment
            char = agent.get("character", {})
            char_domain = char.get("domain", "")

            # Assign topic (round-robin if more agents than topics)
            if topics:
                topic_idx = i % len(topics)
                assigned_topic = topics[topic_idx]
                # Get seed papers for this topic
                seed_papers = topic_papers.get(assigned_topic, [])[:5]
            else:
                # No topics from literature search - use character's domain as topic
                assigned_topic = char_domain
                seed_papers = []

            # Add to agent config
            agent["assigned_topic"] = assigned_topic
            agent["seed_papers"] = seed_papers

            print(f"\n  📌 Agent {i+1} ({char.get('name', 'Unknown')}) assigned to topic: {assigned_topic}")
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

        # Return a fallback plan if LLM fails - create generic characters
        fallback_char_1 = ResearchCharacter(
            character_id="fallback_researcher_1",
            name="General Research Analyst",
            domain="Research Analysis",
            stance="neutral",
            expertise_areas=["Literature review", "Research methods", "Data analysis"],
            typical_venues=["arXiv", "General conferences"],
            preferred_databases=["arxiv", "semantic_scholar", "openalex"],
            background="General research expert for broad topic analysis",
            communication_style="Balanced and analytical",
            description="Fallback research character for general analysis",
            sub_domains=["Research methods", "Analysis"]
        )

        fallback_char_2 = ResearchCharacter(
            character_id="fallback_researcher_2",
            name="Applied Research Expert",
            domain="Applied Research",
            stance="constructive",
            expertise_areas=["Practical applications", "Case studies", "Implementation"],
            typical_venues=["Applied conferences", "Practical journals"],
            preferred_databases=["scopus", "crossref"],
            background="Expert in practical research applications",
            communication_style="Pragmatic and application-focused",
            description="Fallback character for applied research",
            sub_domains=["Applications", "Implementation"]
        )

        fallback_plan = {
            "research_strategy": "Use generic research analysts for multi-perspective exploration",
            "proposed_agents": [
                {
                    "character": fallback_char_1.model_dump(),
                    "search_scope": ["research", "analysis", research_topic],
                    "rationale": "Provides general analytical perspective on the research topic"
                },
                {
                    "character": fallback_char_2.model_dump(),
                    "search_scope": ["applications", "implementation", research_topic],
                    "rationale": "Explores practical applications and real-world relevance"
                }
            ],
            "interdisciplinary_connections": "Agents will collaborate on theoretical and practical aspects"
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
        char = agent.get('character', {})
        print(f"   Agent {idx}: {char.get('name', 'Unknown')} - {char.get('domain', 'Unknown')}")

    # Extract character configurations from approved plan
    # Characters are now embedded in each agent proposal
    character_configs = []

    for agent_proposal in proposed_plan.get("proposed_agents", []):
        # Get the embedded character
        character = agent_proposal.get("character", {})

        char_config = {
            "character": character,  # Full character object embedded
            "search_scope": agent_proposal.get("search_scope", []),
            "assigned_topic": agent_proposal.get("assigned_topic", ""),  # Topic from broad search
            "seed_papers": agent_proposal.get("seed_papers", []),  # Seed papers for this topic
            "rationale": agent_proposal.get("rationale", "")
        }
        character_configs.append(char_config)

    print(f"\n✅ Plan approved with {len(character_configs)} agents")
    print("\n🔍 DEBUG: Final character_configs:")
    for idx, config in enumerate(character_configs, 1):
        char = config['character']
        print(f"   Config {idx}: {char.get('name', 'Unknown')} ({char.get('domain', 'Unknown')})")

    # CRITICAL FIX: Return the proposed_research_plan in the update to ensure
    # it overwrites the checkpoint state. LangGraph's Command(resume={...}) doesn't
    # automatically merge nested dicts - we must explicitly return it.
    return {
        "plan_approved": True,
        "character_configs": character_configs,
        "proposed_research_plan": proposed_plan  # Explicitly return to overwrite checkpoint
    }
