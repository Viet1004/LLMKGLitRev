"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation  
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from llmkglitrev.agents.tools import get_today_str
from llmkglitrev.agents.prompts.research_planning import plan_research_full_agent
from llmkglitrev.agents.prompts.research_summary import research_agent_keyword_extractor
from llmkglitrev.agents.states import AgentState, AgentInputState, SupervisorState, KeyWordsList
from llmkglitrev.agents.research_supervisor import create_interactive_supervisor
from llmkglitrev.agents.research_planner import propose_research_plan, process_plan_approval
from llmkglitrev.agents.academic_search_tools import (
    _search_arxiv_internal,
    _deduplicate_papers
)
from typing import Union, List, Dict
from langchain.chat_models import init_chat_model
import uuid
# writer_model = init_chat_model(model="openai:gpt-4o", max_tokens=16000) # model="anthropic:claude-sonnet-4-20250514", max_tokens=64000
writer_model = init_chat_model(model="deepseek:deepseek-chat")

summarize_model = init_chat_model(model="deepseek:deepseek-chat").with_structured_output(KeyWordsList)
async def format_question(state:AgentState):
    """
    Generate research keywords and initialize session.
    """
    query = research_agent_keyword_extractor.format(
        research_prompt=state.get('messages', "")
    )

    keywords = await summarize_model.ainvoke(query)

    # Extract research topic from messages
    messages = state.get('messages', [])
    research_topic = str(messages[0].content) if messages else ""

    # Generate session ID if not present
    session_id = state.get('session_id', '') or str(uuid.uuid4())

    return {
        "supervisor_messages": [HumanMessage(content=f"{state['messages']}.")],
        "research_keywords": keywords.keywords,
        "research_topic": research_topic,
        "session_id": session_id,
        "plan_approved": False,
        "active_characters": [],
        "conversation_artifacts": [],
        "character_configs": []
    }

async def broad_literature_search(state: AgentState):
    """
    Broad literature search using arXiv only.

    Replaces Neo4j vector search with direct arXiv academic source search.
    Returns 20-40 papers across multiple topics to give broad landscape.
    """
    keywords = state.get('research_keywords', [])
    research_topic = state.get('research_topic', '')

    print(f"\n🔍 Broad Literature Search (arXiv)")
    print(f"   Keywords: {', '.join(keywords[:5])}")
    print("="*70)

    try:
        all_papers = []

        # Search using top 3 keywords
        search_terms = keywords[:3] if len(keywords) >= 3 else [research_topic]

        for i, keyword in enumerate(search_terms, 1):
            print(f"\n📚 Search {i}/{len(search_terms)}: '{keyword}'")

            # arXiv search only
            print("   🔎 Searching arXiv...")
            arxiv_papers = _search_arxiv_internal(
                query=keyword,
                max_results=15,  # Increased from 10 since we're only using arXiv
                date_from="2018-01-01"
            )
            print(f"      Found {len(arxiv_papers)} papers")
            all_papers.extend(arxiv_papers)

        # Deduplicate papers
        print(f"\n🔄 Deduplicating {len(all_papers)} total papers...")
        unique_papers = _deduplicate_papers(all_papers)

        # Sort by relevance (citations + recency)
        unique_papers.sort(
            key=lambda x: (
                x.get("citations", 0),
                x.get("year", 0),
                x.get("relevance_score", 0)
            ),
            reverse=True
        )

        # Extract topics from papers
        print(f"\n🏷️  Extracting topics from {len(unique_papers)} papers...")
        topics = await extract_topics_from_papers(unique_papers[:20])  # Use top 20 for topic extraction

        # Cluster papers by topic
        print(f"📊 Clustering papers into {len(topics)} topics...")
        topic_papers = cluster_papers_by_topic(unique_papers, topics)

        # Format for LLM context
        literature_context = format_papers_for_llm(unique_papers[:15])  # Use top 15 for context

        print(f"\n✅ Broad search complete:")
        print(f"   • {len(unique_papers)} unique papers found")
        print(f"   • {len(topics)} topics identified: {', '.join(topics[:5])}")
        print(f"   • Papers per topic: {', '.join([f'{t}: {len(topic_papers[t])}' for t in topics[:3]])}")
        print("="*70)

        return {
            "broad_papers": unique_papers,
            "topics": topics,
            "topic_papers": topic_papers,
            "literature_context": literature_context,
            "retrieved_papers": unique_papers  # Backward compatibility
        }

    except Exception as e:
        print(f"\n⚠️  Error in broad literature search: {e}")
        import traceback
        traceback.print_exc()
        return {
            "broad_papers": [],
            "topics": [],
            "topic_papers": {},
            "literature_context": "Literature search unavailable. Please check arXiv installation.",
            "retrieved_papers": []
        }


async def extract_topics_from_papers(papers: List[Dict]) -> List[str]:
    """
    Extract main topics from paper titles and abstracts using LLM.

    Args:
        papers: List of paper dictionaries

    Returns:
        List of 3-5 topic strings
    """
    if not papers:
        return []

    # Prepare paper summaries
    paper_summaries = []
    for i, paper in enumerate(papers[:20], 1):  # Use top 20 papers
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")[:200]  # First 200 chars
        paper_summaries.append(f"{i}. {title}\n   Abstract: {abstract}...")

    papers_text = "\n\n".join(paper_summaries)

    # Ask LLM to extract topics
    prompt = f"""Based on these academic papers, identify 3-5 main research topics or themes.

Papers:
{papers_text}

Provide 3-5 concise topic names (each 3-6 words) that capture the main themes.
Format: Return ONLY a comma-separated list of topics, nothing else.

Example: "Transfer learning methods, Medical image classification, Few-shot learning approaches"
"""

    try:
        response = await writer_model.ainvoke([HumanMessage(content=prompt)])
        topics_text = response.content.strip()

        # Parse topics
        topics = [t.strip() for t in topics_text.split(",")]
        topics = [t for t in topics if t]  # Remove empty

        # Limit to 5 topics
        return topics[:5]

    except Exception as e:
        print(f"   ⚠️ Error extracting topics: {e}")
        # Fallback: use keywords
        return state.get('research_keywords', [])[:3]


def cluster_papers_by_topic(papers: List[Dict], topics: List[str]) -> Dict[str, List[Dict]]:
    """
    Cluster papers by topic using simple keyword matching.

    Args:
        papers: List of paper dictionaries
        topics: List of topic strings

    Returns:
        Dictionary mapping topic -> list of papers
    """
    topic_papers = {topic: [] for topic in topics}

    for paper in papers:
        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", "").lower()
        paper_text = f"{title} {abstract}"

        # Find best matching topic
        best_topic = None
        best_score = 0

        for topic in topics:
            # Count keyword matches
            topic_keywords = topic.lower().split()
            score = sum(1 for kw in topic_keywords if kw in paper_text)

            if score > best_score:
                best_score = score
                best_topic = topic

        # Assign to best topic (or first topic if no match)
        if best_topic and best_score > 0:
            topic_papers[best_topic].append(paper)
        else:
            # Assign to first topic as fallback
            topic_papers[topics[0]].append(paper)

    return topic_papers


def format_papers_for_llm(papers: List[Dict]) -> str:
    """
    Format papers for LLM context (similar to Neo4j format).

    Args:
        papers: List of paper dictionaries

    Returns:
        Formatted string for LLM prompt
    """
    if not papers:
        return "No relevant papers found in literature search."

    formatted = f"## Retrieved Literature ({len(papers)} papers from Google Scholar + arXiv):\n\n"

    for i, paper in enumerate(papers, 1):
        formatted += f"**[{i}] {paper.get('title', 'Untitled')}**\n"

        # Authors
        authors = paper.get('authors', [])
        if authors:
            author_str = ', '.join(authors[:3])
            if len(authors) > 3:
                author_str += f" et al. ({len(authors)} authors)"
            formatted += f"   Authors: {author_str}\n"

        # Year and venue
        year = paper.get('year', 'Unknown')
        venue = paper.get('venue', 'Unknown venue')
        formatted += f"   Year: {year} | Venue: {venue}\n"

        # Citations
        citations = paper.get('citations', 0)
        if citations:
            formatted += f"   Citations: {citations}\n"

        # Abstract (truncated)
        abstract = paper.get('abstract', 'No abstract available')
        if len(abstract) > 300:
            abstract = abstract[:300] + "..."
        formatted += f"   Abstract: {abstract}\n"

        # URL
        url = paper.get('url', '')
        if url:
            formatted += f"   URL: {url}\n"

        # PDF URL (if arXiv)
        pdf_url = paper.get('pdf_url', '')
        if pdf_url:
            formatted += f"   PDF: {pdf_url}\n"

        formatted += "\n"

    return formatted

def save_artifacts(state: AgentState) -> dict:
    """
    Save conversation artifacts to disk.

    This node persists artifacts from the supervisor to the session directory
    so they can be accessed later for dialogue or review.
    """
    from llmkglitrev.characters import ConversationArtifactManager, ConversationArtifact

    session_id = state.get("session_id", "")
    conversation_artifacts = state.get("conversation_artifacts", [])

    if not session_id:
        print("\n⚠️  Warning: No session ID found, artifacts not saved")
        return {}

    if not conversation_artifacts:
        print("\n⚠️  Warning: No conversation artifacts to save")
        return {}

    # Initialize artifact manager
    artifact_manager = ConversationArtifactManager()

    print(f"\n💾 Saving {len(conversation_artifacts)} artifacts to sessions/{session_id}/")

    # Save each artifact
    saved_count = 0
    for artifact_dict in conversation_artifacts:
        try:
            # Convert dict back to ConversationArtifact
            artifact = ConversationArtifact.model_validate(artifact_dict)

            # Save to disk
            artifact_manager.save_artifact(session_id, artifact)
            saved_count += 1

            print(f"  ✅ Saved artifact for {artifact.domain} ({artifact.character_id})")

        except Exception as e:
            print(f"  ❌ Error saving artifact: {e}")

    print(f"✅ Saved {saved_count}/{len(conversation_artifacts)} artifacts\n")

    return {}  # No state changes needed


async def final_research_proposal(state:AgentState | SupervisorState):
    """
    Supervisor Synthesis: Create Unified Research Proposal (Phase 4)

    This is where the supervisor synthesizes insights from ALL character agents
    into a single, coherent research proposal. This is YOUR proposal to present.

    Process:
    1. Collect all research findings from character agents (via artifacts)
    2. Synthesize into unified narrative
    3. Create comprehensive research proposal
    4. Save proposal to state

    IMPORTANT: After this step, artifacts are already saved. Characters in Phase 5-7
    will LOAD these artifacts (not do new research) for dialogue and validation.

    Args:
        state: Contains research_proposals, raw_notes, and conversation_artifacts

    Returns:
        final_proposal: Unified research proposal synthesized from all agents
    """
    proposals = state.get("research_proposals", [])
    notes = state.get("raw_notes", [])
    conversation_artifacts = state.get("conversation_artifacts", [])

    print("\n" + "="*70)
    print("🎓 PHASE 4: SUPERVISOR SYNTHESIS")
    print("="*70)
    print(f"\n📊 Synthesizing insights from {len(conversation_artifacts)} character agents")
    print(f"   • Research proposals: {len(proposals)}")
    print(f"   • Raw notes: {len(notes)}")

    # Combine all findings
    findings = "\n".join(proposals)

    # Generate unified proposal
    print("\n✍️  Creating unified research proposal...")
    final_research_proposal_prompt = plan_research_full_agent.format(
        research_topic=state.get("research_topic", ""),
        findings=findings,
        notes=notes,
        date=get_today_str()
    )
    final_proposal = await writer_model.ainvoke([HumanMessage(content=final_research_proposal_prompt)])

    print(f"\n✅ Unified research proposal created ({len(final_proposal.content)} characters)")
    print("="*70)
    print("\n📝 This proposal will now be used for:")
    print("   • Phase 5: Socratic dialogue (characters ask critical questions)")
    print("   • Phase 6: Cross-domain validation (identify knowledge gaps)")
    print("   • Phase 7: Industry partner review (optional)")
    print("="*70)

    return {
        "final_proposal": final_proposal.content,
        "messages": ["Here is the final proposal: " + final_proposal.content],
    }
from langgraph.checkpoint.memory import MemorySaver


def route_after_plan_approval(state: AgentState) -> str:
    """
    Route to supervisor if plan is approved, otherwise loop back to propose_research_plan.
    """
    if state.get("plan_approved", False):
        return "instantiate_agents"
    else:
        # Plan was rejected or needs modification - re-propose
        return "propose_research_plan"


def instantiate_agents(state: AgentState) -> dict:
    """
    Instantiate character-based agents from the approved research plan.

    This converts the approved plan into actual character configurations
    that the supervisor can use to spawn agents.
    """
    from llmkglitrev.characters import CharacterManager, ResearchCharacter

    print("\n" + "="*70)
    print("🎭 INSTANTIATING RESEARCH AGENTS")
    print("="*70)

    character_configs = state.get("character_configs", [])
    char_manager = CharacterManager()

    active_characters = []

    for config in character_configs:
        char_id = config.get("character_id")
        domain = config.get("domain")
        stance = config.get("stance", "neutral")
        assigned_topic = config.get("assigned_topic", "")
        seed_papers = config.get("seed_papers", [])

        print(f"\n📌 Configuring agent for {domain} ({stance})")
        if assigned_topic:
            print(f"   🎯 Assigned topic: {assigned_topic}")
            print(f"   📄 Seed papers: {len(seed_papers)}")

        if char_id != "custom":
            # Load existing character template
            try:
                character = char_manager.load_character(char_id)
                print(f"   ✅ Loaded template: {character.name}")
            except Exception as e:
                print(f"   ⚠️  Failed to load {char_id}, using fallback")
                # Create a basic character if template fails
                character = ResearchCharacter(
                    character_id=f"agent_{domain.lower().replace(' ', '_')}",
                    name=f"{domain} Expert",
                    domain=domain,
                    stance=stance,
                    system_prompt_template=f"You are a {domain} expert with a {stance} perspective.",
                    expertise_areas=[domain],
                    research_style=stance
                )
        else:
            # Create custom character from config
            custom_config = config.get("custom_config", {})
            character = ResearchCharacter(
                character_id=custom_config.get("character_id", f"custom_{domain.lower()}"),
                name=custom_config.get("name", f"{domain} Expert"),
                domain=domain,
                stance=stance,
                system_prompt_template=custom_config.get("system_prompt", ""),
                expertise_areas=custom_config.get("expertise_areas", [domain]),
                research_style=custom_config.get("research_style", stance)
            )
            print(f"   ✅ Created custom character: {character.name}")

        # Add to active characters list with topic and seed papers
        character_dict = character.model_dump()
        character_dict["assigned_topic"] = assigned_topic  # NEW: Topic from broad search
        character_dict["seed_papers"] = seed_papers  # NEW: Seed papers for deep research
        active_characters.append(character_dict)

    print(f"\n✅ Instantiated {len(active_characters)} research agents")
    print("="*70)

    return {
        "active_characters": active_characters
    }


supervisor_agent = create_interactive_supervisor()
agent_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add nodes
agent_builder.add_node("format_question", format_question)
agent_builder.add_node("broad_literature_search", broad_literature_search)
agent_builder.add_node("propose_research_plan", propose_research_plan)  # NEW: Propose agents
agent_builder.add_node("process_plan_approval", process_plan_approval)  # NEW: Process approval
agent_builder.add_node("instantiate_agents", instantiate_agents)  # NEW: Create character agents
agent_builder.add_node("supervisor_subgraph", supervisor_agent)
agent_builder.add_node("save_artifacts", save_artifacts)  # NEW: Save artifacts to disk
agent_builder.add_node("final_research_proposal", final_research_proposal)

# Add edges - NEW WORKFLOW:
# 1. Format question and extract keywords
agent_builder.add_edge(START, "format_question")

# 2. Broad literature search (replaces Neo4j)
agent_builder.add_edge("format_question", "broad_literature_search")

# 3. Propose research plan with interrupt for approval
agent_builder.add_edge("broad_literature_search", "propose_research_plan")

# 4. Process approval (runs after interrupt is resumed)
agent_builder.add_edge("propose_research_plan", "process_plan_approval")

# 5. Route based on approval: instantiate agents or re-propose
agent_builder.add_conditional_edges(
    "process_plan_approval",
    route_after_plan_approval,
    {
        "instantiate_agents": "instantiate_agents",
        "propose_research_plan": "propose_research_plan"
    }
)

# 6. Run supervisor with instantiated agents
agent_builder.add_edge("instantiate_agents", "supervisor_subgraph")

# 7. Save conversation artifacts to disk
agent_builder.add_edge("supervisor_subgraph", "save_artifacts")

# 8. Generate final proposal
agent_builder.add_edge("save_artifacts", "final_research_proposal")

# 9. End
agent_builder.add_edge("final_research_proposal", END)

# Compile with MemorySaver checkpointer to support interrupts
# Configure to interrupt BEFORE process_plan_approval for user approval
checkpointer = MemorySaver()
proposal_generator_agent = agent_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["process_plan_approval"]
)

# Import interactive runner for standalone usage
from llmkglitrev.agents.interactive_runner import (
    run_research_interactive,
    run_research_interactive_sync,
    resume_research_interactive
)
    