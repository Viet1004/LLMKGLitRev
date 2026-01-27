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
    _search_scopus_internal,
    _search_ieee_internal,
    _search_semantic_scholar_internal,
    _search_openalex_internal,
    _search_crossref_internal,
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
    Broad literature search using arXiv, Scopus, and IEEE.

    Uses sophisticated search strategies:
    - Boolean queries (AND/OR operators)
    - Keyword combinations for broader coverage
    - Multiple search patterns per database

    Searches multiple academic databases to provide comprehensive coverage.
    Returns 30-90 papers across multiple search strategies.
    """
    keywords = state.get('research_keywords', [])
    research_topic = state.get('research_topic', '')

    print(f"\n🔍 Broad Literature Search (arXiv + Scopus + IEEE)")
    print(f"   Keywords: {', '.join(keywords[:7])}")
    print("="*70)

    try:
        all_papers = []

        # Construct sophisticated search queries
        # Strategy 1: Core concept combinations (AND operator)
        # Strategy 2: Alternative approaches (OR operator)
        # Strategy 3: Specific methods within domain

        search_queries = []

        if len(keywords) >= 3:
            # Strategy 1: Primary keywords combined with AND (precise, focused)
            search_queries.append({
                'query': f"{keywords[0]} AND {keywords[1]}",
                'description': f"Core concept: {keywords[0]} + {keywords[1]}",
                'strategy': 'focused'
            })

            # Strategy 2: Broader search with OR (captures variations)
            if len(keywords) >= 4:
                search_queries.append({
                    'query': f"({keywords[0]} OR {keywords[1]}) AND {keywords[2]}",
                    'description': f"Broader: ({keywords[0]} OR {keywords[1]}) + {keywords[2]}",
                    'strategy': 'broad'
                })

            # Strategy 3: Method-focused search
            if len(keywords) >= 5:
                search_queries.append({
                    'query': f"{keywords[0]} AND ({keywords[3]} OR {keywords[4]})",
                    'description': f"Method-focused: {keywords[0]} + ({keywords[3]} OR {keywords[4]})",
                    'strategy': 'method'
                })

            # Strategy 4: Alternative primary concepts
            if len(keywords) >= 3:
                search_queries.append({
                    'query': f"{keywords[2]}",
                    'description': f"Alternative approach: {keywords[2]}",
                    'strategy': 'alternative'
                })
        else:
            # Fallback: use research topic
            search_queries = [{
                'query': research_topic,
                'description': f"Main topic: {research_topic}",
                'strategy': 'fallback'
            }]

        print(f"\n📋 Constructed {len(search_queries)} search strategies:")
        for idx, sq in enumerate(search_queries, 1):
            print(f"   {idx}. [{sq['strategy']}] {sq['description']}")
        print()

        # Execute searches across all databases with error handling
        # Track which sources succeeded/failed
        source_stats = {
            'arxiv': {'success': 0, 'failed': 0, 'papers': 0},
            'semantic_scholar': {'success': 0, 'failed': 0, 'papers': 0},
            'openalex': {'success': 0, 'failed': 0, 'papers': 0},
            'crossref': {'success': 0, 'failed': 0, 'papers': 0},
            'scopus': {'success': 0, 'failed': 0, 'papers': 0},
            'ieee': {'success': 0, 'failed': 0, 'papers': 0}
        }

        for i, search_item in enumerate(search_queries, 1):
            query = search_item['query']
            description = search_item['description']

            print(f"\n📚 Search {i}/{len(search_queries)}: {description}")
            print(f"   Query: '{query}'")

            # arXiv search
            print("   🔎 Searching arXiv...")
            try:
                arxiv_papers = _search_arxiv_internal(
                    query=query,
                    max_results=10,
                    date_from="2018-01-01"
                )
                print(f"      ✅ Found {len(arxiv_papers)} papers")
                all_papers.extend(arxiv_papers)
                source_stats['arxiv']['success'] += 1
                source_stats['arxiv']['papers'] += len(arxiv_papers)
            except Exception as e:
                print(f"      ❌ arXiv search failed: {str(e)}")
                source_stats['arxiv']['failed'] += 1

            # Semantic Scholar search (free, no API key)
            print("   🔎 Searching Semantic Scholar...")
            try:
                ss_papers = _search_semantic_scholar_internal(
                    query=query,
                    max_results=10,
                    year_from=2018
                )
                print(f"      ✅ Found {len(ss_papers)} papers")
                all_papers.extend(ss_papers)
                source_stats['semantic_scholar']['success'] += 1
                source_stats['semantic_scholar']['papers'] += len(ss_papers)
            except Exception as e:
                print(f"      ❌ Semantic Scholar search failed: {str(e)}")
                source_stats['semantic_scholar']['failed'] += 1

            # OpenAlex search (free, no API key)
            print("   🔎 Searching OpenAlex...")
            try:
                oa_papers = _search_openalex_internal(
                    query=query,
                    max_results=10,
                    year_from=2018
                )
                print(f"      ✅ Found {len(oa_papers)} papers")
                all_papers.extend(oa_papers)
                source_stats['openalex']['success'] += 1
                source_stats['openalex']['papers'] += len(oa_papers)
            except Exception as e:
                print(f"      ❌ OpenAlex search failed: {str(e)}")
                source_stats['openalex']['failed'] += 1

            # CrossRef search (free, no API key)
            print("   🔎 Searching CrossRef...")
            try:
                cr_papers = _search_crossref_internal(
                    query=query,
                    max_results=10,
                    year_from=2018
                )
                print(f"      ✅ Found {len(cr_papers)} papers")
                all_papers.extend(cr_papers)
                source_stats['crossref']['success'] += 1
                source_stats['crossref']['papers'] += len(cr_papers)
            except Exception as e:
                print(f"      ❌ CrossRef search failed: {str(e)}")
                source_stats['crossref']['failed'] += 1

            # # Scopus search (requires API key)
            # print("   🔎 Searching Scopus...")
            # try:
            #     scopus_papers = _search_scopus_internal(
            #         query=query,
            #         max_results=10,
            #         year_from=2018
            #     )
            #     print(f"      ✅ Found {len(scopus_papers)} papers")
            #     all_papers.extend(scopus_papers)
            #     source_stats['scopus']['success'] += 1
            #     source_stats['scopus']['papers'] += len(scopus_papers)
            # except Exception as e:
            #     print(f"      ⚠️ Scopus search failed: {str(e)}")
            #     print(f"         (Scopus requires API key and institutional access)")
            #     source_stats['scopus']['failed'] += 1

            # IEEE search (requires API key)
            # print("   🔎 Searching IEEE Xplore...")
            # try:
            #     ieee_papers = _search_ieee_internal(
            #         query=query,
            #         max_results=10,
            #         year_from=2018
            #     )
            #     print(f"      ✅ Found {len(ieee_papers)} papers")
            #     all_papers.extend(ieee_papers)
            #     source_stats['ieee']['success'] += 1
            #     source_stats['ieee']['papers'] += len(ieee_papers)
            # except Exception as e:
            #     print(f"      ⚠️ IEEE search failed: {str(e)}")
            #     print(f"         (IEEE requires API key)")
            #     source_stats['ieee']['failed'] += 1

        # Print source statistics
        print(f"\n📊 Search Source Statistics:")
        print("=" * 70)
        for source, stats in source_stats.items():
            total_attempts = stats['success'] + stats['failed']
            if total_attempts > 0:
                status = "✅" if stats['failed'] == 0 else "⚠️" if stats['success'] > 0 else "❌"
                print(f"   {status} {source:20s}: {stats['success']}/{total_attempts} successful, {stats['papers']} papers")
        print("=" * 70)

        # Deduplicate papers
        print(f"\n🔄 Deduplicating {len(all_papers)} total papers...")
        unique_papers = _deduplicate_papers(all_papers)
        # Sort by relevance (citations + recency)
        # unique_papers.sort(
        #     key=lambda x: (
        #         x.get("citations", 0),
        #         x.get("year", 0),
        #         x.get("relevance_score", 0)
        #     ),
        #     reverse=True
        # )

        # Extract topics from papers
        print(f"\n🏷️  Extracting topics from {len(unique_papers)} papers...")
        topics = await extract_topics_from_papers(unique_papers)  # Use top 20 for topic extraction

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
        title = (paper.get("title") or " ")
        abstract = (paper.get("abstract") or " ")[:200]  # First 200 chars
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
        title = (paper.get("title") or " ").lower()
        abstract = (paper.get("abstract") or " ").lower()
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
        if abstract and len(abstract) > 300:
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

    NEW: Uses unified format - updates existing character entries with research artifacts
    """
    from llmkglitrev.characters.artifact import UnifiedCharacterManager, ConversationArtifact

    session_id = state.get("session_id", "")
    conversation_artifacts = state.get("conversation_artifacts", [])

    if not session_id:
        print("\n⚠️  Warning: No session ID found, artifacts not saved")
        return {}

    if not conversation_artifacts:
        print("\n⚠️  Warning: No conversation artifacts to save")
        return {}

    # Initialize unified manager
    unified_manager = UnifiedCharacterManager()

    print(f"\n💾 Saving {len(conversation_artifacts)} artifacts to sessions/{session_id}/")

    # Update each character with its research artifact
    saved_count = 0
    for artifact_dict in conversation_artifacts:
        try:
            # Convert dict back to ConversationArtifact
            artifact = ConversationArtifact.model_validate(artifact_dict)

            # Update existing character entry with artifact
            unified_manager.update_artifact(
                session_id=session_id,
                character_id=artifact.character_id,
                artifact=artifact
            )
            saved_count += 1

            print(f"  ✅ Updated {artifact.character_id} with research artifact")

        except Exception as e:
            print(f"  ❌ Error saving artifact for {artifact_dict.get('character_id', 'unknown')}: {e}")

    print(f"✅ Saved {saved_count}/{len(conversation_artifacts)} artifacts\n")

    return {}  # No state changes needed


async def extract_ontology_concepts(state: AgentState) -> dict:
    """
    Extract ontology concepts from final proposal and agent artifacts.

    This runs AFTER research completes to save tokens.
    Extracts concepts only from final proposal, then maps back to which agents mentioned them.

    Process:
    1. Load domain ontology (EDAM or custom)
    2. Extract concepts from final proposal
    3. For each concept, find which agents mentioned it
    4. Build relationships between concepts
    5. Build hierarchical tree structure
    6. Save to session directory

    Args:
        state: Contains final_proposal, conversation_artifacts, session_id

    Returns:
        Dictionary with ontology_data key
    """
    from llmkglitrev.ontology import DomainOntologyManager
    from pathlib import Path
    import json

    session_id = state.get("session_id", "")
    final_proposal = state.get("final_proposal", "")
    conversation_artifacts = state.get("conversation_artifacts", [])

    print("\n" + "="*70)
    print("🧠 EXTRACTING ONTOLOGY CONCEPTS")
    print("="*70)

    if not final_proposal:
        print("⚠️  No final proposal found, skipping ontology extraction")
        return {}

    # Check if any character has a custom ontology URL
    custom_ontology_url = None
    for artifact in conversation_artifacts:
        character_config = artifact.get("character_config")
        if character_config and isinstance(character_config, dict):
            ont_url = character_config.get("ontology_url", "")
            if ont_url:
                custom_ontology_url = ont_url
                print(f"   Found custom ontology URL from character '{character_config.get('name')}': {ont_url}")
                break

    # ONLY load ontology if user provided a URL
    if not custom_ontology_url:
        print("ℹ️  No custom ontology URL provided - skipping ontology extraction")
        print("   Socratic dialogue will use priority-based ordering only (no taxonomy)")
        return {
            "ontology_data": {
                "ontology_concepts": [],
                "concept_labels": {},
                "concept_relationships": [],
                "concept_hierarchy": {},
                "concept_clusters": {},
                "concept_to_agents": {}
            }
        }

    # Load custom ontology
    print(f"\n📥 Loading custom ontology from: {custom_ontology_url}")
    ontology = DomainOntologyManager(ontology_source="custom", custom_url=custom_ontology_url)
    success = ontology.load()

    if not success:
        print("⚠️  Failed to load custom ontology - skipping ontology extraction")
        print("   Socratic dialogue will use priority-based ordering only (no taxonomy)")
        return {
            "ontology_data": {
                "ontology_concepts": [],
                "concept_labels": {},
                "concept_relationships": [],
                "concept_hierarchy": {},
                "concept_clusters": {},
                "concept_to_agents": {}
            }
        }

    # Extract concepts from FINAL PROPOSAL only
    print(f"\n🔍 Extracting concepts from final proposal...")
    proposal_concepts = ontology.extract_concepts(final_proposal)
    print(f"   Found {len(proposal_concepts)} concepts in final proposal")

    if not proposal_concepts:
        print("⚠️  No concepts found in final proposal")
        return {}

    # For each concept, find which agents mentioned it
    print(f"\n👥 Mapping concepts to agents...")
    concept_to_agents = {}
    for concept in proposal_concepts:
        concept_to_agents[concept] = []

        for artifact in conversation_artifacts:
            research_output = artifact.get("research_output", "")
            if concept.lower() in research_output.lower():
                concept_to_agents[concept].append(artifact.get("character_id"))

    # Build relationships between concepts
    print(f"\n🔗 Building concept relationships...")
    relationships = []
    for i, concept in enumerate(proposal_concepts):
        # Find related concepts using ontology
        related_concepts = ontology.find_related_concepts(concept, max_results=5)

        for rel_data in related_concepts:
            rel_concept = rel_data.get("concept")
            if rel_concept and rel_concept in proposal_concepts:  # Only include if also in proposal
                relationship_type = ontology._find_relationship_type(concept, rel_concept)
                relation_class = _classify_relation(relationship_type)

                relationships.append({
                    "source": concept,
                    "target": rel_concept,
                    "relation": relationship_type,
                    "relation_type": relation_class
                })

    print(f"   Found {len(relationships)} relationships")

    # Build hierarchical tree
    print(f"\n🌳 Building concept hierarchy...")
    hierarchy = ontology.build_concept_tree(proposal_concepts)

    # Cluster concepts by domain (top-level parent)
    print(f"\n📊 Clustering concepts...")
    clusters = {}
    for concept in proposal_concepts:
        top_parent = ontology.get_top_level_parent(concept)

        if top_parent not in clusters:
            clusters[top_parent] = []

        clusters[top_parent].append(concept)

    print(f"   Created {len(clusters)} clusters")

    # Create concept labels mapping (ID -> human-readable label)
    print(f"\n🏷️  Creating concept labels...")
    concept_labels = {}
    for concept in proposal_concepts:
        label = ontology.get_concept_label(concept)
        concept_labels[concept] = label

    # Save ontology data to session directory
    if session_id:
        ontology_data = {
            "ontology_concepts": proposal_concepts,
            "concept_labels": concept_labels,  # NEW: ID -> Label mapping
            "concept_relationships": relationships,
            "concept_hierarchy": hierarchy,
            "concept_clusters": clusters,
            "concept_to_agents": concept_to_agents
        }

        # Save to file
        session_path = Path(f"sessions/{session_id}")
        session_path.mkdir(parents=True, exist_ok=True)

        ontology_file = session_path / "ontology_data.json"
        with open(ontology_file, "w") as f:
            json.dump(ontology_data, f, indent=2)

        print(f"\n💾 Ontology data saved to: {ontology_file}")
    else:
        print("\n⚠️  No session ID, ontology data not saved to disk")
        ontology_data = {}

    print("✅ Ontology extraction complete\n")
    print("="*70)

    return {
        "ontology_data": ontology_data
    }


def _classify_relation(relation: str) -> str:
    """Classify relationship type for visualization."""
    if relation in ["is_a", "subClassOf"]:
        return "hierarchy"
    elif relation in ["part_of", "hasPart"]:
        return "part_of"
    else:
        return "association"


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

    # Save final proposal to disk
    session_id = state.get("session_id", "")
    if session_id:
        from pathlib import Path
        proposal_file = Path(f"sessions/{session_id}/final_proposal.md")

        # Create parent directory if needed
        proposal_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(proposal_file, "w") as f:
            f.write(final_proposal.content)

        print(f"💾 Final proposal saved to: {proposal_file}")
    else:
        print("⚠️  Warning: No session ID found, final proposal not saved to disk")

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

    NEW: Uses unified character format - saves characters to sessions/{session_id}/conversations/
    """
    from llmkglitrev.characters import ResearchCharacter
    from llmkglitrev.characters.artifact import UnifiedCharacterManager
    from llmkglitrev.characters.system_templates import get_system_template, SYSTEM_TEMPLATES

    print("\n" + "="*70)
    print("🎭 INSTANTIATING RESEARCH AGENTS")
    print("="*70)

    character_configs = state.get("character_configs", [])
    session_id = state.get("session_id")
    unified_manager = UnifiedCharacterManager()

    active_characters = []

    for config in character_configs:
        # Extract embedded character object
        character_data = config.get("character", {})
        assigned_topic = config.get("assigned_topic", "")
        seed_papers = config.get("seed_papers", [])

        # Validate and create ResearchCharacter from embedded data
        try:
            character = ResearchCharacter.model_validate(character_data)
        except Exception as e:
            print(f"   ⚠️  Error validating character: {e}")
            # Create fallback character
            domain = character_data.get("domain", "Research")
            character = ResearchCharacter(
                character_id=f"agent_{domain.lower().replace(' ', '_')}",
                name=character_data.get("name", f"{domain} Expert"),
                domain=domain,
                stance=character_data.get("stance", "neutral"),
                system_prompt_template=f"You are a {domain} expert.",
                sub_domains=[domain],
                communication_style="academic",
                expertise_areas=[],
                typical_venues=[],
                background=""
            )

        print(f"\n📌 Configuring agent: {character.name}")
        print(f"   Domain: {character.domain} ({character.stance})")
        if assigned_topic:
            print(f"   🎯 Assigned topic: {assigned_topic}")
            print(f"   📄 Seed papers: {len(seed_papers)}")
        print(f"   🎓 Expertise: {', '.join(character.expertise_areas[:3])}..." if character.expertise_areas else "")
        print(f"   📚 Venues: {', '.join(character.typical_venues[:3])}..." if character.typical_venues else "")

        # Save character to session (artifact=None since research hasn't started yet)
        unified_manager.save_character(
            session_id=session_id,
            character=character,
            artifact=None  # No artifact yet - will be added after research
        )
        print(f"   💾 Saved to sessions/{session_id}/conversations/{character.character_id}.json")

        # Add to active characters list with topic and seed papers
        character_dict = character.model_dump()
        character_dict["assigned_topic"] = assigned_topic
        character_dict["seed_papers"] = seed_papers
        active_characters.append(character_dict)

    print(f"\n✅ Instantiated {len(active_characters)} research agents")
    print("="*70)

    return {
        "active_characters": active_characters
    }


#  ===== PHASE 5-7: GAP IDENTIFICATION, SOCRATIC DIALOGUE, INDUSTRY REVIEW =====

async def identify_research_gaps(state: AgentState) -> dict:
    """Phase 5: Identify research gaps from ontology exploration.

    Characters explore their own ontologies to:
    1. Find research gaps in the unified proposal
    2. Suggest new research ideas from different perspectives
    3. Present gaps for user validation

    Artifacts are loaded from sessions/{session_id}/conversations/
    """
    session_id = state.get("session_id")
    final_proposal = state.get("final_proposal", "")
    conversation_artifacts = state.get("conversation_artifacts", [])

    print(f"\n{'='*70}")
    print(f"PHASE 5: RESEARCH GAP IDENTIFICATION")
    print(f"{'='*70}")
    print(f"Session ID: {session_id}")
    print(f"Loading character artifacts from disk...")
    print(f"Characters will explore their own ontologies to find research gaps")
    print(f"{'='*70}\n")

    # Load artifacts from disk
    from llmkglitrev.characters.artifact import ConversationArtifactManager, ConversationArtifact
    from llmkglitrev.characters import CharacterManager
    from llmkglitrev.agents.character_agent import CharacterBasedResearchAgent
    from langchain_core.messages import HumanMessage

    artifact_manager = ConversationArtifactManager()
    loaded_artifacts = artifact_manager.load_artifacts(session_id)

    if not loaded_artifacts:
        print("⚠️  No artifacts found on disk! Using in-memory artifacts...")
        loaded_artifacts = []
        for artifact_dict in conversation_artifacts:
            artifact = ConversationArtifact.model_validate(artifact_dict)
            loaded_artifacts.append(artifact)

    print(f"✓ Loaded {len(loaded_artifacts)} artifacts")

    # For each character, identify gaps using THEIR OWN ontology
    manager = CharacterManager()
    identified_gaps = []

    for artifact in loaded_artifacts:
        character = manager.load_character(artifact.character_id)

        # Create agent with artifact
        agent = CharacterBasedResearchAgent(
            character=character,
            session_id=session_id,
            literature_subset=[]
        )
        agent.artifact = artifact

        print(f"🔍 {character.name} exploring ontology for gaps...")

        # Use ontology to identify gaps
        gap_identification_prompt = f"""
Review the unified research proposal and your research findings.

Using YOUR domain ontology, identify:
1. Research gaps - what's missing or unexplored
2. Alternative perspectives - different ways to approach this
3. New research ideas - extensions or related questions

Unified Proposal:
{final_proposal}

Your Research:
{artifact.research_output[:2000]}...  # Truncate for context

For each gap, provide:
- Gap description
- Why it's important (from YOUR ontology perspective)
- Suggested research direction
- Feasibility assessment
"""

        # Get LLM response
        response = await agent.agent_executor.ainvoke({
            "messages": [HumanMessage(content=gap_identification_prompt)]
        })

        gap_text = response["messages"][-1].content

        identified_gaps.append({
            "character_id": artifact.character_id,
            "character_name": character.name,
            "domain": artifact.domain,
            "gaps_identified": gap_text,
            "ontology_used": "own"  # Using own ontology
        })

        print(f"✓ {character.name} identified gaps\n")

    print(f"\n✅ Gap identification complete!")
    print(f"Identified gaps from {len(identified_gaps)} characters\n")

    return {
        "identified_gaps": identified_gaps,
        "gap_identification_complete": True
    }


async def run_socratic_dialogue(state: AgentState) -> dict:
    """Phase 6: Socratic dialogue using own ontology perspectives.

    Characters ask critical questions using their own ontology for presentation prep.
    Uses existing dialogue_coordinator infrastructure.
    """
    session_id = state.get("session_id")

    print(f"\n{'='*70}")
    print(f"PHASE 6: SOCRATIC DIALOGUE")
    print(f"{'='*70}")
    print(f"Session ID: {session_id}")
    print(f"Characters will ask critical questions using their own ontologies")
    print(f"{'='*70}\n")

    # Load artifacts (same as Phase 5)
    from llmkglitrev.characters.artifact import ConversationArtifactManager

    artifact_manager = ConversationArtifactManager()
    loaded_artifacts = artifact_manager.load_artifacts(session_id)

    if not loaded_artifacts:
        print("⚠️  No artifacts found! Using in-memory artifacts...")
        conversation_artifacts = state.get("conversation_artifacts", [])
        from llmkglitrev.characters.artifact import ConversationArtifact
        loaded_artifacts = []
        for artifact_dict in conversation_artifacts:
            artifact = ConversationArtifact.model_validate(artifact_dict)
            loaded_artifacts.append(artifact)

    print(f"✓ Loaded {len(loaded_artifacts)} artifacts")

    # Use dialogue_coordinator (existing code)
    from llmkglitrev.agents.dialogue_coordinator import create_dialogue_coordinator

    coordinator = create_dialogue_coordinator(min_priority=7)

    dialogue_state = {
        "session_id": session_id,
        "conversation_artifacts": [a.model_dump() for a in loaded_artifacts],
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

    config = {"configurable": {"thread_id": f"dialogue-{session_id}"}}

    # Run dialogue with interrupts for user feedback
    print("Starting Socratic dialogue coordinator...")
    async for event in coordinator.astream(dialogue_state, config=config):
        # Just track progress, actual interaction happens in Streamlit via interrupts
        pass

    # Get final state
    final_state = coordinator.get_state(config)
    dialogue_history = final_state.values.get("dialogue_history", [])

    print(f"\n✅ Socratic dialogue complete!")
    print(f"Recorded {len(dialogue_history)} question-answer exchanges\n")

    return {
        "dialogue_history": dialogue_history,
        "dialogue_complete": True
    }


async def industry_partner_review(state: AgentState) -> dict:
    """Phase 7: Industry partners review using OTHER researchers' ontologies.

    Industrial partners ask operational questions about:
    - Project alignment and objectives
    - Implementation steps and resources
    - Practical feasibility and risks

    Key difference: Partners review using OTHER researchers' ontologies (not their own).
    """
    session_id = state.get("session_id")
    final_proposal = state.get("final_proposal", "")

    print(f"\n{'='*70}")
    print(f"PHASE 7: INDUSTRY PARTNER REVIEW")
    print(f"{'='*70}")
    print(f"Session ID: {session_id}")
    print(f"Industry partners will review using OTHER researchers' ontologies")
    print(f"{'='*70}\n")

    # Load artifacts
    from llmkglitrev.characters.artifact import ConversationArtifactManager

    artifact_manager = ConversationArtifactManager()
    loaded_artifacts = artifact_manager.load_artifacts(session_id)

    if not loaded_artifacts:
        print("⚠️  No artifacts found! Using in-memory artifacts...")
        conversation_artifacts = state.get("conversation_artifacts", [])
        from llmkglitrev.characters.artifact import ConversationArtifact
        loaded_artifacts = []
        for artifact_dict in conversation_artifacts:
            artifact = ConversationArtifact.model_validate(artifact_dict)
            loaded_artifacts.append(artifact)

    print(f"✓ Loaded {len(loaded_artifacts)} artifacts")

    # Create industry partner personas
    industry_partners = [
        {
            "name": "Project Manager",
            "focus": "Project alignment, timelines, deliverables",
            "ontology_perspective": "operational"
        },
        {
            "name": "Technical Lead",
            "focus": "Implementation steps, technical feasibility",
            "ontology_perspective": "technical"
        },
        {
            "name": "Stakeholder",
            "focus": "ROI, impact, resource allocation",
            "ontology_perspective": "business"
        }
    ]

    industry_feedback = []

    # Use LLM to generate feedback from each partner
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage

    review_model = init_chat_model(model="deepseek:deepseek-chat")

    for partner in industry_partners:
        print(f"👔 {partner['name']} reviewing...")

        # Partner reviews using OTHER researchers' ontologies
        ontology_list = "\n".join([
            f"- {a.character_id} ({a.domain}): {a.research_output[:500]}..."
            for a in loaded_artifacts
        ])

        prompt = f"""
You are a {partner['name']} reviewing this research proposal.

IMPORTANT: Review using the ontologies and perspectives of these researchers:
{ontology_list}

Proposal:
{final_proposal[:2000]}...

From your {partner['focus']} perspective, ask operational questions about:
- How does this align with project objectives?
- What are the implementation steps?
- What resources are needed?
- What are the risks and mitigations?
- What is the timeline for deliverables?

Provide specific, actionable feedback focused on practical feasibility.
"""

        response = await review_model.ainvoke([HumanMessage(content=prompt)])

        industry_feedback.append({
            "partner_name": partner['name'],
            "focus_area": partner['focus'],
            "ontologies_reviewed": [a.character_id for a in loaded_artifacts],
            "feedback": response.content
        })

        print(f"✓ {partner['name']} review complete\n")

    print(f"\n✅ Industry partner review complete!")
    print(f"Collected feedback from {len(industry_feedback)} partners\n")

    return {
        "industry_feedback": industry_feedback,
        "industry_review_complete": True
    }


# ===== GRAPH CONSTRUCTION =====

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
agent_builder.add_node("extract_ontology_concepts", extract_ontology_concepts)  # Extract ontology concepts
agent_builder.add_node("identify_research_gaps", identify_research_gaps)  # PHASE 5
agent_builder.add_node("run_socratic_dialogue", run_socratic_dialogue)  # PHASE 6
agent_builder.add_node("industry_partner_review", industry_partner_review)  # PHASE 7

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

# 9. Extract ontology concepts (after proposal is generated)
agent_builder.add_edge("final_research_proposal", "extract_ontology_concepts")

# 10. End (Phases 1-4 complete)
agent_builder.add_edge("extract_ontology_concepts", END)

# NOTE: Phases 5-7 nodes are defined above but not connected to the main workflow yet.
# They will be triggered manually from the Streamlit UI tabs.
# To enable automatic execution, uncomment these edges:
#
# agent_builder.add_edge("final_research_proposal", "identify_research_gaps")  # Phase 5
# agent_builder.add_edge("identify_research_gaps", "run_socratic_dialogue")  # Phase 6
# agent_builder.add_edge("run_socratic_dialogue", "industry_partner_review")  # Phase 7
# agent_builder.add_edge("industry_partner_review", END)

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
    