"""
Interactive Research Workflow - Streamlit Application

This application provides a complete interactive workflow for:
1. User research input with literature search
2. Research plan generation and approval (Interrupt #1)
3. Character-based research execution with real-time progress
4. Supervisor feedback during research (Interrupt #2)
5. Results and artifact review
6. Socratic dialogue interaction

Author: Research Assistant System
"""

import streamlit as st
import asyncio
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llmkglitrev.api_endpoint.workflow_utils import (
    initialize_research_session,
    generate_research_plan,
    resume_after_plan_approval,
    execute_research_with_monitoring,
    resume_after_supervisor_feedback,
    get_available_characters,
    load_character_details,
    save_custom_character,
    modify_character_in_plan,
    load_session_artifacts,
    get_session_summary,
    export_session_to_json,
    extract_high_priority_dialogue_notes,
    validate_research_plan,
    format_elapsed_time
)
from llmkglitrev.characters import CharacterManager, ResearchCharacter
from llmkglitrev.agents.dialogue_coordinator import create_dialogue_coordinator
from llmkglitrev.utils.visualizations import (
    visualize_concept_relationships,
    visualize_reasoning_trace,
    create_research_dashboard_data,
    visualize_cross_domain_mapping,
    export_visualization
)
from llmkglitrev.ontology.concept_mapper import ConceptMapper, CrossDomainGapIdentifier
from llmkglitrev.ontology.ontology_manager import DomainOntologyManager
from langchain_core.messages import HumanMessage
import plotly.graph_objects as go
import streamlit.components.v1 as components


# ===== PAGE CONFIGURATION =====

st.set_page_config(
    page_title="Research Workflow System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .stage-indicator {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 20px;
    }
    .character-card {
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .activity-log {
        max-height: 300px;
        overflow-y: scroll;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ===== SESSION STATE INITIALIZATION =====

def init_session_state():
    """Initialize all session state variables."""
    # Workflow state
    if "workflow_stage" not in st.session_state:
        st.session_state.workflow_stage = "input"  # input, planning, research, results, dialogue

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"web-{uuid.uuid4()}"

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "start_time" not in st.session_state:
        st.session_state.start_time = datetime.now()

    # User input
    if "research_topic" not in st.session_state:
        st.session_state.research_topic = ""

    if "research_keywords" not in st.session_state:
        st.session_state.research_keywords = []

    if "retrieved_papers" not in st.session_state:
        st.session_state.retrieved_papers = []

    # Research plan
    if "proposed_research_plan" not in st.session_state:
        st.session_state.proposed_research_plan = None

    if "modified_plan" not in st.session_state:
        st.session_state.modified_plan = None

    # Research execution
    if "research_in_progress" not in st.session_state:
        st.session_state.research_in_progress = False

    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []

    if "pending_supervisor_feedback" not in st.session_state:
        st.session_state.pending_supervisor_feedback = None

    # Results
    if "conversation_artifacts" not in st.session_state:
        st.session_state.conversation_artifacts = []

    if "research_complete" not in st.session_state:
        st.session_state.research_complete = False

    # Visualization & Validation
    if "concept_graph_data" not in st.session_state:
        st.session_state.concept_graph_data = None

    if "cross_domain_mappings" not in st.session_state:
        st.session_state.cross_domain_mappings = None

    if "ontology_managers" not in st.session_state:
        st.session_state.ontology_managers = {}

    # Dialogue
    if "dialogue_started" not in st.session_state:
        st.session_state.dialogue_started = False

    if "current_dialogue_index" not in st.session_state:
        st.session_state.current_dialogue_index = 0

    if "dialogue_history" not in st.session_state:
        st.session_state.dialogue_history = []

    if "dialogue_notes" not in st.session_state:
        st.session_state.dialogue_notes = []

    # Approved plan (final version after user modifications)
    if "approved_plan" not in st.session_state:
        st.session_state.approved_plan = None

    # Settings
    if "enable_literature_search" not in st.session_state:
        st.session_state.enable_literature_search = True

    if "top_k_papers" not in st.session_state:
        st.session_state.top_k_papers = 10

    if "enable_ontology" not in st.session_state:
        st.session_state.enable_ontology = True


init_session_state()


# ===== HELPER FUNCTIONS =====

def reset_session():
    """Reset session to start fresh."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
    st.rerun()


def load_previous_session(previous_session_id: str):
    """
    Load a previous session's artifacts to enable Phases 5-7 without re-running research.

    This allows you to:
    1. Load saved artifacts from a completed research session
    2. Skip directly to Phase 5 (Gap Identification)
    3. Run Phases 6-7 (Socratic Dialogue, Industry Review) on existing research

    Args:
        previous_session_id: The session ID to load from sessions/ directory
    """
    try:
        # Load artifacts from the previous session
        artifacts = load_session_artifacts(previous_session_id)

        if not artifacts:
            st.error(f"No artifacts found for session {previous_session_id[:12]}...")
            return

        # Extract session information from artifacts
        if len(artifacts) > 0:
            # Get session_id from artifacts (this is the workflow's session_id)
            workflow_session_id = artifacts[0].get("session_id")

            # Load any saved final_proposal if it exists
            session_path = Path(f"sessions/{previous_session_id}")
            proposal_file = session_path / "final_proposal.md"

            if proposal_file.exists():
                with open(proposal_file, "r") as f:
                    final_proposal = f.read()
            else:
                # Try to extract from artifacts
                final_proposal = artifacts[0].get("research_output", "")
                if not final_proposal:
                    final_proposal = "Final proposal not found. Using first artifact's research output."

            # Update session state
            st.session_state.conversation_artifacts = artifacts
            st.session_state.final_proposal = final_proposal
            st.session_state.research_complete = True
            st.session_state.workflow_stage = "results"

            # Set default session values (keep new Streamlit session ID but reference old workflow)
            st.session_state.loaded_from_session = previous_session_id

            add_activity_log(f"Loaded session {previous_session_id[:12]}... with {len(artifacts)} artifacts", "success")
            st.success(f"✅ Loaded session {previous_session_id[:12]}... with {len(artifacts)} research artifacts!")
            st.info("You can now go to the Results tab and run Phases 5-7 (Gap Identification, Socratic Dialogue, Industry Review)")

            st.rerun()
        else:
            st.error("Artifacts are empty")

    except Exception as e:
        st.error(f"Error loading session: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def add_activity_log(message: str, level: str = "info"):
    """Add message to activity log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }.get(level, "ℹ️")

    st.session_state.activity_log.append(f"{timestamp} {emoji} {message}")


def get_stage_progress() -> tuple:
    """Get current stage and progress percentage."""
    stage_map = {
        "input": (1, 20, "User Input & Literature Search"),
        "planning": (2, 40, "Research Plan Approval"),
        "research": (3, 60, "Research Execution"),
        "results": (4, 80, "Results & Artifacts"),
        "dialogue": (5, 100, "Socratic Dialogue")
    }
    return stage_map.get(st.session_state.workflow_stage, (1, 20, "Unknown"))


# ===== STAGE INDICATOR =====

def render_stage_indicator():
    """Render workflow stage indicator."""
    stage_num, progress, stage_name = get_stage_progress()

    st.markdown(f"""
    <div class="stage-indicator">
        <strong>Stage {stage_num}/5:</strong> {stage_name}
    </div>
    """, unsafe_allow_html=True)

    st.progress(progress / 100)
    st.markdown("---")


# ===== SIDEBAR =====

def render_sidebar():
    """Render sidebar with session info and controls."""
    with st.sidebar:
        st.title("🔬 Research Workflow")
        st.markdown("---")

        # Session info
        st.subheader("📊 Session Info")
        st.caption(f"**ID:** `{st.session_state.session_id[:12]}...`")
        st.caption(f"**Thread:** `{st.session_state.thread_id[:12]}...`")

        # Show if session was loaded from a previous session
        if st.session_state.get("loaded_from_session"):
            st.caption(f"**Loaded from:** `{st.session_state.loaded_from_session[:12]}...`")
            st.caption("✓ **Ready for Phases 5-7**")

        stage_num, _, stage_name = get_stage_progress()
        st.caption(f"**Stage:** {stage_num}/5 - {stage_name}")

        elapsed = format_elapsed_time(st.session_state.start_time)
        st.caption(f"**Elapsed:** {elapsed}")

        st.markdown("---")

        # Character library
        st.subheader("👥 Character Library")

        characters = get_available_characters()
        st.caption(f"Available: {len(characters)} characters")

        if st.button("🔍 Browse Characters", width='stretch'):
            st.session_state.show_character_browser = True

        # Show character browser modal
        if st.session_state.get("show_character_browser", False):
            render_character_browser_modal(characters)

        st.markdown("---")

        # Advanced settings
        st.subheader("⚙️ Settings")

        st.session_state.enable_literature_search = st.checkbox(
            "Enable Literature Search",
            value=st.session_state.enable_literature_search,
            help="Search Neo4j database for relevant papers"
        )

        if st.session_state.enable_literature_search:
            st.session_state.top_k_papers = st.slider(
                "Papers to retrieve",
                min_value=5,
                max_value=20,
                value=st.session_state.top_k_papers
            )

        st.session_state.enable_ontology = st.checkbox(
            "Enable Ontology Validation",
            value=st.session_state.enable_ontology,
            help="Validate research terminology against domain ontology"
        )

        st.markdown("---")

        # Session management
        st.subheader("🔄 Session Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("New Session", width='stretch'):
                reset_session()

        with col2:
            if st.button("Export", width='stretch'):
                export_data = export_session_to_json(
                    st.session_state.session_id,
                    st.session_state.research_topic
                )
                st.download_button(
                    label="Download JSON",
                    data=export_data,
                    file_name=f"session_{st.session_state.session_id[:8]}.json",
                    mime="application/json",
                    width='stretch'
                )

        st.markdown("---")

        # Load Previous Session
        st.subheader("📂 Load Previous Session")
        st.caption("Load saved research to run Phases 5-7")

        # Get available sessions
        sessions_path = Path("sessions")
        if sessions_path.exists():
            available_sessions = []
            for session_dir in sorted(sessions_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if session_dir.is_dir():
                    # Check if session has artifacts
                    artifacts_path = session_dir / "conversations"
                    if artifacts_path.exists() and any(artifacts_path.glob("*.json")):
                        # Get timestamp from directory
                        timestamp = datetime.fromtimestamp(session_dir.stat().st_mtime)
                        available_sessions.append({
                            "id": session_dir.name,
                            "timestamp": timestamp,
                            "display": f"{timestamp.strftime('%m/%d %H:%M')}"
                        })

            if available_sessions:
                # Session selector
                selected_session = st.selectbox(
                    "Select session:",
                    options=[None] + available_sessions,
                    format_func=lambda x: "-- Select --" if x is None else f"{x['display']} ({x['id'][:8]}...)",
                    key="session_selector",
                    help="Choose a previous session to load its research artifacts"
                )

                if selected_session:
                    if st.button("📥 Load Session", use_container_width=True):
                        load_previous_session(selected_session["id"])
            else:
                st.caption("No saved sessions")
        else:
            st.caption("No sessions found")

        st.markdown("---")

        # Help
        st.subheader("📚 Help")
        with st.expander("How to use"):
            st.markdown("""
            **Workflow Steps:**

            1. **Input**: Enter your research question
            2. **Planning**: Review and approve proposed research plan
            3. **Research**: Monitor research execution in real-time
            4. **Results**: Review artifacts and ontology validation
            5. **Dialogue**: Engage in Socratic dialogue

            **Tips:**
            - You can modify proposed agents before approval
            - Add custom characters during planning
            - Provide feedback during research if supervisor requests it
            """)


def render_character_browser_modal(characters: List[Dict]):
    """Render character browser as expander."""
    with st.expander("📚 Available Characters", expanded=True):
        for char_info in characters:
            char = load_character_details(char_info["id"])
            if char:
                st.markdown(f"""
                **{char.name}**
                *{char.domain}* | *{char.stance}*
                {char.sub_domains[:100]}...
                """)
                st.markdown("---")

        if st.button("Close"):
            st.session_state.show_character_browser = False
            st.rerun()


# ===== STAGE 1: USER INPUT =====

def render_input_stage():
    """Render user input and literature search stage."""
    st.header("🔍 Research Topic & Literature Search")

    st.markdown("""
    Enter your research question below. The system will:
    1. Extract key research keywords
    2. Search the literature database for relevant papers
    3. Generate a specialized research plan
    """)

    st.markdown("---")

    # Research topic input
    research_topic = st.text_area(
        "Research Question",
        value=st.session_state.research_topic,
        height=150,
        placeholder="E.g., What are the latest developments in medical AI for diagnostic imaging?",
        help="Be specific about your research interest"
    )

    st.session_state.research_topic = research_topic

    # Example topics
    with st.expander("💡 Example Topics"):
        examples = [
            "Transfer learning techniques for medical image classification with limited data",
            "Explainable AI methods for clinical decision support systems",
            "Federated learning approaches for privacy-preserving healthcare analytics"
        ]
        for ex in examples:
            if st.button(ex, key=f"example_{ex[:20]}"):
                st.session_state.research_topic = ex
                st.rerun()

    st.markdown("---")

    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button(
            "🚀 Start Research Workflow",
            type="primary",
            width='stretch',
            disabled=not research_topic.strip()
        ):
            # Process user input
            with st.spinner("Processing your research topic..."):
                try:
                    result = asyncio.run(initialize_research_session(
                        research_topic=research_topic,
                        session_id=st.session_state.session_id,
                        enable_literature_search=st.session_state.enable_literature_search,
                        top_k_papers=st.session_state.top_k_papers
                    ))

                    # Store results
                    st.session_state.research_keywords = result.get("research_keywords", [])
                    st.session_state.retrieved_papers = result.get("retrieved_papers", [])
                    st.session_state.research_topic = research_topic  # Store the topic
                    st.session_state.input_processed = True  # Flag that we've processed input

                    # Add to activity log
                    add_activity_log("Research topic processed", "success")
                    add_activity_log(f"Extracted {len(st.session_state.research_keywords)} keywords", "info")
                    add_activity_log(f"Retrieved {len(st.session_state.retrieved_papers)} papers", "info")

                    st.rerun()  # Rerun to show results

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    add_activity_log(f"Error: {str(e)}", "error")

    # Show results if input has been processed
    if st.session_state.get("input_processed", False):
        st.success("✅ Research topic processed successfully!")

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("Keywords Extracted", len(st.session_state.research_keywords))

        with col_b:
            st.metric("Papers Found", len(st.session_state.retrieved_papers))

        # Display keywords
        if st.session_state.research_keywords:
            st.markdown("**Keywords:**")
            st.write(", ".join(st.session_state.research_keywords))

        # Display papers summary
        if st.session_state.retrieved_papers:
            with st.expander(f"📄 View Retrieved Papers ({len(st.session_state.retrieved_papers)})"):
                for i, paper in enumerate(st.session_state.retrieved_papers[:5], 1):
                    st.markdown(f"**{i}.** {paper.get('title', 'Unknown')}")
                    st.caption(f"Authors: {', '.join(paper.get('authors', [])[:3])}")

        # Continue button - now outside the button handler
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Continue to Research Planning ➡️", type="primary", width='stretch'):
                st.session_state.workflow_stage = "planning"
                add_activity_log("Moving to research planning", "info")
                st.rerun()


# ===== STAGE 2: RESEARCH PLAN APPROVAL =====

def render_planning_stage():
    """Render research plan approval stage (Interrupt #1)."""
    st.header("📋 Research Plan Approval")

    # Generate plan if not already done
    if st.session_state.proposed_research_plan is None:
        st.info("Generating research plan based on your topic and literature...")

        with st.spinner("🤔 AI is analyzing and proposing research plan..."):
            try:
                # Generate plan (agent will handle state internally)
                plan = asyncio.run(generate_research_plan(
                    research_topic=st.session_state.research_topic,
                    thread_id=st.session_state.thread_id
                ))

                if plan:
                    st.session_state.proposed_research_plan = plan
                    st.session_state.modified_plan = plan.copy()
                    add_activity_log("Research plan generated", "success")
                    st.rerun()
                else:
                    st.error("Failed to generate research plan")
                    add_activity_log("Plan generation failed", "error")
                    return

            except Exception as e:
                st.error(f"Error generating plan: {str(e)}")
                add_activity_log(f"Error: {str(e)}", "error")
                return

    # Display proposed plan - always use modified_plan for display and editing
    # Initialize modified_plan if not exists
    if not st.session_state.modified_plan and st.session_state.proposed_research_plan:
        st.session_state.modified_plan = st.session_state.proposed_research_plan.copy()

    plan = st.session_state.modified_plan

    st.markdown("---")

    # Research strategy (DIRECTLY EDITABLE - no toggle)
    st.subheader("🎯 Research Strategy")

    current_strategy = plan.get("research_strategy", "No strategy provided")

    # Editable text area (always shown)
    new_strategy = st.text_area(
        "Research Strategy (edit directly):",
        value=current_strategy,
        height=150,
        key="strategy_editor",
        help="Edit the research strategy directly. Changes are saved automatically when you approve the plan."
    )

    # Auto-save to modified_plan on change
    if new_strategy != current_strategy:
        st.session_state.modified_plan["research_strategy"] = new_strategy
        add_activity_log("Research strategy modified", "info")

    st.markdown("---")

    # Proposed agents - HORIZONTAL LAYOUT
    st.subheader("👥 Research Agents")

    if "proposed_agents" in plan and plan["proposed_agents"]:
        agents = plan["proposed_agents"]

        # Show agent count
        st.caption(f"**{len(agents)} agent(s)** configured")

        # Display all agents in one horizontal row
        cols = st.columns(len(agents))

        for idx, (col, agent) in enumerate(zip(cols, agents)):
            with col:
                render_agent_card_horizontal(agent, idx)
    else:
        st.warning("No agents configured. Add custom agents below.")

    st.markdown("---")

    # Add custom character
    with st.expander("➕ Add Custom Character"):
        render_custom_character_form()

    st.markdown("---")

    # Approval actions (SIMPLIFIED)
    st.subheader("✅ Plan Review")

    st.info("""
    **Review your plan:**
    - Edit the research strategy text above
    - Click "View/Edit" on any agent card to modify or remove agents
    - Add custom agents using the form below
    - When ready, click "Approve & Start Research"
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve & Start Research", type="primary", use_container_width=True):
            # Store the current modified_plan as the approved_plan
            st.session_state.approved_plan = st.session_state.modified_plan.copy()
            st.session_state.plan_approval_action = "approve"
            st.session_state.workflow_stage = "research"
            
            # Log approval with agent count for verification
            num_agents = len(st.session_state.approved_plan['proposed_agents'])
            add_activity_log(f"Plan approved with {num_agents} agents", "success")
            
            # Log agent details for debugging
            for idx, agent in enumerate(st.session_state.approved_plan['proposed_agents'], 1):
                add_activity_log(f"Agent {idx}: {agent['domain']} ({agent['recommended_character']})", "info")
            
            st.rerun()

    with col2:
        if st.button("💾 Download Plan", use_container_width=True):
            plan_json = json.dumps(st.session_state.modified_plan, indent=2)
            st.download_button(
                "⬇️ Save as JSON",
                data=plan_json,
                file_name=f"research_plan_{st.session_state.session_id[:8]}.json",
                mime="application/json",
                use_container_width=True
            )


def render_agent_card_horizontal(agent: Dict, index: int):
    """Render agent card in horizontal layout with modal editing (Fix 2)."""
    domain = agent.get('domain', 'Unknown Domain')
    character = agent.get('recommended_character', 'Not specified')
    stance = agent.get('stance', 'neutral')
    assigned_topic = agent.get('assigned_topic', '')
    seed_papers = agent.get('seed_papers', [])
    seed_papers_count = len(seed_papers)

    # Card styling
    st.markdown(f"""
    <div style="
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        height: 200px;
        background-color: #f9f9f9;
        margin-bottom: 10px;
    ">
        <h3 style="color: #1f77b4; margin-bottom: 10px;">🎭</h3>
        <h4 style="margin: 5px 0;">{domain}</h4>
        <p style="color: #666; font-size: 12px; margin: 5px 0;">
            {character}<br/>
            <strong>Stance:</strong> {stance}
        </p>
        {f'<p style="font-size: 11px; color: #888;">📄 {seed_papers_count} seed papers</p>' if seed_papers_count > 0 else ''}
    </div>
    """, unsafe_allow_html=True)

    # Click to edit (opens modal)
    if st.button("View/Edit", key=f"view_{index}", use_container_width=True):
        render_agent_modal(agent, index)


@st.dialog("Agent Configuration")
def render_agent_modal(agent: Dict, index: int):
    """Render agent details and editing in modal dialog (Fix 2)."""
    st.subheader(f"🎭 {agent.get('domain', 'Unknown Domain')}")

    # Display current configuration
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Character:** {agent.get('recommended_character', 'Not specified')}")
        st.write(f"**Stance:** {agent.get('stance', 'neutral')}")

        # assigned_topic = agent.get('assigned_topic', '')
        # if assigned_topic:
        #     st.write(f"**Assigned Topic:** {assigned_topic}")

    with col2:
        search_scope = agent.get('search_scope', [])
        if search_scope:
            st.write("**Search Scope:**")
            for scope in search_scope[:5]:
                st.write(f"  • {scope}")

        seed_papers = agent.get('seed_papers', [])
        if seed_papers:
            st.write(f"**Seed Papers:** {len(seed_papers)}")

    st.write("**Rationale:**")
    st.info(agent.get('rationale', 'No rationale provided'))

    st.markdown("---")

    # Edit form
    with st.expander("✏️ Edit Configuration", expanded=False):
        render_agent_editor(agent, index)

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Remove Agent", use_container_width=True, key=f"remove_modal_{index}"):
            # Ensure modified_plan exists and is a copy
            if not st.session_state.modified_plan:
                st.session_state.modified_plan = st.session_state.proposed_research_plan.copy()

            # Remove the agent
            st.session_state.modified_plan["proposed_agents"].pop(index)

            # Log the change
            add_activity_log(f"Removed agent at position {index+1}", "warning")

            # Force close modal and rerun
            st.rerun()

    with col2:
        if st.button("✖️ Close", use_container_width=True, key=f"close_modal_{index}"):
            st.rerun()


def render_agent_card(agent: Dict, index: int):
    """Render individual agent configuration card (DEPRECATED - use render_agent_card_horizontal)."""
    with st.container():
        st.markdown(f"""
        <div class="character-card">
            <strong>{agent.get('domain', 'Unknown Domain')}</strong>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write(f"**Character:** {agent.get('recommended_character', 'Not specified')}")
            st.write(f"**Stance:** {agent.get('stance', 'neutral')}")
            st.write(f"**Rationale:** {agent.get('rationale', 'No rationale provided')}")

            search_scope = agent.get('search_scope', [])
            if search_scope:
                st.write(f"**Search Scope:** {', '.join(search_scope)}")

        with col2:
            if st.button("✏️ Edit", key=f"edit_{index}"):
                st.session_state[f"editing_agent_{index}"] = True

            if st.button("🗑️ Remove", key=f"remove_{index}"):
                plan = st.session_state.modified_plan
                plan["proposed_agents"].pop(index)
                st.rerun()

        # Edit mode
        if st.session_state.get(f"editing_agent_{index}", False):
            with st.expander("Edit Agent", expanded=True):
                render_agent_editor(agent, index)


def render_agent_editor(agent: Dict, index: int):
    """Render agent editor form."""
    # Load available characters
    characters = get_available_characters()
    char_options = {c["id"]: c["name"] for c in characters}

    # Character selection
    current_char = agent.get("recommended_character", "")
    selected_char = st.selectbox(
        "Character",
        options=list(char_options.keys()),
        format_func=lambda x: char_options[x],
        index=list(char_options.keys()).index(current_char) if current_char in char_options else 0,
        key=f"char_select_{index}"
    )

    # Stance selection
    stance = st.selectbox(
        "Stance",
        options=["critical", "constructive", "neutral"],
        index=["critical", "constructive", "neutral"].index(agent.get("stance", "neutral")),
        key=f"stance_select_{index}"
    )

    # Search scope
    search_scope = st.text_input(
        "Search Scope (comma-separated)",
        value=", ".join(agent.get("search_scope", [])),
        key=f"scope_{index}"
    )

    # Ontology selection
    ontology = st.selectbox(
        "Ontology",
        options=["edam", "ml-schema", "auto"],
        index=0,
        key=f"ontology_{index}",
        help="Domain ontology for validation"
    )

    # Save button
    if st.button("💾 Save Changes", key=f"save_modal_{index}"):
        # Ensure modified_plan exists and is a copy
        if not st.session_state.modified_plan:
            st.session_state.modified_plan = st.session_state.proposed_research_plan.copy()

        modifications = {
            "recommended_character": selected_char,
            "stance": stance,
            "search_scope": [s.strip() for s in search_scope.split(",")],
            "ontology": ontology
        }

        plan = modify_character_in_plan(
            st.session_state.modified_plan,
            index,
            modifications
        )
        st.session_state.modified_plan = plan

        # Log the change
        add_activity_log(f"Modified agent at position {index+1}", "success")

        st.success("✅ Changes saved!")
        st.rerun()


def render_custom_character_form():
    """Render form to create custom character."""
    st.markdown("**Create a new research character**")

    name = st.text_input("Character Name", placeholder="e.g., Ethics Expert")
    domain = st.text_input("Domain", placeholder="e.g., Research Ethics")
    stance = st.selectbox("Stance", ["critical", "constructive", "neutral"])

    expertise = st.text_area(
        "Expertise Areas (one per line)",
        placeholder="e.g.,\nResearch ethics\nClinical trials\nPatient privacy",
        height=100
    )

    communication_style = st.text_input(
        "Communication Style",
        placeholder="e.g., Thoughtful and questioning"
    )

    background = st.text_area(
        "Background (optional)",
        placeholder="Brief background about this character's perspective",
        height=80
    )

    if st.button("➕ Add to Plan", type="primary"):
        if name and domain:
            # Create character data
            char_data = {
                "character_id": f"custom_{uuid.uuid4().hex[:8]}",
                "name": name,
                "domain": domain,
                "stance": stance,
                "expertise_areas": [e.strip() for e in expertise.split("\n") if e.strip()],
                "communication_style": communication_style or "Professional",
                "background": background or f"Expert in {domain}",
                "sub_domains": []
            }

            # Save character permanently
            char_id = save_custom_character(char_data)

            # Add to plan
            new_agent = {
                "domain": domain,
                "recommended_character": char_id,
                "stance": stance,
                "search_scope": char_data["expertise_areas"],
                "rationale": f"Custom character for {domain}",
                "custom_config": char_data
            }

            plan = st.session_state.modified_plan
            plan["proposed_agents"].append(new_agent)

            st.success(f"✅ Added {name} to plan and saved permanently!")
            add_activity_log(f"Custom character '{name}' created", "success")
            st.rerun()
        else:
            st.warning("Please provide at least name and domain")


# ===== STAGE 3: RESEARCH EXECUTION =====

def render_research_stage():
    """Render research execution with real-time progress."""
    st.header("🔬 Research Execution")

    # Check for supervisor feedback interrupt
    if st.session_state.pending_supervisor_feedback:
        render_supervisor_feedback_ui()
        return

    # Start research if not already running
    if not st.session_state.research_in_progress and not st.session_state.research_complete:
        st.info("Starting research execution...")

        # Check if we need to resume from plan approval
        resume_with_approval = st.session_state.get("plan_approval_action") == "approve"

        if resume_with_approval:
            st.info("Resuming workflow with approved plan...")
            add_activity_log("Starting research with approved plan", "success")
            st.session_state.plan_approval_action = None  # Clear flag

        # Create placeholders for real-time updates
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        log_placeholder = st.empty()

        try:
            # Execute research with monitoring
            async def run_research():
                # Use approved_plan (which contains all user modifications)
                plan_to_use = st.session_state.approved_plan if resume_with_approval else None
                
                # Debug log to verify plan
                if plan_to_use:
                    num_agents = len(plan_to_use.get('proposed_agents', []))
                    add_activity_log(f"Executing research with {num_agents} agents from approved plan", "info")
                
                async for event in execute_research_with_monitoring(
                    st.session_state.thread_id,
                    resume_with_approval=resume_with_approval,
                    modified_plan=plan_to_use
                ):
                    event_type = event.get("type")

                    if event_type == "supervisor_interrupt":
                        # Store pending proposals
                        st.session_state.pending_supervisor_feedback = event.get("pending_proposals", [])
                        st.session_state.research_in_progress = False
                        return

                    elif event_type == "progress":
                        # Update UI with progress
                        active_chars = event.get("active_characters", [])
                        messages = event.get("messages", [])

                        # Update status
                        status_placeholder.success(f"✅ Active Characters: {len(active_chars)}")

                        # Update progress
                        with progress_placeholder.container():
                            for char in active_chars:
                                st.write(f"**{char['name']}** ({char['domain']}) - Conducting research...")

                        # Update log
                        for msg in messages:
                            add_activity_log(str(msg), "info")

                        with log_placeholder.container():
                            render_activity_log()

                    elif event_type == "complete":
                        # Research complete
                        artifacts = event.get("artifacts", [])
                        st.session_state.conversation_artifacts = artifacts

                        # Capture final proposal
                        final_proposal = event.get("final_proposal", "")
                        if final_proposal:
                            st.session_state.final_proposal = final_proposal

                        st.session_state.research_complete = True
                        st.session_state.research_in_progress = False
                        add_activity_log("Research completed!", "success")
                        return

                    elif event_type == "error":
                        error = event.get("error", "Unknown error")
                        add_activity_log(f"Error: {error}", "error")
                        st.session_state.research_in_progress = False
                        return

            st.session_state.research_in_progress = True
            asyncio.run(run_research())
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error during research: {str(e)}")
            add_activity_log(f"Error: {str(e)}", "error")
            st.session_state.research_in_progress = False

    # Display activity log - COLLAPSIBLE (Fix 3)
    st.markdown("---")
    with st.expander("📋 Activity Log & Tool Calls", expanded=False):
        st.caption("Expand to see detailed research activity, queries, and tool calls")
        render_activity_log()

    # If complete, show continue button
    if st.session_state.research_complete:
        st.success("✅ Research completed successfully!")

        if st.button("View Results ➡️", type="primary"):
            st.session_state.workflow_stage = "results"
            st.rerun()


def render_supervisor_feedback_ui():
    """Render UI for supervisor feedback (Interrupt #2)."""
    st.warning("🤔 The supervisor is requesting your feedback on research directions")

    st.markdown("---")

    st.subheader("📋 Proposed Research Directions")

    proposals = st.session_state.pending_supervisor_feedback

    if proposals:
        for i, proposal in enumerate(proposals, 1):
            with st.expander(f"Direction {i}", expanded=True):
                st.write(proposal)

    st.markdown("---")

    st.subheader("💬 Your Feedback")

    feedback = st.text_area(
        "Provide feedback or guidance:",
        placeholder="E.g., 'Approved, please proceed' or 'Focus more on clinical validation studies'",
        height=150
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Submit Feedback", type="primary", width='stretch'):
            if feedback.strip():
                with st.spinner("Resuming research with your feedback..."):
                    try:
                        asyncio.run(resume_after_supervisor_feedback(
                            thread_id=st.session_state.thread_id,
                            feedback=feedback
                        ))

                        st.session_state.pending_supervisor_feedback = None
                        add_activity_log("Feedback provided, research continuing", "success")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please provide feedback before submitting")

    with col2:
        if st.button("⏭️ Approve As-Is", width='stretch'):
            with st.spinner("Continuing research..."):
                try:
                    asyncio.run(resume_after_supervisor_feedback(
                        thread_id=st.session_state.thread_id,
                        feedback="Approved, please proceed with proposed directions"
                    ))

                    st.session_state.pending_supervisor_feedback = None
                    add_activity_log("Directions approved", "success")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")


def render_activity_log():
    """Render activity log with scroll."""
    if st.session_state.activity_log:
        log_html = "<div class='activity-log'>"
        for entry in st.session_state.activity_log[-20:]:  # Last 20 entries
            log_html += f"{entry}<br>"
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.info("No activity yet")


# ===== STAGE 4: RESULTS =====

def render_results_stage():
    """Render results and artifacts with visualization and validation."""
    st.header("📊 Research Results")

    # ===== FIX 1: Display Final Proposal Prominently =====
    st.subheader("📝 Unified Research Proposal")

    # Get final proposal from session state
    final_proposal = st.session_state.get("final_proposal", None)

    if final_proposal:
        file_path = Path(f"sessions/{st.session_state.session_id}/final_proposal.md")
        if not file_path.exists():
            # Create the parent directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # Save final proposal to session directory
            with open(file_path, "w") as f:
                f.write(final_proposal)
            # Display in nice card
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px;
                    border-left: 5px solid #1f77b4; margin-bottom: 30px;">
        """, unsafe_allow_html=True)

        st.markdown(final_proposal)

        st.markdown("</div>", unsafe_allow_html=True)

        # Download button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                "📥 Download Proposal",
                data=final_proposal,
                file_name=f"research_proposal_{st.session_state.session_id[:8]}.md",
                mime="text/markdown"
            )
    else:
        st.warning("Final proposal not available yet")

    st.markdown("---")

    # Load artifacts if not already loaded
    if not st.session_state.conversation_artifacts:
        st.session_state.conversation_artifacts = load_session_artifacts(st.session_state.session_id)

    artifacts = st.session_state.conversation_artifacts

    if not artifacts:
        st.warning("No artifacts found for this session")
        return

    # Session summary at top
    summary = get_session_summary(st.session_state.session_id)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Artifacts", summary["artifact_count"])

    with col2:
        st.metric("Characters", len(summary["characters_used"]))

    with col3:
        st.metric("Dialogue Notes", summary["total_dialogue_notes"])

    with col4:
        st.metric("Papers", len(summary["papers_consulted"]))

    st.markdown("---")

    # Tabbed interface for different views (Fix 3: Activity log moved to tab)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📄 Research Artifacts",
        "🔍 Concept Visualization",
        "🔗 Cross-Domain Validation",
        # "📊 Dashboard",
        "🔧 Activity Log",
        "💡 Gap Identification",
        "🗣️ Socratic Dialogue",
        "🏢 Industry Review"
    ])

    with tab1:
        render_artifacts_tab(artifacts)

    with tab2:
        render_visualization_tab(artifacts)

    with tab3:
        render_cross_domain_validation_tab(artifacts)

    # with tab4:
    #     render_dashboard_tab(artifacts)

    with tab4:
        st.caption("Detailed research activity, queries, and tool calls")
        render_activity_log()

    with tab5:
        render_gap_identification_tab()

    with tab6:
        render_socratic_dialogue_tab()

    with tab7:
        render_industry_review_tab()

    st.markdown("---")

    # Next steps (always visible at bottom)
    st.subheader("🎯 Next Steps")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💬 Start Dialogue", type="primary", width='stretch'):
            # Extract dialogue notes
            st.session_state.dialogue_notes = extract_high_priority_dialogue_notes(artifacts)
            st.session_state.workflow_stage = "dialogue"
            st.rerun()

    with col2:
        if st.button("📥 Export Session", width='stretch'):
            export_data = export_session_to_json(
                st.session_state.session_id,
                st.session_state.research_topic
            )
            st.download_button(
                "Download JSON",
                data=export_data,
                file_name=f"session_{st.session_state.session_id[:8]}.json",
                mime="application/json"
            )

    with col3:
        if st.button("🔄 New Session", width='stretch'):
            reset_session()


def render_artifacts_tab(artifacts: List[Dict]):
    """Render traditional artifacts view."""
    st.subheader("📄 Research Artifacts")

    artifact_tabs = st.tabs([f"{art.get('domain', 'Unknown')}" for art in artifacts])

    for tab, artifact in zip(artifact_tabs, artifacts):
        with tab:
            render_artifact_details(artifact)


def render_visualization_tab(artifacts: List[Dict]):
    """Render concept relationship visualizations."""
    st.subheader("🔍 Concept Relationship Visualization")

    st.markdown("""
    This visualization shows the ontology concepts used during research and their relationships.
    """)

    # Extract concepts and relationships from artifacts
    all_concepts = []
    all_relationships = []
    reasoning_steps = []

    for artifact in artifacts:
        # Get grounded concepts if available
        grounded_concepts = artifact.get("grounded_concepts", [])
        all_concepts.extend(grounded_concepts)

        # Get concept relationships
        relationships = artifact.get("concept_relationships", [])
        all_relationships.extend(relationships)

        # Get reasoning trace
        trace = artifact.get("reasoning_trace", [])
        if trace:
            reasoning_steps.extend([
                f"[{artifact.get('domain', 'Unknown')}] {step}"
                for step in trace
            ])

    # Remove duplicates
    all_concepts = list(set(all_concepts))

    if not all_concepts:
        st.info("No ontology-grounded concepts found in artifacts. Enable ontology validation to see concept relationships.")
        return

    st.success(f"Found {len(all_concepts)} unique concepts across {len(artifacts)} research artifacts")

    # Visualization options
    col1, col2 = st.columns(2)

    with col1:
        viz_type = st.selectbox(
            "Visualization Type",
            ["Concept Graph", "Reasoning Timeline"],
            help="Choose how to visualize the research concepts"
        )

    with col2:
        if st.button("🔄 Regenerate Visualization"):
            st.session_state.concept_graph_data = None
            st.rerun()

    st.markdown("---")

    # Generate and display visualization
    if viz_type == "Concept Graph":
        try:
            fig = visualize_concept_relationships(
                concepts=all_concepts,
                relationships=all_relationships,
                title=f"Research Concept Network - {st.session_state.research_topic[:50]}..."
            )

            st.plotly_chart(fig, width='stretch')

            # Export option
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("💾 Export HTML"):
                    output_path = f"visualizations/session_{st.session_state.session_id[:8]}_concepts.html"
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    export_visualization(fig, output_path, format="html")
                    st.success(f"Saved to {output_path}")

            with col2:
                if st.button("💾 Export PNG"):
                    output_path = f"visualizations/session_{st.session_state.session_id[:8]}_concepts.png"
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        export_visualization(fig, output_path, format="png")
                        st.success(f"Saved to {output_path}")
                    except Exception as e:
                        st.warning(f"PNG export requires kaleido package. Saved as HTML instead.")

        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")

    elif viz_type == "Reasoning Timeline":
        if reasoning_steps:
            try:
                # Create concepts per step data
                concepts_per_step = []
                accumulated_concepts = set()

                for step in reasoning_steps:
                    # Extract concepts mentioned in this step
                    step_concepts = [c for c in all_concepts if c.lower() in step.lower()]
                    accumulated_concepts.update(step_concepts)
                    concepts_per_step.append(list(accumulated_concepts))

                fig = visualize_reasoning_trace(
                    reasoning_steps=reasoning_steps,
                    concepts_per_step=concepts_per_step
                )

                st.plotly_chart(fig, width='stretch')

            except Exception as e:
                st.error(f"Error generating timeline: {str(e)}")
        else:
            st.info("No reasoning traces found. Enable ontology grounding to track reasoning steps.")


def render_cross_domain_validation_tab(artifacts: List[Dict]):
    """Render cross-domain concept mapping and validation."""
    st.subheader("🔗 Cross-Domain Validation")

    st.markdown("""
    Map concepts between different domain ontologies to validate multidisciplinary research
    or check alignment with industry/stakeholder perspectives.
    """)

    # Domain selection
    st.markdown("### Select Domains for Validation")

    col1, col2 = st.columns(2)

    with col1:
        source_domain = st.selectbox(
            "Source Domain (Academic Research)",
            ["machine_learning", "bioinformatics", "medical_imaging"],
            help="The domain of your research"
        )

    with col2:
        target_domain = st.selectbox(
            "Target Domain (Stakeholder Perspective)",
            ["information_systems", "healthcare_systems", "industry_applications"],
            help="The domain you want to validate against"
        )

    # Load ontologies button
    if st.button("🔄 Load Ontologies & Perform Validation"):
        with st.spinner(f"Loading ontologies and mapping concepts..."):
            try:
                # Load ontology managers
                from pathlib import Path
                ontology_path = Path(__file__).parent / "ontologies"

                # Initialize ontology managers (simplified - in production would load from config)
                st.session_state.ontology_managers = {}

                # For demonstration, use EDAM for bioinformatics/ML
                if source_domain in ["machine_learning", "bioinformatics"]:
                    edam_path = ontology_path / "edam.owl"
                    if edam_path.exists():
                        st.session_state.ontology_managers[source_domain] = DomainOntologyManager(
                            str(edam_path),
                            ontology_name=source_domain.replace("_", " ").title()
                        )

                # Note: In production, you'd load appropriate ontologies for each domain
                st.success("✅ Ontologies loaded")

            except Exception as e:
                st.error(f"Error loading ontologies: {str(e)}")
                st.info("Make sure ontology files are available in the ontologies/ directory")
                return

    # Perform mapping if ontologies are loaded
    if st.session_state.ontology_managers:
        st.markdown("---")
        st.markdown("### Concept Mapping Results")

        # Extract concepts from artifacts
        research_concepts = []
        for artifact in artifacts:
            grounded_concepts = artifact.get("grounded_concepts", [])
            research_concepts.extend(grounded_concepts)

        research_concepts = list(set(research_concepts))[:10]  # Limit to 10 for demo

        if not research_concepts:
            st.warning("No grounded concepts found. Enable ontology validation during research.")
            return

        st.info(f"Mapping {len(research_concepts)} concepts from research...")

        try:
            # Create concept mapper
            mapper = ConceptMapper(st.session_state.ontology_managers)

            # Map concepts
            mappings = {}
            for concept in research_concepts:
                try:
                    mapping = mapper.map_concept(
                        concept=concept,
                        source_domain=source_domain,
                        target_domain=target_domain
                    )
                    mappings[concept] = mapping
                except Exception as e:
                    st.warning(f"Could not map {concept}: {str(e)}")

            # Display mapping statistics
            stats = mapper.get_mapping_statistics(mappings)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Concepts", stats["total_concepts"])

            with col2:
                st.metric("Mapped", stats["mapped"], f"{stats['mapping_rate']*100:.0f}%")

            with col3:
                st.metric("Exact Matches", stats["exact_matches"])

            with col4:
                st.metric("Avg Confidence", f"{stats['average_confidence']:.0%}")

            st.markdown("---")

            # Display individual mappings
            st.markdown("### Mapping Details")

            for concept, mapping in mappings.items():
                confidence_color = (
                    "🟢" if mapping.mapping_confidence > 0.8
                    else "🟡" if mapping.mapping_confidence > 0.6
                    else "🔴"
                )

                with st.expander(f"{confidence_color} {concept} → {', '.join(mapping.target_concepts) if mapping.target_concepts else 'No mapping'}"):
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.write(f"**Relationship:** {mapping.relationship_type}")
                        st.write(f"**Confidence:** {mapping.mapping_confidence:.0%}")

                    with col_b:
                        st.write(f"**Source:** {mapping.source_domain}")
                        st.write(f"**Target:** {mapping.target_domain}")

                    st.write(f"**Explanation:** {mapping.explanation}")

                    if mapping.suggested_alternatives:
                        st.write(f"**Alternatives:** {', '.join(mapping.suggested_alternatives)}")

            # Identify gaps
            st.markdown("---")
            st.markdown("### Cross-Domain Gaps & Validation Questions")

            gap_identifier = CrossDomainGapIdentifier(mapper)
            gaps = gap_identifier.identify_gaps(
                source_domain=source_domain,
                target_domain=target_domain,
                proposal_concepts=research_concepts,
                mapped_concepts=mappings
            )

            if gaps:
                questions = gap_identifier.generate_cross_domain_questions(gaps, max_questions=5)

                st.warning(f"⚠️ Identified {len(gaps)} potential gaps")

                for i, q in enumerate(questions, 1):
                    priority_stars = "⭐" * q["priority"]
                    st.markdown(f"""
                    **{i}. [{q['gap_type']}]** {priority_stars}

                    💡 {q['question']}

                    *{q['explanation']}*
                    """)
                    st.markdown("---")

                # Visualize mapping
                if st.button("📊 Visualize Cross-Domain Mapping"):
                    source_concepts = list(mappings.keys())
                    target_concepts = list(set([
                        tc for m in mappings.values()
                        for tc in m.target_concepts
                    ]))

                    mapping_data = [
                        {
                            "source": m.source_concept,
                            "target": m.target_concepts[0] if m.target_concepts else None,
                            "confidence": m.mapping_confidence
                        }
                        for m in mappings.values()
                        if m.target_concepts
                    ]

                    fig = visualize_cross_domain_mapping(
                        source_concepts=source_concepts,
                        target_concepts=target_concepts,
                        mappings=mapping_data,
                        source_domain=source_domain,
                        target_domain=target_domain
                    )

                    st.plotly_chart(fig, width='stretch')

            else:
                st.success("✅ No significant gaps identified - concepts map well between domains!")

        except Exception as e:
            st.error(f"Error during concept mapping: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def render_dashboard_tab(artifacts: List[Dict]):
    """Render research dashboard with multiple coordinated views."""
    st.subheader("📊 Research Dashboard")

    st.markdown("""
    Multi-panel view of research activities, concept usage, and collaboration patterns.
    """)

    try:
        # Generate dashboard data
        dashboard_figs = create_research_dashboard_data(artifacts)

        # Display each figure
        if "concepts_by_character" in dashboard_figs:
            st.markdown("### Concepts Used by Each Character")
            st.plotly_chart(dashboard_figs["concepts_by_character"], width='stretch')

        if "note_priorities" in dashboard_figs:
            st.markdown("### Dialogue Note Priorities")
            st.plotly_chart(dashboard_figs["note_priorities"], width='stretch')

        if "concept_overlap" in dashboard_figs:
            st.markdown("### Concept Overlap Between Characters")
            st.plotly_chart(dashboard_figs["concept_overlap"], width='stretch')

    except Exception as e:
        st.error(f"Error generating dashboard: {str(e)}")
        st.info("Dashboard requires artifacts with grounded concepts and dialogue notes.")


def render_artifact_details(artifact: Dict):
    """Render detailed artifact information."""
    st.markdown(f"**Character:** {artifact.get('character_id', 'Unknown')}")
    st.markdown(f"**Domain:** {artifact.get('domain', 'Unknown')}")

    st.markdown("---")

    # Research output
    st.subheader("📝 Research Output")
    research_output = artifact.get("research_output", "No research output")
    st.markdown(research_output)

    st.markdown("---")

    # Dialogue notes
    st.subheader("💬 Dialogue Notes")
    notes = artifact.get("dialogue_notes", [])

    if notes:
        # Sort by priority
        notes_sorted = sorted(notes, key=lambda x: x.get("priority", 0), reverse=True)

        for note in notes_sorted[:5]:  # Show top 5
            priority = note.get("priority", 0)
            note_type = note.get("type", "unknown")
            question = note.get("suggested_question", "No question")

            st.markdown(f"""
            **Priority {priority}** | *{note_type}*
            {question}
            """)
            st.markdown("---")

        if len(notes) > 5:
            with st.expander(f"View all {len(notes)} notes"):
                for note in notes_sorted:
                    st.write(f"**P{note.get('priority')}:** {note.get('suggested_question')}")
    else:
        st.info("No dialogue notes generated")

    # Papers consulted
    papers = artifact.get("papers_consulted", [])
    if papers:
        with st.expander(f"📄 Papers Consulted ({len(papers)})"):
            for paper in papers:
                st.write(f"- {paper}")


# ===== PHASE 5-7 TAB RENDERERS =====

def render_gap_identification_tab():
    """Phase 5: Display identified research gaps."""
    st.subheader("🔍 Research Gap Identification")
    st.caption("Characters explore their own ontologies to identify gaps and suggest new research ideas")

    identified_gaps = st.session_state.get("identified_gaps", [])

    if not identified_gaps:
        st.info("Phase 5 not yet started. Research gaps will be identified by characters using their own ontologies.")

        # Check if prerequisites exist
        final_proposal = st.session_state.get("final_proposal", "")
        artifacts = st.session_state.get("conversation_artifacts", [])

        # Validate prerequisites
        if not final_proposal:
            st.warning("⚠️ No final proposal found. Please complete Phase 4 (research execution) first.")
            return

        if not artifacts:
            st.warning("⚠️ No research artifacts found. Please complete Phase 4 (research execution) first.")
            return

        # Extract session_id from artifacts (workflow's session_id, not Streamlit's)
        session_id = artifacts[0].get("session_id") if artifacts else None

        if not session_id:
            st.warning("⚠️ No session ID found in artifacts. Please complete Phase 4 (research execution) first.")
            return

        # Button to trigger Phase 5
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Phase 5: Gap Identification", type="primary", use_container_width=True):
                # Run Phase 5 as standalone conversation
                import asyncio
                from llmkglitrev.agents.post_research_phases import identify_research_gaps_standalone

                with st.spinner("Characters exploring their ontologies for research gaps..."):
                    try:
                        # Run Phase 5 using workflow's session_id from artifacts
                        gaps = asyncio.run(identify_research_gaps_standalone(
                            session_id=session_id,
                            final_proposal=final_proposal
                        ))

                        st.session_state.identified_gaps = gaps
                        st.success(f"✓ Phase 5 complete! {len(gaps)} characters identified gaps.")
                        add_activity_log(f"Phase 5 complete: {len(gaps)} gap analyses", "success")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error in Phase 5: {str(e)}")
                        add_activity_log(f"Phase 5 error: {str(e)}", "error")
        return

    # Display gaps from each character
    for gap_item in identified_gaps:
        character_name = gap_item.get('character_name', 'Unknown')
        domain = gap_item.get('domain', 'Unknown')
        gaps_text = gap_item.get('gaps_identified', '')
        ontology_used = gap_item.get('ontology_used', 'own')

        with st.expander(f"🎭 {character_name} ({domain})", expanded=True):
            st.markdown(gaps_text)
            st.caption(f"✓ Using: {character_name}'s {ontology_used} ontology")

    # User validation section
    st.markdown("---")
    st.subheader("💭 Your Validation & Feedback")

    st.info("Review the identified gaps and provide your thoughts:")

    user_feedback = st.text_area(
        "Do these gaps make sense? Any new ideas sparked?",
        height=150,
        placeholder="Share your thoughts on the identified gaps, new perspectives, or research ideas...",
        key="gap_feedback_input"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Validated - Proceed to Phase 6", type="primary", use_container_width=True):
            st.session_state.gap_feedback = user_feedback
            st.session_state.gap_identification_validated = True
            st.success("Feedback recorded! You can now go to the Socratic Dialogue tab.")

    with col2:
        if st.button("🔄 Re-run Gap Analysis", use_container_width=True):
            # Clear current gaps to re-run
            st.session_state.identified_gaps = []
            st.rerun()


def render_socratic_dialogue_tab():
    """Phase 6: Display Socratic dialogue Q&A."""
    st.subheader("🗣️ Socratic Dialogue")
    st.caption("Characters ask critical questions based on ontology gaps from Phase 5")

    dialogue_questions = st.session_state.get("dialogue_questions", [])

    if not dialogue_questions:
        st.info("Phase 6 not yet started. Characters will generate critical questions based on their ontology gap analysis.")

        # Check prerequisites
        final_proposal = st.session_state.get("final_proposal", "")
        identified_gaps = st.session_state.get("identified_gaps", [])

        # Validate prerequisites
        if not final_proposal:
            st.warning("⚠️ No final proposal found. Please complete Phase 4 (research execution) first.")
            return

        if not identified_gaps:
            st.warning("⚠️ No gap identification found. Please complete Phase 5 (Gap Identification) first.")
            return

        # Extract session_id from artifacts (workflow's session_id, not Streamlit's)
        artifacts = st.session_state.get("conversation_artifacts", [])
        session_id = artifacts[0].get("session_id") if artifacts else None

        if not session_id:
            st.warning("⚠️ No session ID found in artifacts. Please complete Phase 4 (research execution) first.")
            return

        # Show start button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Phase 6: Generate Questions", type="primary", use_container_width=True):
                with st.spinner("Characters generating questions based on ontology gaps..."):
                    import asyncio
                    from llmkglitrev.agents.post_research_phases import run_socratic_dialogue_standalone

                    # Run Phase 6
                    dialogue_questions = asyncio.run(run_socratic_dialogue_standalone(
                        session_id=session_id,
                        final_proposal=final_proposal,
                        identified_gaps=identified_gaps,
                        num_questions_per_character=2
                    ))

                    st.session_state.dialogue_questions = dialogue_questions
                    st.success(f"✓ Phase 6 complete! {len(dialogue_questions)} characters generated questions.")
                    st.rerun()
        return

    # Display generated questions with hidden reasoning
    st.markdown("### 💬 Questions from Research Characters")
    st.caption("Each character has generated critical questions based on their ontology gap analysis")

    for char_idx, char_questions in enumerate(dialogue_questions):
        character_name = char_questions.get('character_name', 'Character')
        domain = char_questions.get('domain', '')
        questions_text = char_questions.get('questions_text', '')

        with st.expander(f"🎭 {character_name} ({domain})", expanded=True):
            st.markdown(f"**Ontology Used:** {character_name}'s own ontology")
            st.markdown("---")

            # Display the questions with formatting
            st.markdown(questions_text)

            # User response area
            st.markdown("---")
            st.markdown("**💭 Your Response to These Questions:**")

            user_response = st.text_area(
                "Reflect on these questions and provide your thoughts:",
                height=150,
                key=f"response_{char_idx}",
                placeholder="Consider the hidden reasoning and expected insights when formulating your response..."
            )

            if user_response:
                # Save response
                if f"responses_{char_idx}" not in st.session_state:
                    st.session_state[f"responses_{char_idx}"] = user_response

            st.markdown("---")

    # Action buttons
    st.markdown("### 📝 Summary")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Re-generate Questions", use_container_width=True):
            st.session_state.dialogue_questions = []
            st.rerun()

    with col2:
        if st.button("➡️ Proceed to Phase 7: Industry Review", type="primary", use_container_width=True):
            # Save all responses
            all_responses = []
            for idx in range(len(dialogue_questions)):
                response_key = f"responses_{idx}"
                if response_key in st.session_state:
                    all_responses.append({
                        "character_name": dialogue_questions[idx].get("character_name"),
                        "response": st.session_state[response_key]
                    })

            st.session_state.dialogue_responses = all_responses
            st.success("✓ Responses saved! You can now proceed to Phase 7.")
            # User can manually go to Phase 7 tab


def render_industry_review_tab():
    """Phase 7: Display industry partner feedback."""
    st.subheader("🏢 Industry Partner Review")
    st.caption("Industrial partners review the proposal using OTHER researchers' ontologies")

    industry_feedback = st.session_state.get("industry_feedback", [])

    if not industry_feedback:
        st.info("Phase 7 not yet started. Industry partners will review using other researchers' ontologies.")

        # Check if prerequisites exist
        final_proposal = st.session_state.get("final_proposal", "")
        artifacts = st.session_state.get("conversation_artifacts", [])

        # Validate prerequisites
        if not final_proposal:
            st.warning("⚠️ No final proposal found. Please complete Phase 4 (research execution) first.")
            return

        if not artifacts:
            st.warning("⚠️ No research artifacts found. Please complete Phase 4 (research execution) first.")
            return

        # Extract session_id from artifacts (workflow's session_id, not Streamlit's)
        session_id = artifacts[0].get("session_id") if artifacts else None

        if not session_id:
            st.warning("⚠️ No session ID found in artifacts. Please complete Phase 4 (research execution) first.")
            return

        # Phase 7 can run independently - doesn't require Phase 6
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Phase 7: Industry Review", type="primary", use_container_width=True):
                # Run Phase 7 as standalone conversation
                import asyncio
                from llmkglitrev.agents.post_research_phases import industry_partner_review_standalone

                with st.spinner("Industry partners reviewing proposal..."):
                    try:
                        # Run Phase 7 using workflow's session_id from artifacts
                        feedback = asyncio.run(industry_partner_review_standalone(
                            session_id=session_id,
                            final_proposal=final_proposal
                        ))

                        st.session_state.industry_feedback = feedback
                        st.success(f"✓ Phase 7 complete! {len(feedback)} partners provided feedback.")
                        add_activity_log(f"Phase 7 complete: {len(feedback)} partner reviews", "success")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error in Phase 7: {str(e)}")
                        add_activity_log(f"Phase 7 error: {str(e)}", "error")
        return

    # Display feedback from each industry partner
    st.markdown("### 👔 Partner Feedback")

    for feedback_item in industry_feedback:
        partner_name = feedback_item.get('partner_name', 'Partner')
        focus_area = feedback_item.get('focus_area', 'General')
        ontologies_reviewed = feedback_item.get('ontologies_reviewed', [])
        feedback_text = feedback_item.get('feedback', '')

        with st.expander(f"👔 {partner_name} - {focus_area}", expanded=True):
            # Show which ontologies they reviewed
            st.markdown(f"**Ontologies Reviewed:** {', '.join(ontologies_reviewed)}")
            st.caption("✓ Using OTHER researchers' ontologies (not their own)")

            st.markdown("---")

            # Feedback content
            st.markdown(feedback_text)

    # Completion actions
    st.markdown("---")
    st.success("✅ All phases complete! The research workflow has finished.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download Full Report", width='stretch'):
            st.info("Would generate and download complete report...")

    with col2:
        if st.button("📊 Generate Summary", width='stretch'):
            st.info("Would generate executive summary...")

    with col3:
        if st.button("🔄 Start New Session", width='stretch'):
            st.info("Would start new research session...")


# ===== STAGE 5: DIALOGUE =====

def render_dialogue_stage():
    """Render Socratic dialogue stage."""
    st.header("💬 Socratic Dialogue")

    notes = st.session_state.dialogue_notes

    if not notes:
        st.warning("No high-priority dialogue notes found")
        if st.button("Back to Results"):
            st.session_state.workflow_stage = "results"
            st.rerun()
        return

    st.info(f"Found {len(notes)} high-priority questions to explore")

    st.markdown("---")

    # Current question
    current_idx = st.session_state.current_dialogue_index

    if current_idx >= len(notes):
        st.success("✅ All questions explored!")

        if st.button("View Results"):
            st.session_state.workflow_stage = "results"
            st.rerun()

        return

    current_note = notes[current_idx]

    # Display question
    st.subheader(f"Question {current_idx + 1} of {len(notes)}")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.metric("Priority", current_note.get("priority", 0))
        st.caption(f"Type: {current_note.get('type', 'unknown')}")
        st.caption(f"From: {current_note.get('character_domain', 'Unknown')}")

    with col2:
        st.markdown(f"### {current_note.get('suggested_question', 'No question')}")
        st.caption(f"**Context:** {current_note.get('context', 'No context')}")

    st.markdown("---")

    # Character's answer (mock for now - in real system would call character agent)
    st.subheader("💡 Character's Perspective")

    # For demonstration, show a sample answer
    sample_answer = f"""
    Based on the research conducted, this question touches on {current_note.get('type', 'an important aspect')}.

    The {current_note.get('character_domain', 'domain')} perspective suggests that we need to consider...

    [In a real implementation, this would be generated by the character agent using their research findings and domain expertise]
    """

    st.info(sample_answer)

    st.markdown("---")

    # User feedback
    st.subheader("📝 Your Response")

    user_response = st.text_area(
        "Your thoughts or follow-up questions:",
        height=150,
        placeholder="Share your thoughts, ask follow-up questions, or type 'continue' to move to next question..."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💬 Submit Response", type="primary", width='stretch'):
            if user_response.strip():
                # Record dialogue
                st.session_state.dialogue_history.append({
                    "question": current_note.get("suggested_question"),
                    "character_answer": sample_answer,
                    "user_response": user_response,
                    "timestamp": datetime.now().isoformat()
                })

                add_activity_log("Dialogue response recorded", "success")

                # Move to next question
                st.session_state.current_dialogue_index += 1
                st.rerun()
            else:
                st.warning("Please provide a response")

    with col2:
        if st.button("⏭️ Skip Question", width='stretch'):
            st.session_state.current_dialogue_index += 1
            add_activity_log("Question skipped", "info")
            st.rerun()

    with col3:
        if st.button("🏁 End Dialogue", width='stretch'):
            st.session_state.workflow_stage = "results"
            add_activity_log("Dialogue ended by user", "info")
            st.rerun()

    # Progress indicator
    st.progress((current_idx + 1) / len(notes))
    st.caption(f"Question {current_idx + 1} of {len(notes)}")


# ===== MAIN APP =====

def main():
    """Main application entry point."""
    # Render sidebar
    render_sidebar()

    # Render header
    st.title("🔬 Interactive Research Workflow System")
    st.caption(f"Session: {st.session_state.session_id[:12]}... | Thread: {st.session_state.thread_id[:12]}...")

    # Render stage indicator
    render_stage_indicator()

    # Render current stage
    stage = st.session_state.workflow_stage

    if stage == "input":
        render_input_stage()
    elif stage == "planning":
        render_planning_stage()
    elif stage == "research":
        render_research_stage()
    elif stage == "results":
        render_results_stage()
    elif stage == "dialogue":
        render_dialogue_stage()
    else:
        st.error(f"Unknown workflow stage: {stage}")


if __name__ == "__main__":
    main()
