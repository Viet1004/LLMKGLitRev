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

            # CRITICAL: Load ontology data if available
            ontology_file = session_path / "ontology_data.json"
            if ontology_file.exists():
                import json
                with open(ontology_file, "r") as f:
                    st.session_state.ontology_data = json.load(f)
                print("✅ Loaded ontology data from session")
            else:
                print("⚠️  No ontology data found in session")
                st.session_state.ontology_data = {}

            # NOTE: Ontology manager is NOT loaded by default
            # It will be loaded ONLY if user provided ontology URLs in character configs
            # This happens automatically during concept extraction (Phase 4)
            # See: research_proposal_generator.py:extract_ontology_concepts()
            st.session_state.ontology_manager = None
            print("ℹ️  No default ontology loaded - will use character-specific ontologies if provided")

            # NOTE: Character configs are already stored in session artifacts at:
            # sessions/{session_id}/conversations/{character_id}.json
            # They will be loaded by UnifiedCharacterManager when needed (e.g., during dialogue)
            # No need to copy them to the characters/ folder
            print(f"ℹ️  Character configs available in session artifacts for {len(artifacts)} characters")

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
                character_data = agent.get('character', {})
                char_name = character_data.get('name', 'Unknown')
                char_domain = character_data.get('domain', 'Unknown Domain')
                add_activity_log(f"Agent {idx}: {char_domain} ({char_name})", "info")
            
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
    """Render agent card in horizontal layout with modal editing."""
    # Get embedded character
    character_data = agent.get('character', {})

    character_name = character_data.get('name', 'Unknown Character')
    domain = character_data.get('domain', 'Unknown Domain')
    stance = character_data.get('stance', 'neutral')
    assigned_topic = agent.get('assigned_topic', '')
    typical_venues = character_data.get('typical_venues', [])
    seed_papers = agent.get('seed_papers', [])
    expertise = character_data.get('expertise_areas', [])

    seed_papers_count = len(seed_papers)
    venues_display = ", ".join(typical_venues[:3]) if typical_venues else ""
    venues_text = f'<p style="font-size: 11px; color: #666; margin: 5px 0;">📍 {venues_display}{"..." if len(typical_venues) > 3 else ""}</p>' if venues_display else ''

    # Show expertise if no venues
    expertise_display = ", ".join(expertise[:2]) if expertise and not venues_display else ""
    expertise_text = f'<p style="font-size: 11px; color: #666; margin: 5px 0;">🎓 {expertise_display}{"..." if len(expertise) > 2 else ""}</p>' if expertise_display else ''

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
        <h4 style="margin: 5px 0;">{character_name}</h4>
        <p style="color: #666; font-size: 12px; margin: 5px 0;">
            {domain}<br/>
            <strong>Stance:</strong> {stance.title()}
        </p>
        {venues_text}
        {expertise_text}
        {f'<p style="font-size: 11px; color: #888;">📄 {seed_papers_count} seed papers</p>' if seed_papers_count > 0 else ''}
    </div>
    """, unsafe_allow_html=True)

    # Click to edit (opens modal)
    if st.button("View/Edit", key=f"view_{index}", use_container_width=True):
        render_agent_modal(agent, index)


@st.dialog("Agent Configuration")
def render_agent_modal(agent: Dict, index: int):
    """Render agent details and editing in modal dialog."""
    # Get embedded character
    character = agent.get('character', {})

    st.subheader(f"🎭 {character.get('name', 'Unknown Character')}")
    st.caption(f"*{character.get('domain', 'Unknown Domain')}*")

    # Display current configuration
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Stance:** {character.get('stance', 'neutral').title()}")

        expertise = character.get('expertise_areas', [])
        if expertise:
            st.write("**Expertise:**")
            for exp in expertise[:3]:
                st.write(f"  • {exp}")

        databases = character.get('preferred_databases', [])
        if databases:
            st.write(f"**Databases:** {', '.join(databases[:3])}")

    with col2:
        search_scope = agent.get('search_scope', [])
        if search_scope:
            st.write("**Search Keywords:**")
            for scope in search_scope[:5]:
                st.write(f"  • {scope}")

        venues = character.get('typical_venues', [])
        if venues:
            st.write("**Publication Venues:**")
            for venue in venues[:3]:
                st.write(f"  • {venue}")

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
    # Extract character data from embedded object
    character_data = agent.get('character', {})
    character_name = character_data.get('name', 'Unknown Character')
    domain = character_data.get('domain', 'Unknown Domain')
    stance = character_data.get('stance', 'neutral')

    with st.container():
        st.markdown(f"""
        <div class="character-card">
            <strong>{domain}</strong>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write(f"**Character:** {character_name}")
            st.write(f"**Stance:** {stance}")
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
    """Render agent editor form with embedded character fields."""
    # Get the embedded character
    character = agent.get("character", {})

    st.markdown("### Character Profile")

    # Character name
    char_name = st.text_input(
        "Character Name",
        value=character.get("name", ""),
        key=f"char_name_{index}",
        help="Descriptive name for this research character"
    )

    # Domain
    domain = st.text_input(
        "Research Domain",
        value=character.get("domain", ""),
        key=f"domain_{index}",
        help="Specific research domain (e.g., 'Privacy-Preserving Machine Learning')"
    )

    # Stance selection
    stance_options = ["critical", "constructive", "neutral", "pragmatic"]
    current_stance = character.get("stance", "neutral")
    stance = st.selectbox(
        "Stance",
        options=stance_options,
        index=stance_options.index(current_stance) if current_stance in stance_options else 0,
        key=f"stance_select_{index}",
        help="critical=skeptical, constructive=supportive, neutral=balanced"
    )

    col1, col2 = st.columns(2)

    with col1:
        # Expertise areas
        expertise = st.text_area(
            "Expertise Areas (one per line)",
            value="\n".join(character.get("expertise_areas", [])),
            height=100,
            key=f"expertise_{index}",
            help="Specific technical skills and methods"
        )

    with col2:
        # Sub-domains
        sub_domains = st.text_area(
            "Sub-domains (one per line)",
            value="\n".join(character.get("sub_domains", [])),
            height=100,
            key=f"subdomains_{index}",
            help="Related research sub-fields"
        )

    # Typical venues
    typical_venues = st.text_area(
        "Publication Venues (one per line)",
        value="\n".join(character.get("typical_venues", [])),
        height=100,
        key=f"venues_{index}",
        help="Key conferences and journals (e.g., NeurIPS, ICML, Nature)"
    )

    # Preferred databases
    all_databases = ["arxiv", "scopus", "ieee", "semantic_scholar", "openalex", "crossref"]
    current_dbs = character.get("preferred_databases", ["arxiv", "semantic_scholar"])
    preferred_dbs = st.multiselect(
        "Preferred Databases",
        options=all_databases,
        default=[db for db in current_dbs if db in all_databases],
        key=f"databases_{index}",
        help="Which academic sources to prioritize"
    )

    # Background
    background = st.text_area(
        "Background",
        value=character.get("background", ""),
        height=80,
        key=f"background_{index}",
        help="2-3 sentences about this character's perspective"
    )

    # Communication style
    comm_style = st.text_input(
        "Communication Style",
        value=character.get("communication_style", "academic"),
        key=f"comm_style_{index}",
        help="How this character communicates (e.g., 'Technical and rigorous')"
    )

    st.markdown("---")
    st.markdown("### 🧠 Domain Ontology (Optional)")
    st.caption("Select a predefined ontology or provide your own custom URL")

    # Import predefined ontologies
    from llmkglitrev.ontologies.predefined_ontologies import (
        get_ontology_dropdown_options,
        get_ontology_url_from_selection,
        get_ontology_info_from_selection,
        get_ontology_selection_from_url
    )

    # Get current ontology URL and determine selection
    current_ontology_url = character.get("ontology_url", "")
    current_selection = get_ontology_selection_from_url(current_ontology_url)

    # Dropdown selection
    ontology_options = get_ontology_dropdown_options()
    default_index = ontology_options.index(current_selection) if current_selection in ontology_options else 0

    ontology_selection = st.selectbox(
        "Select Ontology",
        options=ontology_options,
        index=default_index,
        help="Choose from predefined ontologies or select 'Other' to provide custom URL",
        key=f"ontology_selection_{index}"
    )

    # Show ontology info if predefined ontology selected
    if ontology_selection not in ["None (No ontology)", "Other (Custom URL)"]:
        onto_info = get_ontology_info_from_selection(ontology_selection)
        if onto_info:
            with st.expander("ℹ️ Ontology Details", expanded=False):
                st.markdown(f"**Description:** {onto_info.description}")
                st.markdown(f"**Format:** {onto_info.format}")
                st.markdown(f"**Source:** {onto_info.source}")
                st.markdown(f"**URL:** `{onto_info.url}`")

    # Custom URL input if "Other" selected
    custom_ontology_url = ""
    if ontology_selection == "Other (Custom URL)":
        custom_ontology_url = st.text_input(
            "Custom Ontology URL",
            value=current_ontology_url if current_selection == "Other (Custom URL)" else "",
            placeholder="e.g., http://edamontology.org/EDAM.owl",
            help="Provide URL to your custom OWL/RDF ontology file",
            key=f"custom_ontology_url_{index}"
        )

    # Determine final ontology URL
    if ontology_selection == "Other (Custom URL)":
        ontology_url = custom_ontology_url.strip()
    else:
        ontology_url = get_ontology_url_from_selection(ontology_selection)

    st.markdown("---")
    st.markdown("### Agent Configuration")

    # Search scope
    search_scope = st.text_input(
        "Search Keywords (comma-separated)",
        value=", ".join(agent.get("search_scope", [])),
        key=f"scope_{index}",
        help="Keywords this agent should focus on during research"
    )

    # Rationale
    rationale = st.text_area(
        "Rationale",
        value=agent.get("rationale", ""),
        height=80,
        key=f"rationale_{index}",
        help="Why this agent is needed (2-3 sentences)"
    )

    # Save button
    if st.button("💾 Save Changes", key=f"save_modal_{index}"):
        # Ensure modified_plan exists and is a copy
        if not st.session_state.modified_plan:
            st.session_state.modified_plan = st.session_state.proposed_research_plan.copy()

        # Build updated character object
        updated_character = {
            "character_id": character.get("character_id", f"custom_{index}"),
            "name": char_name,
            "domain": domain,
            "stance": stance,
            "expertise_areas": [e.strip() for e in expertise.split("\n") if e.strip()],
            "sub_domains": [s.strip() for s in sub_domains.split("\n") if s.strip()],
            "typical_venues": [v.strip() for v in typical_venues.split("\n") if v.strip()],
            "preferred_databases": preferred_dbs,
            "background": background,
            "communication_style": comm_style,
            "description": character.get("description", ""),
            "typical_methods": character.get("typical_methods", []),
            "typical_datasets": character.get("typical_datasets", []),
            "theoretical_foundations": character.get("theoretical_foundations", []),
            "focus_areas": character.get("focus_areas", []),
            "question_types_to_ask": character.get("question_types_to_ask", []),
            "created_by": character.get("created_by", "user"),
            "created_at": character.get("created_at", ""),
            "last_modified": character.get("last_modified", ""),
            "version": character.get("version", "1.0"),
            "system_prompt_template": character.get("system_prompt_template", ""),
            "ontology_url": ontology_url.strip() if ontology_url else ""
        }

        # Update the agent with modified character and config
        modifications = {
            "character": updated_character,
            "search_scope": [s.strip() for s in search_scope.split(",") if s.strip()],
            "rationale": rationale
        }

        plan = modify_character_in_plan(
            st.session_state.modified_plan,
            index,
            modifications
        )
        st.session_state.modified_plan = plan

        # Log the change
        add_activity_log(f"Modified agent: {char_name}", "success")

        st.success("✅ Changes saved!")
        st.rerun()


def render_custom_character_form():
    """Render form to create custom character and add to plan."""
    st.markdown("**Create a new research character**")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Character Name", placeholder="e.g., Clinical AI Ethics Researcher")
        domain = st.text_input("Domain", placeholder="e.g., AI Ethics in Healthcare")
        stance = st.selectbox("Stance", ["critical", "constructive", "neutral", "pragmatic"])

    with col2:
        communication_style = st.text_input(
            "Communication Style",
            placeholder="e.g., Thoughtful and questioning",
            value="Professional"
        )
        background = st.text_area(
            "Background",
            placeholder="Brief background about this character's perspective (2-3 sentences)",
            height=100
        )

    expertise = st.text_area(
        "Expertise Areas (one per line)",
        placeholder="e.g.,\nAI fairness and bias\nClinical decision support\nPatient privacy\nRegulatory compliance",
        height=100,
        help="Specific technical skills and methods"
    )

    typical_venues = st.text_area(
        "Publication Venues (one per line)",
        placeholder="e.g.,\nFAccT (Fairness, Accountability, Transparency)\nNature Medicine\nJAMA\nIEEE Security & Privacy",
        height=100,
        help="Key conferences and journals for this domain"
    )

    # Preferred databases
    all_databases = ["arxiv", "scopus", "ieee", "semantic_scholar", "openalex", "crossref"]
    preferred_dbs = st.multiselect(
        "Preferred Databases",
        options=all_databases,
        default=["arxiv", "semantic_scholar", "scopus"],
        help="Which academic sources to prioritize"
    )

    search_keywords = st.text_input(
        "Search Keywords (comma-separated)",
        placeholder="e.g., AI ethics, healthcare AI, clinical decision support",
        help="Keywords this agent should focus on during research"
    )

    # Ontology URL (optional) - Dropdown with predefined options
    st.markdown("---")
    st.markdown("**🧠 Domain Ontology (Optional)**")
    st.caption("Select a predefined ontology or provide your own custom URL")

    # Import predefined ontologies
    from llmkglitrev.ontologies.predefined_ontologies import (
        get_ontology_dropdown_options,
        get_ontology_url_from_selection,
        get_ontology_info_from_selection
    )

    # Dropdown selection
    ontology_options = get_ontology_dropdown_options()
    ontology_selection = st.selectbox(
        "Select Ontology",
        options=ontology_options,
        index=0,  # Default to "None (No ontology)"
        help="Choose from predefined ontologies or select 'Other' to provide custom URL",
        key="ontology_selection_custom"
    )

    # Show ontology info if predefined ontology selected
    if ontology_selection not in ["None (No ontology)", "Other (Custom URL)"]:
        onto_info = get_ontology_info_from_selection(ontology_selection)
        if onto_info:
            with st.expander("ℹ️ Ontology Details", expanded=False):
                st.markdown(f"**Description:** {onto_info.description}")
                st.markdown(f"**Format:** {onto_info.format}")
                st.markdown(f"**Source:** {onto_info.source}")
                st.markdown(f"**URL:** `{onto_info.url}`")

    # Custom URL input if "Other" selected
    custom_ontology_url = ""
    if ontology_selection == "Other (Custom URL)":
        custom_ontology_url = st.text_input(
            "Custom Ontology URL",
            placeholder="e.g., http://edamontology.org/EDAM.owl",
            help="Provide URL to your custom OWL/RDF ontology file",
            key="custom_ontology_url"
        )

    # Determine final ontology URL
    if ontology_selection == "Other (Custom URL)":
        ontology_url = custom_ontology_url.strip()
    else:
        ontology_url = get_ontology_url_from_selection(ontology_selection)

    rationale = st.text_area(
        "Rationale",
        placeholder="Why is this agent needed for your research? (2-3 sentences)",
        height=80
    )

    if st.button("➕ Add to Plan", type="primary"):
        if name and domain:
            # Build complete character object
            char_data = {
                "character_id": f"custom_{uuid.uuid4().hex[:8]}",
                "name": name,
                "domain": domain,
                "stance": stance,
                "expertise_areas": [e.strip() for e in expertise.split("\n") if e.strip()],
                "sub_domains": [],
                "typical_venues": [v.strip() for v in typical_venues.split("\n") if v.strip()],
                "preferred_databases": preferred_dbs,
                "communication_style": communication_style or "Professional",
                "background": background or f"Expert in {domain}",
                "description": f"Custom {domain} researcher",
                "typical_methods": [],
                "typical_datasets": [],
                "theoretical_foundations": [],
                "focus_areas": [],
                "question_types_to_ask": [],
                "created_by": "user",
                "created_at": "",
                "last_modified": "",
                "version": "1.0",
                "system_prompt_template": "",
                "ontology_url": ontology_url.strip() if ontology_url else ""  # NEW: Optional ontology URL
            }

            # Build agent with embedded character
            new_agent = {
                "character": char_data,
                "search_scope": [s.strip() for s in search_keywords.split(",") if s.strip()] if search_keywords else char_data["expertise_areas"],
                "rationale": rationale or f"Custom character for {domain}",
                "assigned_topic": "",
                "seed_papers": []
            }

            # Add to modified plan
            if not st.session_state.modified_plan:
                st.session_state.modified_plan = st.session_state.proposed_research_plan.copy()

            plan = st.session_state.modified_plan
            plan["proposed_agents"].append(new_agent)

            st.success(f"✅ Added {name} to research plan!")
            add_activity_log(f"Custom character '{name}' added to plan", "success")
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

                        # Capture ontology data
                        ontology_data = event.get("ontology_data", {})
                        if ontology_data:
                            st.session_state.ontology_data = ontology_data

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


# ===== ONTOLOGY VISUALIZATION FUNCTIONS =====

def render_concept_network_graph(
    concepts: List[str],
    relationships: List[Dict],
    clusters: Dict[str, List[str]],
    concept_to_agents: Dict[str, List[str]],
    selected_agents: List[str],
    concept_labels: Optional[Dict[str, str]] = None
):
    """
    Render interactive network graph of ontology concepts.

    Features:
    - Nodes = concepts (colored by cluster)
    - Edges = relationships (shown on hover)
    - Clustering with visual boundaries
    - Filter by agent
    - Small, manageable size
    """
    import plotly.graph_objects as go
    import networkx as nx

    # Filter concepts by selected agents
    if selected_agents:
        filtered_concepts = [
            c for c in concepts
            if any(agent in concept_to_agents.get(c, []) for agent in selected_agents)
        ]
    else:
        filtered_concepts = concepts

    if not filtered_concepts:
        st.warning("No concepts found for selected agents")
        return None

    # Build NetworkX graph
    G = nx.Graph()

    # Add nodes
    for concept in filtered_concepts:
        G.add_node(concept)

    # Add edges (only between filtered concepts)
    for rel in relationships:
        if rel["source"] in filtered_concepts and rel["target"] in filtered_concepts:
            G.add_edge(
                rel["source"],
                rel["target"],
                relation=rel["relation"],
                relation_type=rel["relation_type"]
            )

    # Layout with clustering (spring layout with moderate spacing)
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Prepare node traces (one per cluster for coloring)
    cluster_colors = {
        "Data": "#1f77b4",
        "Operation": "#ff7f0e",
        "Topic": "#2ca02c",
        "Format": "#d62728",
        "Identifier": "#9467bd",
        "Other": "#8c564b"
    }

    node_traces = []

    for cluster_name, cluster_concepts in clusters.items():
        # Filter to only concepts in filtered_concepts
        cluster_filtered = [c for c in cluster_concepts if c in filtered_concepts]

        if not cluster_filtered:
            continue

        # Get positions
        node_x = [pos[c][0] for c in cluster_filtered]
        node_y = [pos[c][1] for c in cluster_filtered]

        # Get agents for each concept (for sizing)
        node_sizes = [
            10 + 5 * len(concept_to_agents.get(c, []))
            for c in cluster_filtered
        ]

        # Hover text with labels
        if concept_labels:
            node_text = [
                f"<b>{concept_labels.get(c, c)}</b><br>ID: {c}<br>Cluster: {cluster_name}<br>Agents: {', '.join(concept_to_agents.get(c, []))}"
                for c in cluster_filtered
            ]
            # Text shown on graph (labels only)
            node_labels = [concept_labels.get(c, c) for c in cluster_filtered]
        else:
            node_text = [
                f"<b>{c}</b><br>Cluster: {cluster_name}<br>Agents: {', '.join(concept_to_agents.get(c, []))}"
                for c in cluster_filtered
            ]
            node_labels = cluster_filtered

        trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',  # Show text on graph
            name=cluster_name,
            text=node_labels,
            textposition='top center',
            textfont=dict(size=9),
            marker=dict(
                size=node_sizes,
                color=cluster_colors.get(cluster_name, "#8c564b"),
                line=dict(width=2, color='white')
            ),
            hovertext=node_text,
            hoverinfo='text',
            showlegend=True
        )

        node_traces.append(trace)

    # Edge traces (lighter color, shown on hover)
    edge_x = []
    edge_y = []
    edge_info = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        relation = edge[2].get('relation', 'related')
        edge_info.append(f"{edge[0]} → {relation} → {edge[1]}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        showlegend=False,
        opacity=0.3
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=dict(
                text="Concept Network (from Final Proposal)",
                font=dict(size=16)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            height=500,  # Small, manageable size
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )

    return fig


def print_taxonomy_tree(hierarchy: Dict, concept_labels: Optional[Dict[str, str]] = None) -> str:
    """
    Print taxonomy tree in readable hierarchical format.

    Returns formatted string showing General → Specific structure.
    """
    lines = []

    def traverse(node, indent=0, is_last=False):
        name = node.get("name", "Unknown")
        level = node.get("level", 0)

        # Get human-readable label
        display_label = concept_labels.get(name, name) if concept_labels else name

        # Create tree structure with proper indentation
        prefix = "    " * indent
        if indent > 0:
            connector = "└── " if is_last else "├── "
        else:
            connector = ""

        # Format line with level indicator
        line = f"{prefix}{connector}{display_label} [Level {level}]"
        lines.append(line)

        # Recurse through children
        children = node.get("children", [])
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            traverse(child, indent + 1, is_last_child)

    if hierarchy:
        traverse(hierarchy)

    return "\n".join(lines)


def render_concept_tree(hierarchy: Dict, concept_to_agents: Dict, concept_labels: Optional[Dict[str, str]] = None):
    """
    Render hierarchical tree of concepts (general → specific).

    Used for understanding complexity levels for question generation.
    """
    import plotly.graph_objects as go

    # Flatten tree for treemap
    labels = []
    parents = []
    values = []
    levels = []
    hover_texts = []

    def traverse(node, parent_name="", parent_label=""):
        name = node.get("name", "Unknown")
        level = node.get("level", 0)

        # Get human-readable label
        display_label = concept_labels.get(name, name) if concept_labels else name

        labels.append(display_label)
        parents.append(parent_label)
        values.append(1)  # Equal size for all nodes
        levels.append(level)

        # Hover text with both label and ID
        agents = concept_to_agents.get(name, [])
        if concept_labels and name in concept_labels:
            hover_text = f"<b>{display_label}</b><br>ID: {name}<br>Level: {level}<br>Agents: {', '.join(agents) if agents else 'None'}"
        else:
            hover_text = f"<b>{name}</b><br>Level: {level}<br>Agents: {', '.join(agents) if agents else 'None'}"
        hover_texts.append(hover_text)

        # Recurse
        for child in node.get("children", []):
            traverse(child, parent_name=name, parent_label=display_label)

    if hierarchy:
        traverse(hierarchy)

    if not labels:
        st.warning("No hierarchy data available")
        return None

    # Create treemap
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=levels,  # Use the levels for color mapping
            colorscale='Viridis',
            cmid=2,
            colorbar=dict(title="Complexity Level")
        ),
        text=hover_texts,
        hoverinfo='text',
        textposition='middle center'
    ))

    fig.update_layout(
        title="Concept Hierarchy (General → Specific)",
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig


# ===== STAGE 4: RESULTS =====

def render_results_stage():
    """Render results and artifacts with visualization and validation."""
    st.header("📊 Research Results")

    # ===== FIX 1: Display Final Proposal Prominently =====
    st.subheader("📝 Unified Research Proposal")

    # Get final proposal from session state
    final_proposal = st.session_state.get("final_proposal", None)

    if final_proposal:
        # Final proposal is already saved to disk by the workflow
        # Just display it here
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
    tab3, tab4 = st.tabs([
        # "📄 Research Artifacts",
        # "🔍 Concept Visualization",
        # "🔗 Cross-Domain Validation",
        # "📊 Dashboard",
        # "🔧 Activity Log",
        # "💡 Gap Identification",
        "🗣️ Socratic Dialogue",
        "🏢 Industry Review"
    ])

    # with tab1:
        # render_artifacts_tab(artifacts)

    # with tab1:
    #     render_visualization_tab(artifacts)

    # with tab1:
        # render_cross_domain_validation_tab(artifacts)

    # with tab4:
    #     render_dashboard_tab(artifacts)

    # with tab3:
    #     st.caption("Detailed research activity, queries, and tool calls")
    #     render_activity_log()

    # with tab2:
        # render_gap_identification_tab()

    with tab3:
        render_socratic_dialogue_tab()

    with tab4:
        render_industry_review_tab()

    st.markdown("---")

    # Next steps (always visible at bottom)
    st.subheader("🎯 Next Steps")

    col1, col2 = st.columns(2)

    # with col1:
    #     if st.button("💬 Start Dialogue", type="primary", width='stretch'):
    #         # Extract dialogue notes
    #         st.session_state.dialogue_notes = extract_high_priority_dialogue_notes(artifacts)
    #         st.session_state.workflow_stage = "dialogue"
    #         st.rerun()

    with col1:
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

    with col2:
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
    """Render ontology concept visualization (Network Graph + Tree)."""
    st.subheader("🧠 Ontology Concept Visualization")

    st.markdown("""
    This visualization shows ontology concepts that appear in the **final research proposal**
    and their relationships according to domain ontologies.
    """)

    # Load ontology data from session
    ontology_data = st.session_state.get("ontology_data", {})

    if not ontology_data:
        # Try to load from file
        session_id = st.session_state.get("session_id", "")
        if session_id:
            ontology_file = Path(f"sessions/{session_id}/ontology_data.json")

            # Debug logging
            st.write(f"🔍 DEBUG: Looking for ontology file at: {ontology_file}")
            st.write(f"🔍 DEBUG: File exists: {ontology_file.exists()}")

            if ontology_file.exists():
                import json
                with open(ontology_file, "r") as f:
                    ontology_data = json.load(f)
                    st.session_state.ontology_data = ontology_data
                st.success(f"✅ Loaded ontology data from file")
            else:
                st.info(f"Ontology concepts not yet extracted. Expected file at: {ontology_file}")
                st.info("Complete research workflow to generate ontology visualization.")
                return
        else:
            st.info("No session data available. Complete research first.")
            return

    # Extract data
    concepts = ontology_data.get("ontology_concepts", [])
    concept_labels = ontology_data.get("concept_labels", {})
    relationships = ontology_data.get("concept_relationships", [])
    hierarchy = ontology_data.get("concept_hierarchy", {})
    clusters = ontology_data.get("concept_clusters", {})
    concept_to_agents = ontology_data.get("concept_to_agents", {})

    if not concepts:
        st.info("ℹ️ **No ontology concepts extracted**")
        st.markdown("""
        This is expected if:
        - No custom ontology URL was provided during character creation
        - The custom ontology failed to load

        **Socratic dialogue (Phase 6) will still work** using priority-based question ordering.

        To use taxonomy-based ordering:
        1. Create a new research session
        2. Provide a custom ontology URL when creating characters
        3. The ontology will be used to order questions from general → specific
        """)
        return

    st.success(f"Found {len(concepts)} ontology concepts in final proposal")

    # Visualization type selector
    viz_type = st.radio(
        "Visualization Type:",
        ["Network Graph", "Hierarchy Tree"],
        horizontal=True,
        help="Network Graph shows concept relationships and clusters. Hierarchy Tree shows complexity levels."
    )

    st.markdown("---")

    if viz_type == "Network Graph":
        st.markdown("**Network Graph**: Shows concept relationships and clustering from final proposal")
        st.caption("💡 Hover over nodes to see which agents mentioned each concept. Relationships shown on edges.")

        # Agent filter
        all_agents = list(set([
            agent
            for agents in concept_to_agents.values()
            for agent in agents
        ]))

        if all_agents:
            selected_agents = st.multiselect(
                "Filter by Agent:",
                options=all_agents,
                default=all_agents,
                help="Show only concepts mentioned by selected agents"
            )
        else:
            selected_agents = []
            st.info("No agent information available for filtering")

        # Render graph
        try:
            fig = render_concept_network_graph(
                concepts,
                relationships,
                clusters,
                concept_to_agents,
                selected_agents,
                concept_labels
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Export HTML"):
                        output_path = f"sessions/{st.session_state.session_id}/concept_network.html"
                        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                        fig.write_html(output_path)
                        st.success(f"Exported to {output_path}")

                # Stats
                st.markdown("---")
                st.markdown(f"**Total Concepts**: {len(concepts)}")
                st.markdown(f"**Relationships**: {len(relationships)}")
                st.markdown(f"**Clusters**: {len(clusters)}")

        except Exception as e:
            st.error(f"Error rendering network graph: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    else:  # Hierarchy Tree
        st.markdown("**Hierarchy Tree**: Shows complexity levels (general → specific)")
        st.caption("💡 This structure will be used for generating Socratic questions at different complexity levels")

        # Render tree
        try:
            fig = render_concept_tree(hierarchy, concept_to_agents, concept_labels)

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Export
                if st.button("💾 Export HTML"):
                    output_path = f"sessions/{st.session_state.session_id}/concept_tree.html"
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    fig.write_html(output_path)
                    st.success(f"Exported to {output_path}")

                # Taxonomy Tree Printout
                st.markdown("---")
                st.markdown("### 📋 Taxonomy Hierarchy (General → Specific)")
                st.caption("This structure shows the hierarchical order of concepts for Phase 6 question generation")

                taxonomy_text = print_taxonomy_tree(hierarchy, concept_labels)
                st.code(taxonomy_text, language="")

                # Explanation
                st.markdown("---")
                st.markdown("### Using This for Question Generation")
                st.markdown("""
                - **Level 0-1**: General foundational questions (broad concepts)
                - **Level 2-3**: Specific technical questions (focused concepts)
                - **Level 4+**: Deep implementation questions (detailed concepts)

                In Phase 6, characters will ask questions starting from general concepts (top of tree)
                and progressively move to more specific concepts (bottom of tree).
                """)

        except Exception as e:
            st.error(f"Error rendering tree: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


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

                # Initialize ontology managers (optional feature for cross-domain validation)
                st.session_state.ontology_managers = {}

                # NOTE: This is a DEMO feature - loads local ontology files if available
                # In production, ontologies should be specified in character configs, not loaded here
                if source_domain in ["machine_learning", "bioinformatics"]:
                    edam_path = ontology_path / "edam.owl"
                    if edam_path.exists():
                        # This is for demo purposes only - not part of main workflow
                        st.info("ℹ️ Demo feature: Loading local EDAM ontology for cross-domain mapping")
                        st.session_state.ontology_managers[source_domain] = DomainOntologyManager(
                            ontology_source="custom",
                            custom_url=str(edam_path)
                        )
                    else:
                        st.warning("⚠️ Local EDAM file not found. Cross-domain mapping disabled.")
                        st.info("This is a demo feature - ontologies should be specified in character configs instead.")

                if st.session_state.ontology_managers:
                    st.success("✅ Demo ontologies loaded for cross-domain mapping")

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


def render_question_queue(pending_notes, hierarchy_available):
    """Show upcoming questions with priority and taxonomy levels."""
    st.markdown("### 📊 Question Queue")

    if not pending_notes:
        st.success("✅ All questions answered!")
        return

    st.info(f"⏳ {len(pending_notes)} questions remaining")

    st.markdown("**Next up:**")
    for i, note in enumerate(pending_notes[:3], 1):
        priority = note.get("priority", 0)
        level = note.get("complexity_level", "N/A")
        question = note.get("suggested_question", "")

        if hierarchy_available and level != "N/A":
            complexity_badge = f"Level {level}"
            if level <= 2:
                complexity_label = "(General)"
            elif level <= 4:
                complexity_label = "(Specific)"
            else:
                complexity_label = "(Detailed)"
        else:
            complexity_badge = "No taxonomy"
            complexity_label = ""

        st.markdown(f"`{i}.` **[Priority {priority}, {complexity_badge}]** {question[:80]}... {complexity_label}")


def render_chat_message(qa_item, is_current=False):
    """Render a single Q&A exchange in chat format."""
    # Character message
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"**{qa_item.get('character_name', 'Character')}**")
        priority = qa_item.get('priority', 0)
        level = qa_item.get('complexity_level', 'N/A')
        st.caption(f"Priority {priority} • Level {level}")
        st.markdown(f"**Q:** {qa_item['question']}")

        if is_current and not qa_item.get('answer'):
            # Show loading for current question being answered
            with st.spinner("Character is thinking..."):
                pass
        elif qa_item.get('answer'):
            st.markdown(f"**A:** {qa_item['answer']}")

    # User feedback (if provided)
    if qa_item.get('user_feedback') and qa_item['user_feedback'] not in ['continue', 'skip', 'stop']:
        with st.chat_message("user", avatar="👤"):
            st.markdown(qa_item['user_feedback'])


def render_feedback_input():
    """Render input area for user feedback with action buttons."""
    st.markdown("### ✍️ Your Response")

    user_input = st.text_area(
        "Provide feedback or type a command:",
        height=100,
        placeholder="Share your thoughts, or type 'continue', 'skip', or 'stop'",
        key="dialogue_user_input"
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    response = None

    with col1:
        if st.button("⏭️ Skip Question", use_container_width=True):
            response = "skip"

    with col2:
        if st.button("✅ Submit Feedback", type="primary", use_container_width=True):
            response = user_input if user_input else "continue"

    with col3:
        if st.button("🛑 Stop Dialogue", use_container_width=True):
            response = "stop"

    return response


def run_interactive_dialogue(session_id, artifacts):
    """Run simple character-based Socratic dialogue.

    User selects a character, character reads proposal and generates questions,
    user discusses unclear points with the character.
    """
    from llmkglitrev.agents.dialogue_coordinator import SocraticDialogue
    import asyncio
    from datetime import datetime

    # Initialize dialogue with thread support
    if "dialogue_session" not in st.session_state:
        with st.spinner("Loading dialogue session..."):
            st.session_state.dialogue_session = SocraticDialogue(session_id)
            st.session_state.conversation_history = []  # Legacy - keep for compatibility
            st.session_state.current_character = None
            st.session_state.generated_questions = []
            # NEW: Question thread management
            st.session_state.question_threads = {}  # {question_index: [conversation_history]}
            st.session_state.current_question_thread = None  # Currently selected question index

    dialogue = st.session_state.dialogue_session

    # ===== STEP 1: Character Selection =====
    st.markdown("### 1️⃣ Select a Character to Talk With")

    available_characters = dialogue.get_available_characters()

    if not available_characters:
        st.warning("No characters found in this session. Please complete research first.")
        return

    # Show character cards
    cols = st.columns(min(len(available_characters), 3))

    for idx, char_info in enumerate(available_characters):
        col = cols[idx % 3]
        with col:
            is_selected = st.session_state.current_character == char_info["character_id"]

            if st.button(
                f"🎭 **{char_info['name']}**\n\n{char_info['domain'][:50]}...",
                key=f"select_char_{char_info['character_id']}",
                type="primary" if is_selected else "secondary",
                use_container_width=True
            ):
                # Switch character
                if st.session_state.current_character != char_info["character_id"]:
                    st.session_state.current_character = char_info["character_id"]
                    st.session_state.generated_questions = []
                    st.session_state.conversation_history = []
                    # NEW: Reset question threads when switching characters
                    st.session_state.question_threads = {}
                    st.session_state.current_question_thread = None
                    st.rerun()

    # If no character selected, stop here
    if not st.session_state.current_character:
        st.info("👆 Please select a character to start the dialogue")
        return

    current_char_info = next(
        c for c in available_characters
        if c["character_id"] == st.session_state.current_character
    )

    st.markdown(f"**Current Character:** {current_char_info['name']} ({current_char_info['domain']})")

    # ===== STEP 2: Generate Questions with Thread System =====
    st.markdown("---")
    st.markdown("### 2️⃣ Character Questions")

    if not st.session_state.generated_questions:
        st.info(f"{current_char_info['name']} will read the proposal and identify unclear points from their perspective.")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            num_questions = st.slider("How many questions?", 1, 5, 3, key="num_socratic_questions")

            if st.button("🔍 Generate Questions", type="primary", use_container_width=True, key="gen_socratic_questions"):
                with st.spinner(f"Reading proposal from {current_char_info['domain']} perspective..."):
                    try:
                        questions = asyncio.run(
                            dialogue.generate_questions(
                                st.session_state.current_character,
                                num_questions=num_questions
                            )
                        )
                        st.session_state.generated_questions = questions
                        # Initialize thread for each question
                        st.session_state.question_threads = {i: [] for i in range(len(questions))}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating questions: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
        return

    # Show generated questions as clickable thread cards
    st.markdown(f"**{current_char_info['name']}** identified these unclear points - Click to Discuss:")

    # Safety check: ensure question_threads is initialized
    if "question_threads" not in st.session_state:
        st.session_state.question_threads = {i: [] for i in range(len(st.session_state.generated_questions))}

    for i, question in enumerate(st.session_state.generated_questions):
        # Check if this question has an active thread
        has_conversation = len(st.session_state.question_threads.get(i, [])) > 0
        is_current = st.session_state.current_question_thread == i

        # Question card
        col1, col2 = st.columns([5, 1])
        with col1:
            # Button to select this question thread
            button_label = f"{'📍' if is_current else '💬'} Q{i+1}: {question[:100]}{'...' if len(question) > 100 else ''}"
            if has_conversation:
                msg_count = len([m for m in st.session_state.question_threads[i] if m.get('speaker') == 'user'])
                button_label += f" ({msg_count} msg{'s' if msg_count != 1 else ''})"

            if st.button(
                button_label,
                key=f"select_socratic_question_{i}",
                type="primary" if is_current else "secondary",
                use_container_width=True
            ):
                st.session_state.current_question_thread = i
                st.rerun()

        with col2:
            # Show status indicator
            if is_current:
                st.markdown("**🟢 Active**")
            elif has_conversation:
                st.markdown("✅ Discussed")
            else:
                st.markdown("⚪ New")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Regenerate All", key="regen_socratic_questions"):
            st.session_state.generated_questions = []
            st.session_state.question_threads = {}
            st.session_state.current_question_thread = None
            st.rerun()
    with col2:
        if st.button("❌ Clear All", key="clear_socratic_questions"):
            st.session_state.generated_questions = []
            st.session_state.question_threads = {}
            st.session_state.current_question_thread = None
            st.rerun()

    # ===== STEP 3: Conversation Thread =====
    st.markdown("---")
    st.markdown("### 3️⃣ Discuss with Character")

    # DEBUG: Show current state
    with st.expander("🔧 Debug Info (click to expand)", expanded=False):
        st.write(f"current_question_thread: {st.session_state.get('current_question_thread', 'Not set')}")
        st.write(f"question_threads keys: {list(st.session_state.get('question_threads', {}).keys())}")
        st.write(f"Number of questions: {len(st.session_state.generated_questions)}")

    # Check if a question thread is selected
    if st.session_state.current_question_thread is not None:
        thread_idx = st.session_state.current_question_thread
        current_question = st.session_state.generated_questions[thread_idx]

        # Display the current question being discussed
        st.info(f"**Question {thread_idx + 1}:** {current_question}")

        # Get conversation history for this thread
        thread_history = st.session_state.question_threads.get(thread_idx, [])

        # Display conversation history for this thread
        if thread_history:
            st.markdown("**Conversation History:**")
            for msg in thread_history:
                speaker = msg.get("speaker", "Unknown")
                message = msg.get("message", "")
                timestamp = msg.get("timestamp", "")

                if speaker == "user":
                    st.markdown(f"**You:** {message}")
                else:
                    st.markdown(f"**{current_char_info['name']}:** {message}")
                st.caption(f"_{timestamp}_")
                st.markdown("")

            st.markdown("---")

        # User input for this thread
        st.markdown("**Ask about this unclear point:**")

        user_question = st.text_area(
            "Your question or follow-up:",
            placeholder=f"Discuss Question {thread_idx + 1} with {current_char_info['name']}...",
            height=100,
            key=f"socratic_user_question_thread_{thread_idx}"
        )

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            if st.button("📤 Send", type="primary", disabled=not user_question.strip(), key=f"send_socratic_thread_{thread_idx}"):
                # Add user message to this thread
                st.session_state.question_threads[thread_idx].append({
                    "speaker": "user",
                    "message": user_question,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })

                # Get character's answer with thread context
                with st.spinner(f"{current_char_info['name']} is thinking..."):
                    try:
                        # Build context: original question + thread history
                        context_messages = [
                            {"speaker": "system", "message": f"Original unclear point: {current_question}"}
                        ] + st.session_state.question_threads[thread_idx]

                        answer = asyncio.run(
                            dialogue.answer_question(
                                st.session_state.current_character,
                                user_question,
                                conversation_history=context_messages
                            )
                        )

                        # Add character's answer to this thread
                        st.session_state.question_threads[thread_idx].append({
                            "speaker": current_char_info["name"],
                            "message": answer,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })

                        st.rerun()

                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())

        with col2:
            if st.button("🔄 Clear Thread", key=f"clear_socratic_thread_{thread_idx}"):
                st.session_state.question_threads[thread_idx] = []
                st.rerun()

        with col3:
            if st.button("⬅️ Back", key=f"back_from_socratic_thread_{thread_idx}"):
                st.session_state.current_question_thread = None
                st.rerun()

        with col4:
            if st.button("💾 Export", key=f"export_socratic_thread_{thread_idx}"):
                import json
                from pathlib import Path
                export_dir = Path(f"sessions/{session_id}")
                export_dir.mkdir(parents=True, exist_ok=True)
                export_file = export_dir / f"dialogue_Q{thread_idx+1}_{st.session_state.current_character}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                export_data = {
                    "character": current_char_info,
                    "question": current_question,
                    "question_index": thread_idx + 1,
                    "conversation": st.session_state.question_threads[thread_idx]
                }

                with open(export_file, "w") as f:
                    json.dump(export_data, f, indent=2)

                st.success(f"Exported thread to {export_file}")

    else:
        # No question selected - show prompt
        if st.session_state.generated_questions:
            st.info("👆 Select a question above to start or continue a conversation thread")
        else:
            st.info("Generate questions above, then click on a question to start discussing it")

        # Free-form conversation option
        st.markdown("---")
        st.markdown("**Or ask a free-form question:**")

        free_question = st.text_area(
            "Your question:",
            height=100,
            placeholder=f"Ask {current_char_info['name']} anything...",
            key="socratic_free_question"
        )

        if st.button("📤 Send Free-Form Question", type="secondary", disabled=not free_question.strip(), key="send_socratic_free_question"):
            # Create a new thread for free-form question
            new_thread_idx = len(st.session_state.generated_questions)
            st.session_state.generated_questions.append(f"[Free-form] {free_question}")
            st.session_state.question_threads[new_thread_idx] = []
            st.session_state.current_question_thread = new_thread_idx

            # Add user message
            st.session_state.question_threads[new_thread_idx].append({
                "speaker": "user",
                "message": free_question,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

            # Get answer
            with st.spinner(f"{current_char_info['name']} is thinking..."):
                try:
                    answer = asyncio.run(
                        dialogue.answer_question(
                            st.session_state.current_character,
                            free_question,
                            conversation_history=st.session_state.question_threads[new_thread_idx]
                        )
                    )

                    st.session_state.question_threads[new_thread_idx].append({
                        "speaker": current_char_info["name"],
                        "message": answer,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })

                    st.rerun()
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())


# Kept for backward compatibility but not used anymore
def run_interactive_dialogue_old(session_id, artifacts):
    """OLD VERSION - kept for reference, not used."""
    st.warning("This is the old dialogue system. Using new simplified version above.")




def render_socratic_dialogue_tab():
    """Phase 6: Interactive Socratic dialogue."""
    st.subheader("🗣️ Socratic Dialogue")
    st.caption("Interactive conversation with research characters using taxonomy-ordered questions")

    # Check prerequisites
    final_proposal = st.session_state.get("final_proposal", "")
    artifacts = st.session_state.get("conversation_artifacts", [])

    if not final_proposal or not artifacts:
        st.warning("⚠️ Please complete Phase 4 (research execution) first.")
        return

    session_id = artifacts[0].get("session_id") if artifacts else None

    if not session_id:
        st.warning("⚠️ No session ID found.")
        return

    # Mode selector
    dialogue_mode = st.radio(
        "Dialogue Mode:",
        ["Interactive Chat (LangGraph)", "Simple Batch Mode"],
        index=0,
        help="Interactive: One question at a time with natural conversation. Simple: All questions upfront."
    )

    st.markdown("---")

    if dialogue_mode == "Interactive Chat (LangGraph)":
        run_interactive_dialogue(session_id, artifacts)
    else:
        # Keep existing simple batch implementation for backward compatibility
        render_simple_dialogue_mode(session_id, artifacts)


def render_simple_dialogue_mode(session_id, artifacts):
    """Simple batch mode - generate all questions upfront (backward compatible)."""
    dialogue_questions = st.session_state.get("dialogue_questions", [])

    if not dialogue_questions:
        st.info("Simple batch mode: Characters will generate all questions upfront based on dialogue notes.")

        final_proposal = st.session_state.get("final_proposal", "")

        # Show start button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Simple Dialogue", type="primary", use_container_width=True):
                with st.spinner("Generating questions from dialogue notes..."):
                    # Simple approach: just display dialogue notes from artifacts
                    all_questions = []
                    for artifact in artifacts:
                        character_id = artifact.get("character_id", "unknown")
                        domain = artifact.get("domain", "")
                        notes = artifact.get("dialogue_notes", [])

                        # Filter high-priority notes
                        high_priority_notes = [n for n in notes if n.get("priority", 0) >= 7]

                        # Sort by priority
                        high_priority_notes.sort(key=lambda x: x.get("priority", 0), reverse=True)

                        questions_text = "\n\n".join([
                            f"{i+1}. **Question:** {note.get('suggested_question', '')}\n   **Priority:** {note.get('priority', 0)}"
                            for i, note in enumerate(high_priority_notes[:5])
                        ])

                        all_questions.append({
                            "character_id": character_id,
                            "character_name": character_id,
                            "domain": domain,
                            "questions_text": questions_text,
                            "num_questions": len(high_priority_notes[:5])
                        })

                    st.session_state.dialogue_questions = all_questions
                    st.success(f"✓ Generated questions from {len(all_questions)} characters.")
                    st.rerun()
        return

    # Display generated questions
    st.markdown("### 💬 Questions from Research Characters")

    for char_idx, char_questions in enumerate(dialogue_questions):
        character_name = char_questions.get('character_name', 'Character')
        domain = char_questions.get('domain', '')
        questions_text = char_questions.get('questions_text', '')

        with st.expander(f"🎭 {character_name} ({domain})", expanded=True):
            st.markdown(questions_text)

            # User response area
            st.markdown("---")
            st.markdown("**💭 Your Response:**")

            user_response = st.text_area(
                "Provide your thoughts:",
                height=150,
                key=f"response_{char_idx}",
                placeholder="Share your thoughts on these questions..."
            )

            if user_response:
                st.session_state[f"responses_{char_idx}"] = user_response

    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Re-generate Questions", use_container_width=True):
            st.session_state.dialogue_questions = []
            st.rerun()

    with col2:
        if st.button("➡️ Proceed to Phase 7", type="primary", use_container_width=True):
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


def render_industry_review_tab():
    """Phase 7: Interactive industry partner dialogue."""
    st.subheader("🏢 Industry Partner Review")
    st.caption("Talk with industry partners about project fit, alignment, and cost")

    # Check prerequisites
    final_proposal = st.session_state.get("final_proposal", "")
    artifacts = st.session_state.get("conversation_artifacts", [])

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

    # Document upload section (always show)
    st.markdown("### 📄 Project Documentation (Optional)")
    st.caption("Upload project documents (PDF, DOCX) for context-aware answers. Works without documents too!")

    uploaded_files = st.file_uploader(
        "Upload Project Documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Industry partners will use these documents to ask context-aware questions",
        key="industry_docs_uploader"
    )

    # Process documents if uploaded
    if uploaded_files:
        from llmkglitrev.utils.document_rag import ProjectDocumentRAG

        try:
            rag = ProjectDocumentRAG(session_id)

            # Check if index already exists
            if rag.load_index():
                st.success(f"✓ Using existing document index ({rag.get_document_summary()['num_documents']} documents)")
            else:
                # Build new index
                if st.button("🔨 Build Document Index", key="build_index_btn"):
                    with st.spinner("Processing documents..."):
                        for uploaded_file in uploaded_files:
                            file_path = rag.docs_dir / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            rag.add_document(file_path, metadata={"uploaded_name": uploaded_file.name})

                        rag.build_index()
                        st.success(f"✓ Indexed {len(uploaded_files)} documents")
                        st.rerun()

            # Show document summary
            if rag.vector_store:
                summary = rag.get_document_summary()
                with st.expander("📊 Document Summary"):
                    st.markdown(f"**Total Documents:** {summary['num_documents']}")
                    for doc in summary['documents']:
                        st.markdown(f"- {doc['name']} ({doc['type']}) - {doc['length']:,} characters")

        except ImportError as e:
            st.error(f"Missing dependencies: {str(e)}")
            st.info("Install with: `pip install PyPDF2 python-docx langchain langchain-openai langchain-community faiss-cpu`")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

    st.markdown("---")

    # Initialize industry dialogue
    from llmkglitrev.agents.industry_dialogue import IndustryDialogue
    import asyncio
    from datetime import datetime

    if "industry_dialogue" not in st.session_state:
        with st.spinner("Loading industry partners..."):
            st.session_state.industry_dialogue = IndustryDialogue(session_id)
            st.session_state.industry_conversation_history = []
            st.session_state.current_partner = None
            st.session_state.generated_industry_questions = []
            # NEW: Question thread management
            st.session_state.industry_question_threads = {}  # {question_index: [conversation_history]}
            st.session_state.current_question_thread = None  # Currently selected question index

    dialogue = st.session_state.industry_dialogue

    # STEP 1: Partner Selection
    st.markdown("### 1️⃣ Select an Industry Partner to Talk With")

    available_partners = dialogue.get_available_partners()

    # Display partners with theme badges
    cols = st.columns(3)
    for idx, partner_info in enumerate(available_partners):
        col = cols[idx]
        with col:
            is_selected = st.session_state.current_partner == partner_info["partner_id"]

            # Theme emoji
            theme_emoji = {
                "project_fit": "🎯",
                "alignment": "🔗",
                "cost": "💰"
            }.get(partner_info["conversation_theme"], "👔")

            if st.button(
                f"{theme_emoji} **{partner_info['name']}**\n\n{partner_info['focus_area']}",
                key=f"select_partner_{partner_info['partner_id']}",
                type="primary" if is_selected else "secondary",
                use_container_width=True
            ):
                if st.session_state.current_partner != partner_info["partner_id"]:
                    st.session_state.current_partner = partner_info["partner_id"]
                    st.session_state.generated_industry_questions = []
                    st.session_state.industry_conversation_history = []
                    # NEW: Reset question threads when switching partners
                    st.session_state.industry_question_threads = {}
                    st.session_state.current_question_thread = None
                    st.rerun()

    if not st.session_state.current_partner:
        st.info("👆 Select a partner above to start the conversation")
        return

    # Get current partner info
    current_partner_info = next(
        (p for p in available_partners if p["partner_id"] == st.session_state.current_partner),
        None
    )

    if not current_partner_info:
        st.error("Partner not found")
        return

    st.markdown(f"**Current Partner:** {current_partner_info['name']}")
    st.caption(f"Focus: {current_partner_info['focus_area']}")

    st.markdown("---")

    # STEP 2: Generate Questions with Thread System
    with st.expander("🔍 Generate Starter Questions", expanded=not st.session_state.generated_industry_questions):
        if not st.session_state.generated_industry_questions:
            num_questions = st.slider("How many questions?", 1, 5, 3, key="num_industry_questions")

            if st.button("🔍 Generate Questions", type="primary", key="gen_industry_questions"):
                with st.spinner(f"{current_partner_info['name']} is reviewing the proposal..."):
                    questions = asyncio.run(
                        dialogue.generate_questions(
                            st.session_state.current_partner,
                            num_questions=num_questions
                        )
                    )
                    st.session_state.generated_industry_questions = questions
                    # Initialize thread for each question
                    st.session_state.industry_question_threads = {i: [] for i in range(len(questions))}
                    st.rerun()
        else:
            st.markdown("**Generated Questions - Click to Start Conversation:**")

            # Safety check: ensure industry_question_threads is initialized
            if "industry_question_threads" not in st.session_state:
                st.session_state.industry_question_threads = {i: [] for i in range(len(st.session_state.generated_industry_questions))}

            # Display questions as clickable cards
            for i, q in enumerate(st.session_state.generated_industry_questions):
                # Check if this question has an active thread
                has_conversation = len(st.session_state.industry_question_threads.get(i, [])) > 0
                is_current = st.session_state.current_question_thread == i

                # Question card
                col1, col2 = st.columns([5, 1])
                with col1:
                    # Button to select this question thread
                    button_label = f"{'📍' if is_current else '💬'} Q{i+1}: {q[:100]}{'...' if len(q) > 100 else ''}"
                    if has_conversation:
                        msg_count = len([m for m in st.session_state.industry_question_threads[i] if m.get('speaker') == 'user'])
                        button_label += f" ({msg_count} msg{'s' if msg_count != 1 else ''})"

                    if st.button(
                        button_label,
                        key=f"select_question_{i}",
                        type="primary" if is_current else "secondary",
                        use_container_width=True
                    ):
                        st.session_state.current_question_thread = i
                        st.rerun()

                with col2:
                    # Show status indicator
                    if is_current:
                        st.markdown("**🟢 Active**")
                    elif has_conversation:
                        st.markdown("✅ Discussed")
                    else:
                        st.markdown("⚪ New")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Regenerate All", key="regen_industry_questions"):
                    st.session_state.generated_industry_questions = []
                    st.session_state.industry_question_threads = {}
                    st.session_state.current_question_thread = None
                    st.rerun()
            with col2:
                if st.button("❌ Clear All", key="clear_industry_questions"):
                    st.session_state.generated_industry_questions = []
                    st.session_state.industry_question_threads = {}
                    st.session_state.current_question_thread = None
                    st.rerun()

    st.markdown("---")

    # STEP 3: Conversation Thread
    st.markdown("### 💬 Conversation")

    # DEBUG: Show current state
    with st.expander("🔧 Debug Info (click to expand)", expanded=False):
        st.write(f"current_question_thread: {st.session_state.get('current_question_thread', 'Not set')}")
        st.write(f"industry_question_threads keys: {list(st.session_state.get('industry_question_threads', {}).keys())}")
        st.write(f"Number of questions: {len(st.session_state.generated_industry_questions)}")

    # Check if a question thread is selected
    if st.session_state.current_question_thread is not None:
        thread_idx = st.session_state.current_question_thread
        current_question = st.session_state.generated_industry_questions[thread_idx]

        # Display the current question being discussed
        st.info(f"**Question {thread_idx + 1}:** {current_question}")

        # Get conversation history for this thread
        thread_history = st.session_state.industry_question_threads.get(thread_idx, [])

        # Display conversation history for this thread
        if thread_history:
            st.markdown("**Conversation History:**")
            for msg in thread_history:
                speaker = msg.get("speaker", "Unknown")
                message = msg.get("message", "")
                timestamp = msg.get("timestamp", "")

                if speaker == "user":
                    st.markdown(f"**You** *({timestamp})*")
                    st.info(message)
                else:
                    st.markdown(f"**{speaker}** *({timestamp})*")
                    st.success(message)

            st.markdown("---")

        # User input for this thread
        user_question = st.text_area(
            "Your follow-up or new question:",
            height=100,
            placeholder=f"Discuss Question {thread_idx + 1} with {current_partner_info['name']}...",
            key=f"industry_user_question_thread_{thread_idx}"
        )

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            if st.button("📤 Send", type="primary", disabled=not user_question.strip(), key=f"send_question_thread_{thread_idx}"):
                # Add user message to this thread
                st.session_state.industry_question_threads[thread_idx].append({
                    "speaker": "user",
                    "message": user_question,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })

                # Get partner's answer with thread context
                with st.spinner(f"{current_partner_info['name']} is thinking..."):
                    try:
                        # Build context: original question + thread history
                        context_messages = [
                            {"speaker": "system", "message": f"Original question: {current_question}"}
                        ] + st.session_state.industry_question_threads[thread_idx]

                        answer = asyncio.run(
                            dialogue.answer_question(
                                st.session_state.current_partner,
                                user_question,
                                conversation_history=context_messages
                            )
                        )

                        # Add partner's answer to this thread
                        st.session_state.industry_question_threads[thread_idx].append({
                            "speaker": current_partner_info["name"],
                            "message": answer,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })

                        st.rerun()

                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")

        with col2:
            if st.button("🔄 Clear Thread", key=f"clear_thread_{thread_idx}"):
                st.session_state.industry_question_threads[thread_idx] = []
                st.rerun()

        with col3:
            if st.button("⬅️ Back", key=f"back_from_thread_{thread_idx}"):
                st.session_state.current_question_thread = None
                st.rerun()

        with col4:
            if st.button("💾 Export", key=f"export_thread_{thread_idx}"):
                import json
                export_data = {
                    "session_id": session_id,
                    "partner": current_partner_info,
                    "question": current_question,
                    "question_index": thread_idx + 1,
                    "conversation": st.session_state.industry_question_threads[thread_idx],
                    "timestamp": datetime.now().isoformat()
                }
                st.download_button(
                    "📥 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"industry_thread_Q{thread_idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key=f"download_thread_{thread_idx}"
                )

    else:
        # No question selected - show prompt
        if st.session_state.generated_industry_questions:
            st.info("👆 Select a question above to start or continue a conversation thread")
        else:
            st.info("Generate questions above, then click on a question to start discussing it")

        # Free-form conversation option
        st.markdown("---")
        st.markdown("**Or ask a free-form question:**")

        free_question = st.text_area(
            "Your question:",
            height=100,
            placeholder=f"Ask {current_partner_info['name']} anything about {current_partner_info['conversation_theme'].replace('_', ' ')}...",
            key="industry_free_question"
        )

        if st.button("📤 Send Free-Form Question", type="secondary", disabled=not free_question.strip(), key="send_free_question"):
            # Create a new thread for free-form question
            new_thread_idx = len(st.session_state.generated_industry_questions)
            st.session_state.generated_industry_questions.append(f"[Free-form] {free_question}")
            st.session_state.industry_question_threads[new_thread_idx] = []
            st.session_state.current_question_thread = new_thread_idx

            # Add user message
            st.session_state.industry_question_threads[new_thread_idx].append({
                "speaker": "user",
                "message": free_question,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

            # Get answer
            with st.spinner(f"{current_partner_info['name']} is thinking..."):
                try:
                    answer = asyncio.run(
                        dialogue.answer_question(
                            st.session_state.current_partner,
                            free_question,
                            conversation_history=st.session_state.industry_question_threads[new_thread_idx]
                        )
                    )

                    st.session_state.industry_question_threads[new_thread_idx].append({
                        "speaker": current_partner_info["name"],
                        "message": answer,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })

                    st.rerun()
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")


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
