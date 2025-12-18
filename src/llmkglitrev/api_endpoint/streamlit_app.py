import streamlit as st
import requests
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Research Proposal Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL_START = "http://localhost:8000/research/start"
API_URL_RESUME = "http://localhost:8000/research/resume"
API_TIMEOUT = 600  # Research agent may take longer

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # API status check
    st.subheader("API Status")
    try:
        response = requests.get("http://localhost:8000/docs", timeout=2)
        st.success("✅ API Connected")
    except:
        st.error("❌ API Disconnected")
        st.info("Make sure FastAPI is running on port 8000")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Info section
    st.subheader("ℹ️ About")
    st.markdown("""
    This system generates comprehensive research proposals using a multi-agent approach.
    
    **Features:**
    - Multi-agent research coordination
    - Web search for relevant literature
    - Parallel investigation of different aspects
    - Synthesized research proposals
    - Quality evaluation and refinement
    
    **How it works:**
    1. Supervisor coordinates research agents
    2. Agents investigate different aspects
    3. Web searches gather current literature
    4. Findings are evaluated and synthesized
    5. Final proposal is generated
    """)

# Main title
st.title("🔬 Research Proposal Generator")
st.markdown("Generate comprehensive research proposals using a multi-agent system that investigates different aspects and synthesizes findings.")
st.markdown("---")
st.info("⏱️ **Note:** Research proposals can take 3-10 minutes to generate as the system conducts web searches and coordinates multiple AI agents.", icon="ℹ️")

# ===== FUNCTION DEFINITIONS =====

# Function to start research
def start_research(prompt: str, thread_id: Optional[str] = None) -> Optional[dict]:
    """Start a new research workflow."""
    try:
        response = requests.post(
            API_URL_START,
            json={"query": prompt, "thread_id": thread_id},
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.Timeout:
        return {"status": "error", "error": "⚠️ Request timed out. The research agent took too long to respond (>10 minutes)."}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "❌ Could not connect to the API. Make sure the FastAPI server is running on port 8000."}
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        return {"status": "error", "error": f"❌ API returned an error: {error_detail}"}
    except Exception as e:
        return {"status": "error", "error": f"❌ An error occurred: {str(e)}"}

# Function to resume research with feedback
def resume_research(thread_id: str, feedback: str) -> Optional[dict]:
    """Resume an interrupted research workflow with human feedback."""
    try:
        response = requests.post(
            API_URL_RESUME,
            json={"thread_id": thread_id, "feedback": feedback},
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.Timeout:
        return {"status": "error", "error": "⚠️ Request timed out."}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "❌ Could not connect to the API."}
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        return {"status": "error", "error": f"❌ API error: {error_detail}"}
    except Exception as e:
        return {"status": "error", "error": f"❌ Error: {str(e)}"}

def display_research_process(result: dict):
    """Display the complete research process with expandable sections."""
    
    # Check for errors
    if "error" in result:
        st.error(result["error"])
        return
    
    # 1. Display Keywords
    if result.get("research_keywords"):
        with st.expander("🔑 Step 1: Extracted Research Keywords", expanded=False):
            keywords = result["research_keywords"]
            st.write("Keywords used for literature search:")
            st.markdown("**Keywords:** " + ", ".join(f"`{kw}`" for kw in keywords))
    
    # 2. Display Retrieved Literature
    if result.get("retrieved_papers"):
        papers = result["retrieved_papers"]
        with st.expander(f"📚 Step 2: Retrieved Literature ({len(papers)} papers)", expanded=False):
            st.write("Relevant papers found in the literature database:")
            for i, paper in enumerate(papers, 1):
                with st.container():
                    st.markdown(f"**{i}. {paper.get('title', 'Untitled')}**")
                    if paper.get('authors'):
                        st.markdown(f"*Authors:* {paper['authors']}")
                    if paper.get('year'):
                        st.markdown(f"*Year:* {paper['year']}")
                    if paper.get('venue'):
                        st.markdown(f"*Venue:* {paper['venue']}")
                    if paper.get('abstract'):
                        abstract = paper['abstract']
                        if len(abstract) > 300:
                            abstract = abstract[:300] + "..."
                        st.markdown(f"*Abstract:* {abstract}")
                    if paper.get('score'):
                        st.markdown(f"*Relevance Score:* {paper['score']:.3f}")
                    st.markdown("---")
    
    # 3. Display Research Proposals from Sub-agents
    if result.get("research_proposals") and len(result["research_proposals"]) > 0:
        proposals = result["research_proposals"]
        with st.expander(f"🔬 Step 3: Sub-agent Research Proposals ({len(proposals)} proposals)", expanded=False):
            st.write("Individual research proposals from specialized agents:")
            for i, proposal in enumerate(proposals, 1):
                with st.container():
                    st.markdown(f"### Proposal {i}")
                    # Check if proposal is a dict (ResearchSummary) or string
                    if isinstance(proposal, dict):
                        st.markdown(f"**Topic:** {proposal.get('topic', 'N/A')}")
                        st.markdown(f"**Research Question:** {proposal.get('research_question', 'N/A')}")
                        st.markdown(f"**Method:** {proposal.get('method', 'N/A')}")
                        if proposal.get('disciplinaries'):
                            st.markdown(f"**Disciplines:** {', '.join(proposal['disciplinaries'])}")
                    else:
                        st.markdown(proposal)
                    st.markdown("---")
    
    # 4. Display Final Research Proposal
    if result.get("final_proposal"):
        with st.expander("📋 Step 4: Final Synthesized Research Proposal", expanded=True):
            st.markdown(result["final_proposal"])
    
    # Return a formatted summary for chat history
    summary = f"### Research Proposal Generated\n\n"
    if result.get("research_keywords"):
        summary += f"**Keywords:** {', '.join(result['research_keywords'])}\n\n"
    if result.get("retrieved_papers"):
        summary += f"**Literature Retrieved:** {len(result['retrieved_papers'])} papers\n\n"
    if result.get("research_proposals"):
        summary += f"**Sub-agent Proposals:** {len(result['research_proposals'])} proposals\n\n"
    summary += "**Final Proposal:** See expanded sections above"
    
    return summary

# ===== MAIN APP =====

# Initialize session state
import uuid as uuid_lib

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid_lib.uuid4())

if "waiting_for_feedback" not in st.session_state:
    st.session_state.waiting_for_feedback = False

if "interrupt_data" not in st.session_state:
    st.session_state.interrupt_data = None

# Check if we're waiting for feedback - show interrupt UI
if st.session_state.waiting_for_feedback and st.session_state.interrupt_data:
    st.info("🔔 **The research agent is requesting your feedback**", icon="ℹ️")
    
    interrupt_data = st.session_state.interrupt_data
    
    with st.container():
        st.markdown(f"### {interrupt_data.get('interrupt_question', 'Please provide feedback')}")
        
        if interrupt_data.get('interrupt_instructions'):
            st.markdown(f"*{interrupt_data['interrupt_instructions']}*")
        
        # Display proposals if available
        if interrupt_data.get('interrupt_proposals'):
            st.markdown("#### Proposed Research Directions:")
            for i, proposal in enumerate(interrupt_data['interrupt_proposals'], 1):
                st.markdown(f"{i}. {proposal}")
        
        st.markdown("---")
        
        # Feedback input
        feedback = st.text_area(
            "Your feedback:",
            height=150,
            placeholder="You can approve, modify, or provide specific guidance for each research direction...",
            key="feedback_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("✅ Submit Feedback", use_container_width=True):
                if feedback.strip():
                    # Resume research with feedback
                    with st.spinner("Processing your feedback and continuing research..."):
                        result = resume_research(st.session_state.thread_id, feedback)
                        
                        if result.get("status") == "interrupted":
                            # Another interrupt - update state
                            st.session_state.interrupt_data = result
                            st.rerun()
                        elif result.get("status") == "complete":
                            # Research complete
                            st.session_state.waiting_for_feedback = False
                            st.session_state.interrupt_data = None
                            
                            # Add to messages
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "Feedback received. Research completed!",
                                "result": result
                            })
                            st.rerun()
                        else:
                            # Error
                            st.error(result.get("error", "Unknown error"))
                else:
                    st.warning("Please provide feedback before submitting.")
        
        with col2:
            if st.button("🔄 Approve All", use_container_width=True):
                # Auto-approve
                with st.spinner("Approving and continuing research..."):
                    result = resume_research(st.session_state.thread_id, "Approved. Proceed with all proposed research directions.")
                    
                    if result.get("status") == "complete":
                        st.session_state.waiting_for_feedback = False
                        st.session_state.interrupt_data = None
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Research completed!",
                            "result": result
                        })
                        st.rerun()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "result" in message:
            # Display full research process for assistant messages with results
            display_research_process(message["result"])
        else:
            # Display simple text for user messages
            st.markdown(message["content"])

# Example prompts
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 Example Research Topics:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("LLMs for Literature Review", use_container_width=True):
            st.session_state.example_prompt = "How can Large Language Models improve literature review processes and knowledge graph construction?"
            st.rerun()
    
    with col2:
        if st.button("AI in Healthcare", use_container_width=True):
            st.session_state.example_prompt = "Using AI and machine learning to improve early disease detection in medical imaging"
            st.rerun()
    
    with col3:
        if st.button("Climate Change Mitigation", use_container_width=True):
            st.session_state.example_prompt = "Novel approaches to carbon capture and storage using machine learning optimization"
            st.rerun()

# Handle example prompt (only if not waiting for feedback)
if "example_prompt" in st.session_state and not st.session_state.waiting_for_feedback:
    prompt = st.session_state.example_prompt
    del st.session_state.example_prompt
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Start research
    with st.chat_message("assistant"):
        with st.spinner("🔬 Starting research... This may take 3-10 minutes..."):
            result = start_research(prompt, st.session_state.thread_id)
            
            if result.get("status") == "interrupted":
                # Store interrupt data and show feedback UI
                st.session_state.waiting_for_feedback = True
                st.session_state.interrupt_data = result
                st.info("Research paused - your feedback is needed!")
                st.rerun()
            elif result.get("status") == "complete":
                # Display complete results
                summary = display_research_process(result)
                st.session_state.messages.append({"role": "assistant", "content": summary, "result": result})
                st.rerun()
            else:
                # Error
                st.error(result.get("error", "Unknown error"))
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {result.get('error')}"})
                st.rerun()

# Chat input (only if not waiting for feedback)
if not st.session_state.waiting_for_feedback:
    if prompt := st.chat_input("Enter your research topic or question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Start research
        with st.chat_message("assistant"):
            with st.spinner("🔬 Starting research... This may take 3-10 minutes..."):
                result = start_research(prompt, st.session_state.thread_id)
                
                if result and result.get("status") == "interrupted":
                    # Store interrupt data and show feedback UI
                    st.session_state.waiting_for_feedback = True
                    st.session_state.interrupt_data = result
                    st.info("Research paused - your feedback is needed!")
                    st.rerun()
                elif result and result.get("status") == "complete":
                    # Display complete results
                    summary = display_research_process(result)
                    st.session_state.messages.append({"role": "assistant", "content": summary, "result": result})
                else:
                    # Error
                    error_msg = result.get("error", "Unknown error") if result else "No response from server"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
