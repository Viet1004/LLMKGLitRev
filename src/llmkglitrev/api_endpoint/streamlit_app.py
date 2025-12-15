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
API_URL = "http://localhost:8000/research"
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
        response = requests.get(API_URL.replace("/research", "/docs"), timeout=2)
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
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

# Function to call API
def get_api_response(prompt: str) -> Optional[str]:
    """Call the FastAPI endpoint and return the response."""
    try:
        response = requests.post(
            API_URL,
            json={"query": prompt},
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        return result["final_proposal"]
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. The research agent took too long to respond (>10 minutes)."
    except requests.exceptions.ConnectionError:
        return "❌ Could not connect to the API. Make sure the FastAPI server is running on port 8000."
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        return f"❌ API returned an error: {error_detail}"
    except Exception as e:
        return f"❌ An error occurred: {str(e)}"

# Handle example prompt
if "example_prompt" in st.session_state:
    prompt = st.session_state.example_prompt
    del st.session_state.example_prompt
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔬 Generating research proposal... This may take 3-10 minutes..."):
            response = get_api_response(prompt)
            st.markdown(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Chat input
if prompt := st.chat_input("Enter your research topic or question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔬 Generating research proposal... This may take 3-10 minutes..."):
            response = get_api_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
