"""
Example: Streamlit app with graph visualization capabilities.
This shows how to integrate visualizations with the chat interface.
"""

import streamlit as st
import requests
from typing import Optional
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

# Page configuration
st.set_page_config(
    page_title="Research Idea Quality Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://localhost:8000/chat"
API_TIMEOUT = 310

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

def create_sample_knowledge_graph():
    """Create a sample knowledge graph for demonstration."""
    G = nx.Graph()
    
    # Add nodes
    nodes = [
        "Research Idea",
        "Literature Review",
        "Knowledge Graph",
        "LLM",
        "Evaluation",
        "Metrics"
    ]
    G.add_nodes_from(nodes)
    
    # Add edges
    edges = [
        ("Research Idea", "Literature Review"),
        ("Research Idea", "Evaluation"),
        ("Literature Review", "Knowledge Graph"),
        ("Knowledge Graph", "LLM"),
        ("Evaluation", "Metrics"),
        ("LLM", "Metrics")
    ]
    G.add_edges_from(edges)
    
    return G

def plot_knowledge_graph(G):
    """Plot knowledge graph using plotly."""
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=20,
            color='#1976d2',
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
    )
    
    return fig

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # API status check
    st.subheader("API Status")
    try:
        response = requests.get(API_URL.replace("/chat", "/docs"), timeout=2)
        st.success("✅ API Connected")
    except:
        st.error("❌ API Disconnected")
        st.info("Make sure FastAPI is running on port 8000")
    
    st.markdown("---")
    
    # Visualization toggle
    st.subheader("📊 Visualization")
    show_graph = st.checkbox("Show Knowledge Graph", value=False)
    
    if show_graph:
        st.markdown("**Sample Knowledge Graph**")
        G = create_sample_knowledge_graph()
        st.metric("Nodes", G.number_of_nodes())
        st.metric("Edges", G.number_of_edges())
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", width='stretch'):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Info section
    st.subheader("ℹ️ About")
    st.markdown("""
    This assistant helps evaluate the quality of research ideas.
    
    **Features:**
    - Research idea evaluation
    - Quality assessment
    - Knowledge graph visualization
    - Improvement suggestions
    """)

# Main content area
if show_graph:
    # Two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("🔬 Research Idea Quality Assistant")
    
    with col2:
        st.markdown("### 📊 Knowledge Graph")
else:
    st.title("🔬 Research Idea Quality Assistant")

st.markdown("A helpful research assistant that judges the quality of research ideas.")
st.markdown("---")

# Show knowledge graph if enabled
if show_graph:
    with st.expander("🔍 View Knowledge Graph", expanded=True):
        G = create_sample_knowledge_graph()
        fig = plot_knowledge_graph(G)
        st.plotly_chart(fig, width='stretch')
        
        st.info("💡 This is a sample graph. Connect your actual knowledge graph data here!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Example prompts
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 Example Questions:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What makes a good research idea?", width='stretch'):
            st.session_state.example_prompt = "What makes a good research idea?"
            st.rerun()
    
    with col2:
        if st.button("Evaluate: Using LLMs for literature review", width='stretch'):
            st.session_state.example_prompt = "Can you evaluate this research idea: Using LLMs to automate literature reviews and knowledge graph construction"
            st.rerun()
    
    with col3:
        if st.button("Key criteria for research novelty?", width='stretch'):
            st.session_state.example_prompt = "What are the key criteria for evaluating research novelty?"
            st.rerun()

# Function to call API
def get_api_response(prompt: str) -> Optional[str]:
    """Call the FastAPI endpoint and return the response."""
    try:
        response = requests.get(
            API_URL,
            params={"prompt": prompt},
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. The API took too long to respond."
    except requests.exceptions.ConnectionError:
        return "❌ Could not connect to the API. Make sure the FastAPI server is running on port 8000."
    except requests.exceptions.HTTPError as e:
        return f"❌ API returned an error: {e.response.status_code}"
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
        with st.spinner("Thinking..."):
            response = get_api_response(prompt)
            st.markdown(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about research ideas..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_api_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
