import streamlit as st
from typing import Optional
import asyncio
from dotenv import load_dotenv
load_dotenv()
from llmkglitrev.agents.research_supervisor import create_interactive_supervisor

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Page configuration
st.set_page_config(
    page_title="Interactive Research Agent",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Interactive Research Proposal Generator")
st.markdown("Generate research proposals with interactive feedback")
st.markdown("---")

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session-1"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None
if "awaiting_feedback" not in st.session_state:
    st.session_state.awaiting_feedback = False
if "proposed_research" not in st.session_state:
    st.session_state.proposed_research = []

# Create agent with checkpointer for interrupts
agent = create_interactive_supervisor()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show proposed research if awaiting feedback
if st.session_state.awaiting_feedback:
    st.warning("🤔 Agent is proposing research directions. Please review and provide feedback below:")
    
    if st.session_state.proposed_research:
        st.markdown("**Proposed Research Topics:**")
        for i, topic in enumerate(st.session_state.proposed_research, 1):
            st.markdown(f"{i}. {topic}")
    
    feedback = st.text_area(
        "Your feedback:",
        placeholder="Type 'approve' to proceed, or suggest modifications/additional topics...",
        key="feedback_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Submit Feedback", type="primary"):
            if feedback:
                # Resume the agent with feedback
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"Feedback: {feedback}"
                })
                
                async def resume_agent():
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    
                    # Stream the agent execution
                    async for event in agent.astream(
                        Command(resume=feedback),
                        config,
                        stream_mode="values"
                    ):
                        # Check if interrupted again or completed
                        if hasattr(event, '__interrupted__'):
                            interrupt_data = event.__interrupted__
                            st.session_state.proposed_research = interrupt_data.get("proposed_research", [])
                            st.session_state.awaiting_feedback = True
                            return
                    
                    # Get final result
                    final_state = agent.get_state(config)
                    if final_state.values.get("research_proposals"):
                        proposals = final_state.values["research_proposals"]
                        result = "\n\n".join(proposals)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result
                        })
                        st.session_state.awaiting_feedback = False
                
                with st.spinner("Processing your feedback and conducting research..."):
                    try:
                        asyncio.run(resume_agent())
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.session_state.awaiting_feedback = False
            else:
                st.warning("Please provide feedback before submitting.")
    
    with col2:
        if st.button("Skip Feedback (Use Proposals As-Is)"):
            feedback = "approve"
            st.session_state.messages.append({
                "role": "user", 
                "content": "Approved proposals as-is"
            })
            st.rerun()

# Chat input (only show if not awaiting feedback)
elif prompt := st.chat_input("Enter your research topic or question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Run agent
    with st.chat_message("assistant"):
        async def start_agent():
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Start agent execution
            events = []
            async for event in agent.astream(
                {"supervisor_messages": [HumanMessage(content=prompt)]},
                config,
                stream_mode="values"
            ):
                events.append(event)
                
                # Check if interrupted for feedback
                state = agent.get_state(config)
                if state.next and "get_human_feedback" in state.next:
                    # Extract proposed research from state's pending_proposals
                    pending_proposals = state.values.get("pending_proposals", [])
                    
                    if pending_proposals:
                        st.session_state.proposed_research = pending_proposals
                        st.session_state.awaiting_feedback = True
                        return
        
        with st.spinner("🔬 Analyzing and proposing research directions..."):
            try:
                asyncio.run(start_agent())
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    
    if st.button("🗑️ New Conversation", width='stretch'):
        st.session_state.messages = []
        st.session_state.thread_id = f"streamlit-session-{len(st.session_state.messages)}"
        st.session_state.awaiting_feedback = False
        st.session_state.proposed_research = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### How it works:
    1. 📝 Enter research topic
    2. 🤖 Agent proposes research directions
    3. 👤 You provide feedback
    4. 🔍 Agent conducts approved research
    5. 📊 Final proposal generated
    """)
