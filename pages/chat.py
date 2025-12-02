import streamlit as st
from agent import agent_graph
from langchain.schema import HumanMessage

st.set_page_config(page_title="MediVision AI - Chat", page_icon="💬")

st.title("💬 Chat with MediVision AI")
st.markdown("Ask about symptoms, health advice, find pharmacies, doctors, etc.")

# User ID
user_id = st.sidebar.text_input("User ID", value="demo_user", key="user_id")

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat interface
user_input = st.text_input("Your message:", key="user_input")

if st.button("Send") and user_input:
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Invoke agent
    with st.spinner("Thinking..."):
        response = agent_graph.invoke({
            "messages": st.session_state.chat_history,
            "user_id": user_id
        })

    # Add agent response to history
    agent_response = response["messages"][-1].content
    st.session_state.chat_history.append(response["messages"][-1])

    # Display response
    st.write("**MediVision AI:**")
    st.write(agent_response)

# Display Chat History
if st.session_state.chat_history:
    st.header("Chat History")
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.write(f"**You:** {msg.content}")
        else:
            st.write(f"**MediVision AI:** {msg.content}")
