"""
Streamlit Discussion Interface for Conversational AI Retrieval System.

This module provides a web-based chat interface that supports real-time streaming
of AI responses, including separate display of thinking processes and final answers.
The interface allows users to configure server connections and model settings.
"""

import streamlit as st

from utils.utils import extract_think_and_answer
from pages.utils import is_ollama_client_available, is_connected
from core.agents import RetrievalAgent
from core.configuration import load_config
from constant import OLLAMA_LOCALHOST


############################## Initialization ##############################


st.set_page_config(
    page_title="Discussion",
    layout="centered",
)

if "baseConfig" not in st.session_state:
    st.session_state.baseConfig = load_config() 
if "retrievalAgent" not in st.session_state:
    st.session_state.retrievalAgent = RetrievalAgent() # Instantiate the retrieval agent
if "ollama_host" not in st.session_state:
    from constant import OLLAMA_CLIENT
    st.session_state.ollama_host = OLLAMA_CLIENT

_THREAD = 1


############################## Private methods ##############################


def _stream_with_thinking_separation(query: str):
    """
    Stream AI response with thinking detection and answer streaming.
    
    This function:
    1. Accumulates the full response while detecting thinking content
    2. Shows expander with thinking content when answer begins to stream
    3. Streams only the final answer in the main message area
    
    Args:
        query: The user query to process
    """
    accumulated_text = ""
    if (
        st.session_state.baseConfig.response_model == st.session_state.models["server_reasoning"]
        or st.session_state.baseConfig.response_model == st.session_state.models["local_reasoning"]
    ):
        thinking_placeholder = st.expander("Show Thinking")
    response_placeholder = st.empty()
    
    try:
        # Stream response chunks
        for chunk in st.session_state.retrievalAgent.stream(
            query=query,
            configuration=st.session_state.baseConfig,
            thread_id=_THREAD,
        ):
            accumulated_text += chunk
            
            # Check if thinking is complete
            if "</think>" in accumulated_text.lower():
                # Extract thinking and reset the accumulated text to get the actual answer
                # (only one thinking part)
                thinking_content, accumulated_text = extract_think_and_answer(accumulated_text)

                # Show expander with thinking content if it exists
                if thinking_content:
                    thinking_placeholder.markdown(thinking_content)
                
                # Start streaming the answer part
                response_placeholder.markdown(accumulated_text)

            elif "<think>" not in accumulated_text.lower():
                # No thinking content detected, stream normally
                response_placeholder.markdown(accumulated_text)
                
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        response_placeholder.markdown(error_message)


############################## Page ##############################


# Retrieve existing messages
messages = st.session_state.retrievalAgent.get_messages(
    configuration=st.session_state.baseConfig,
    thread_id=_THREAD,
)

for message in messages:
    from langchain_core.messages import AIMessage
    from langchain_core.messages.human import HumanMessage
    
    if isinstance(message, AIMessage):
        name = "ai"
        # Extract thinking and response content
        thoughts, answer = extract_think_and_answer(message.content)
        
        with st.chat_message(name):
            # Display thinking content in expander if available
            if thoughts:
                with st.expander("Show thinking"):
                    st.write(thoughts)
            # Display the main response
            st.markdown(answer if answer else message.content)

    elif isinstance(message, HumanMessage):
        name = "human"
        with st.chat_message(name):
            st.markdown(message.content)
    else:
        name = "system"
        with st.chat_message(name):
            st.markdown(message.content)


# Display the chat input box for user queries
query = st.chat_input("Enter your query:")
if query:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    # Display assistant response with streaming thinking separation
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            _stream_with_thinking_separation(query)


############################## Sidebar ##############################


# Server connection toggle
connectionButton = st.sidebar.toggle(
    label="Server execution",
    value=is_connected(st.session_state)
)

# Configure server host based on connection preference
if connectionButton:
    conn = is_ollama_client_available(st.session_state.ollama_host)
    if conn:
        st.session_state.baseConfig.ollama_host = st.session_state.ollama_host
    else:
        st.sidebar.warning(f"Could not connect to {st.session_state.ollama_host}")
        st.session_state.baseConfig.ollama_host = OLLAMA_LOCALHOST
else:
    st.session_state.baseConfig.ollama_host = OLLAMA_LOCALHOST

# Display the current server connection status
st.sidebar.write(f"Connected to: {st.session_state.baseConfig.ollama_host}")
    
# Model selection toggle
reasoningModelButton = st.sidebar.toggle(
        label="Reasoning model",
    )

# Configure model based on server type and thinking preference
if reasoningModelButton and st.session_state.baseConfig.ollama_host == OLLAMA_LOCALHOST:
    st.session_state.baseConfig.response_model = st.session_state.models["local_reasoning"]

elif not reasoningModelButton and st.session_state.baseConfig.ollama_host == OLLAMA_LOCALHOST:
    st.session_state.baseConfig.response_model = st.session_state.models["local_standard"]

elif reasoningModelButton and st.session_state.baseConfig.ollama_host == st.session_state.ollama_host:
    st.session_state.baseConfig.response_model = st.session_state.models["server_reasoning"]

else:
    st.session_state.baseConfig.response_model = st.session_state.models["server_standard"]

# Display the currently selected model in the sidebar
st.sidebar.write(f"using model {st.session_state.baseConfig.response_model}")