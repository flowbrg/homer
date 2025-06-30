"""
Streamlit Discussion Interface for Conversational AI Retrieval System.

This module provides a web-based chat interface that supports real-time streaming
of AI responses, including separate display of thinking processes and final answers.
The interface allows users to configure server connections and model settings.
"""

import streamlit as st

from src.utils.utils import is_connected, extract_think_and_answer
from src.core.agents import RetrievalAgent
from src.core.configuration import load_config
from src.env import OLLAMA_CLIENT


def _init():
    """
    Initialize Streamlit application and session state variables.
    
    Sets up the page configuration and initializes required session state:
    - baseConfig: Application configuration loaded from file
    - retrievalAgent: AI agent for processing queries
    - currentThread: Current conversation thread ID
    """
    st.set_page_config(page_title="Discussion")

    if "baseConfig" not in st.session_state:
        st.session_state.baseConfig = load_config()
    if "retrievalAgent" not in st.session_state:
        st.session_state.retrievalAgent = RetrievalAgent()
    if "currentThread" not in st.session_state:
        st.session_state.currentThread = 1


def _display_conversation(thread_id: int):
    """
    Display conversation history for the specified thread.
    
    Shows all messages in the current thread with proper formatting:
    - Human messages: displayed as user chat messages
    - AI messages: displayed as assistant chat messages with thinking expanders
    - System messages: displayed as system chat messages
    
    Args:
        thread_id: The conversation thread to display
    """
    if thread_id is None:
        return
    
    messages = st.session_state.retrievalAgent.get_messages(
        configuration=st.session_state.baseConfig,
        thread_id=st.session_state.currentThread,
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


def _is_ollama_client_available(url: str) -> bool:
    """
    Check if Ollama server is available at the given URL.
    
    Args:
        url: The Ollama server URL to test
        
    Returns:
        True if server responds successfully, False otherwise
    """
    import requests
    try:
        response = requests.get(url, timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def _build_sidebar():
    """
    Build sidebar interface for configuration options.
    
    Provides controls for:
    - Server execution toggle (local vs remote)
    - Model selection toggle (thinking vs standard models)
    - Connection status display
    - Current model display
    """
    # Server connection toggle
    connectionButton = st.sidebar.toggle(
        label="Server execution",
        value=is_connected(st.session_state)
    )

    # Configure server host based on connection preference
    if connectionButton:
        conn = _is_ollama_client_available(OLLAMA_CLIENT)
        if conn:
            st.session_state.baseConfig.ollama_host = OLLAMA_CLIENT
        else:
            st.sidebar.warning(f"Could not connect to {OLLAMA_CLIENT}")
            st.session_state.baseConfig.ollama_host = "http://127.0.0.1:11434/"
    else:
        st.session_state.baseConfig.ollama_host = "http://127.0.0.1:11434/"
    
    st.sidebar.write(f"Connected to: {st.session_state.baseConfig.ollama_host}")
        
    # Model selection toggle
    reasoningModelButton = st.sidebar.toggle(label="Thinking model")

    # Configure model based on server type and thinking preference
    if reasoningModelButton and st.session_state.baseConfig.ollama_host == "http://127.0.0.1:11434/":
        st.session_state.baseConfig.response_model = "qwen3:0.6b"
    elif not reasoningModelButton and st.session_state.baseConfig.ollama_host == "http://127.0.0.1:11434/":
        st.session_state.baseConfig.response_model = "gemma3:1b"
    elif reasoningModelButton and st.session_state.baseConfig.ollama_host == OLLAMA_CLIENT:
        st.session_state.baseConfig.response_model = "qwen3:30b-a3b"
    else:
        st.session_state.baseConfig.response_model = "gemma3:12b"

    st.sidebar.write(f"using model {st.session_state.baseConfig.response_model}")


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
    if st.session_state.baseConfig.response_model == "qwen3:30b-a3b" or st.session_state.baseConfig.response_model == "qwen3:0.6b":
        thinking_placeholder = st.expander("Show Thinking")
    response_placeholder = st.empty()
    
    try:
        # Stream response chunks
        for chunk in st.session_state.retrievalAgent.stream(
            query=query,
            configuration=st.session_state.baseConfig,
            thread_id=st.session_state.currentThread,
        ):
            accumulated_text += chunk
            
            # Check if thinking is complete
            if "</think>" in accumulated_text.lower():
                # Extract thinking and reset the accumulated text to get the actual answer (only one thinking part)
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


def _build_chat_input():
    """
    Build chat input interface with streaming response handling.
    
    Creates the chat input widget and processes user queries with:
    - Immediate display of user messages
    - Real-time streaming of AI responses
    - Separation of thinking content and final answers
    - Error handling with user feedback
    """
    query = st.chat_input("Enter your query:")
    if query:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(query)

        # Display assistant response with streaming thinking separation
        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                _stream_with_thinking_separation(query)


if __name__ == "__main__":
    _init()
    
    # Display current conversation
    _display_conversation(thread_id=st.session_state.currentThread)

    # Build sidebar controls
    _build_sidebar()

    # Build chat input interface
    _build_chat_input()