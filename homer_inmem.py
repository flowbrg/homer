import streamlit as st

from typing import Any

from langchain_core.messages.human import HumanMessage
from langchain_core.messages import AnyMessage

#from src.core.application import Application
from src.core.agents import RetrievalAgent
from src.core.configuration import load_config
from src.env import OLLAMA_CLIENT



def _init():
    if "baseConfig" not in st.session_state:
        st.session_state.baseConfig = load_config()
    #if "backend" not in st.session_state:
    #    st.session_state.backend = Application(config=st.session_state.baseConfig)
    if "retrievalAgent" not in st.session_state:
        st.session_state.retrievalAgent = RetrievalAgent()
    if "currentThread" not in st.session_state:
        st.session_state.currentThread = 1

st.set_page_config(page_title="Homer")

def _display_conversation(thread_id: int):
    if thread_id is None:
        return
    
    messages = st.session_state.retrievalAgent.get_messages(
        configuration = st.session_state.baseConfig,
        thread_id = st.session_state.currentThread,
    )

    for message in messages:
        from langchain_core.messages import AIMessage
        from langchain_core.messages.human import HumanMessage
        if isinstance(message, AIMessage):
            name = "ai"
        elif isinstance(message, HumanMessage):
            name = "human"
        else:
            name = "system"  # fallback for other message types
        
        with st.chat_message(name):
            st.markdown(message.content)

def _is_ollama_client_available(url: str) -> bool:
    import requests
    try:
        response = requests.get(url, timeout=2)
        return response.ok
    except requests.RequestException:
        return False

def main():
    # Display current conversation
    _display_conversation(thread_id=st.session_state.currentThread)
    
    # Sidebar button

    connectionButton = st.sidebar.toggle(
        label = "Server execution"
    )

    if connectionButton:
        conn = _is_ollama_client_available(OLLAMA_CLIENT)
        if conn:
            st.sidebar.write(f"using distant ollama client {OLLAMA_CLIENT}")
            st.session_state.baseConfig.ollama_host=OLLAMA_CLIENT
        else:
            st.sidebar.warning(f"Could not connect to {OLLAMA_CLIENT}")
            st.session_state.baseConfig.ollama_host="http://127.0.0.1:11434/"
    else:
        st.sidebar.write(f"using localhost")
        st.session_state.baseConfig.ollama_host="http://127.0.0.1:11434/"
        
    modelButton = st.sidebar.toggle(
        label = "Thinking model",
    )

    if modelButton and st.session_state.baseConfig.ollama_host == "http://127.0.0.1:11434/":
        st.session_state.baseConfig.response_model = "qwen3:0.6b"
    elif not modelButton and st.session_state.baseConfig.ollama_host == "http://127.0.0.1:11434/":
        st.session_state.baseConfig.response_model = "gemma3:1b"
    elif modelButton and st.session_state.baseConfig.ollama_host == OLLAMA_CLIENT:
        st.session_state.baseConfig.response_model = "qwen3:30b-a3b"
    else:
        st.session_state.baseConfig.response_model = "gemma3:12b"

    st.sidebar.write(f"using model {st.session_state.baseConfig.response_model}")


    # Chat input
    query = st.chat_input("Enter your query:")
    if query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            streamed_text = ""
            try:
                with st.spinner("Processing your query..."):
                    for chunk in st.session_state.retrievalAgent.stream(
                        query=query,
                        configuration=st.session_state.baseConfig,
                        thread_id=st.session_state.currentThread,
                    ):
                        streamed_text += chunk
                        response_placeholder.markdown(streamed_text)

            except Exception as e:
                error_message = f"Error processing query: {str(e)}"
                response_placeholder.markdown(error_message)

if __name__ == "__main__":
    _init()
    main()