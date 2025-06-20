import streamlit as st

from src.core.application import Application
from src.core.configuration import load_config

def _init():
    if "baseConfig" not in st.session_state:
        st.session_state.baseConfig = load_config()
    if "backend" not in st.session_state:
        st.session_state.backend = Application(config=st.session_state.baseConfig)
    if "currentThread" not in st.session_state:
        st.session_state.currentThread = 1

st.set_page_config(page_title="Homer")

def _display_conversation(thread_id: int):
    if thread_id is None:
        return
    
    for message in st.session_state.backend.get_messages(thread_id=thread_id):
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

def main():
    # Display current conversation
    _display_conversation(thread_id=st.session_state.currentThread)
    
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
                    for chunk in st.session_state.backend.stream_retrieval_graph(
                        query=query,
                        thread_id=st.session_state.currentThread
                    ):
                        streamed_text += chunk
                        response_placeholder.markdown(streamed_text)

            except Exception as e:
                error_message = f"Error processing query: {str(e)}"
                response_placeholder.markdown(error_message)

if __name__ == "__main__":
    _init()
    main()