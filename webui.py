import streamlit as st
from src.core import database as db
from src.core.application import Application
from src.core.configuration import load_config

def _init():
    if "baseConfig" not in st.session_state:
        st.session_state.baseConfig = load_config()
    if "backend" not in st.session_state:
        st.session_state.backend = Application(config=st.session_state.baseConfig)
    if "threads" not in st.session_state:
        st.session_state.threads = db.get_all_threads()
    # Only set currentThread if it doesn't exist or if there are no threads
    if "currentThread" not in st.session_state:
        st.session_state.currentThread = st.session_state.threads[-1] if st.session_state.threads else None

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

def _create_new_thread():
    """Create a new thread with a unique ID"""
    # Generate a new thread ID (you might want to use a different strategy)
    new_thread_id = st.session_state.threads[-1][0] + 1 if st.session_state.threads else 1
    db.new_thread(thread_id=new_thread_id, thread_name="New Thread")
    st.session_state.threads = db.get_all_threads()
    st.session_state.currentThread = st.session_state.threads[-1]

def _delete_current_thread():
    """Delete the current thread"""
    db.delete_thread(st.session_state.currentThread)
    st.session_state.threads = db.get_all_threads()
    
    # Set current thread to the last available thread, or None if no threads exist
    if st.session_state.threads:
        st.session_state.currentThread = st.session_state.threads[-1]
    else:
        st.session_state.currentThread = None

def _build_sidebar():
    st.sidebar.title("Chats")
    
    # Thread selection
    if st.session_state.threads:
        thread_options = [name for _, name in st.session_state.threads]
        thread_ids = [thread_id for thread_id, _ in st.session_state.threads]
        
        # Find current thread index
        current_idx = 0
        if st.session_state.currentThread in thread_ids:
            current_idx = thread_ids.index(st.session_state.currentThread)
        
        selected_idx = st.sidebar.selectbox(
            label="Select Thread:",
            options=range(len(thread_options)),
            format_func=lambda x: thread_options[x],
            index=current_idx,
            key="thread_selector"
        )
        
        # Update current thread when selection changes
        if selected_idx is not None:
            st.session_state.currentThread = st.session_state.threads[selected_idx][0]
    else:
        st.sidebar.write("No threads available")
    
    # Buttons
    if st.sidebar.button("New Thread", key="new_thread_btn"):
        _create_new_thread()
    
    if st.sidebar.button("Delete", key="delete_thread_btn", 
                disabled=(st.session_state.currentThread is None)):
        _delete_current_thread()

def _name_discussion(query: str):
    """Name the discussion based on the first query"""
    chat_name = st.session_state.backend.invoke_simple_query_graph(query=query)
    db.edit_thread_name(thread_id=st.session_state.currentThread, thread_name=chat_name)
    st.session.currentThread[1] = chat_name  # Update the current thread name in session state
    st.session_state.threads = db.get_all_threads()

def main():
    # Handle the case where we need to create a new thread
    if st.session_state.currentThread is None and st.session_state.threads:
        st.session_state.currentThread = st.session_state.threads[-1][0]
    elif st.session_state.currentThread is None:
        _create_new_thread()

    # Display the sidebar
    _build_sidebar()

    # Display current conversation
    if st.session_state.currentThread is not None:
        _display_conversation(st.session_state.currentThread)
    
    # Chat input
    query = st.chat_input("Enter your query:")
    if query:
        # If this is a new thread that hasn't been named yet, name it
        if st.session.currentThread[1] == "New Thread":
            _name_discussion(query)
        
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
            st.rerun(scope="scope")

if __name__ == "__main__":
    _init()
    print(f"[info] using configuration: {st.session_state.baseConfig}")
    print(f"[info] available threads: {st.session_state.threads}")
    print(f"[info] starting with thread: {st.session_state.currentThread}\n")
    main()