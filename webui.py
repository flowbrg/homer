import streamlit as st
from src.core import database as db
from src.core.application import Application
from src.core.configuration import load_config

def init():
    if "baseConfig" not in st.session_state:
        st.session_state.baseConfig = load_config()
    if "backend" not in st.session_state:
        st.session_state.backend = Application(config=st.session_state.baseConfig)
    if "threads" not in st.session_state:
        st.session_state.threads = db.get_all_threads()
    if "currentThread" not in st.session_state:
        st.session_state.currentThread = st.session_state.threads[-1][0] if st.session_state.threads else 1

st.set_page_config(page_title="Homer")


def _display_conversation(thread_id:int):
    for message in st.session_state.backend.get_messages(thread_id = thread_id):
        from langchain_core.messages import AIMessage
        from langchain_core.messages.human import HumanMessage
        print(message)
        if isinstance(message, AIMessage):
            name = "ai"
        if isinstance(message, HumanMessage):
            name = "human"
        with st.chat_message(name):
            st.markdown(message.content)


def _build_sidebar():
    st.sidebar.title("Chats")
    if st.session_state.threads:
        thread_options = [name for _, name in st.session_state.threads]
        selected_idx = st.sidebar.selectbox(
            label = "Select Thread:",
            options = range(len(thread_options)),
            format_func = lambda x: thread_options[x],
            accept_new_options=True,
        )
        st.session_state.currentThread = st.session_state.threads[selected_idx][0]


def main():
    _build_sidebar()
    _display_conversation(st.session_state.currentThread)

    query = st.chat_input("Enter your query:")
    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # Placeholder to stream the assistant response
            response_placeholder = st.empty()
            streamed_text = ""
            if st.session_state.currentThread not in [t[0] for t in st.session_state.threads]:
                chat_name = st.session_state.backend.invoke_simple_query_graph(query=query)
                db.new_thread(thread_id = st.session_state.currentThread, thread_name= chat_name)
                st.session_state.threads = db.get_all_threads()
                _build_sidebar()
            try:
                # Show spinner during processing
                with st.spinner("Processing your query..."):
                    # Stream chunks from the generator
                    for chunk in st.session_state.backend.stream_retrieval_graph(
                        query = query,
                        thread_id= st.session_state.currentThread
                    ):
                        streamed_text += chunk
                        response_placeholder.markdown(streamed_text)

            except Exception as e:
                error_message = f"Error processing query: {str(e)}"
                response_placeholder.markdown(error_message)

if __name__ == "__main__":
    init()
    print(f"[info] using configuration: {st.session_state.baseConfig}")
    print(f"[info] available threads: {st.session_state.threads}")
    print(f"[info] starting with thread: {st.session_state.currentThread}")
    print(f"[info] messages in current conversation: {st.session_state.backend.get_messages(st.session_state.currentThread)}")
    main()