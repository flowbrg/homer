import streamlit as st
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.messages.human import HumanMessage

# Import your existing components
from src.core.configuration import Configuration
from src.core.agents.retrival_graph import get_retrieval_graph

# Initialize session state
if 'retrieval_graph' not in st.session_state:
    st.session_state.retrieval_graph = get_retrieval_graph()

if 'config' not in st.session_state:
    st.session_state.config = Configuration()  # Assuming default constructor

if 'discussions' not in st.session_state:
    st.session_state.discussions = {}

if 'current_thread_id' not in st.session_state:
    st.session_state.current_thread_id = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Constants
DOCUMENTS_DIR = "uploaded_documents"
DISCUSSIONS_FILE = "discussions.json"

# Ensure directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

def load_discussions():
    """Load discussions from file"""
    if os.path.exists(DISCUSSIONS_FILE):
        with open(DISCUSSIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_discussions():
    """Save discussions to file"""
    with open(DISCUSSIONS_FILE, 'w') as f:
        json.dump(st.session_state.discussions, f, indent=2)

def create_new_discussion(title: str = None) -> str:
    """Create a new discussion thread"""
    thread_id = str(len(st.session_state.discussions) + 1)
    if not title:
        title = f"Discussion {thread_id}"
    
    st.session_state.discussions[thread_id] = {
        'title': title,
        'created_at': datetime.now().isoformat(),
        'messages': []
    }
    save_discussions()
    return thread_id

def get_thread_messages(thread_id: str) -> List[Dict]:
    """Get messages for a specific thread using the retrieval graph"""
    try:
        config = {"configurable": st.session_state.config.asdict() | {"thread_id": thread_id}}
        graph_state = st.session_state.retrieval_graph.get_state(config=config)
        messages = graph_state.values.get("messages", []) if graph_state.values else []
        return messages
    except Exception as e:
        st.error(f"Error retrieving messages: {e}")
        return []

def stream_response(query: str, thread_id: str):
    """Stream response from the retrieval graph"""
    try:
        config = {"configurable": st.session_state.config.asdict() | {"thread_id": thread_id}}
        
        # Create a placeholder for streaming content
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in st.session_state.retrieval_graph.stream(
            input=HumanMessage(query), 
            stream_mode="messages", 
            config=config
        ):
            message_chunk, metadata = chunk
            if message_chunk.content and metadata["langgraph_node"] == "respond":
                full_response += message_chunk.content
                message_placeholder.markdown(full_response + "▋")
        
        # Final message without cursor
        message_placeholder.markdown(full_response)
        
        return full_response
    except Exception as e:
        st.error(f"Error streaming response: {e}")
        return "Sorry, I encountered an error processing your request."

# Sidebar navigation
st.sidebar.title("💬 Claude-like Chat")

# Navigation
page = st.sidebar.selectbox(
    "Navigate",
    ["💬 Discussion", "📊 Report Creation", "📁 Add Documents"],
    key="navigation"
)

# Load discussions
st.session_state.discussions = load_discussions()

# DISCUSSION PAGE
if page == "💬 Discussion":
    st.title("💬 Discussion")
    
    # Discussion selection at the top
    col1, col2 = st.columns([3, 1])
    
    with col1:
        discussion_options = ["Create New Discussion"] + [
            f"{thread_id}: {info['title']}" 
            for thread_id, info in st.session_state.discussions.items()
        ]
        
        selected_discussion = st.selectbox(
            "Select or Create Discussion",
            discussion_options,
            key="discussion_selector"
        )
    
    with col2:
        if st.button("🗑️ Delete", help="Delete current discussion"):
            if st.session_state.current_thread_id and st.session_state.current_thread_id in st.session_state.discussions:
                del st.session_state.discussions[st.session_state.current_thread_id]
                save_discussions()
                st.session_state.current_thread_id = None
                st.rerun()
    
    # Handle discussion selection
    if selected_discussion == "Create New Discussion":
        # Show input for new discussion title
        new_title = st.text_input("Discussion Title (optional):", placeholder="Enter a title for your new discussion")
        if st.button("Create Discussion") or (new_title and len(new_title.strip()) > 0):
            thread_id = create_new_discussion(new_title.strip() if new_title else None)
            st.session_state.current_thread_id = thread_id
            st.rerun()
    else:
        # Extract thread_id from selection
        thread_id = selected_discussion.split(":")[0]
        st.session_state.current_thread_id = thread_id
    
    # Chat interface
    if st.session_state.current_thread_id:
        st.subheader(f"Discussion: {st.session_state.discussions[st.session_state.current_thread_id]['title']}")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            messages = get_thread_messages(st.session_state.current_thread_id)
            
            for message in messages:
                if hasattr(message, 'type'):
                    if message.type == "human":
                        with st.chat_message("user"):
                            st.write(message.content)
                    elif message.type == "ai":
                        with st.chat_message("assistant"):
                            st.write(message.content)
        
        # Chat input at the bottom
        if prompt := st.chat_input("Ask me anything..."):
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                response = stream_response(prompt, st.session_state.current_thread_id)
    else:
        st.info("👆 Please select or create a discussion to start chatting!")

# REPORT CREATION PAGE
elif page == "📊 Report Creation":
    st.title("📊 Report Creation")
    
    st.info("🔄 Report generation functionality is in development")
    
    # Report input
    report_query = st.text_area(
        "Enter your report requirements:",
        placeholder="Describe what kind of report you need...",
        height=150
    )
    
    if st.button("🚀 Generate Report", disabled=True):
        st.warning("Report generation is not implemented yet.")
        # Future implementation:
        # with st.spinner("Generating report..."):
        #     result = invoke_report_graph(report_query)
        #     st.success("Report generated successfully!")
        #     st.write(result)

# ADD DOCUMENTS PAGE
elif page == "📁 Add Documents":
    st.title("📁 Add Documents")
    
    # File upload section
    st.subheader("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'md', 'json', 'csv'],
        help="Drag and drop files here or click to browse"
    )
    
    if uploaded_files:
        st.write(f"📄 {len(uploaded_files)} file(s) selected:")
        
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size} bytes)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Files"):
                try:
                    saved_files = []
                    for file in uploaded_files:
                        file_path = os.path.join(DOCUMENTS_DIR, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        saved_files.append(file.name)
                    
                    st.success(f"✅ Saved {len(saved_files)} file(s) to {DOCUMENTS_DIR}")
                    for filename in saved_files:
                        st.write(f"- {filename}")
                        
                except Exception as e:
                    st.error(f"❌ Error saving files: {e}")
        
        with col2:
            if st.button("🔄 Process Documents", disabled=True):
                st.warning("Document processing is not implemented yet.")
                # Future implementation:
                # with st.spinner("Processing documents..."):
                #     populate_knowledge()
                #     st.success("Documents processed successfully!")
    
    # Show existing documents
    st.subheader("📚 Existing Documents")
    
    if os.path.exists(DOCUMENTS_DIR):
        files = os.listdir(DOCUMENTS_DIR)
        if files:
            st.write(f"Found {len(files)} document(s):")
            
            for file in files:
                file_path = os.path.join(DOCUMENTS_DIR, file)
                file_size = os.path.getsize(file_path)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"📄 {file}")
                
                with col2:
                    st.write(f"{file_size} bytes")
                
                with col3:
                    if st.button("🗑️", key=f"delete_{file}", help=f"Delete {file}"):
                        try:
                            os.remove(file_path)
                            st.success(f"Deleted {file}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting {file}: {e}")
        else:
            st.info("No documents found. Upload some documents to get started!")
    else:
        st.info("Documents directory not found. Upload some documents to create it!")
    
    # Clear all documents
    if st.button("🧹 Clear All Documents", type="secondary"):
        if st.session_state.get('confirm_clear', False):
            try:
                shutil.rmtree(DOCUMENTS_DIR)
                os.makedirs(DOCUMENTS_DIR, exist_ok=True)
                st.success("All documents cleared!")
                st.session_state.confirm_clear = False
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing documents: {e}")
        else:
            st.session_state.confirm_clear = True
            st.warning("⚠️ Click again to confirm clearing all documents!")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Status")
st.sidebar.info(f"Documents: {len(os.listdir(DOCUMENTS_DIR)) if os.path.exists(DOCUMENTS_DIR) else 0}")
st.sidebar.info(f"Discussions: {len(st.session_state.discussions)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("*Built with Streamlit & LangChain*")