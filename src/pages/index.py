import streamlit as st
import os

from pathlib import Path

from core.configuration import load_config
from core.agents import IndexAgent
from core.retrieval import delete_documents, get_existing_documents
from constant import UPLOAD_DIR, OLLAMA_LOCALHOST
from core.retrieval import get_existing_documents
from pages.utils import is_ollama_client_available, is_connected


############################## Initialize session state ##############################


st.set_page_config(
    page_title="Documents",
    layout="centered",
)

if "baseConfig" not in st.session_state:
    st.session_state.baseConfig = load_config()
if "indexAgent" not in st.session_state:
    st.session_state.indexAgent = IndexAgent()
if "ollama_host" not in st.session_state:
    from constant import OLLAMA_CLIENT
    st.session_state.ollama_host = OLLAMA_CLIENT

st.markdown("# Documents")

# Define constants
# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


############################## Private methods ##############################


def _reset_vector_store():
    with st.spinner("Updating database..."):
            try:
                documents = get_existing_documents()
                for doc in documents:
                    delete_documents(docs=doc)
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
            else:
                st.success("Database has been cleared.")


def _process_files(uploaded_files):
    success_count = 0
    error_count = 0
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        try:
            # Check if the file is a PDF
            if uploaded_file.name.lower().endswith('.pdf'):
                # Save the file to the selected category directory
                save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                success_count += 1
            else:
                st.warning(f"Skipped {uploaded_file.name} - not a PDF file")
                error_count += 1
                
        except Exception as e:
            st.error(f"Error saving {uploaded_file.name}: {str(e)}")
            error_count += 1

    with st.spinner("Updating database..."):
        try:
            st.session_state.indexAgent.invoke(path = UPLOAD_DIR, configuration = st.session_state.baseConfig)
        except Exception as e:
            st.error(f"Error updating database: {str(e)}")
        
        for file in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, file)
            os.remove(file_path)
        st.success("Database has been updated.")
    
    # Show results
    if error_count > 0:
        st.warning(f"Failed to upload {error_count} file(s)")


############################## Sidebar ##############################


conn = is_ollama_client_available(st.session_state.ollama_host)

connectionButton = st.sidebar.toggle(
    label = "Server execution",
    value = is_connected(st.session_state),
    key="indexConnectionButton",
)

if connectionButton:
    if conn:
        st.session_state.baseConfig.ollama_host=st.session_state.ollama_host
    else:
        st.sidebar.warning(f"Could not connect to {st.session_state.ollama_host}")
        st.session_state.baseConfig.ollama_host=OLLAMA_LOCALHOST
else:
    st.session_state.baseConfig.ollama_host=OLLAMA_LOCALHOST

st.sidebar.write(f"Connected to: {st.session_state.baseConfig.ollama_host}")

visionParserButton = st.sidebar.toggle(
    label = "Vision Parser",
    value = st.session_state.baseConfig.ocr,,
    key="indexVisionButton",
    help="Enable or disable the vision analysis for PDF documents (only available on the server)",
    disabled=not (connectionButton and conn), # Disable if not connected or Ollama client is not available
)

if visionParserButton:
    st.session_state.baseConfig.ocr = True
else:
    st.session_state.baseConfig.ocr = False


############################## Page ##############################


# Create the file uploader
uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files is not None and len(uploaded_files) > 0:
    
    # Upload button
    uploadButton=st.button(
        label="Upload",
        type="primary",
        #on_click=_process_files(uploaded_files)
    )
    if uploadButton:
        _process_files(uploaded_files)


# Reset button
st.button(
    label="🗑️ Reset database",
    type="primary",
    use_container_width=True,
    on_click=_reset_vector_store
)

# Display existing documents
files = [f for f in get_existing_documents()]
for file in files:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.write(f"📄 {Path(file).stem}")
        with col2:
            if st.button("🗑️ Delete", key=f"delete_{file}"):
                try:
                    delete_documents(docs = file)
                except Exception as e:
                    st.error(f"Error deleting document: {str(e)}")
                st.rerun()