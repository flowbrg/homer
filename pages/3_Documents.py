import streamlit as st
import os

from pathlib import Path

from src.core.configuration import load_config
from src.core.agents import IndexAgent
from src.core.retrieval import delete_documents, get_existing_documents
from src.env import UPLOAD_DIR, OLLAMA_CLIENT
from src.core.retrieval import get_existing_documents

def _init():
    if "baseConfig" not in st.session_state:
        st.session_state.baseConfig = load_config()
    if "indexAgent" not in st.session_state:
        st.session_state.indexAgent = IndexAgent()

st.set_page_config(page_title="Documents")

st.markdown("# Documents")

# Define constants
# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _is_ollama_client_available(url: str) -> bool:
    import requests
    try:
        response = requests.get(url, timeout=2)
        return response.ok
    except requests.RequestException:
        return False

def _build_sidebar():
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

def _list_uploaded_files():
    """List all the uploaded files in the specified folder with delete option."""
                
    files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]

    if files:
        st.subheader("Uploaded documents:")
        for file in files:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"📄 {file}")
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{file}"):
                    try:
                        file_path = os.path.join(UPLOAD_DIR, file)
                        os.remove(file_path)                        
                        st.success(f"Deleted {file}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {str(e)}")
    else:
        st.info(f"No documents uploaded yet.")

def _reset_vector_store():
    with st.spinner("Updating database..."):
            try:
                documents = get_existing_documents()
                delete_documents(docs=documents)
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
            else:
                st.success("Database has been cleared.")
def main():
    # Create the file uploader
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files is not None and len(uploaded_files) > 0:
        
        # Upload button
        if st.button(label="Upload", type="primary"):
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
            
            # Show results
            if success_count > 0:
                st.success(f"Successfully uploaded {success_count} file(s)")
            if error_count > 0:
                st.warning(f"Failed to upload {error_count} file(s)")
    
    # List uploaded files (assuming this function exists)
    _list_uploaded_files()

    if st.sidebar.button(label="update database",type="primary",use_container_width=True):
        with st.spinner("Updating database..."):
            try:
                st.session_state.indexAgent.invoke(path = UPLOAD_DIR)
            except Exception as e:
                st.error(f"Error updating database: {str(e)}")
            
            for file in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, file)
                os.remove(file_path)
            st.success("Database has been updated.")

    resetVectorStoreButton = st.sidebar.button(
        label="reset database",
        type="primary",
        use_container_width=True
    )
        

    
def list_documents():
    files = [f for f in get_existing_documents()]

    selected_files = st.multiselect(
        label = "Available files",
        options= [os.path.basename(os.path.realpath(f)) for f in files]
    )

if __name__ == "__main__":
    _init()
    _build_sidebar()
    main()