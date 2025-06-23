import streamlit as st
import os

from pathlib import Path

from src.core.configuration import load_config
from src.core.application import Application
from src.core.agents.retrieval import delete_documents
from src.env import UPLOAD_DIR
from src.core.retrieval import get_existing_documents

def _init():
    if "baseConfig" not in st.session_state:
        st.session_state.baseConfig = load_config()
    if "backend" not in st.session_state:
        st.session_state.backend = Application(config=st.session_state.baseConfig)

st.set_page_config(page_title="Documents")

st.markdown("# Documents")

# Define constants
# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


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


def main():

    # Create the file uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Check if the file is a PDF
        if uploaded_file.name.lower().endswith('.pdf') and st.button(label="upload",type="primary"):
            
            # Save the file to the selected category directory
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File {uploaded_file.name} has been saved")

    _list_uploaded_files()

    if st.sidebar.button(label="update database",type="primary",use_container_width=True):
        with st.spinner("Updating database..."):
            try:
                st.session_state.backend.invoke_index_graph(path = UPLOAD_DIR)
            except Exception as e:
                st.error(f"Error updating database: {str(e)}")
            
            for file in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, file)
                os.remove(file_path)
            st.success("Database has been updated.")

    if st.sidebar.button(label="reset database",type="primary",use_container_width=True):
        with st.spinner("Updating database..."):
            try:
                delete_documents(docs="")
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
            else:
                st.success("Database has been cleared.")

def list_documents():
    files = [f for f in get_existing_documents()]

    selected_files = st.multiselect(
        label = "Available files",
        options= [os.path.basename(os.path.realpath(f)) for f in files]
    )

if __name__ == "__main__":
    _init()
    main()