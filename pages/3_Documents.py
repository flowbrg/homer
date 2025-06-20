import streamlit as st
import os

from pathlib import Path

from src.core.configuration import load_config
from src.core.application import Application
from src.core.agents.retrieval import delete_documents
from src.env import UPLOAD_DIR
from src.core.agents.retrieval import get_existing_documents

def _init():
    if "baseConfig" not in st.session_state:
        st.session_state.baseConfig = load_config()
    if "indexGraph" not in st.session_state:
        st.session_state.indexGraph = Application(config=st.session_state.baseConfig)

st.set_page_config(page_title="Documents")

st.markdown("# Documents")

# Define constants
# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


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

    if st.sidebar.button(label="update database",type="primary",use_container_width=True):
        with st.spinner("Updating database..."):
            try:
                st.session_state.backend.invoke_index_graph()
            except Exception as e:
                st.error(f"Error updating database: {str(e)}")
            else:
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
    """List all the uploaded files in the specified folder with delete option."""
          
    files = [f for f in get_existing_documents()]

    selected_files = st.multiselect(
        label = "Available files",
        options= [os.path.basename(os.path.realpath(f)) for f in files]
    )

if __name__ == "__main__":
    main()