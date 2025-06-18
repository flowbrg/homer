import streamlit as st
import os
from populate_database import populate
from populate_database import clear_database

st.set_page_config(page_title="Documents", page_icon="📄")

st.markdown("# Documents")
st.sidebar.header("Documents")

# Create a session state variable to track if the user is logged in
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
# Define constants
UPLOAD_DIR = "data"  # This can be changed later
# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Determine which access levels this user can access
accessible_levels = []
if st.session_state.level == "confidential":
    accessible_levels = ["confidential", "non_confidential"]
    info="With confidential access, you can manage both confidential and non-confidential documents."
else:
    accessible_levels = ["non_confidential"]
    info="With non-confidential access, you can only manage non-confidential documents."

def main():
    st.info(info)
    # Only show category selection for confidential users
    if st.session_state.level == "confidential":
        selected_level = st.selectbox("Select document category:", accessible_levels)
    else:
        selected_level = "non_confidential"
        st.write(f"Document category: **{selected_level.upper()}**")

    # Create the file uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Check if the file is a PDF
        if uploaded_file.name.lower().endswith('.pdf') and st.button(label="upload",type="primary"):
            # Make sure the target directory exists
            target_dir = os.path.join(UPLOAD_DIR, selected_level)
            os.makedirs(target_dir, exist_ok=True)
            
            # Save the file to the selected category directory
            save_path = os.path.join(target_dir, uploaded_file.name)
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File {uploaded_file.name} has been saved to {selected_level} category")

    list_uploaded_files()

    if st.sidebar.button(label="update database",type="primary",use_container_width=True):
        with st.spinner("Updating database..."):
            try:
                populate()
            except Exception as e:
                st.error(f"Error updating database: {str(e)}")
            else:
                st.success("Database has been updated.")

    if st.sidebar.button(label="reset database",type="primary",use_container_width=True):
        with st.spinner("Updating database..."):
            try:
                clear_database()
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
            else:
                st.success("Database has been cleared.")

def list_uploaded_files():
    """List all the uploaded files in the specified folder with delete option."""
    for level in accessible_levels:
        path = os.path.join(UPLOAD_DIR, level)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)  # Create directory if it doesn't exist
            st.info(f"No documents found in {level} category.")
            continue  # Continue to the next level instead of returning
            
        files = [f for f in os.listdir(path) if f.lower().endswith(".pdf")]

        if files:
            st.subheader(f"{level.title()} Documents:")
            for file in files:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.write(f"📄 {file}")
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{level}_{file}"):
                        try:
                            file_path = os.path.join(path, file)
                            os.remove(file_path)
                            # Mark that documents have changed
                            st.session_state.docs_changed = True
                            
                            st.success(f"Deleted {file}")
                            st.warning("⚠️ Database update required to reflect this change in searches.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {str(e)}")
        else:
            st.info(f"No documents found in {level} category.")

if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.error("You are not logged in")
    elif not st.session_state.logged_in:
        st.error("You are not logged in")
    else:
        main()