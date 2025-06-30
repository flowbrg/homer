import streamlit as st
from dataclasses import fields
from typing import get_origin, get_args
import requests

# Assuming your Configuration class is imported
from src.core.configuration import Configuration, load_config

def validate_url(url: str) -> bool:
    """Validate if the URL is reachable."""
    try:
        response = requests.get(url, timeout=2)
        return response.ok
    except requests.RequestException:
        return False

def init_config():
    """Initialize configuration in session state if not present."""
    if 'baseConfig' not in st.session_state:
        st.session_state.baseConfig = load_config()

def render_config_editor():
    """Render the configuration editor interface."""
    st.title("🔧 Configuration Editor")
    st.markdown("Configure your indexing and retrieval settings below.")
    
    # Initialize config
    init_config()
    config = st.session_state.baseConfig
    
    # Create form for configuration
    with st.form("config_form"):
        st.subheader("📊 Report Configuration")
        
        # Number of parts
        number_of_parts = st.number_input(
            "Number of Report Parts",
            min_value=1,
            max_value=20,
            value=config.number_of_parts,
            help="The number of parts in the report outline. Must be a positive integer."
        )
        
        st.subheader("🔗 Ollama Configuration")
        
        # Ollama host
        ollama_host = st.text_input(
            "Ollama Host URL",
            value=config.ollama_host,
            help="The host URL for the Ollama service. Must be a valid URL."
        )
        
        # Test connection button (outside form to avoid form submission)
        col1, col2 = st.columns([3, 1])
        with col1:
            if ollama_host:
                if ollama_host.startswith(('http://', 'https://')):
                    st.success("✅ Valid URL format")
                else:
                    st.warning("⚠️ URL should start with http:// or https://")
        
        st.subheader("🤖 Model Configuration")
        
        # Model configurations
        embedding_model = st.text_input(
            "Embedding Model",
            value=config.embedding_model,
            help="Model used for generating embeddings"
        )
        
        response_model = st.text_input(
            "Response Model",
            value=config.response_model,
            help="Model used for generating responses"
        )
        
        query_model = st.text_input(
            "Query Model",
            value=config.query_model,
            help="Model used for processing queries"
        )
        
        outline_model = st.text_input(
            "Outline Model", 
            value=config.outline_model,
            help="Model used for generating outlines"
        )
        
        # Form submission buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            submitted = st.form_submit_button("💾 Save Configuration", type="primary")
        
        with col2:
            reset = st.form_submit_button("🔄 Reset to Defaults")
            
        with col3:
            test_connection = st.form_submit_button("🧪 Test Ollama Connection")
    
    # Handle form submissions
    if submitted:
        # Update configuration
        config.number_of_parts = number_of_parts
        config.ollama_host = ollama_host
        config.embedding_model = embedding_model
        config.response_model = response_model
        config.query_model = query_model
        config.outline_model = outline_model
        
        st.session_state.baseConfig = config
        st.success("✅ Configuration saved successfully!")
        st.rerun()
    
    if reset:
        # Reset to default configuration
        st.session_state.baseConfig = load_config()
        st.success("🔄 Configuration reset to defaults!")
        st.rerun()
    
    if test_connection:
        # Test Ollama connection
        with st.spinner("Testing connection..."):
            if validate_url(ollama_host):
                st.success(f"✅ Successfully connected to {ollama_host}")
            else:
                st.error(f"❌ Failed to connect to {ollama_host}")

def show_current_config():
    """Display current configuration in an expandable section."""
    if 'baseConfig' in st.session_state:
        with st.expander("📋 Current Configuration (JSON)", expanded=False):
            config_dict = st.session_state.baseConfig.asdict()
            st.json(config_dict)

def show_config_info():
    """Show information about configuration fields."""
    with st.expander("ℹ️ Configuration Field Information", expanded=False):
        st.markdown("""
        **Report Configuration:**
        - **Number of Parts**: Controls how many sections your report will have
        
        **Ollama Configuration:**
        - **Host URL**: The endpoint where your Ollama service is running
        
        **Model Configuration:**
        - **Embedding Model**: Used to convert text into vector representations
        - **Response Model**: Generates the final responses to queries
        - **Query Model**: Processes and understands user queries
        - **Outline Model**: Creates structured outlines for reports
        
        **Common Models:**
        - `nomic-embed-text`: Good for embeddings
        - `gemma3:1b`: Lightweight model for various tasks
        - `qwen3:0.6b`: Alternative lightweight model
        """)

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Configuration Editor",
        page_icon="⚙️",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stForm {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    render_config_editor()
    
    # Show additional information
    col1, col2 = st.columns(2)
    with col1:
        show_current_config()
    with col2:
        show_config_info()

if __name__ == "__main__":
    main()