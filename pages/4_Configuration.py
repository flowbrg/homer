import streamlit as st
from dataclasses import fields
from typing import get_origin, get_args
import requests

from src.core.configuration import Configuration, load_config






def validate_url(url: str) -> bool:
    """Validate if the URL is reachable."""
    try:
        response = requests.get(url, timeout=2)
        return response.ok
    except requests.RequestException:
        return False

def render_config_editor():
    """Render the configuration editor interface."""
    st.title("🔧 Configuration Editor")
    st.markdown("Configure your indexing and retrieval settings below.")
    
    config = st.session_state.baseConfig
    
    # Create form for configuration
    with st.form("config_form"):
        # Ollama host
        ollama_host = st.text_input(
            "Ollama Host URL",
            value=st.session_state.ollama_host,
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
    #    st.subheader("📊 Report Configuration")
    #    
    #    # Number of parts
    #    number_of_parts = st.number_input(
    #        "Number of Report Parts",
    #        min_value=1,
    #        max_value=20,
    #        value=config.number_of_parts,
    #        help="The number of parts in the report outline. Must be a positive integer."
    #    )
    #    
    #    st.subheader("🔗 Ollama Configuration")
    #    
    #    # Ollama host
    #    ollama_host = st.text_input(
    #        "Ollama Host URL",
    #        value=config.ollama_host,
    #        help="The host URL for the Ollama service. Must be a valid URL."
    #    )
    #    
    #    # Test connection button (outside form to avoid form submission)
    #    col1, col2 = st.columns([3, 1])
    #    with col1:
    #        if ollama_host:
    #            if ollama_host.startswith(('http://', 'https://')):
    #                st.success("✅ Valid URL format")
    #            else:
    #               st.warning("⚠️ URL should start with http:// or https://")
    #    
        st.subheader("Model Configuration")
       
        # Model configurations
        local_reasoning_model = st.text_input(
            "Local Reasoning Model",
            value=st.session_state.models["local_reasoning"],
            help="Reasoning model for local execution"
        )
        local_standard_model = st.text_input(
            "Local Reasoning Model",
            value=st.session_state.models["local_standard"],
            help="Standard model for local execution"
        )
        server_reasoning_model = st.text_input(
            "Local Reasoning Model",
            value=st.session_state.models["server_reasoning"],
            help="Reasoning model for server execution"
        )
        
        server_standard_model = st.text_input(
            "Local Reasoning Model",
            value=st.session_state.models["server_standard"],
            help="Reasoning model for server execution"
        )
        

        submitted = st.form_submit_button("💾 Save Configuration", type="primary")
    
    # Handle form submissions
    if submitted:
        # Update configuration
        st.session_state.models = {
        "server_reasoning": server_reasoning_model,
        "server_standard": server_standard_model,
        "local_reasoning": local_reasoning_model,
        "local_standard": local_standard_model,            
        }        
        st.session_state.ollama_host = ollama_host
        st.success("✅ Configuration saved successfully!")
        st.rerun()

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

if __name__ == "__main__":
    main()