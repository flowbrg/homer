import streamlit as st
import ollama

from core.configuration import load_config
from constant import OLLAMA_CLIENT


############################## Initialization ##############################


st.set_page_config(
        page_title="Configuration Editor",
        layout="centered"
    )

# Default values of the models for server/local execution and classic/reasoning
DEFAULT_MODELS = {
    "server_reasoning": "qwen3:30b-a3b",
    "server_standard": "gemma3:4b-it-qat", 
    "local_reasoning": "qwen3:0.6b",
    "local_standard": "gemma3n:e2b"
}

# Ensure session variables are instantiated

if "baseConfig" not in st.session_state:
    st.session_state.baseConfig = load_config()
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = OLLAMA_CLIENT
if "models" not in st.session_state:
    st.session_state.models = DEFAULT_MODELS.copy()


######################################## Form ########################################


st.title("Configuration Editor")
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
                st.success("Valid URL format")
            else:
                st.warning("URL should start with http:// or https://")    

    st.subheader("Model Configuration")
    
    # Model configurations
    local_reasoning_model = st.text_input(
        "Local Reasoning Model",
        value=st.session_state.models["local_reasoning"],
        help="Reasoning model for local execution"
    )
    local_standard_model = st.text_input(
        "Local Standard Model",
        value=st.session_state.models["local_standard"],
        help="Standard model for local execution"
    )
    server_reasoning_model = st.text_input(
        "Server Reasoning Model",
        value=st.session_state.models["server_reasoning"],
        help="Reasoning model for server execution"
    )
    server_standard_model = st.text_input(
        "Server Standard Model",
        value=st.session_state.models["server_standard"],
        help="Reasoning model for server execution"
    )
    server_vision_model = st.text_input(
        "Vision model",
        value=config.vision_model,
        help="Vision model used to parse pdf, only if connected to server ollama client"
    )
    

    submitted = st.form_submit_button("Save Configuration", type="primary")

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
    st.session_state.baseConfig.vision_model = server_vision_model

    st.success("Configuration saved successfully!")
    st.rerun()