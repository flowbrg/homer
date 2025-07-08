import streamlit as st
import ollama

from tqdm import tqdm
from src.utils.utils import is_connected
from src.core.configuration import load_config
from src.constant import OLLAMA_CLIENT, OLLAMA_LOCALHOST


############################## Initialization ##############################


st.set_page_config(
    page_title="Model",
    layout="wide",
)

if "baseConfig" not in st.session_state:
    st.session_state.baseConfig = load_config()
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = OLLAMA_CLIENT


############################## Private methods ##############################


def _is_ollama_client_available(url: str) -> bool:
    """
    Check if Ollama server is available at the given URL.
    
    Args:
        url: The Ollama server URL to test
        
    Returns:
        True if server responds successfully, False otherwise
    """
    import requests
    try:
        response = requests.get(url, timeout=2)
        return response.ok
    except requests.RequestException:
        return False
    

############################## Page builders ##############################


def _build_sidebar():
    """
    Build sidebar interface for configuration options.
    
    Provides controls for:
    - Server execution toggle (local vs remote)
    - Model selection toggle (thinking vs standard models)
    - Connection status display
    - Current model display
    """
    # Server connection toggle
    connectionButton = st.sidebar.toggle(
        label="Server execution",
        value=is_connected(st.session_state)
    )

    # Configure server host based on connection preference
    if connectionButton:
        conn = _is_ollama_client_available(st.session_state.ollama_host)
        if conn:
            st.session_state.baseConfig.ollama_host = st.session_state.ollama_host
        else:
            st.sidebar.warning(f"Could not connect to {st.session_state.ollama_host}")
            st.session_state.baseConfig.ollama_host = OLLAMA_LOCALHOST
    else:
        st.session_state.baseConfig.ollama_host = OLLAMA_LOCALHOST
    
    st.sidebar.write(f"Connected to: {st.session_state.baseConfig.ollama_host}")


def _build_model_input():
    model = st.text_input("Enter the model you want to pull:")
    pullButton = st.button("Pull")
    if pullButton and model:
        host = st.session_state.baseConfig.ollama_host
        # Create Ollama client connecting to custom host/port
        client = ollama.Client(
            host = host
        )
        try:
            
            # Check if model already exists locally
            client.show(model)
            st.success(f"Model {model} is already available on host: {host}")
        except ollama.ResponseError as e:
            # If model doesn't exist (404 error), download it
            if e.status_code == 404:
                # Initialize tracking variables for progress bars
                current_digest, bars = "", {}
                
                try:
                    with st.spinner(f"Pulling model {model}"):
                        # Stream the model download process to get real-time updates
                        for progress in client.pull(model, stream=True):
                            # Get the digest (unique ID) for current file chunk being downloaded
                            digest = progress.get("digest", "")
                            
                            # If we've moved to a new chunk, close the previous progress bar
                            if digest != current_digest and current_digest in bars:
                                bars[current_digest].close()

                            # Create new progress bar for this chunk if it doesn't exist and has size info
                            if digest not in bars and (total := progress.get("total")):
                                bars[digest] = tqdm(
                                    total=total,                          # Total bytes to download for this chunk
                                    desc=f"pulling {digest[7:19]}",       # Show first 12 chars of digest as description
                                    unit="B",                             # Display units in bytes
                                    unit_scale=True,                      # Auto-scale to KB/MB/GB
                                )

                            # Update progress bar with newly downloaded bytes
                            if completed := progress.get("completed"):
                                # Update only the difference between current completed and last position
                                bars[digest].update(completed - bars[digest].n)

                            # Track current digest for next iteration
                            current_digest = digest
                        st.success(f"Model {model} pulled successfully on {host}")
                except ollama.ResponseError as e:
                    if e.status_code == 500:
                        st.warning(f"Model {model} does not exist, visit https://ollama.com/ to find a compatible model.")
        except ollama.ResponseError as e:
            if e.status_code == 500:
                st.warning(f"The model {model} does not exist, visit")



_build_sidebar()
_build_model_input()