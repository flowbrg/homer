"""
Utility functions for Streamlit pages.
"""

######################################## Ollama client availability check ########################################


def is_ollama_client_available(url: str) -> bool:
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
    

######################################## Streamlit connection button state ########################################


from streamlit.runtime.state.session_state_proxy import SessionStateProxy
from constant import OLLAMA_CLIENT

def is_connected(session_state: SessionStateProxy) -> bool:
    if "baseConfig" not in session_state:
        raise Exception("config not loaded in the session state")
    elif session_state.baseConfig.ollama_host == OLLAMA_CLIENT:
        return True
    return False