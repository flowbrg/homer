import streamlit as st
from PIL import Image

from src.core.configuration import load_config
from src.constant import OLLAMA_CLIENT

############################## Initialization ##############################

from src.utils.logging import setup_logging, get_logger
setup_logging("INFO")  # or "DEBUG" for more detailed logs

# Default values of the models for server/local execution and classic/reasoning
_SERVER_REASONING_MODEL = "qwen3:30b-a3b" 
_SERVER_MODEL = "gemma3:4b-it-qat"
_LOCAL_REASONING_MODEL = "qwen3:0.6b"
_LOCAL_MODEL = "gemma3n:e2b"

if "baseConfig" not in st.session_state:
    st.session_state.baseConfig = load_config()
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = OLLAMA_CLIENT
if "models" not in st.session_state:
    st.session_state.models = {
        "server_reasoning": _SERVER_REASONING_MODEL,
        "server_standard": _SERVER_MODEL,
        "local_reasoning": _LOCAL_REASONING_MODEL,
        "local_standard": _LOCAL_MODEL,
    }

############################## Page builder ##############################

def build():
    logo = Image.open("static/homerlogo-nobg.png")
    st.markdown("""
        <br>
        <br>
        
        
        """,unsafe_allow_html=True)
    # Image on the left, text on the right
    col1, col2 = st.columns([1, 2])  # [1,2] = Image takes 1/3, text 2/3
    
    with col1:
        st.image(logo, width=150)

    with col2:
        st.markdown("""
        ### Welcome to HOMER  
        <br>
        <div style="display: flex; align-items: center; gap: 15px;">
            <span>Start by uploading your documents</span>
            <a href="./Documents" target="_self">
                <button style="padding:0.3em 0.8em; font-size:16px; background-color:#512967; color:white; border:none; border-radius:5px; cursor:pointer;">
                    HERE
                </button>
            </a>
        </div>
        
        <br>
        Then, you can either ask a simple question or generate a full structured report based on the content of your files.
        """, unsafe_allow_html=True)
    
    st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    text-align: center;
    font-size: 12px;
    color: black;
    background-color: #f5f5f5;
    padding: 10px 0;
    border-top: 1px solid #ddd;
    z-index: 100;
}
</style>

<div class="footer">
    Designed by Florent Bergé & Mathieu de la Barre (IMT Atlantique) for SCK CEN
</div>
""", unsafe_allow_html=True)
    



if __name__ == "__main__":
    build()