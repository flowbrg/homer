import streamlit as st


############################## Initialization ##############################


from utils.logging import setup_logging
from constant import LOG_LEVEL
setup_logging(LOG_LEVEL)  # or "DEBUG" for more detailed logs

# Default values of the models for server/local execution and classic/reasoning
_SERVER_REASONING_MODEL = "qwen3:30b-a3b" 
_SERVER_MODEL = "gemma3:4b-it-qat"
_LOCAL_REASONING_MODEL = "qwen3:0.6b"
_LOCAL_MODEL = "gemma3n:e2b"

if "baseConfig" not in st.session_state:
    from core.configuration import load_config
    st.session_state.baseConfig = load_config()
if "ollama_host" not in st.session_state:
    from constant import OLLAMA_CLIENT
    st.session_state.ollama_host = OLLAMA_CLIENT
if "models" not in st.session_state:
    st.session_state.models = {
        "server_reasoning": _SERVER_REASONING_MODEL,
        "server_standard": _SERVER_MODEL,
        "local_reasoning": _LOCAL_REASONING_MODEL,
        "local_standard": _LOCAL_MODEL,
    }


############################## Page navigation ##############################


homePage = st.Page("./pages/home.py", title="Home", icon=":material/home:")
discussionPage = st.Page("./pages/discussion.py", title="Discussion", icon=":material/chat_bubble:")
reportPage = st.Page("./pages/report.py", title="Report", icon=":material/edit:")
documentsPage = st.Page("./pages/index.py", title="Documents", icon=":material/database_upload:")
configPage = st.Page("./pages/config.py", title="Configuration", icon=":material/settings:")
modelsPage = st.Page("./pages/models.py", title="Models", icon=":material/download:")

pg = st.navigation([
    homePage,
    discussionPage,
    reportPage,
    documentsPage,
    configPage,
    modelsPage
])

pg.run()