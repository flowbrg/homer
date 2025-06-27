import streamlit as st

from src.core.agents import RetrievalAgent
from src.core.configuration import load_config

def _init():
    if "baseConfig" not in st.session_state:
        st.session_state.baseConfig = load_config()
    #if "backend" not in st.session_state:
    #    st.session_state.backend = Application(config=st.session_state.baseConfig)

def main():
    st.info("PLACEHOLDER")


if __name__ == "__main__":
    _init()
    main()