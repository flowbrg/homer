import streamlit as st

from src.core.application import Application
from src.core.configuration import load_config

def _init():
    if "baseConfig" not in st.session_state:
        st.session_state.baseConfig = load_config()
    if "backend" not in st.session_state:
        st.session_state.backend = Application(config=st.session_state.baseConfig)

st.set_page_config(page_title="Report")


def main():
    # Create the query input area at the bottom
    query = st.chat_input("Enter your query:")

    if query:
        with st.spinner("Updating database..."):
            try:
                st.session_state.backend.invoke_report_graph(query=query)
            except Exception as e:
                st.error(f"Error generating the report: {str(e)}")
            else:
                st.success("Report generated.")
        
        st.info()

if __name__ == "__main__":
    _init()
