import streamlit as st

from pathlib import Path
from datetime import datetime

from src.core.agents import ReportAgent
from src.core.configuration import load_config
from src.utils.dict_to_pdf import str_to_pdf
from src.utils.utils import is_connected
from src.constant import OUTPUT_DIR, OLLAMA_LOCALHOST


############################## Initialization ##############################


st.set_page_config(
    page_title="Report Generator",
    layout="centered"
)

if "baseConfig" not in st.session_state:
    st.session_state.baseConfig = load_config()
if "reportAgent" not in st.session_state:
    st.session_state.reportAgent = ReportAgent()
if "report_history" not in st.session_state:
    st.session_state.report_history = []
if "ollama_host" not in st.session_state:
    from src.constant import OLLAMA_CLIENT
    st.session_state.ollama_host = OLLAMA_CLIENT
    

############################## Private methods ##############################


def _is_ollama_client_available(url: str) -> bool:
    import requests
    try:
        response = requests.get(url, timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def _create_report(query=str):
    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.pdf"
    output_path = Path(OUTPUT_DIR) / filename
    
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Generate the report
    output = st.session_state.reportAgent.invoke(
        query=query,
        writing_style= "general",
        number_of_parts= 3,
        configuration=st.session_state.baseConfig,
    )

    if output:
        
        # Generate PDF
        with st.spinner("Creating PDF..."):
            pdf_path = str_to_pdf(
                data = output,
                filename = filename,
                output_dir = OUTPUT_DIR,
            )
        
        # Success message
        st.success(f"Report generated successfully!")
        st.info(f"Saved to: {pdf_path}")
        
        # Add to history
        st.session_state.report_history.append({
            "query": query,
            "timestamp": timestamp,
            "path": str(output_path)
        })
        
        # Offer download
        with open(output_path, "rb") as pdf_file:
            st.download_button(
                label="Download Report",
                data=pdf_file.read(),
                file_name=filename,
                mime="application/pdf"
            )
    else:
        st.error("No report content was generated.")

############################## Page builders ##############################

def _build_sidebar():
    connectionButton = st.sidebar.toggle(
        label = "Server execution",
        value = is_connected(st.session_state)
    )

    if connectionButton:
        conn = _is_ollama_client_available(st.session_state.ollama_host)
        if conn:
            st.session_state.baseConfig.ollama_host=st.session_state.ollama_host
        else:
            st.sidebar.warning(f"Could not connect to {st.session_state.ollama_host}")
            st.session_state.baseConfig.ollama_host=OLLAMA_LOCALHOST
    else:
        st.session_state.baseConfig.ollama_host=OLLAMA_LOCALHOST
    
    st.sidebar.write(f"Connected to: {st.session_state.baseConfig.ollama_host}")


def _display_reports():
    st.title("Report Generator")
    
    # Display previous reports if any
    if st.session_state.report_history:
        with st.expander("Previous Reports"):
            for idx, report_info in enumerate(st.session_state.report_history):
                st.write(f"{idx + 1}. {report_info['query']} - {report_info['timestamp']}")


def _build_query_input():
    # Create the query input area
    query = st.chat_input(
        placeholder="Enter your query:",
        #disabled=True,
        )

    if query:
        # Display user query
        with st.chat_message("user"):
            st.write(query)
        
        # Generate report with progress tracking
        with st.chat_message("assistant"):
            progress_container = st.container()
            
            with progress_container:
                with st.spinner("Generating report..."):
                    try:
                        _create_report(query=query)                            
                    except Exception as e:
                        st.error(f"Error generating the report: {str(e)}")
                        st.info("Please check the logs for more details.")


if __name__ == "__main__":
    
    _build_sidebar()

    _display_reports()

    _build_query_input()