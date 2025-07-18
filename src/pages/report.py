"""Streamlit page to generate reports based on user queries.

This page allows users to input queries, generates reports using the ReportAgent,
and provides options to download the generated reports in PDF format.
"""


import streamlit as st

from typing import Literal
from pathlib import Path
from datetime import datetime

from core.agents import ReportAgent
from core.configuration import load_config
from utils.converter import dict_to_pdf
from pages.utils import is_ollama_client_available, is_connected
from constant import OUTPUT_DIR, OLLAMA_LOCALHOST


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
    from constant import OLLAMA_CLIENT
    st.session_state.ollama_host = OLLAMA_CLIENT
    

############################## Private methods ##############################


def _create_report(query:str):
    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.pdf"
    output_path = Path(OUTPUT_DIR) / filename
    
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Generate the report
    output, header = st.session_state.reportAgent.invoke(
        query=query,
        configuration=st.session_state.baseConfig,
    )

    if output and header:
        
        # Generate PDF
        with st.spinner("Creating PDF..."):
            pdf_path = dict_to_pdf(
                data = output,
                output_filename = filename,
                output_dir = OUTPUT_DIR,
                header = header
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
        st.error(f"No report content was generated: output:{output}")


############################## Sidebar ##############################


# Connection button
connectionButton = st.sidebar.toggle(
    label = "Server execution",
    value = is_connected(st.session_state)
)

if connectionButton:
    conn = is_ollama_client_available(st.session_state.ollama_host)
    if conn:
        st.session_state.baseConfig.ollama_host=st.session_state.ollama_host
    else:
        st.sidebar.warning(f"Could not connect to {st.session_state.ollama_host}")
        st.session_state.baseConfig.ollama_host=OLLAMA_LOCALHOST
else:
    st.session_state.baseConfig.ollama_host=OLLAMA_LOCALHOST

st.sidebar.write(f"Connected to: {st.session_state.baseConfig.ollama_host}")

st.sidebar.divider()

# Writing style buttons
writingControl = st.sidebar.segmented_control(
    label="Writing style",
    options=["general","technical"],
    default="general",
)

if writingControl == "technical":
    st.sidebar.info(
        "This mode is designed to generate detailed technical sections strictly based on the retrieved context.\n\n"
        "Use this mode when your audience expects precision, completeness, and domain accuracy."
    )
else:
    st.sidebar.info(
        "This mode focuses on producing accessible and informative content using the available context.\n\n"
        "Use this mode when aiming for clarity and synthesis for a broader audience."
    )

# Update the configuration
st.session_state.baseConfig.writing_style = writingControl

st.sidebar.divider()

# Number of parts slider
numberOfPartsSlider = st.sidebar.slider(
    label="Number of parts",
    min_value=1,
    max_value=10,
    value=3,
)

# Update the configuration
st.session_state.baseConfig.number_of_parts = numberOfPartsSlider

st.sidebar.divider()

# Number of parts slider
retriverKSlider = st.sidebar.slider(
    label="Number of documents retrieved",
    min_value=1,
    max_value=20,
    value=4,
)

# Update the configuration
#st.session_state.baseConfig.retriever_k = retriverKSlider

############################## Page ##############################


st.title("Report Generator")

# Display previous reports if any
if st.session_state.report_history:
    with st.expander("Previous Reports"):
        for idx, report_info in enumerate(st.session_state.report_history):
            st.write(f"{idx + 1}. {report_info['query']} - {report_info['timestamp']}")


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