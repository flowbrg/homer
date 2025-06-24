
import streamlit as st
import logging

from pathlib import Path
from datetime import datetime, timezone

from src.core.agents import ReportAgent
from src.core.configuration import load_config
from src.resources.dict_to_pdf import dict_to_pdf
from src.env import OUTPUT_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _init():
    """Initialize session state variables."""
    if "baseConfig" not in st.session_state:
        st.session_state.baseConfig = load_config()
    if "reportAgent" not in st.session_state:
        st.session_state.reportAgent = ReportAgent()
    if "report_history" not in st.session_state:
        st.session_state.report_history = []

st.set_page_config(page_title="Report Generator", layout="wide")


def main():
    st.title("Report Generator")
    
    # Display previous reports if any
    if st.session_state.report_history:
        with st.expander("Previous Reports"):
            for idx, report_info in enumerate(st.session_state.report_history):
                st.write(f"{idx + 1}. {report_info['query']} - {report_info['timestamp']}")
    
    # Create the query input area
    query = st.chat_input("Enter your query:")
    
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
                        # Create a unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"report_{timestamp}.pdf"
                        output_path = Path(OUTPUT_DIR) / filename
                        
                        # Ensure output directory exists
                        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
                        
                        # Generate the report
                        report = st.session_state.reportAgent.invoke(
                            query=query, 
                            configuration=st.session_state.baseConfig
                        )
                    
                        if report:
                            # Display report preview
                            st.subheader("Report Preview")
                            for section in report:
                                with st.expander(section.get("title", "Untitled Section")):
                                    st.write(section.get("content", "No content"))
                            
                            # Generate PDF
                            with st.spinner("📄 Creating PDF..."):
                                dict_to_pdf(data=report, output_filename=str(output_path))
                            
                            # Success message
                            st.success(f"✅ Report generated successfully!")
                            st.info(f"📁 Saved to: {output_path}")
                            
                            # Add to history
                            st.session_state.report_history.append({
                                "query": query,
                                "timestamp": timestamp,
                                "path": str(output_path)
                            })
                            
                            # Offer download
                            with open(output_path, "rb") as pdf_file:
                                st.download_button(
                                    label="📥 Download Report",
                                    data=pdf_file.read(),
                                    file_name=filename,
                                    mime="application/pdf"
                                )
                        else:
                            st.error("No report content was generated.")
                            
                    except Exception as e:
                        logger.error(f"Error generating report: {str(e)}", exc_info=True)
                        st.error(f"❌ Error generating the report: {str(e)}")
                        st.info("Please check the logs for more details.")


if __name__ == "__main__":
    _init()
    main()