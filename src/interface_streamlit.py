from main import main
import streamlit as st
import pandas as pd
import structlog

from mapping import map_and_generate_pre_report
from rag import enrich_with_rag

logger = structlog.get_logger(__file__)

st.title("HERMES Clinical Reporting Pipeline")

uploaded_file = st.file_uploader("Upload Patient Metrics CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} rows.")
    
    # Let the user pick a row visually
    row_idx = st.number_input("Select Patient Row Index", min_value=0, max_value=len(df)-1, value=0)
    
    if st.button("Generate Reports"):
        with st.spinner("Processing through HERMES Agents..."):
            # Generate reports
            patient_row_data = df.iloc[row_idx]
            pre_report = map_and_generate_pre_report(patient_row_data)
            with st.success("Pre-report generation complete!"):
                final_report = enrich_with_rag(pre_report)
            st.success("Analysis Complete!")
            
            # Show output tabs
            tab1, tab2 = st.tabs(["Agent 1 Pre-Report", "Agent 2 Final Enriched Report"])
            with tab1:
                st.markdown(pre_report)
            with tab2:
                st.markdown(final_report)
                st.download_button("Download Enriched Report", data=final_report, file_name="final_report.md")
