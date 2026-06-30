import os
import structlog

import streamlit as st
import pandas as pd

from core.mapping import map_and_generate_pre_report
from core.rag import enrich_with_rag
from core.semantics import verify_dataset_clinical_context
from config.config import FINAL_PDF_FILEPATH
from src.utils.app_utils import download_pdf_report

logger = structlog.get_logger(__file__)

# Page configuration
st.set_page_config(
    page_title="Data Analysis Dashboard",
    layout="wide",  # Use full width
    initial_sidebar_state="expanded",
)

st.title("HERMES Clinical Reporting Pipeline")

# Create two columns with custom width ratio
# col1, col2 = st.columns([2, 1])  # Left column 2x wider than right
col1, col2 = st.columns([1, 1])
is_valid_clinical_context = True

with col1:
    uploaded_file = st.file_uploader("Upload Patient Metrics CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} rows.")

        # Display dataframe with row selection enabled
        selected_data = st.dataframe(
            df,
            width="stretch",  # Fixed: was 'stretch' (string needed)
            height=400,
            hide_index=False,
            key="patient_dataframe",
            on_select="rerun",  # Enable row selection
            selection_mode="single-row",  # Allow only single row selection
        )

        # Validate dataframe context
        is_valid_clinical_context = verify_dataset_clinical_context(df)

        if is_valid_clinical_context:
            # Get the selected row index from dataframe selection
            selected_rows = (
                selected_data.selection.rows if selected_data.selection else []
            )

            # Determine the row index: use selected row if available, otherwise use manual input
            if selected_rows:
                # User clicked on a row - use that index
                auto_selected_idx = selected_rows[0]
                row_idx = st.number_input(
                    "Select Patient Row Index",
                    min_value=0,
                    max_value=len(df) - 1,
                    value=auto_selected_idx,
                    help="Index updated automatically when you click a row above",
                )
            else:
                # No row selected - use manual input
                row_idx = st.number_input(
                    "Select Patient Row Index",
                    min_value=0,
                    max_value=len(df) - 1,
                    value=0,
                    help="Enter index manually or click a row above",
                )

            # Display the selected row information
            st.write("**Selected Patient data:**")
            st.write(df.iloc[row_idx])
        else:
            st.write("**❌ Error: dataset context is not clinically-related!**")
if is_valid_clinical_context:
    with col2:
        st.write("**Generated Reports:**")
        if st.button("Create Reports"):
            with st.spinner("Processing through HERMES Agents..."):
                # Generate reports
                patient_row_data = df.iloc[row_idx]
                pre_report = map_and_generate_pre_report(patient_row_data)
                with st.success("Pre-report generation complete!"):
                    clinical_report = enrich_with_rag(pre_report)
                    # Generate final PDF
                    download_pdf_report(pre_report, clinical_report)
                st.success("Analysis Complete!")
                if clinical_report:
                    st.download_button(
                        "Download Full Report",
                        on_click="ignore",
                        data=open(FINAL_PDF_FILEPATH, "rb").read(),
                        file_name=os.path.basename(FINAL_PDF_FILEPATH),
                    )
                # Show output tabs
                tab1, tab2 = st.tabs(
                    ["Detected Biomarkers Report", "Enriched Clinical Report"]
                )
                with tab1:
                    st.markdown(pre_report)
                with tab2:
                    st.markdown(clinical_report)
