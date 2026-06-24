import random
import structlog

import pandas as pd

from config.config import AGENT2_OUTPUT_REPORT_FILEPATH, MOCK_PATIENT_RECORDS
from pipelines.demo_agents_pipeline import run_hermes_pipeline
from utils.llm_utils import load_prompt

logger = structlog.get_logger(__file__)


def main():
    df_patients = pd.DataFrame(MOCK_PATIENT_RECORDS)
    # Select a completely random record from the dataset
    random_idx = random.randint(0, len(df_patients) - 1)
    target_patient = df_patients.iloc[random_idx]
    
    # Execute full pipeline
    final_markdown_report = run_hermes_pipeline(target_patient)
    
    # Output to console
    print("\n======================= FINAL COMBINED REPORT =======================")
    print(final_markdown_report)
    print("=====================================================================")
    
    # Save output to a markdown file for the portfolio presentation
    output_filename = AGENT2_OUTPUT_REPORT_FILEPATH
    with open(output_filename, "w", encoding="utf-8") as out_file:
        out_file.write(final_markdown_report)
    logger.info(f"💾 Report successfully compiled and saved as: '{output_filename}'")


if __name__ == "__main__":
    main()