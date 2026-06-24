import random
import structlog

from config.config import AGENT1_OUTPUT_REPORT_FILEPATH, AGENT2_OUTPUT_REPORT_FILEPATH
from mapping import map_and_generate_pre_report
from rag import enrich_with_rag
from utils.dataset_utils import load_mimic_dataset, load_mock_dataset

logger = structlog.get_logger(__file__)

def main(save_report_files: bool = False):
    # Load selected dataset
    df = load_mimic_dataset()
    # df = load_mock_dataset()

    # Choose random patient row
    random_idx = random.randint(0, len(df) - 1)
    patient_row_data = df.iloc[random_idx]

    # Generate reports
    pre_report = map_and_generate_pre_report(patient_row_data)
    
    logger.info("\n================== AGENT 1 PRE-REPORT OUTPUT ==================")
    logger.info(pre_report)
    logger.info("===============================================================")

    final_report = enrich_with_rag(pre_report)
    
    logger.info("\n======================= OUTPUT PREVIEW =======================")
    logger.info(final_report[:500] + "\n\n[... Remaining Report Content Saved To Disk ...]")

    if save_report_files:
        for filepath, content in (
            (AGENT1_OUTPUT_REPORT_FILEPATH, pre_report),
            (AGENT2_OUTPUT_REPORT_FILEPATH, final_report),
        ):
            with open(filepath, "w") as file:
                file.write(content)

        logger.info(f"💾 Reports successfully compiled and saved to:\n\t'{AGENT1_OUTPUT_REPORT_FILEPATH}'\n\t'{AGENT2_OUTPUT_REPORT_FILEPATH}'")

if __name__ == "__main__":
    main(save_report_files=True)