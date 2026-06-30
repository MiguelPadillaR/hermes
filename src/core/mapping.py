import json
import random
import structlog

import pandas as pd

from config.config import (
    AGENT1_MAPPING_PROMPT_FILEPATH,
    AGENT1_OUTPUT_REPORT_FILEPATH,
    LLM_MODEL_NAME,
    REPORT_TEMPLATE,
    TARGET_UNITS
    )

from utils.llm_utils import init_client
from utils.dataset_utils import load_mimic_dataset, load_mock_dataset

logger = structlog.get_logger(__file__)

# 1. CORE API CONFIGURATION
client = init_client()
MODEL_NAME =  LLM_MODEL_NAME

def map_and_generate_pre_report(patient_row_data: pd.Series):
    """
    Provided patient clinical data, the LLM maps the keys & values semantically to target schema.
    It also takes any extra keys and generated an Additional Contextual Metrics section with them.
    Args:
        patient_row_data (pd.Series): Patient row from clinical dataframe.
    Returns:
        pre_report (str): The filled-out target schema with the added section.
    """
    with open(AGENT1_MAPPING_PROMPT_FILEPATH, "r", encoding="utf-8") as f:
        sys_prompt = f.read()

    # Generate payload with raw keys and values
    if not isinstance(patient_row_data, dict):
        patient_row_data = patient_row_data.to_dict()
    logger.info(f"RAW DATA:\n{patient_row_data}")
    user_payload = (
        f"--- INCOMING DATA FOR DYNAMIC MAPPING ---\n"
        f"Incoming Columns & Values:\n{json.dumps(patient_row_data, indent=2)}\n"
    )

    logger.info("🤖 HERMES is analyzing columns and generating the pre-report...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_payload}
        ],
        temperature=0.2 # Keep it strictly deterministic
    )
    
    raw_llm_response = response.choices[0].message.content

    # 1. Isolate the mapping block explicitly using the boundary tags
    try:
        mapping_text = raw_llm_response.split("[MAPPING_BLOCK]")[1].split("[/MAPPING_BLOCK]")[0].strip()
        additional_context = raw_llm_response.split("[/MAPPING_BLOCK]")[1].strip()
    except IndexError:
        # Fallback guard clause in case the text format is malformed
        mapping_text = ""
        additional_context = "Error parsing additional metrics structural block."

    # 2. Build the dictionary deterministically
    mapped_columns = {}
    for line in mapping_text.split("\n"):
        line = line.strip()
        if ":" in line:
            target_col, incoming_col = line.split(":", 1) # Max split 1 to protect against messy inputs
            
            # Strip string out values and map to None if missing
            incoming_col = incoming_col.strip()
            mapped_columns[target_col.strip()] = None if incoming_col == "None" else incoming_col

    # 3. Validation Check
    logger.info("Deterministic Mapped Columns Dictionary:")
    logger.info(mapped_columns)

    logger.info("\nMarkdown Fragment for Pre-Report:")
    logger.info(additional_context)
    
    # Build and return report
    pre_report = build_pre_report(patient_row_data, mapped_columns, extra_metrics=additional_context)
    return pre_report

def build_pre_report(patient_row_data: dict, mapped_columns: dict, template: str = REPORT_TEMPLATE, extra_metrics: str = "") -> str:
    """
    Looks up mapped column names in the raw data row. If a target key 
    points to None, it defaults to 'Unreported'. Then substitutes placeholders.
    Args:
        patient_row_data (dict): Patient row from clinical dataframe.
        mapped_columns (dict): Dict with the patien data keys matched to the target schema keys.
        template (str): Pre-report schema template.
        extra_metrics (str): Any additional key-values detected in the patient data to add to the report
    Returns:
        pre_report (str): The filled-out target schema with the added section.

    """
    templated_values = {}
    
    for target_key, incoming_column in mapped_columns.items():
        if incoming_column and incoming_column in patient_row_data:
            raw_value = patient_row_data[incoming_column]
            unit = TARGET_UNITS.get(target_key, "")
            
            # Grab actual value from the raw dataset row
            templated_values[target_key] = f"{raw_value} {unit}".strip()
        else:
            # Safe fallback if column was missing or explicitly mapped to None
            templated_values[target_key] = "Data Unreported"
            
    # Render the base template using dictionary unpacking
    rendered_base = template.format(**templated_values)
    
    # Append the additional markdown block safely
    pre_report = f"{rendered_base}\n{extra_metrics}"
    return pre_report

if __name__ == "__main__":
    df = load_mimic_dataset()
    # df = load_mock_dataset()
    random_idx = random.randint(0, len(df) - 1)
    patient_row_data = df.iloc[random_idx]

    pre_report = map_and_generate_pre_report(patient_row_data)
    
    logger.info("\n================== AGENT 1 PRE-REPORT OUTPUT ==================")
    logger.info(pre_report)
    logger.info("===============================================================")
    
    # Save pre-report to file
    with open(AGENT1_OUTPUT_REPORT_FILEPATH, "w", encoding="utf-8") as f:
        f.write(pre_report)