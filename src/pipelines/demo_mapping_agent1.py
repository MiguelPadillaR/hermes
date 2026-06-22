import json
import random
import structlog

import pandas as pd

from openai import OpenAI

from config.config import (
    AGENT1_MAPPING_PROMPT_FILEPATH,
    AGENT1_OUTPUT_REPORT_FILEPATH,
    LLM_API_KEY, LLM_BASE_URL,
    LLM_MODEL_NAME,
    REPORT_TEMPLATE,
    TARGET_UNITS
    )

from utils.dataset_utils import load_mimic_dataset, load_mock_dataset
from utils.llm_utils import load_prompt

logger = structlog.get_logger(__file__)

# 1. CORE API CONFIGURATION
client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY
)

MODEL_NAME =  LLM_MODEL_NAME

def run_mapping_test(raw_data: pd.Series):
    with open(AGENT1_MAPPING_PROMPT_FILEPATH, "r", encoding="utf-8") as f:
        sys_prompt = f.read()

    # We format the payload showing the model the raw keys and the values
    raw_data = raw_data.to_dict()
    logger.info(f"RAW DATA:\n{raw_data}")
    user_payload = (
        f"--- INCOMING DATA FOR DYNAMIC MAPPING ---\n"
        f"Incoming Columns & Values:\n{json.dumps(raw_data, indent=2)}\n"
    )

    logger.info("🤖 HERMES is analyzing columns and generating the pre-report...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_payload}
        ],
        temperature=0.1 # Keep it strictly deterministic
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
    
    return build_pre_report(raw_data, mapped_columns, extra_metrics=additional_context)

def build_pre_report(row_data: dict, mapping: dict, template: str = REPORT_TEMPLATE, extra_metrics: str = "") -> str:
    """
    Looks up mapped column names in the raw data row. If a target key 
    points to None, it defaults to 'Unreported'. Then substitutes placeholders.
    """
    templated_values = {}
    
    for target_key, incoming_column in mapping.items():
        if incoming_column and incoming_column in row_data:
            raw_value = row_data[incoming_column]
            unit = TARGET_UNITS.get(target_key, "")
            
            # Grab actual value from the raw dataset row
            templated_values[target_key] = f"{raw_value} {unit}".strip()
        else:
            # Safe fallback if column was missing or explicitly mapped to None
            templated_values[target_key] = "Data Unreported"
            
    # Render the base template using dictionary unpacking
    rendered_base = template.format(**templated_values)
    
    # Append the additional markdown block safely
    final_report = f"{rendered_base}\n{extra_metrics}"
    return final_report

if __name__ == "__main__":
    # df = load_mimic_dataset()
    df = load_mock_dataset()
    random_idx = random.randint(0, len(df) - 1)
    mimic_incoming_row = df.iloc[random_idx]

    # mimic_incoming_row = {
    #     "subject_id": 10005817,
    #     "charttime": "2132-12-15 20:15:00",
    #     "heartrate": 88,            # Matches target: heart_rate
    #     "sbp": 122,                 # Arterial Systolic Pressure -> Matches target
    #     "dbp": 74,                  # Arterial Diastolic Pressure -> Matches target
    #     "spo2": 96,                 # Matches target: oxygen_saturation
    #     "temperature_f": 98.6,      # Matches target: skin_temperature (but in Fahrenheit!)
    #     "glucose_infusion_rate": 4.5, # Leftover: Highly relevant clinical context!
    #     "caregiver_signature_id": 928, # Leftover: Administrative metadata (Not clinically relevant)
    # }


    pre_report = run_mapping_test(mimic_incoming_row)
    
    logger.info("\n================== AGENT 1 PRE-REPORT OUTPUT ==================")
    logger.info(pre_report)
    logger.info("===============================================================")
    
    # Save the file out so Agent 2 can grab it next
    with open(AGENT1_OUTPUT_REPORT_FILEPATH, "w", encoding="utf-8") as f:
        f.write(pre_report)