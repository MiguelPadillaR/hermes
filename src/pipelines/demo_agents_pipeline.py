import random
import structlog

import pandas as pd

from openai import OpenAI

from config.config import (
    AGENT1_SYSTEM_PROMPT_FILEPATH,
    AGENT2_OUTPUT_REPORT_FILEPATH,
    AGENT2_SYSTEM_PROMPT_FILEPATH,
    LLM_API_KEY, LLM_BASE_URL,
    LLM_MODEL_NAME,
    MOCK_KNOWLEDGE_BASE,
    MOCK_PATIENT_RECORDS
    )
from utils.llm_utils import load_prompt

logger = structlog.get_logger(__file__)

# 1. CORE API CONFIGURATION
client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY
)

MODEL_NAME =  LLM_MODEL_NAME

# 2. MOCK PATIENT RECORDS INGESTION
df_patients = pd.DataFrame(MOCK_PATIENT_RECORDS)

# 3. PIPELINE EXECUTION ENGINE
def run_hermes_pipeline(patient_row: pd.Series):
    # Load system prompts dynamically
    agent_1_sys = load_prompt(AGENT1_SYSTEM_PROMPT_FILEPATH)
    
    patient_dict = patient_row.to_dict()
    diagnosis = patient_dict['Primary_Diagnosis']
    
    logger.info(f"Evaluating Patient: {patient_dict['Name']} | Diagnosis: {diagnosis}")
    
    # ----------------------------------------------------
    # STEP 1: Serialize raw data & Execute Agent #1
    # ----------------------------------------------------
    patient_data_input = (
        f"Patient ID: {patient_dict['Patient_ID']} ({patient_dict['Name']})\n"
        f"Admitted For: {diagnosis}\n"
        f"Heart Rate Change: {patient_dict['Vitals_HR_Delta']}\n"
        f"Current Temperature: {patient_dict['Vitals_Temp_Celsius']}°C\n"
        f"Latest Shift Notes: {patient_dict['Recent_Nurse_Notes']}\n"
    )
    
    logger.info("🤖 Running Agent #1: Creating Clinical Status Summary...")
    a1_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": agent_1_sys},
            {"role": "user", "content": patient_data_input}
        ],
        temperature=0.1
    )
    agent_1_output = a1_response.choices[0].message.content
    
    # ----------------------------------------------------
    # STEP 2: Mock RAG / Knowledge Base Retrieval
    # ----------------------------------------------------
    agent_2_sys = load_prompt(AGENT2_SYSTEM_PROMPT_FILEPATH)
    
    logger.info("🔍 Performing RAG Search: Checking internal medical database...")
    # Get protocol if it exists, otherwise fall back to a default empty string context
    retrieved_protocol = MOCK_KNOWLEDGE_BASE.get(
        diagnosis, 
        "No specific advanced clinical guideline found in local vector repository."
    )
    
    # ----------------------------------------------------
    # STEP 3: Combine Context & Execute Agent #2
    # ----------------------------------------------------
    logger.info("🤖 Running Agent #2: Enriching report with Advisory Protocols...")
    agent_2_input = (
        f"### AGENT 1 BASE SUMMARY ###\n{agent_1_output}\n\n"
        f"### RETRIEVED INTERNAL MEDICAL REFERENCE GUIDE ###\n{retrieved_protocol}\n"
    )
    
    a2_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": agent_2_sys},
            {"role": "user", "content": agent_2_input}
        ],
        temperature=0.2
    )
    agent_2_output = a2_response.choices[0].message.content
    
    # Return the fully appended output
    final_combined_report = f"{agent_1_output}\n\n{agent_2_output}"
    return final_combined_report

# 4. RUN THE PIPELINE
if __name__ == "__main__":
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