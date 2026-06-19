import random
import structlog

import pandas as pd

from openai import OpenAI

from config.config import AGENT1_OUTPUT_REPORT_FILEPATH, AGENT1_SYSTEM_PROMPT_FILEPATH, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME, MOCK_PATIENT_RECORDS
from utils.llm_utils import load_prompt

logger = structlog.get_logger(__file__)


# 1. CONFIGURE CLIENT & PORTABILITY 
# Point this to your workplace IP and port. 
# It mimics the OpenAI library standard perfectly.

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY
)

MODEL_NAME =  LLM_MODEL_NAME


# 2. READ MOCK DATASET (Pandas DataFrame)
df_patients = pd.DataFrame(MOCK_PATIENT_RECORDS)

# 3. STATELESS PROCESSING PIPELINE
def run_hermes_pipeline(patient_row: pd.Series, system_prompt: str):
    """
    Executes a single, stateless request to the inference server. 
    Does not store chat history, conserving local computing resources.
    """
    patient_dict = patient_row.to_dict()
    
    # Serialize the dataframe row explicitly into structured text
    patient_data_input = (
        f"--- CURRENT CLINICAL DATA FOR SUMMARY ---\n"
        f"Patient ID: {patient_dict['Patient_ID']} ({patient_dict['Name']}), Age: {patient_dict['Age']}\n"
        f"Admitted For: {patient_dict['Primary_Diagnosis']}\n"
        f"Heart Rate Change (24h): {patient_dict['Vitals_HR_Delta']}\n"
        f"Current Temperature: {patient_dict['Vitals_Temp_Celsius']}°C\n"
        f"Latest Shift Notes: {patient_dict['Recent_Nurse_Notes']}\n"
    )
    
    logger.info(f"🎲 Selected Patient: {patient_dict['Name']} [{patient_dict['Patient_ID']}]")
    logger.info(f"🚀 Prompting stateless engine '{MODEL_NAME}'...")

    # We send a clean array containing exactly one system rule and one payload.
    # No history tracking overhead.
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": patient_data_input}
        ],
        temperature=0.1  # Low temperature forces deterministic, factual outputs
    )
    
    return response.choices[0].message.content


# 4. EXECUTION
if __name__ == "__main__":
    prompt_path = AGENT1_SYSTEM_PROMPT_FILEPATH

    # Securely load prompt
    sys_prompt = load_prompt(prompt_path)
    
    # Pick a completely random index from our Pandas DataFrame size
    random_index = random.randint(0, len(df_patients) - 1)
    selected_patient = df_patients.iloc[random_index]
    
    # Generate report
    report_output = run_hermes_pipeline(selected_patient, sys_prompt)
    
    with open(AGENT1_OUTPUT_REPORT_FILEPATH, "w") as f:
        f.write(report_output)

    print("\n================== HERMES OUTPUT REPORT ==================")
    print(report_output)
    print("==========================================================")