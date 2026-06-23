import os

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# LLM CREDENTIALS & MODEL
LLM_BASE_URL= os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY= os.environ.get("LLM_API_KEY", "")
LLM_MODEL_NAME=os.environ.get("LLM_MODEL_NAME", "")

# DIR & FILE PATHS
ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
CONFIG_DIR = ROOT_DIR / "config"
REPORTS_DIR = ROOT_DIR / "reports"
HERMES_DB_PATH = ROOT_DIR / "hermes_vector_db"

AGENT1_SYSTEM_PROMPT_FILEPATH = ASSETS_DIR / "agent1" / "SYSTEM.md"
AGENT1_MAPPING_PROMPT_FILEPATH = ASSETS_DIR / "agent1" / "MAPPING.md"
AGENT2_SYSTEM_PROMPT_FILEPATH = ASSETS_DIR / "agent2" / "SYSTEM.md"

AGENT1_OUTPUT_REPORT_FILEPATH = REPORTS_DIR / "agent1_report.md"
AGENT2_OUTPUT_REPORT_FILEPATH = REPORTS_DIR / "agent2_report.md"

# REPORTING
REPORT_TEMPLATE = """# HERMES Pre-Visit Clinical Report

## Core Biomarkers & Vital Signs
- **Vein Pressure:** {vein_pressure}
- **Heart Rate:** {heart_rate}
- **Temperature:** {temperature}
- **Arterial Pressure:** {arterial_pressure}
- **Oxygen Saturation:** {oxygen_saturation}
- **Pain Level:** {pain_level}
- **Blood Sugar:** {blood_sugar}
"""

TARGET_UNITS = {
    "vein_pressure": "cmH2O",
    "heart_rate": "bpm",
    "temperature": "°C",
    "arterial_pressure": "cmH2O",  # Adjusted to match your hospital's specific data schema
    "oxygen_saturation": "%",
    "pain_level": "/10",
    "blood_sugar": "mg/dl"
}

# MOCK DATA (REMOVE)
MOCK_PATIENT_RECORDS = {
    "Patient_ID": ["P-101", "P-102", "P-103", "P-104", "P-105"],
    "Name": ["Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince", "Evan Wright"],
    "Age": [45, 67, 34, 29, 82],
    "Primary_Diagnosis": [
        "Post-Op Appendix Removal", 
        "Acute Decompensated Heart Failure",
        "Diabetic Ketoacidosis (DKA)",
        "Severe Migraine / Rule out Hemorrhage",
        "Community-Acquired Pneumonia"
    ],
    "Vitals_HR_Delta": [
        "+4 bpm (Stable)", 
        "+22 bpm (Critical Spike)", 
        "-12 bpm (Bradycardia observation)", 
        "+2 bpm (Stable)", 
        "+18 bpm (Elevated)"
    ],
    "Vitals_Temp_Celsius": [36.8, 38.5, 37.1, 36.6, 39.4],
    "Recent_Nurse_Notes": [
        "Patient resting comfortably. Pain managed well via IV medication.",
        "Patient reports shortness of breath when sitting up. Mild wheezing noted during auscultation.",
        "Blood glucose stabilized over last 4 hours. Patient requesting oral fluids.",
        "Photophobia present. Patient resting in a darkened room, pain scores slightly decreasing.",
        "Productive cough with thick sputum. Oxygen saturation dipping to 92% on room air. Started supplemental O2."
    ]
}
MOCK_KNOWLEDGE_BASE = {
    "Acute Decompensated Heart Failure": (
        "Protocol HF-2026: Initiate strict fluid balance tracking (Input/Output logs). "
        "Monitor continuous oxygen saturation. If saturation drops below 93% or heart rate spikes "
        "past baseline by >20bpm, evaluate for urgent IV diuretic adjustment (e.g., Furosemide) "
        "and order an immediate portable chest X-ray to check for worsening pulmonary edema."
    ),
    "Community-Acquired Pneumonia": (
        "Protocol PNEU-09: Maintain continuous pulse oximetry. For oxygen saturation dipping below 93%, "
        "escalate supplemental O2 via nasal cannula to hit target 94-98%. Ensure blood cultures are drawn "
        "prior to subsequent antibiotic adjustments. Monitor temperature intervals; if fever exceeds 39.5°C, "
        "administer scheduled antipyretics and evaluate for secondary pleural effusion via ultrasound."
    ),
    "Diabetic Ketoacidosis (DKA)": (
        "Protocol DKA-4: Continuous hourly blood glucose and venous blood gas (VBG) tracking. "
        "Maintain strict insulin infusion protocols alongside electrolyte replacement (specifically Potassium monitoring)."
    )
}

