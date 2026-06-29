from io import BytesIO
import os
import structlog

import numpy as np
import pandas as pd

from config.config import DATASETS_DIR

logger = structlog.get_logger()

def load_mimic_dataset() ->pd.DataFrame:
    """
    Loads and processes the MIMIC-IV ICU dataset.
    Processing takes the vitals' values for each patient, translates them from ID to str, makes them the new DF columns and copies the values in them.
    Patients are rows and vitasl are columns.
    Only most recent values for each vital are used.
    Returns:
        df (pd.DataFrame): Processed MIMIC dataframe. 
    """
    logger.info(f"Selected MIMIC-IV ICU dataset.")
    
    base_path = DATASETS_DIR / 'montassarba/mimic-iv-clinical-database-demo-2-2/mimic-iv-clinical-database-demo-2.2'
    df_filepath = f'{base_path}/icu/chartevents.csv'
    df = pd.read_csv(df_filepath)

    vitals_map = {
        # -- CORE VITALS ---
        220045: 'heart_rate',          # Heart Rate (bpm)
        220210: 'resp_rate',           # Respiratory Rate (breaths/min)
        224690: 'resp_rate_total',     # Respiratory Rate Total (Ventilator + Spontaneous)
        220277: 'spo2',                # Pulse Oximetry Peripheral Oxygen Saturation (%)
        223761: 'temp_f',              # Temperature Fahrenheit
        223762: 'temp_c',              # Temperature Celsius

        # --- NON-INVASIVE BLOOD PRESSURE (NIBP) ---
        220179: 'sbp_non_invasive',    # Non-Invasive Systolic Blood Pressure (mmHg)
        220180: 'dbp_non_invasive',    # Non-Invasive Diastolic Blood Pressure (mmHg)
        220181: 'mbp_non_invasive',    # Non-Invasive Mean Blood Pressure (mmHg)

        # --- INVASIVE ARTERIAL BLOOD PRESSURE (ABP line) ---
        220050: 'sbp_invasive',        # Arterial Blood Pressure Systolic (mmHg)
        220051: 'dbp_invasive',        # Arterial Blood Pressure Diastolic (mmHg)
        220052: 'mbp_invasive',        # Arterial Blood Pressure Mean (mmHg)

        # --- VEIN & CENTRAL PRESSURES ---
        220074: 'vein_pressure',       # Central Venous Pressure / CVP (mmHg or cmH2O)
        220059: 'pulmonary_art_sbp',   # Pulmonary Artery Systolic Pressure
        220060: 'pulmonary_art_dbp',   # Pulmonary Artery Diastolic Pressure

        # --- METABOLIC / BLOOD SUGAR ---
        225664: 'blood_sugar_finger',  # Glucose Fingerstick (mg/dL)
        220621: 'blood_sugar_serum',   # Glucose Serum (mg/dL)
        226537: 'blood_sugar_blood',   # Glucose Whole Blood (mg/dL)

        # --- NEUROLOGICAL & PAIN SCALES ---
        223791: 'pain_level_score',    # Pain Level/Score (Visual Analog / Numeric Scale 0-10)
        224409: 'pain_assessment',     # Critical-Care Pain Observation Tool (CPOT)
        223900: 'gcs_verbal',          # Glasgow Coma Scale - Verbal Response
        223901: 'gcs_motor',           # Glasgow Coma Scale - Motor Response
        223902: 'gcs_eyes'             # Glasgow Coma Scale - Eye Opening
    }
# Filter data
    vitals_df = df[df['itemid'].isin(vitals_map.keys())].copy()
    vitals_df['vital_name'] = vitals_df['itemid'].map(vitals_map)

    # Pivot data
    vitals_pivot = vitals_df.pivot_table(
        index=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'valueuom'],
        columns='vital_name',
        values='valuenum'
    ).reset_index()

    # Ensure all expected columns exist in the pivot table ---
    expected_columns = set(vitals_map.values())
    for col in expected_columns:
        if col not in vitals_pivot.columns:
            vitals_pivot[col] = np.nan

    # --- FIX TEMPERATURE ---
    # Convert F to C if C is missing
    f_to_c = (vitals_pivot['temp_f'] - 32) * 5/9
    vitals_pivot['temp_c'] = vitals_pivot['temp_c'].fillna(f_to_c)
    vitals_pivot = vitals_pivot.drop(columns=['temp_f'])

    # --- CONSOLIDATE REPETITIVE CLINICAL CONCEPTS ---
    # 1. Blood Pressure: Prioritize invasive arterial line readings, fall back to non-invasive cuffs
    vitals_pivot['sbp'] = vitals_pivot['sbp_invasive'].fillna(vitals_pivot['sbp_non_invasive'])
    vitals_pivot['dbp'] = vitals_pivot['dbp_invasive'].fillna(vitals_pivot['dbp_non_invasive'])
    vitals_pivot['mbp'] = vitals_pivot['mbp_invasive'].fillna(vitals_pivot['mbp_non_invasive'])

    # 2. Blood Sugar: Combine fingerstick metrics with serum or whole blood lab readings
    vitals_pivot['blood_sugar'] = (
        vitals_pivot['blood_sugar_finger']
        .fillna(vitals_pivot['blood_sugar_serum'])
        .fillna(vitals_pivot['blood_sugar_blood'])
    )

    # Clean up the pre-consolidated structural columns to keep the data clean
    columns_to_drop = [
        'sbp_invasive', 'sbp_non_invasive', 
        'dbp_invasive', 'dbp_non_invasive', 
        'mbp_invasive', 'mbp_non_invasive',
        'blood_sugar_finger', 'blood_sugar_serum', 'blood_sugar_blood'
    ]
    vitals_pivot = vitals_pivot.drop(columns=columns_to_drop)

    # --- AGGREGATION PATTERN ---
    # We now group by patient ICU stay and collect the latest/first available entries
    aggregation_schema = {
        'heart_rate': 'last',
        'resp_rate': 'last',
        'resp_rate_total': 'last',
        'spo2': 'last',
        'temp_c': 'last',
        'sbp': 'last',
        'dbp': 'last',
        'mbp': 'last',
        'vein_pressure': 'last',
        'pulmonary_art_sbp': 'last',
        'pulmonary_art_dbp': 'last',
        'blood_sugar': 'last',
        'pain_level_score': 'last',
        'pain_assessment': 'last',
        'gcs_verbal': 'last',
        'gcs_motor': 'last',
        'gcs_eyes': 'last'
    }

    df = (
        vitals_pivot
        .groupby(['subject_id', 'hadm_id', 'stay_id'])
        .agg(aggregation_schema)
        .reset_index()
    )

    logger.info(f"Unique patients retrieved: {len(df['subject_id'].unique())}")
    logger.debug(f"Chosen dataset: {os.path.basename(str(df_filepath)).upper()}")
    logger.debug(f"DF head:\n{df.head()}")
    logger.debug(f"DF summary:\n{df.describe()}")
    logger.debug(f"DF cols:\n{df.columns}")

    # UNCOMMENT IF YOU NEED THE PROCESSED MIMIC DF FILE
    # df.to_csv(DATASETS_DIR / 'clean_mimic.csv')

    return df

def load_mock_dataset()->pd.DataFrame:
    """
    Loads the artifically generated test dataframe
    Patients are rows, vitals are columns.
    Returns:
        df (pd.DataFrame): Processed MIMIC dataframe. 
    """
    logger.info(f"Selected MOCK dataset.")
    
    df_filepath = DATASETS_DIR / 'mock_dataset.csv'
    df = pd.read_csv(df_filepath)
    
    logger.info(f"Unique patients retrieved: {len(df['Patient_ID'].unique())}")
    logger.debug(f"Chosen dataset: {os.path.basename(str(df_filepath)).upper()}")
    logger.debug(f"DF head:\n{df.head()}")
    logger.debug(f"DF summary:\n{df.describe()}")
    logger.debug(f"DF cols:\n{df.columns}")

    return df

def load_dataset(uploaded_file: str)->pd.DataFrame:
    """
    Loads the dataframe from CSV file
    Patients must be rows, vitals must be columns.
    Args:
        uploaded_file (str): File path to CSV. 
    Returns:
        df (pd.DataFrame): Processed MIMIC dataframe. 
    """
    uploaded_file.seek(0)

    df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
    
    logger.debug(f"DF head:\n{df.head()}")
    logger.debug(f"DF summary:\n{df.describe()}")
    logger.debug(f"DF cols:\n{df.columns}")

    return df

if __name__ == "__main__":
    df = load_mimic_dataset()