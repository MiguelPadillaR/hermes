import os
import structlog

import numpy as np
import pandas as pd

from config.config import ASSETS_DIR

logger = structlog.get_logger()

def load_mimic_dataset():
    logger.info(f"Selected MIMIC-IV ICU dataset.")
    
    base_path = ASSETS_DIR / 'datasets/montassarba/mimic-iv-clinical-database-demo-2-2/mimic-iv-clinical-database-demo-2.2'
    df_filepath = f'{base_path}/icu/chartevents.csv'
    df = pd.read_csv(df_filepath)

    vitals_map = {
        220045: 'heart_rate',
        220179: 'sbp',
        220180: 'dbp',
        220210: 'resp_rate',
        220277: 'spo2',
        223761: 'temp_f', # Temperature Fahrenheit
        223762: 'temp_c'  # Temperature Celsius
    }
    logger.debug(f"DF Columns: {df.columns}")
    # Filter
    vitals_df = df[df['itemid'].isin(vitals_map.keys())].copy()
    vitals_df['vital_name'] = vitals_df['itemid'].map(vitals_map)

    # Pivot
    vitals_pivot = vitals_df.pivot_table(
        index=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'valueuom'],
        columns='vital_name',
        values='valuenum'
    ).reset_index()

    # --- FIX TEMPERATURE ---
    # 1. If temp_c is missing but temp_f exists, convert F to C
    if 'temp_f' in vitals_pivot.columns:
        # Formula: (F - 32) * 5/9
        f_to_c = (vitals_pivot['temp_f'] - 32) * 5/9
        
        # Fill missing Celsius values with converted Fahrenheit values
        if 'temp_c' not in vitals_pivot.columns:
            vitals_pivot['temp_c'] = f_to_c
        else:
            vitals_pivot['temp_c'] = vitals_pivot['temp_c'].fillna(f_to_c)
        
        # Drop temp_f as we don't need it anymore
        vitals_pivot = vitals_pivot.drop(columns=['temp_f'])

    # --- ENSURE TEMP_C EXISTS ---
    if 'temp_c' not in vitals_pivot.columns:
        vitals_pivot['temp_c'] = np.nan
    
    df = (
        vitals_pivot
        .groupby(['subject_id', 'hadm_id', 'stay_id'])
        .agg({
            'heart_rate': 'first',
            'sbp': 'first',
            'dbp': 'first',
            'resp_rate': 'first',
            'spo2': 'first',
            'temp_c': 'first'
        })
        .reset_index()
    )

    logger.info(f"Unique patients retrieved: {len(df['subject_id'].unique())}")
    logger.debug(f"Chosen dataset: {os.path.basename(str(df_filepath)).upper()}")
    logger.debug(f"DF head:\n{df.head()}")
    logger.debug(f"DF summary:\n{df.describe()}")
    logger.debug(f"DF cols:\n{df.columns}")

    return df

def load_mock_dataset():
    logger.info(f"Selected MOCK dataset.")
    
    df_filepath = ASSETS_DIR / 'datasets/mock_dataset.csv'
    df = pd.read_csv(ASSETS_DIR / 'datasets/mock_dataset.csv')
    
    logger.info(f"Unique patients retrieved: {len(df['Patient_ID'].unique())}")
    logger.debug(f"Chosen dataset: {os.path.basename(str(df_filepath)).upper()}")
    logger.debug(f"DF head:\n{df.head()}")
    logger.debug(f"DF summary:\n{df.describe()}")
    logger.debug(f"DF cols:\n{df.columns}")

    return df

