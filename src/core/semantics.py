import structlog

import pandas as pd

from rapidfuzz import process, fuzz

from config.config import CLINICAL_ONTOLOGY_MAP, LLM_MODEL_NAME
from utils.dataset_utils import load_mimic_dataset, load_mock_dataset
from utils.llm_utils import init_client

logger = structlog.get_logger()

# 1. CORE API CONFIGURATION
client = init_client()
MODEL_NAME =  LLM_MODEL_NAME

def verify_dataset_clinical_context(df: pd.DataFrame, score_threshold: float = 85.0) -> bool:
    """
    Verify via onthology mapping matching whether the DataFrame contains valid clincal data.
    Args:
        df (pd.DataFrame): The input DataFrame.
        score_threshold (float): The matching score threshold for a column to be considered clinical. Max should be `100.0`.
    Returns:
        bool: If `True`, the input DataFrame contains sufficient clinical data to be considered as such.
    """
    # Get mapping dict
    mapping = hybrid_schema_extractor(df,score_threshold)

    # Check waht keys contains actual values (not None)
    matched_count = sum(1 for val in mapping.values() if val is not None)

    # If more than half the values (4 out of 7) are valid, return True
    return matched_count > 3

def hybrid_schema_extractor(df: pd.DataFrame, score_threshold: float = 85.0) -> dict[str, str | None]:
    """
    Analyzes dataframe columns using an ontology-aware fuzzy matching system.
    Safely bridges short variables (like 'hr' or 'vp') to our required schema.
    Args:
        df (pd.DataFrame): The input DataFrame.
        score_threshold (float): The matching score threshold for a column to be considered clinical. Max should be `100.0`.
    Returns:
        dict: A clean map of {target_schema_key: matched_dataframe_column_or_None}
    """
    incoming_columns = [str(col).strip() for col in df.columns]
    
    # Initialize the output map
    extracted_mapping = {key: None for key in CLINICAL_ONTOLOGY_MAP.keys()}
    
    print()

    logger.info("⚡ Executing Hybrid Fuzzy-Ontology Extraction loop...")

    # We iterate over our expected targets and search the dataframe columns
    match = None
    for target_key, synonyms in CLINICAL_ONTOLOGY_MAP.items():
        best_match_col = None
        highest_score = 0.0
        
        # Cross-reference each synonym against incoming columns using edit-distance
        for synonym in synonyms:
            # extractOne finds the closest string match in a collection
            match = process.extractOne(
                synonym, 
                incoming_columns, 
                scorer=fuzz.WRatio # WRatio handles case, substring, and order variations gracefully
            )
            
            if match:
                matched_string, score, _ = match
                if score > highest_score and score >= score_threshold:
                    highest_score = score
                    best_match_col = matched_string
                    
        if best_match_col:
            extracted_mapping[target_key] = best_match_col
            logger.info(f"🎯 Mapped Target [{target_key}] ➔  Column [{best_match_col}] (Confidence: {highest_score:.1f}%)")
            # Remove mapped column to avoid double assignment across close indicators
            incoming_columns.remove(best_match_col)
        else:
            logger.warning(f"⚠️  Failed to match Target [{target_key}] to a fuzzy column. Closest match was '{matched_string}' (Confidence: {score:.1f}%)")
    print()
    return extracted_mapping

if __name__ == "__main__":
    df = load_mimic_dataset()
    # df = load_mock_dataset()
    mapping = hybrid_schema_extractor(df)
    print()
    logger.debug(f"{mapping}")
    matched_count = sum(1 for val in mapping.values() if val is not None)
    logger.debug(f"matched_count {matched_count}")
    logger.debug(f"Clinical related DF? {matched_count>3}")