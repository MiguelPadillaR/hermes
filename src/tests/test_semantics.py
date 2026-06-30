import pandas as pd

# Import the core modules we want to protect
from core.semantics import hybrid_schema_extractor, verify_dataset_clinical_context

# =========================================================================
# TEST SUITE 1: TESTING THE HYBRID FUZZY ONTOLOGY MAPPER
# =========================================================================


def test_hybrid_schema_extractor_exact_match():
    """
    Validates that if a user uploads a CSV with perfect, expected headers,
    the fuzzy extractor aligns them with 100% precision.
    """
    # Create a dummy dataframe with ideal columns
    data = {"heart_rate": [72], "temperature": [36.5], "blood_sugar": [90]}
    df = pd.DataFrame(data)

    # Run our extraction logic
    mapping = hybrid_schema_extractor(df, score_threshold=80.0)

    # Assertions check if the output matches our exact expectations
    assert mapping["heart_rate"] == "heart_rate"
    assert mapping["temperature"] == "temperature"
    assert mapping["blood_sugar"] == "blood_sugar"
    assert mapping["vein_pressure"] is None  # Should remain None if completely absent


def test_hybrid_schema_extractor_fuzzy_shorthand():
    """
    Validates that our clinical ontology successfully maps messy, shorthand variables
    (like 'hr' or 'temp_c') down to our standard core keys using edit distance.
    """
    # Create a messy clinical dataset layout similar to real hospital telemetry
    data = {
        "hr": [80],  # Shorthand for heart_rate
        "temp_c": [37.1],  # Alternative representation for temperature
        "spo2": [98],  # Shorthand for oxygen_saturation
        "random_id": [102],  # Non-medical column
    }
    df = pd.DataFrame(data)

    mapping = hybrid_schema_extractor(df, score_threshold=80.0)

    # Verify the fuzzy mapper bridges abbreviations down to standard targets
    assert mapping["heart_rate"] == "hr"
    assert mapping["temperature"] == "temp_c"
    assert mapping["oxygen_saturation"] == "spo2"
    assert mapping["blood_sugar"] is None


# =========================================================================
# TEST SUITE 2: TESTING THE LOCAL CHROMADB EDGE GUARDRAIL
# =========================================================================


def test_verify_dataset_clinical_context_clinical_valid():
    """
    Validates that a legitimate patient vital dataframe scores highly
    and passes the vector semantic similarity guardrail.
    """
    # Sample clinical headers
    valid_df = pd.DataFrame(columns=["patient_id", "hr", "bp", "temperature", "spo2"])

    # Run the guardrail check using the standard threshold
    is_valid = verify_dataset_clinical_context(valid_df, score_threshold=0.40)

    # This dataframe should be recognized as medical data
    assert is_valid is True


def test_verify_dataset_clinical_context_agriculture_invalid():
    """
    Validates that an invalid dataset layout (like crop tracking metrics)
    is recognized as non-clinical and rejected immediately by the mathematical barrier.
    """
    # Create an obviously out-of-domain layout structure
    crop_df = pd.DataFrame(
        columns=["corn_yield_per_acre", "soil_nitrogen_level", "annual_rainfall_inches"]
    )

    # Run the guardrail check
    is_valid = verify_dataset_clinical_context(crop_df, score_threshold=0.40)

    # This should be flagged as INVALID and rejected (False)
    assert is_valid is False
