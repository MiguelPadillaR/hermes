from core.mapping import build_pre_report

def test_build_pre_report_rendering():
    """
    Verifies that build_pre_report properly substitutes matched columns
    and handles additional leftover attributes cleanly.
    """
    patient_row_data = {
        "hr_column": 82,
        "temp_column": 37.0,
        "bp_column": "120/80",
        "spo2_column": 98,
        "pain_column": 2,
        "sugar_column": 90,
        "random_metric": "Elevated"
    }
    
    # Map EVERY required target template key to satisfy template.format()
    mapped_columns = {
        "heart_rate": "hr_column",
        "temperature": "temp_column",
        "arterial_pressure": "bp_column",
        "oxygen_saturation": "spo2_column",
        "pain_level": "pain_column",
        "blood_sugar": "sugar_column",
        "vein_pressure": None  # Explicitly unreported indicator
    }
    
    # Run the real rendering function step
    report = build_pre_report(
        patient_row_data, 
        mapped_columns, 
        extra_metrics="- **random_metric:** Elevated"
    )
    
    # Assertions to verify text replacements
    assert "82" in report
    assert "37.0" in report
    assert "Data Unreported" in report # For vein_pressure
    assert "random_metric" in report
    assert "Elevated" in report