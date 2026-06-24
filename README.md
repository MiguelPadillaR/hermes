# Health Evaluation reporting & Monitoring Engine System (H.E.R.M.E.S)
The Health Evaluation reporting & Monitoring Engine System (HERMES) is a Multi-Agent system that monitors patient status, evaluates current conditions, and relays an overall review to nurses/doctors so they can be prepared before visitations.

## Installation & Setup
- With `uv`:
```
uv sync
source .venv/bin/activate
```
- With `pip`:
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
## Quickstart:
```
```
## Sources
- [MIMIC-IV demo dataset from Kaggle.](https://www.kaggle.com/datasets/montassarba/mimic-iv-clinical-database-demo-2-2)

# New project approach
- Stage 1: 
  - User inputs CSV.
  - LLM fine-tuning indicates what general columns and values we are looking for.
  - LLM evaluates input CSV and retrieves most relevant columns (nearest to target columns)
    - External relevant columns are noted separately
  - Retrieved columns are associated to target columns and used to fill out and schema/json pre-report
- Stage 2: THE SAME
  - LLM perfomrs RAG to enrich pre-report and get final report.