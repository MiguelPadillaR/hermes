# Health Evaluation Reporting & Monitoring Engine System (H.E.R.M.E.S)
The Health Evaluation Reporting & Monitoring Engine System (HERMES) is a LLM-based pipeline that monitors patient status, evaluates current conditions, and relays an overall review to nurses/doctors so they can be prepared before visitations.

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

cd HERMES
```
## Quickstart:
```
python src/main.py
```
## Sources
- [MIMIC-IV demo dataset from Kaggle.](https://www.kaggle.com/datasets/montassarba/mimic-iv-clinical-database-demo-2-2)
