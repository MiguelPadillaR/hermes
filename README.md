# Health Report & Monitoring System (HeRMeS)
The Health Report & Monitoring System (HeRMeS) is a LLM-based pipeline that monitors patient status, evaluates current helath conditions, and relays an overall review to healthcare professionals before visitations. It is specifically aimed to transform machine-friendly files (`.csv`) to a fully fledged medical report.

## Installation & Setup
- Clone and enter the repo:
```bash
git clone https://github.com/MiguelPadillaR/hermes.git
cd hermes
```
- Install dependencies:
  - With `uv`:
  ```bash
  uv sync
  source .venv/bin/activate
  ```
  - With `pip`:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
## Quickstart:
- Pipeline tests with default CSV's:
```bash
python src/main.py
```
- Run the interface an upload your own files:
```bash
streamlit run src/interface_streamlit.py
```
## Sources
- Synthetic datafarme for testing.
- [MIMIC-IV demo dataset from Kaggle.](https://www.kaggle.com/datasets/montassarba/mimic-iv-clinical-database-demo-2-2) (processed to adapt to expected input format).
