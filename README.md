# Smart Burnout Predictor
![Status](https://img.shields.io/badge/status-experimental-yellow)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Summary
`Smart Burnout Predictor` is a reproducible demo that trains a classification model to estimate daily burnout risk for work-from-home users. It combines a documented data pipeline, a Random Forest baseline, and an interactive Streamlit demo to illustrate model predictions and explainability.

This project was collaboratively developed as part of a data science initiative focusing on remote work wellness prediction.

This README is written for maintainers and engineers who will: reproduce training, extend the model, integrate the app into a CI/CD pipeline, or deploy it.

## Quickstart (developer)
1. Clone the repository and open the project root.
```powershell
git clone <repo-url>
cd WFH_Burnout_Prediction
```
2. Create and activate a virtual environment (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
3. Install dependencies:
```powershell
pip install -r requirements.txt
```
4. Run the demo UI:
```powershell
uv run streamlit run Burnout_Prediction.py
```
5. Run unit tests:
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .\.venv\Scripts\Activate.ps1
python -m pytest unit_tests/test_burnout_prediction.py
```

## Project Structure
- `Burnout_Prediction.py` — Streamlit demo and inference UI
- `work_from_home_burnout_dataset.csv` — source dataset (demo)
- `README.md` — this document

Add `src/`, `models/`, and `notebooks/` if you split training and serving code later.

## Design Decisions & Engineering Notes
- Baseline model: Random Forest — reliable, fast to iterate, interpretable via feature importances.
- Class imbalance: SMOTE applied on training folds to reduce bias toward majority class.
- Reproducibility: set random seeds for splits and model training.
- Deployment: Streamlit is used for quick demos; production should separate training and serving, use a REST API or serverless function, and add authentication.

## Data Provenance & Ethics
- Dataset: `work_from_home_burnout_dataset.csv` is included for demo purposes.
- Contains features like:
  - Work Hours
  - Screen Time
  - Meetings Count
  - Breaks Taken
  - Sleep Hours
  - Task Completion Rate
  - Burnout Score
  - Day Type (Weekday/Weekend)

## Reproducing Training (Engineer Checklist)
1. Prepare environment and install dependencies.
2. Inspect `work_from_home_burnout_dataset.csv` and validate schema.
3. Run the training script or notebook; ensure seeds are set.
4. Validate metrics on a holdout set and record them (accuracy, precision, recall, F1 per class).

## Metrics & Monitoring
- Track data drift, prediction distribution, latency, and error rates.
- Log inputs (with PII removed or hashed) and predictions for offline audits.

## Key Insights
- Higher work hours + low sleep → Higher burnout risk
- Frequent breaks reduce burnout probability
- After-hours work significantly increases risk

## Limitations
- Uses synthetic dataset (may not reflect real-world perfectly)
- Limited features (mental health, stress levels not included)
- Model can be improved with real-world data

## Contributors
- Bindhu Sahithi — Model training & prediction logic
- Rishaniya Parthasarathy — Data preprocessing, feature engineering & model evaluation
- Sharaban Tahura — UI design & styling
- Lawrence Jaba Anand — Visualizations
