# HeartScope UI update

## Railway deployment

Upload the extracted project files to a GitHub repository, with `app.py`, `Dockerfile`, `requirements-runtime.txt`, and `railway.json` at the repository root. Connect that repository to a new Railway service. The Docker build installs only dashboard dependencies and trains models during the build. The app listens on Railway's assigned PORT and provides the Streamlit health endpoint. Generate a service domain after the deployment succeeds. Hosting follows your Railway account's usage and billing limits.

The updated dashboard starts from `app.py`. The previous interface is retained in `app_legacy.py`. The data preparation and model training code in `src` is unchanged.

## Run on Windows

Extract this ZIP to a new folder. Open a terminal in the folder containing `app.py`, then run:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

If you use an existing environment, install the updated requirements there before starting the app. Altair powers the interactive charts and is also installed by Streamlit.

## What changed

- Five main sections: Overview, Data explorer, Health insights, Model lab, and Prediction studio.
- Light blue-and-white design, consistent statistic cards, responsive columns, visible keyboard focus and readable labels.
- Interactive charts with hover details, age/sex cohort filters, reset control and an empty-selection message.
- Data-quality tabs and downloadable filtered records and model metrics.
- Grouped prediction inputs, explained units and unknown values, submitted-profile summary and clear model-output wording.
- Models load/train only in Model lab and Prediction studio. First-time training is shown with a loading message.

Cohort filters change exploratory charts and selected-record downloads. Model results always refer to the original full held-out test set. No patient information is sent to an external prediction service.

## Quick verification

Open each navigation section. Change age/sex filters, clear all sex options to see the empty state, and use Reset filters. Inspect Health insights tabs and switch the measurement field. In Model lab change comparison metric and confusion-matrix model. Generate a sample prediction and try zero cholesterol to check median-imputation feedback.

For a phone preview, reduce your browser width below 768px; cards and form columns should stack, and the sidebar can be opened using Streamlit's menu control. Wide data tables scroll horizontally.

The demo guide describes the original eight-section interface; use the five section names above when demonstrating this version.
