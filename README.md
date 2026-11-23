# Heart Disease Prediction - Data Analytics and Data Visualization Project

A complete five-stage project: **Data Loading → EDA → Preprocessing → Model Building → Streamlit Dashboard**.

> **Educational use only.** Predictions are not medical diagnoses.

**Dataset:** [Heart Failure Prediction Dataset (Kaggle, by fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
918 records, 11 features + target `HeartDisease` (1 = heart disease, 0 = no heart disease), built from five real clinical datasets with 272 duplicates already removed by the dataset author.

---

## What is inside

| Path | Purpose |
|---|---|
| `data/` | Put `heart.csv` here (Step 4 below) |
| `notebook/Heart_Disease_Prediction.ipynb` | All five stages, explained step by step |
| `src/data_utils.py` | Loading, cleaning, outlier capping, preprocessing |
| `src/eda.py` | Stage 2 as a script (saves charts to `outputs/eda/`) |
| `src/train_model.py` | Stages 3 and 4 (trains the 3 models, saves results) |
| `src/plots.py` | Chart functions used by the EDA script and the dashboard |
| `app.py` | Stage 5: Streamlit dashboard |
| `run_all.py` | Runs EDA + training + report in one command |
| `make_report.py` | Creates `docs/Project_Report.docx` from YOUR real results |
| `scripts/download_dataset.py` | Optional automatic dataset download |
| `docs/DEMO_GUIDE.md` | How to present and demo the project |
| `docs/VIVA_QUESTIONS.md` | 30 viva questions with simple answers |
| `requirements.txt` | Python packages |

---

## Step-by-step instructions (Windows PowerShell)

### Step 1 - Install Python
Install Python 3.10, 3.11 or 3.12 from https://www.python.org/downloads/ and tick **"Add Python to PATH"** during installation. Check in PowerShell:
```powershell
python --version
```

### Step 2 - Open the project folder
Unzip the project, then in PowerShell:
```powershell
cd "C:\path\to\HeartDiseaseProject"
```

### Step 3 - Create a virtual environment and install packages
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell says *"running scripts is disabled"*, run this once and then activate again:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```
You will see `(venv)` at the start of the line when it is active. Installation can take a few minutes.

### Step 4 - Get the dataset (do this manually - it always works)
1. Open https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction (a free Kaggle account is needed to download).
2. Click **Download** and unzip the file you get.
3. Copy `heart.csv` into the project's **data** folder, so the path is `data\heart.csv`.

*Optional:* try the automatic download instead: `python scripts\download_dataset.py` (it may ask for Kaggle credentials; if it fails, use the manual way).

### Step 5 - Run all stages and create the report
```powershell
python run_all.py
```
This runs the EDA (charts saved in `outputs\eda`), trains the three models, prints the metrics, and creates `docs\Project_Report.docx` filled with your real numbers. Open the report and add your name and roll number.

### Step 6 - Start the dashboard
```powershell
streamlit run app.py
```
Your browser opens at http://localhost:8501. Press `Ctrl + C` in PowerShell to stop it.
(If Streamlit asks for an email on first run, just press Enter.)

### Step 7 - Open the notebook
```powershell
jupyter notebook
```
Open `notebook\Heart_Disease_Prediction.ipynb` and choose **Kernel → Restart & Run All**.

**Google Colab instead:** upload the notebook to https://colab.research.google.com, run it, and upload `heart.csv` when the second code cell asks for it.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized | Reinstall Python and tick "Add Python to PATH", or try `py` instead of `python` |
| `FileNotFoundError ... heart.csv` | The file must be exactly `data\heart.csv` (check it is not `heart.csv.csv`) |
| `streamlit` is not recognized | Activate the environment (`.\venv\Scripts\Activate.ps1`) or run `python -m streamlit run app.py` |
| Packages fail to install | Use Python 3.10-3.12 and check your internet connection |
| Dashboard shows old numbers | Delete the `models` folder and `outputs\results.json`, then run `python run_all.py` again |

---

## How data leakage is avoided
The data is split into training and testing sets **before** medians, outlier limits, scaling and encoding are learned. All of these are learned from the training data only (using a scikit-learn Pipeline). The best model is chosen with cross-validation on the training data, so the test set is used only once, for the final evaluation.

## Honest notes
- The numbers (accuracy, class counts, etc.) are produced when **you** run the code on the real file. This project does not hard-code any result.
- Cholesterol / blood pressure values of 0 are treated as missing (medically impossible).
- The dataset is fairly small, so results can change a few percent with a different split.
