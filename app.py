"""HeartScope dashboard. Run: python -m streamlit run app.py"""
import json
import sys
from html import escape
from pathlib import Path
import joblib
import pandas as pd
import altair as alt
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data_utils import ALL_FEATURES, CHEST_PAIN_NAMES, TARGET, clean_data, load_data, predict_one_patient
from train_model import MODEL_FILE, RESULTS_FILE, train_and_evaluate

st.set_page_config(page_title="HeartScope | Heart Disease Explorer", page_icon="♥", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets' / 'dashboard.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
BLUE, CORAL = "#2563eb", "#ea7759"
COLORS = {"No disease": BLUE, "Disease": CORAL}


def html(text):
    st.markdown(text, unsafe_allow_html=True)


def section(title, subtitle=""):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def stats(items):
    for col, (label, value, detail) in zip(st.columns(len(items)), items):
        with col:
            html(f'<div class="stat"><span>{escape(str(label))}</span><strong>{escape(str(value))}</strong><small>{escape(str(detail))}</small></div>')


def chart(fig):
    fig = fig.properties(height=310).configure_view(strokeWidth=0).configure_axis(
        labelColor="#62738b", titleColor="#62738b", gridColor="#e9eef5", domain=False,
        labelFontSize=11, titleFontSize=12).configure_legend(orient="top", title=None)
    st.altair_chart(fig, use_container_width=True)


def diagnosis_color():
    return alt.Color("Diagnosis:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())), title=None)


def histogram(data, field, title):
    return alt.Chart(data).mark_bar(opacity=.75, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(field + ":Q", bin=alt.Bin(maxbins=16), title=title),
        y=alt.Y("count():Q", title="Records", stack=None), color=diagnosis_color(),
        tooltip=[alt.Tooltip(field + ":Q", bin=True, title=title), "Diagnosis:N", alt.Tooltip("count():Q", title="Records")])


def heatmap(values, row_labels, col_labels, row_title, col_title, correlation=False):
    cells = pd.DataFrame([{row_title: r, col_title: c, "Value": float(values[i][j])}
                          for i, r in enumerate(row_labels) for j, c in enumerate(col_labels)])
    base = alt.Chart(cells).encode(x=alt.X(col_title + ":N", sort=col_labels, axis=alt.Axis(labelAngle=-30)),
                                  y=alt.Y(row_title + ":N", sort=row_labels))
    color_scale = alt.Scale(scheme="redblue", domain=[-1, 1]) if correlation else alt.Scale(scheme="blues")
    rect = base.mark_rect(cornerRadius=4).encode(color=alt.Color("Value:Q", scale=color_scale), tooltip=list(cells.columns))
    text = base.mark_text(fontSize=13).encode(text=alt.Text("Value:Q", format=".2f" if correlation else ".0f"),
        color=alt.condition(alt.datum.Value > (.55 if correlation else cells.Value.max() * .55), alt.value("white"), alt.value("#17243b")))
    return rect + text


@st.cache_data
def get_data():
    raw = load_data(str(ROOT / "data" / "heart.csv"))
    clean, info = clean_data(raw)
    clean["Diagnosis"] = clean[TARGET].map({0: "No disease", 1: "Disease"})
    clean["Gender"] = clean["Sex"].map({"M": "Male", "F": "Female"})
    return raw, clean, info


@st.cache_resource
def get_models():
    if not (Path(MODEL_FILE).exists() and Path(RESULTS_FILE).exists()):
        train_and_evaluate(str(ROOT / "data" / "heart.csv"), verbose=False)
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)
    return joblib.load(MODEL_FILE), results


if not (ROOT / "data" / "heart.csv").exists():
    st.error("Dataset missing. Add heart.csv to the data folder, then reload.")
    st.stop()
raw, df, info = get_data()

with st.sidebar:
    html('<div class="brand"><span class="brand-icon">♥</span><div>HeartScope<small>HEART DISEASE EXPLORER</small></div></div>')
    st.caption("WORKSPACE")
    page = st.radio("Main navigation", ["Overview", "Data explorer", "Health insights", "Model lab", "Prediction studio"], label_visibility="collapsed", key="page")
    st.divider()
    html('<div class="sidebar-note"><b>Explore with context</b><p>Dataset patterns and model outputs for learning and research.</p></div>')
    st.caption("Educational use only · Not a medical diagnosis")

descriptions = {
    "Overview": ("A clearer view of heart health data.", "Explore the cohort, understand patterns and evaluate machine learning models."),
    "Data explorer": ("Know the data behind the model.", "Inspect records, review data quality and download your selected cohort."),
    "Health insights": ("Turn measurements into insights.", "Explore associations across demographics and health measurements."),
    "Model lab": ("Compare evidence, not just accuracy.", "Inspect test metrics, classification errors and feature importance."),
    "Prediction studio": ("Explore a model prediction.", "Enter a sample profile and see how a trained classifier responds."),
}
html(f'<div class="eyebrow">HEARTSCOPE / {page.upper()}</div>')
st.title(descriptions[page][0])
st.markdown(descriptions[page][1])
html('<div class="notice"><b>Educational project</b><span>Model estimates are not diagnoses or validated estimates of personal clinical risk.</span></div>')


def reset_filters():
    st.session_state.age_filter = (int(df.Age.min()), int(df.Age.max()))
    st.session_state.sex_filter = ["Male", "Female"]


def cohort_filters():
    with st.expander("Cohort filters · age and sex"):
        a, b = st.columns(2)
        with a:
            ages = st.slider("Age range (years)", int(df.Age.min()), int(df.Age.max()),
                             (int(df.Age.min()), int(df.Age.max())), key="age_filter")
        with b:
            genders = st.multiselect("Sex recorded in dataset", ["Male", "Female"], default=["Male", "Female"], key="sex_filter")
        st.button("Reset filters", on_click=reset_filters)
    selected = df[df.Age.between(*ages) & df.Gender.isin(genders)]
    st.caption(f"Showing {len(selected):,} of {len(df):,} cleaned records · Charts reflect the selected cohort")
    if selected.empty:
        st.info("No matching records. Choose at least one sex or widen the age range.")
        st.stop()
    return selected


if page in ["Overview", "Data explorer", "Health insights"]:
    cohort = cohort_filters()

if page == "Overview":
    cases = int(cohort[TARGET].sum())
    stats([("Cohort records", f"{len(cohort):,}", "After duplicate removal"),
           ("Disease-labelled records", f"{cases / len(cohort):.1%}", f"{cases:,} records in selected cohort"),
           ("Median age", f"{cohort.Age.median():.0f} years", "Selected cohort"),
           ("Model inputs", len(ALL_FEATURES), "Numeric and categorical features")])
    left, right = st.columns([1, 1.35])
    with left, st.container(border=True):
        section("Cohort composition", "Recorded diagnosis labels, not model predictions")
        counts = cohort.Diagnosis.value_counts().reindex(["No disease", "Disease"], fill_value=0)
        slices = counts.rename_axis("Diagnosis").reset_index(name="Records")
        slices["Share"] = slices.Records / slices.Records.sum()
        fig = alt.Chart(slices).mark_arc(innerRadius=85, outerRadius=120, cornerRadius=5, padAngle=.03).encode(
            theta="Records:Q", color=diagnosis_color(), tooltip=["Diagnosis:N", "Records:Q", alt.Tooltip("Share:Q", format=".1%")])
        center = alt.Chart(pd.DataFrame({"Total": [f"{len(cohort):,}"]})).mark_text(fontSize=30, fontWeight="bold", color="#17243b").encode(text="Total:N")
        chart(fig + center)
    with right, st.container(border=True):
        section("Age profile", "Compare the two groups across age ranges")
        chart(histogram(cohort, "Age", "Age (years)"))
    with st.container(border=True):
        section("Start exploring", "Follow the data from records to a prediction")
        for col, title, desc, dest in zip(st.columns(3), ["Inspect the dataset", "Compare the models", "Try a sample profile"],
                                        ["Review quality checks and source records.", "Understand performance and classification errors.", "Explore an educational model output."],
                                        ["Data explorer", "Model lab", "Prediction studio"]):
            with col:
                st.markdown(f"**{title}**")
                st.caption(desc)
                st.button("Open " + dest.lower(), key=dest, on_click=lambda d=dest: st.session_state.update(page=d), use_container_width=True)

elif page == "Data explorer":
    stats([("Original records", f"{len(raw):,}", "Source CSV"),
           ("Duplicates removed", info["duplicates_removed"], "Exact duplicate rows"),
           ("Missing cells after cleaning", int(df[ALL_FEATURES].isna().sum().sum()), "Includes invalid zero measurements")])
    records, quality, summary = st.tabs(["Records", "Data quality", "Summary statistics"])
    with records:
        st.caption("Cleaned records · blank values are imputed during model preprocessing")
        data = cohort.drop(columns=["Diagnosis", "Gender"])
        st.dataframe(data, hide_index=True, use_container_width=True)
        st.download_button("Download selected records (CSV)", data.to_csv(index=False), "heartscope_cohort.csv", "text/csv")
    with quality:
        st.info("RestingBP and Cholesterol values recorded as zero are treated as missing. The model fills them using training-set medians.")
        cols = ALL_FEATURES + [TARGET]
        st.dataframe(pd.DataFrame({"Field": cols, "Type": [str(raw[c].dtype) for c in cols],
                                  "Missing in original": [int(raw[c].isna().sum()) for c in cols],
                                  "Missing after cleaning": [int(df[c].isna().sum()) for c in cols]}), hide_index=True, use_container_width=True)
        st.caption("Quality checks use the full dataset, regardless of cohort filters.")
    with summary:
        st.dataframe(cohort[ALL_FEATURES].describe().round(2), use_container_width=True)
    st.markdown("[Dataset source: fedesoriano / Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)")

elif page == "Health insights":
    tab1, tab2, tab3 = st.tabs(["Demographics", "Measurements", "Relationships"])
    with tab1:
        a, b = st.columns(2)
        with a, st.container(border=True):
            section("Age by diagnosis")
            chart(alt.Chart(cohort).mark_boxplot(size=45).encode(x="Diagnosis:N", y=alt.Y("Age:Q", title="Age (years)"), color=diagnosis_color()))
        with b, st.container(border=True):
            section("Recorded sex and diagnosis")
            groups = cohort.groupby(["Gender", "Diagnosis"]).size().reset_index(name="Records")
            chart(alt.Chart(groups).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(x="Gender:N", xOffset="Diagnosis:N", y="Records:Q", color=diagnosis_color(), tooltip=list(groups.columns)))
        st.caption("Counts depend on the cohort's composition; differences do not establish causation.")
    with tab2:
        measurement = st.selectbox("Measurement", ["Cholesterol", "RestingBP", "MaxHR", "Oldpeak"])
        units = {"Cholesterol": "Cholesterol (mg/dL)", "RestingBP": "Resting blood pressure (mm Hg)", "MaxHR": "Maximum heart rate (bpm)", "Oldpeak": "ST depression (Oldpeak)"}
        available = cohort.dropna(subset=[measurement])
        st.caption(f"{len(available):,} valid readings · {len(cohort)-len(available):,} missing readings excluded")
        if available.empty:
            st.info("No valid readings for this measurement in the selected cohort.")
        else:
            chart(histogram(available, measurement, units[measurement]))
    with tab3:
        a, b = st.columns(2)
        with a, st.container(border=True):
            section("Chest pain patterns")
            groups = cohort.assign(ChestPain=cohort.ChestPainType.map(CHEST_PAIN_NAMES)).groupby(["ChestPain", "Diagnosis"]).size().reset_index(name="Records")
            chart(alt.Chart(groups).mark_bar().encode(y=alt.Y("ChestPain:N", title=None), x="Records:Q", color=diagnosis_color(), tooltip=list(groups.columns)))
        with b, st.container(border=True):
            section("Numeric correlations", "Pearson correlation · pairwise valid readings")
            corr = cohort[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", TARGET]].corr()
            chart(heatmap(corr.values, list(corr.index), list(corr.columns), "Row", "Column", correlation=True))

elif page == "Model lab":
    with st.spinner("Preparing models and results. The first run may take a little longer…"):
        saved, results = get_models()
    best = results["best_model"]
    m = results["models"][best]
    stats([("Selected model", best, "Highest training cross-validation F1"),
           ("Test accuracy", f'{m["accuracy"]:.1%}', "Full held-out test set"),
           ("Test recall", f'{m["recall"]:.1%}', "Disease-labelled records found"),
           ("Test F1 score", f'{m["f1"]:.3f}', "Balance of precision and recall")])
    compare, errors, features = st.tabs(["Model comparison", "Classification errors", "Feature importance"])
    with compare:
        metric = st.selectbox("Compare metric", ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1_mean"], format_func=lambda s: {"f1": "F1 score", "roc_auc": "ROC AUC", "cv_f1_mean": "Training CV F1"}.get(s, s.title()))
        table = pd.DataFrame([{"Model": n, **{k: val for k, val in v.items() if k != "confusion_matrix"}} for n, v in results["models"].items()])
        fig = alt.Chart(table).mark_bar(color=BLUE, cornerRadiusEnd=5).encode(
            x=alt.X(metric + ":Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            y="Model:N", tooltip=["Model:N", alt.Tooltip(metric + ":Q", format=".3f")])
        chart(fig)
        st.dataframe(table.round(3), hide_index=True, use_container_width=True)
        st.caption(f'Train: {results["dataset"]["n_train"]} · Test: {results["dataset"]["n_test"]} · Majority-class test baseline: {results["baseline_accuracy"]:.1%}')
        st.info("Selection uses the highest mean five-fold F1 on training data. Test metrics use the held-out test set; cohort filters do not change evaluation.")
        st.download_button("Download model metrics (CSV)", table.to_csv(index=False), "heartscope_model_metrics.csv", "text/csv")
    with errors:
        chosen = st.selectbox("Inspect model", list(results["models"]), index=list(results["models"]).index(best))
        cm = results["models"][chosen]["confusion_matrix"]
        a, b = st.columns([1.3, 1])
        with a:
            chart(heatmap(cm, ["No disease", "Disease"], ["No disease", "Disease"], "Actual label", "Predicted label"))
        with b:
            st.markdown(f"**{cm[1][0]} false negatives** — disease-labelled records missed.")
            st.markdown(f"**{cm[0][1]} false positives** — no-disease records predicted as disease.")
            st.markdown(f"**{cm[1][1]} true positives** — disease labels correctly identified.")
            st.markdown(f"**{cm[0][0]} true negatives** — no-disease labels correctly identified.")
    with features:
        weights = pd.Series(results["feature_importance"]).sort_values().tail(12)
        chart(alt.Chart(pd.DataFrame({"Feature": weights.index, "Importance": weights.values})).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(
            x="Importance:Q", y=alt.Y("Feature:N", sort="-x"), tooltip=["Feature:N", alt.Tooltip("Importance:Q", format=".3f")]))
        st.caption("Top 12 encoded features from Random Forest. Importance describes this model, not a causal medical effect.")

else:
    with st.spinner("Preparing prediction models…"):
        saved, results = get_models()
    st.caption("Defaults are a sample profile. Generate a new prediction after changing inputs.")
    with st.form("prediction_form"):
        names = list(saved["models"])
        model = st.selectbox("Prediction model", names, index=names.index(saved["best_model"]))
        st.markdown("#### 1 · Profile and measurements")
        a, b, c = st.columns(3)
        with a:
            age = st.number_input("Age (years)", 20, 90, 54)
            sex = st.selectbox("Sex", ["M", "F"], format_func=lambda x: {"M": "Male", "F": "Female"}[x])
        with b:
            bp = st.number_input("Resting blood pressure (mm Hg)", 0, 250, 130, help="Use 0 if unknown; the model uses a training-set median.")
            chol = st.number_input("Cholesterol (mg/dL)", 0, 700, 240, help="Use 0 if unknown; the model uses a training-set median.")
        with c:
            hr = st.number_input("Maximum heart rate (bpm)", 60, 210, 140)
            sugar = st.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x else "No")
        st.markdown("#### 2 · Symptoms and cardiac tests")
        a, b, c = st.columns(3)
        with a:
            pain = st.selectbox("Chest pain type", list(CHEST_PAIN_NAMES), index=3, format_func=lambda x: CHEST_PAIN_NAMES[x])
            angina = st.selectbox("Exercise-induced angina", ["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No")
        with b:
            ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], help="ST: ST–T wave abnormality. LVH: left ventricular hypertrophy.")
            slope = st.selectbox("ST slope", ["Up", "Flat", "Down"], index=1)
        with c:
            oldpeak = st.number_input("ST depression (Oldpeak)", -3.0, 7.0, 1.0, step=.1)
            st.caption("Use recorded test measurements for a sample profile.")
        submitted = st.form_submit_button("Generate prediction", type="primary", use_container_width=True)
    if submitted:
        patient = dict(Age=age, Sex=sex, ChestPainType=pain, RestingBP=bp, Cholesterol=chol, FastingBS=sugar, RestingECG=ecg, MaxHR=hr, ExerciseAngina=angina, Oldpeak=oldpeak, ST_Slope=slope)
        label, probability = predict_one_patient(saved["models"][model], saved["bounds"], patient)
        with st.container(border=True):
            section("Prediction result", "Snapshot of the submitted profile")
            stats([("Predicted class", "Disease" if label else "No disease", f"{model} · threshold 0.50"),
                   ("Model probability: disease", f"{probability:.1%}", "Not a validated personal risk estimate")])
            st.progress(max(0., min(1., probability)))
            st.caption("Educational output only. This cannot confirm or rule out heart disease.")
            if bp == 0 or chol == 0:
                st.info("Unknown zero measurements were replaced with training-set medians.")
            with st.expander("View submitted profile"):
                st.dataframe(pd.DataFrame([patient]), hide_index=True, use_container_width=True)

st.divider()
st.caption("HeartScope · Educational heart disease exploration · Dataset associations do not establish causation")
