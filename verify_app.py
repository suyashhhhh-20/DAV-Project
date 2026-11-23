"""Smoke checks run during the deployment build."""
from streamlit.testing.v1 import AppTest

app = AppTest.from_file("app.py", default_timeout=120).run()


def check(label):
    errors = [e.message for e in app.exception]
    if errors:
        raise RuntimeError(f"{label}: {errors}")
    print(f"PASS: {label}", flush=True)


check("Overview")
app.multiselect[0].set_value([]).run()
check("Empty cohort")
app.button[0].click().run()
check("Reset cohort")
for page in ["Data explorer", "Health insights", "Model lab", "Prediction studio"]:
    app.radio[0].set_value(page).run()
    check(page)
app.button[0].click().run()
check("Sample prediction")
assert "Prediction result" in [s.value for s in app.subheader]
print("All dashboard smoke checks passed", flush=True)
