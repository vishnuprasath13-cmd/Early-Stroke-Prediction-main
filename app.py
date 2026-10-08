import pandas as pd
import streamlit as st

from model import explain, risk_band, train

st.set_page_config(page_title="Early Stroke Risk Estimator", page_icon="🧠", layout="centered")


@st.cache_resource(show_spinner="Training model on first load...")
def get_model():
    return train()


pipe, auc, base_rate = get_model()

st.title("🧠 Early Stroke Risk Estimator")
st.caption("A machine-learning demo trained on 5,100 patient records. "
           "**Educational only. Not a medical device or a diagnosis.**")

with st.form("patient"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 1, 100, 50)
    gender = c2.selectbox("Gender", ["Female", "Male"])
    glucose = c1.number_input("Average glucose level (mg/dL)", 50.0, 300.0, 105.0)
    bmi = c2.number_input("BMI", 10.0, 60.0, 25.0)
    hypertension = c1.selectbox("Hypertension", ["No", "Yes"])
    heart = c2.selectbox("Heart disease", ["No", "Yes"])
    married = c1.selectbox("Ever married", ["Yes", "No"])
    work = c2.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence = c1.selectbox("Residence", ["Urban", "Rural"])
    smoking = c2.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    submitted = st.form_submit_button("Estimate risk", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "gender": gender, "age": age, "hypertension": int(hypertension == "Yes"),
        "heart_disease": int(heart == "Yes"), "ever_married": married, "work_type": work,
        "Residence_type": residence, "avg_glucose_level": glucose, "bmi": bmi,
        "smoking_status": smoking,
    }])
    p = float(pipe.predict_proba(row)[0, 1])
    band = risk_band(p, base_rate)
    color = {"Low": "green", "Moderate": "orange", "High": "red"}[band]

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated stroke risk", f"{p:.1%}")
    m2.metric("Average in dataset", f"{base_rate:.1%}")
    m3.markdown(f"**Risk band**\n\n### :{color}[{band}]")

    st.subheader("What drove this estimate")
    contrib = explain(pipe, row)
    contrib["direction"] = contrib["contribution"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
    contrib["feature"] = contrib["feature"].str.replace(r"^(num|cat)__", "", regex=True)
    st.bar_chart(contrib.set_index("feature")["contribution"], horizontal=True)
    st.caption("Contribution to log-odds relative to an average person. Positive raises risk, negative lowers it.")

    if band != "Low":
        st.warning("Several factors here are associated with higher stroke risk. "
                   "Consider discussing them with a doctor.")

with st.expander("About this model"):
    st.markdown(f"""
- **Model:** logistic regression with median imputation, scaling and one-hot encoding.
- **Data:** Kaggle *Healthcare Stroke Dataset* (5,109 records, ~{base_rate:.1%} strokes).
- **Performance:** cross-validated ROC AUC **{auc:.2f}**.
- **Limits:** the dataset is small and heavily imbalanced, so treat the number as a
  rough relative indicator, not a clinical prediction. Symptoms such as face drooping,
  arm weakness or slurred speech need emergency care immediately.
""")
