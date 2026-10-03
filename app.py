import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom styling ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        color: #9aa0a6;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
    }
    .result-good {
        background-color: rgba(46, 204, 113, 0.12);
        border: 1px solid rgba(46, 204, 113, 0.4);
    }
    .result-bad {
        background-color: rgba(231, 76, 60, 0.12);
        border: 1px solid rgba(231, 76, 60, 0.4);
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-sub {
        color: #9aa0a6;
        font-size: 0.95rem;
    }
    .summary-box {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        background-color: rgba(127, 127, 127, 0.07);
        border: 1px solid rgba(127, 127, 127, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load model & encoders ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("XGB_credit_model.pkl")
    encoders = {
        col: joblib.load(f"{col}_encoder.pkl")
        for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
    }
    return model, encoders

model, encoders = load_artifacts()

# ---------- Header ----------
st.markdown('<div class="main-header">💳 Credit Risk Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Fill in the applicant details on the left and get an instant risk assessment.</div>',
    unsafe_allow_html=True,
)

# ---------- Sidebar: inputs ----------
with st.sidebar:
    st.header("Applicant Details")

    st.subheader("Personal")
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1,
                           help="0 = unskilled non-resident, 1 = unskilled resident, 2 = skilled, 3 = highly skilled")
    housing = st.selectbox("Housing", ["own", "rent", "free"])

    st.subheader("Financial")
    saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
    checking_accounts = st.selectbox("Checking Accounts", ["little", "moderate", "rich"])
    credit_amount = st.number_input("Credit Amount", min_value=0, value=1000, step=100)
    duration = st.number_input("Duration (months)", min_value=1, value=12)

    st.markdown("---")
    predict_clicked = st.button("🔍 Predict Risk", use_container_width=True)

# ---------- Main area ----------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Application Summary")
    summary_df = pd.DataFrame({
        "Field": ["Age", "Sex", "Job", "Housing", "Saving Accounts",
                  "Checking Accounts", "Credit Amount", "Duration"],
        "Value": [age, sex, job, housing, saving_accounts,
                  checking_accounts, f"${credit_amount:,}", f"{duration} months"],
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

with right:
    st.subheader("Result")
    if predict_clicked:
        try:
            input_df = pd.DataFrame({
                "Age": [age],
                "Sex": [encoders["Sex"].transform([sex])[0]],
                "Job": [job],
                "Housing": [encoders["Housing"].transform([housing])[0]],
                "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
                "Checking account": [encoders["Checking account"].transform([checking_accounts])[0]],
                "Credit amount": [credit_amount],
                "Duration": [duration],
            })

            pred = model.predict(input_df)[0]
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0][pred]

            if pred == 1:
                st.markdown(f"""
                <div class="result-card result-good">
                    <div class="result-title">✅ GOOD Credit Risk</div>
                    <div class="result-sub">{f"Confidence: {proba:.1%}" if proba else ""}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-bad">
                    <div class="result-title">⚠️ BAD Credit Risk</div>
                    <div class="result-sub">{f"Confidence: {proba:.1%}" if proba else ""}</div>
                </div>
                """, unsafe_allow_html=True)

            if proba is not None:
                st.progress(float(proba))

        except ValueError as e:
            st.error(f"Couldn't process this input — one of the fields has a value the model wasn't trained on.\n\n{e}")
    else:
        st.markdown("""
        <div class="summary-box">
            Fill in the details on the left, then click <b>Predict Risk</b> to see the result here.
        </div>
        """, unsafe_allow_html=True)