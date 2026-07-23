"""
Streamlit app for the CatBoost Credit Score model trained in bank.ipynb.

Required files in the same folder as this script:
  - catboost_model.joblib     (already saved by the notebook)
  - label_encoders.joblib     (dict of {column_name: fitted LabelEncoder})

The notebook only saves the CatBoost model. Add this line right after
    joblib.dump(cat_model, "catboost_model.joblib")
and re-run that cell once:

    joblib.dump(label_encoders, "label_encoders.joblib")

Run locally with:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Score Predictor", page_icon="💳", layout="centered")

# ----------------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("catboost_model.joblib")
    except FileNotFoundError:
        model = None
    try:
        encoders = joblib.load("label_encoders.joblib")
    except FileNotFoundError:
        encoders = None
    return model, encoders


model, label_encoders = load_artifacts()

# Exact column order the model was trained on
FEATURE_ORDER = [
    "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Type_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
    "Payment_of_Min_Amount", "Total_EMI_per_month", "Amount_invested_monthly",
    "Payment_Behaviour", "Monthly_Balance",
]

CATEGORICAL_COLS = [
    "Occupation", "Type_of_Loan", "Credit_Mix",
    "Payment_of_Min_Amount", "Payment_Behaviour",
]

OCCUPATION_OPTIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer",
    "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
    "Media_Manager", "Musician", "Scientist", "Teacher", "Writer", "_______",
]
CREDIT_MIX_OPTIONS = ["Good", "Standard", "Bad", "_"]
PAYMENT_MIN_AMOUNT_OPTIONS = ["Yes", "No", "NM"]
PAYMENT_BEHAVIOUR_OPTIONS = [
    "High_spent_Small_value_payments", "High_spent_Medium_value_payments",
    "High_spent_Large_value_payments", "Low_spent_Small_value_payments",
    "Low_spent_Medium_value_payments", "Low_spent_Large_value_payments",
]

st.title("💳 Credit Score Predictor")
st.caption("Predicts Poor / Standard / Good credit score using a CatBoost model.")

if model is None:
    st.error("`catboost_model.joblib` was not found. Place it next to app.py.")
    st.stop()

if label_encoders is None:
    st.warning(
        "`label_encoders.joblib` was not found — categorical inputs will not be "
        "encoded correctly. See the instructions at the top of app.py."
    )

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
with st.form("credit_form"):
    st.subheader("Personal & Income")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS)
        annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
        monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0, step=100.0)
    with c2:
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, value=3)
        num_credit_card = st.number_input("Number of Credit Cards", min_value=0, value=3)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0, value=10)
        num_of_loan = st.number_input("Number of Loans", min_value=0, value=2)

    st.subheader("Loans & Payments")
    c3, c4 = st.columns(2)
    with c3:
        type_of_loan = st.text_input(
            "Type of Loan (comma separated)", value="Not Specified",
            help="e.g. 'Auto Loan, Credit-Builder Loan'"
        )
        delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=0, value=10)
        num_of_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, value=5)
        changed_credit_limit = st.number_input("Changed Credit Limit", value=5.0, step=0.1)
    with c4:
        num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0.0, value=3.0)
        credit_mix = st.selectbox("Credit Mix", CREDIT_MIX_OPTIONS)
        outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=800.0, step=10.0)
        credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, value=30.0)

    st.subheader("History & Balance")
    c5, c6 = st.columns(2)
    with c5:
        credit_history_age = st.number_input(
            "Credit History Age (months)", min_value=0, value=200,
            help="Total months, e.g. 22 years 3 months = 267"
        )
        payment_of_min_amount = st.selectbox("Payment of Minimum Amount", PAYMENT_MIN_AMOUNT_OPTIONS)
        total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, value=100.0, step=10.0)
    with c6:
        amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, value=100.0, step=10.0)
        payment_behaviour = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOUR_OPTIONS)
        monthly_balance = st.number_input("Monthly Balance", value=300.0, step=10.0)

    submitted = st.form_submit_button("Predict Credit Score", use_container_width=True)

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
def safe_encode(value, col):
    """Encode a categorical value; fall back to the first known class if unseen."""
    le = label_encoders[col]
    if value not in le.classes_:
        st.info(f"'{value}' was not seen during training for '{col}' — using a fallback category.")
        value = le.classes_[0]
    return int(le.transform([value])[0])


if submitted:
    row = {
        "Age": age,
        "Occupation": occupation,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_inhand_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_of_loan,
        "Type_of_Loan": type_of_loan,
        "Delay_from_due_date": delay_from_due_date,
        "Num_of_Delayed_Payment": num_of_delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": num_credit_inquiries,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization_ratio,
        "Credit_History_Age": credit_history_age,
        "Payment_of_Min_Amount": payment_of_min_amount,
        "Total_EMI_per_month": total_emi_per_month,
        "Amount_invested_monthly": amount_invested_monthly,
        "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": monthly_balance,
    }

    if label_encoders is not None:
        for col in CATEGORICAL_COLS:
            row[col] = safe_encode(row[col], col)

    input_df = pd.DataFrame([row])[FEATURE_ORDER]

    prediction = model.predict(input_df)
    pred_label = np.array(prediction).flatten()[0]

    st.subheader("Result")
    color = {"Good": "green", "Standard": "orange", "Poor": "red"}.get(pred_label, "blue")
    st.markdown(f"### Predicted Credit Score: :{color}[{pred_label}]")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        classes = model.classes_.flatten()
        proba_df = pd.DataFrame({"Class": classes, "Probability": proba}).sort_values(
            "Probability", ascending=False
        )
        st.bar_chart(proba_df.set_index("Class"))