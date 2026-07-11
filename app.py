import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

add_on_services_col = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]


@st.cache_resource
def load():
    return (joblib.load("best_model.joblib"),
            joblib.load("model_meta.joblib"))


def make_features(d):
    """Recreate the engineered features used at training time."""
    df = pd.DataFrame([d])
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 60, 72],
                                labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"])
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in add_on_services_col)
    df["avg_charges_per_tenure"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["no_protection"] = ((df["OnlineSecurity"] != "Yes") &
                           (df["TechSupport"] != "Yes")).astype(int)
    return df


def main():
    st.set_page_config(page_title="Telco Churn Predictor", page_icon="📞")
    st.title("Customer Churn Predictor")
    st.write("Estimate whether a customer is likely to **churn**, based on their "
             "account profile and subscribed services.")

    model, meta = load()
    st.caption(f"Loaded model: **{meta['best_name']}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with c3:
        payment = st.selectbox("Payment Method",
                               ["Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        total = st.number_input("Total Charges ($)", 0.0, 9000.0, float(monthly * tenure))

    st.write("#### Add-on services")
    sc = st.columns(3)
    services = {}
    opts = ["No", "Yes", "No internet service"]
    for i, svc in enumerate(add_on_services_col):
        services[svc] = sc[i % 3].selectbox(svc, opts)

    if st.button("Predict Churn", type="secondary"):
        row = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multi, "InternetService": internet,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total, **services,
        }
        X = make_features(row)
        proba = model.predict_proba(X)[0, 1]
        pred = "WILL CHURN" if proba >= 0.5 else "will stay"
        if proba >= 0.5:
            st.error(f"### ⚠️ Prediction: customer {pred}")
        else:
            st.success(f"### ✅ Prediction: customer {pred}")
        st.metric("Churn probability", f"{proba:.1%}")
        st.progress(float(proba))


if __name__ == "__main__":
    main()
