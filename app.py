import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("churn_feature_columns.pkl")

st.title("Customer Churn Prediction Dashboard")
st.write("Predict whether a telecom customer is likely to churn.")


def prepare_input():
    raw_input = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": monthly_charges * max(tenure, 1),
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "PaperlessBilling": paperless_billing,
        "TechSupport": tech_support,
        "OnlineSecurity": online_security,
        "StreamingMovies": streaming_movies
    }

    input_df = pd.DataFrame([raw_input])

    input_df = pd.get_dummies(input_df, drop_first=True)

    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    return input_df

left_col, right_col = st.columns([1.2, 0.8], gap="large")


with left_col:
    st.subheader("Customer Information")

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0, 0.1)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    predict_btn = st.button("Predict Churn", use_container_width=True)


with right_col:
    st.subheader("Prediction Result")

    if predict_btn:
        input_df = prepare_input()

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        if probability > 0.6:
            st.error("High Risk of Churn")
        elif probability > 0.4:
            st.warning("Moderate Risk of Churn")
        else:
            st.success("Customer likely to stay")

        st.metric("Churn Probability", f"{probability * 100:.2f}%")

        with st.expander("Input Data Used for Prediction"):
            st.dataframe(input_df, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Customer Profile Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Gender:** {gender}")
            st.write(f"**Tenure:** {tenure} Months")
            st.write(f"**Contract:** {contract}")
        with col2:
            st.write(f"**Monthly:** ${monthly_charges}")

            st.write(f"**Total Bill:** ${monthly_charges * tenure:.2f}")
            st.write(f"**Internet:** {internet_service}")
       
        st.progress(float(probability))
        
        if probability > 0.5:
            st.warning("High risk of losing this customer!")
        else:
            st.info("Customer looks stable for now.")