import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load frequency maps and model
with open("Frequency_maps.joblib", "rb") as file:
    frequency = joblib.load(file)

with open("final_model.joblib", "rb") as file:
    model = joblib.load(file)

st.title("Startup Funding Prediction Model")
st.header(
    "This app uses a machine learning model to estimate how much funding a startup "
    "can attract, using key business attributes such as industry, location, investors, "
    "and funding type."
)

# User inputs
startup_name = st.text_input("Enter Startup Name")
industry_vertical = st.text_input("Enter Industry Vertical")
city_location = st.text_input("Enter City Location")
investors_name = st.text_input("Enter Investor Name")
investment_type = st.text_input("Enter Investment Type")

# Make sure these match training column names exactly
cat_cols = [
    "Startup Name",
    "Industry Vertical",
    "City  Location",   # check spacing!
    "Investors Name",
    "InvestmentnType",  # check spelling!
]

if st.button("PREDICT FUNDING AMOUNT"):
    if not all([startup_name, industry_vertical, city_location, investors_name, investment_type]):
        st.warning("Please fill in all the fields")
    else:
        record = {
            "Startup Name": startup_name,
            "Industry Vertical": industry_vertical,
            "City  Location": city_location,
            "Investors Name": investors_name,
            "InvestmentnType": investment_type,
        }

        df = pd.DataFrame([record])

        # Use stored frequency maps instead of value_counts()
        for col in cat_cols:
            mapping = frequency.get(col, {})
            df[col] = df[col].map(mapping).fillna(0)

        # Predict
        log_pred = model.predict(df)[0]
        amount_pred = np.expm1(log_pred)

        st.subheader("✅ Prediction Result")
        st.write(f"**Predicted log funding value:** `{log_pred:.4f}`")
        st.write(f"**Estimated Funding Amount (USD):** :green[**${amount_pred:,.2f}**]")

        st.info(
            "Note: This is an approximate prediction based on historical data patterns. "
            "Real-world funding can vary due to many qualitative and market factors."
        )

st.markdown("---")
