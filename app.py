import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("linear_model.pkl")

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Wine Quality Prediction App")
st.write(
    "Predict **Wine Quality** using a Linear Regression model "
    "trained on the UCI Wine Quality (Red) dataset."
)

st.subheader("🔢 Enter Wine Chemical Properties")

# Input features (11)
fixed_acidity = st.number_input("Fixed Acidity", 0.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.0)
citric_acid = st.number_input("Citric Acid", 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.0)
chlorides = st.number_input("Chlorides", 0.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0)
density = st.number_input("Density", 0.0)
pH = st.number_input("pH", 0.0)
sulphates = st.number_input("Sulphates", 0.0)
alcohol = st.number_input("Alcohol", 0.0)

if st.button("🍷 Predict Wine Quality"):
    input_data = np.array([[ 
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]])

    prediction = model.predict(input_data)

    # ✅ Cloud-safe scalar extraction
    prediction_value = prediction.ravel()[0]

    st.success(
        f"✅ Predicted Wine Quality: **{round(prediction_value, 2)}**"
    )
    st.info(
        f"🍷 Quality Class (Approx): **{int(round(prediction_value))}**"
    )

st.markdown("---")
st.caption("Model: Linear Regression | Dataset: UCI Wine Quality (Red)")
