import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.joblib")
scaler = joblib.load("heart_scaler.joblib")
columns = joblib.load("heart_columns.joblib")

st.title("Heart Disease Prediction")

age = st.number_input("Age", 18, 100, 50)
restingBP = st.number_input("Resting BP", 80, 220, 120)
chol = st.number_input("Cholesterol", 50, 700, 200)
maxHR = st.number_input("Maximum HR", 60, 220, 150)
oldpeak = st.number_input("Oldpeak", -2.0, 6.0, 1.0)

sex = st.selectbox("Sex", ["F", "M"])
chestPainType = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
exerciseAngina = st.selectbox("Exercise Angina", ["N", "Y"])
stSlope = st.selectbox("ST Slope", ["Down", "Flat", "Up"])

if st.button("Predict"):

    sample_data = {
        "Age": age,
        "RestingBP": restingBP,
        "Cholesterol": chol,
        "MaxHR": maxHR,
        "Oldpeak": oldpeak,
        "Sex": sex,
        "ChestPainType": chestPainType,
        "ExerciseAngina": exerciseAngina,
        "ST_Slope": stSlope
    }

    df = pd.DataFrame([sample_data])

    numeric_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    df = pd.get_dummies(
        df,
        columns=["Sex", "ChestPainType", "ExerciseAngina", "ST_Slope"],
        drop_first=True
    )

    df = df.reindex(columns=columns, fill_value=0)

    pred = model.predict(df)

    if pred[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")