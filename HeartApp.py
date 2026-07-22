import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("heart_model.joblib")
scaler = joblib.load("heart_scaler.joblib")

st.title("Heart Disease Prediction")

age = st.number_input("Age", 18, 100, 50)
chol = st.number_input("Cholesterol", 100, 600, 200)
restingBP = st.number_input("Resting BP",100,160,110)
maxHR = st.number_input("Maximum HR",60,200,120)
oldpeak= st.number_input("Oldpeak",-2,6,1)

sex = st.selectbox("Sex", ["M", "F"])
chestPainType = st.selectbox("Chest pain type",["ATA", "NAP", "ASY", "TA"])
exerciseAngina = st.selectbox("Exercise Angina",["Y","N"])
stSlope = st.selectbox("ST Slope",["Flat","Up","Down"])

if st.button("Predict"):
    sample_data = {
        "Age": age,
        "Sex": sex,
        "Cholesterol": chol,
        "RestingBP": restingBP,
        "MaxHR": maxHR,
        "Oldpeak": oldpeak,
        "ExerciseAngina": exerciseAngina,
        "ST_Slope": stSlope,
        "ChestPainType": chestPainType
    }
    df = pd.DataFrame([sample_data])

    df = pd.get_dummies(df, columns=["Sex", "ChestPainType", "ExerciseAngina", "ST_Slope"])

    numeric_cols = ["Age", "Cholesterol", "RestingBP", "MaxHR", "Oldpeak"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    pred = model.predict(df)
    st.write("Result:", "Heart Disease" if pred[0]==1 else "No Heart Disease")