import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.joblib")
scaler = joblib.load("heart_scaler.joblib")
columns = joblib.load("heart_columns.joblib")
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.joblib")
scaler = joblib.load("heart_scaler.joblib")
columns = joblib.load("heart_columns.joblib")

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
    input_df = pd.DataFrame([sample_data])

    categorical = ["Sex", "ChestPainType", "ExerciseAngina", "ST_Slope"]
    input_df = pd.get_dummies(input_df, columns=categorical)

    input_df = input_df.reindex(columns=columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)
    st.write("Result:", "Heart Disease" if pred[0]==1 else "No Heart Disease")
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
        "age": age,
        "sex": sex,
        "chol": chol,
        "trestbps": restingBP,
        "thalach": maxHR,
        "oldpeak": oldpeak,
        "exang": exerciseAngina,
        "slope": stSlope,
        "cp": chestPainType
    }
    input_df = pd.DataFrame([sample_data])

    categorical = ["sex", "cp", "exang", "slope"]
    input_df = pd.get_dummies(input_df, columns=categorical)

    input_df = input_df.reindex(columns=columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)
    st.write("Result:", "Heart Disease" if pred[0]==1 else "No Heart Disease")