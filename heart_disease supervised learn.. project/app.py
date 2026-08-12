import streamlit as st
import pandas as pd
import joblib as jb

model = jb.load("KNN_heart.pkl")
scaler = jb.load("scaler.pkl")
expect_columns = jb.load("columns.pkl") 

st.title("Heart Disease Prediction")
st.markdown("This app predicts the likelihood of heart disease based on user input.")   

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])   
chestpain_type = st.selectbox("Chest Pain Type", ["ATA", "NPA", "TA", "ASY","none"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["0", "1"])
rest_ecg = st.selectbox("Resting ECG", ["Normal","ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["0", "1"])
oldppeak = st.slider("Oldpeak (ST depression induced by exercise)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "age": age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_blood_sugar,
        'MaxHR': max_hr,
        'Oldpeak': oldppeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chestpain_type: 1,
        'RestingECG_' + rest_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }])

    input_data = input_data.reindex(columns=expect_columns, fill_value=0)
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.error("The model predicts that you are likely to have heart disease. Please consult a healthcare professional for further evaluation.")
    else:
        st.success("The model predicts that you are unlikely to have heart disease. However, please consult a healthcare professional for a comprehensive assessment.")    
