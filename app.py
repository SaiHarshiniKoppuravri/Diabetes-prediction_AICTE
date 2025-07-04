import streamlit as st
import joblib
import numpy as np

classifier = joblib.load('diabetes_svm_model.pkl')

st.title("Diabetes Prediction App")

inputs = []
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

for feature in features:
    value = st.number_input(feature, value=0.0)
    inputs.append(value)

if st.button("Predict"):
    prediction = classifier.predict([np.array(inputs)])
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"Prediction: {result}")
    