import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl


st.title("Obesity Level Predictor")
st.header("Enter your details below")


gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age (years)", min_value=0, max_value=120, step=1)
height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, step=0.01)
weight = st.number_input("Weight (kg)", min_value=10, max_value=300, step=1)
physical_activity = st.slider("Physical activity frequency (hours per week)", min_value=0, max_value=24, step=1)


input_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Height": [height],
    "Weight": [weight],
    "FAF": [physical_activity],
})


with open('model.pkl', 'rb') as f:
    model = pkl.load(f)


input_data = pd.get_dummies(input_data, drop_first=True)  
model_features = model.feature_names_in_  
input_data = input_data.reindex(columns=model_features, fill_value=0)  


if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Predicted Obesity Level: {prediction[0]}")
