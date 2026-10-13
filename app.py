import streamlit as st
import pandas as pd
import joblib

model = joblib.load("decision_tree_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Decision Tree ML Prediction")
st.write("Enter values to get a prediction.")

input_data = {}

for column in model_columns:
    input_data[column] = st.number_input(
        f"Enter {column}",
        value=0.0
    )

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.success(f"Predicted Result: {prediction[0]}")
