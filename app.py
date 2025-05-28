import streamlit as st
import numpy as np
import joblib
import plotly.express as px

# App title
st.set_page_config(page_title="Calories Burn Prediction App", layout="centered")
st.title("🔥 Calories Burn Prediction App")
st.write("Enter the activity details below to estimate burned calories.")

# Loading saved model and scaler
import joblib
model = joblib.load('final_stacking_model.pkl')
scaler = joblib.load('scaler_model.pkl')

# User inputs
age = st.number_input("Age")
height = st.number_input("Height (cm)")
weight = st.number_input("Weight (kg)")
duration = st.number_input("Duration (min)")
heart_rate = st.number_input("Heart Rate")
body_temp = st.number_input("Body Temperature")
bmi = st.number_input("BMI")
steps = st.number_input("Steps")
activity_level = st.selectbox("Activity Level", [1, 2, 3])  # Example if encoded

# Prediction
input_data = np.array([[age, height, weight, duration, heart_rate, body_temp, bmi, steps, activity_level]])
scaled_input = scaler.transform(input_data)

if st.button("Predict Calories Burned"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"🔥 Estimated Calories Burned: **{prediction:.2f} kcal**")

# Optional Visualization with Plotly
st.markdown("### 📊 Calorie Burn vs Duration")

# Generate values for a range of durations
durations = np.arange(5, 125, 5)
temp_inputs = np.array([[age, height, weight, duration, heart_rate, body_temp, bmi, steps, activity_level] for d in durations])
temp_scaled = scaler.transform(temp_inputs)
temp_preds = model.predict(temp_scaled)

fig = px.line(
    x=durations,
    y=temp_preds,
    labels={'x': 'Duration (minutes)', 'y': 'Predicted Calories'},
    title='Calories Burned Over Duration',
)
fig.update_traces(mode='lines+markers', line=dict(color='orange'))
st.plotly_chart(fig)
