import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("models/model.pkl", "rb"))

st.set_page_config(
    page_title="Predictive Maintenance System",
    layout="centered"
)

st.title("🔧 Predictive Maintenance System")

st.write("""
This system predicts whether an industrial machine
is likely to fail based on sensor data.
""")

# Inputs
type_value = st.selectbox(
    "Machine Type",
    [0, 1, 2]
)

air_temp = st.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=400.0
)

process_temp = st.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=0
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0
)

if st.button("Predict Failure"):

    input_data = np.array([[
        type_value,
        air_temp,
        process_temp,
        rotational_speed,
        torque,
        tool_wear
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ Machine Failure Likely!")
    else:
        st.success("✅ Machine Operating Normally")