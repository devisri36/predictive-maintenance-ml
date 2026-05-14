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
Predict industrial machine failures using sensor metrics.
""")

# Inputs
metric1 = st.number_input("Metric 1", value=100000000)
metric2 = st.number_input("Metric 2", value=0)
metric3 = st.number_input("Metric 3", value=0)
metric4 = st.number_input("Metric 4", value=0)
metric5 = st.number_input("Metric 5", value=5)
metric6 = st.number_input("Metric 6", value=300000)
metric7 = st.number_input("Metric 7", value=0)
metric8 = st.number_input("Metric 8", value=0)
metric9 = st.number_input("Metric 9", value=0)

if st.button("Predict Failure"):

    input_data = np.array([[
        metric1,
        metric2,
        metric3,
        metric4,
        metric5,
        metric6,
        metric7,
        metric8,
        metric9
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Machine Failure Likely")
    else:
        st.success("Machine Operating Normally")