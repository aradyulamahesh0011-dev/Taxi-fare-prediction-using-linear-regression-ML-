import streamlit as st
import numpy as np
import pickle
import joblib
import math

# ================== STREAMLIT PAGE SETUP ==================
st.set_page_config(page_title="🚖 Taxi Fare Prediction", layout="centered")
st.title("🚕 Taxi Fare Prediction App")
st.write("Predict taxi fare based on trip details using a trained ML model.")

# ================== LOAD MODEL & ENCODERS ==================
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load("best_model.pkl")
        with open('model_encoder.pkl', 'rb') as f:
            model_encoder = pickle.load(f)
        with open('day_of_week_encoder.pkl', 'rb') as f:
            day_of_week_encoder = pickle.load(f)
        with open('time_category_encoder.pkl', 'rb') as f:
            time_category_encoder = pickle.load(f)
        return model, model_encoder, day_of_week_encoder, time_category_encoder
    except FileNotFoundError:
        st.error("❌ Model or encoder files not found. Ensure .pkl files are in the directory.")
        st.stop()

model, model_encoder, day_of_week_encoder, time_category_encoder = load_model_and_encoders()

# ================== OPTIONS FROM TRAINED ENCODERS ==================
model_options = list(model_encoder.classes_)
day_of_week_options = list(day_of_week_encoder.classes_)
time_category_options = list(time_category_encoder.classes_)

# =============
