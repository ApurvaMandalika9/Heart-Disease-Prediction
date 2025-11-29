import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Paths to model + meta
MODEL_PATH = "models/heart_model.joblib"
META_PATH = "models/model_meta.json"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()
BEST_THRESHOLD = meta.get("threshold", 0.5)

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="🫀")

st.title("Heart Disease Risk Prediction")
st.write(
    "Enter patient information below to estimate the risk of heart disease. "
    "This demo uses a trained Random Forest model from the Kaggle Heart Disease dataset."
)

st.sidebar.header("Input Features")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=55)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox(
    "Chest Pain Type (cp)",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "0 - Typical angina",
        1: "1 - Atypical angina",
        2: "2 - Non-anginal pain",
        3: "3 - Asymptomatic",
    }[x],
)
trestbps = st.sidebar.number_input("Resting BP (trestbps)", min_value=50, max_value=250, value=130)
chol = st.sidebar.number_input("Cholesterol (chol)", min_value=80, max_value=600, value=250)
fbs = st.sidebar.selectbox("Fasting Blood Sugar (fbs)", options=[0, 1], format_func=lambda x: "> 120 mg/dl" if x == 1 else "≤ 120 mg/dl")
restecg = st.sidebar.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate (thalach)", min_value=50, max_value=250, value=150)
exang = st.sidebar.selectbox("Exercise-Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.sidebar.number_input("Oldpeak", min_value=-5.0, max_value=10.0, value=1.0, step=0.1)
slope = st.sidebar.selectbox("Slope", options=[0, 1, 2])
ca = st.sidebar.selectbox("Number of Vessels (ca)", options=[0, 1, 2, 3])
thal = st.sidebar.selectbox("Thal", options=[0, 1, 2])

if st.button("Predict Risk"):
    # Build feature vector in the same order as training
    feature_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    X = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }], columns=feature_names)

    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= BEST_THRESHOLD)

    prob_percent = round(proba * 100, 2)

    if pred == 1:
        st.error(f"High Risk of Heart Disease (probability: {prob_percent}%)")
    else:
        st.success(f"Low Risk of Heart Disease (probability: {prob_percent}%)")

    st.caption(f"Decision threshold used: {BEST_THRESHOLD:.3f}")
