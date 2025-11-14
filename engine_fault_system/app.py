# ==========================================
# 🚗 Vehicle Emission & Fault Monitoring Dashboard
# Author: Amit Mali
# FINAL VERSION — XGBoost Emission Index with Quantile-Based Thresholds
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Vehicle Emission & Fault Dashboard",
    page_icon="🚗",
    layout="wide",
)

# ------------------------------
# HEADER
# ------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🚗 Vehicle Emission & Fault Monitoring Dashboard")
    st.caption("Parallel **Fault Detection** + Cascaded **Multi-Emission Prediction → Emission Index → Emission Level**")
with col2:
    st.image("https://cdn.pixabay.com/photo/2016/02/15/16/27/car-1209912_1280.png", use_container_width=True)

st.markdown("---")

# ------------------------------
# LOAD MODELS
# ------------------------------
base_path = r"C:\Users\ADMIN\Desktop\CI LAB\engine_fault_system\models"

try:
    fault_model = joblib.load(os.path.join(base_path, "fault_detector.pkl"))
    fault_scaler = joblib.load(os.path.join(base_path, "feature_scaler.pkl"))
    emission_model = joblib.load(os.path.join(base_path, "emission_prediction_rf_multi.pkl"))
    emission_preprocessor = joblib.load(os.path.join(base_path, "emission_prediction_preprocessor_rf.pkl"))
    index_model = joblib.load(os.path.join(base_path, "emission_index_xgb_model.pkl"))
    index_preprocessor = joblib.load(os.path.join(base_path, "emission_index_preprocessor.pkl"))
    st.sidebar.success("✅ All models & preprocessors loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ------------------------------
# SIDEBAR INPUTS
# ------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/744/744465.png", width=100)
st.sidebar.header("Vehicle & Environmental Inputs")

# ----- Fault Detection Inputs -----
st.sidebar.subheader("🧠 Fault Detection Parameters")
engine_rpm = st.sidebar.number_input("Engine RPM", 0, 10000, 3000)
lub_oil_pressure = st.sidebar.number_input("Lub Oil Pressure (bar)", 0.0, 10.0, 2.5)
fuel_pressure = st.sidebar.number_input("Fuel Pressure (bar)", 0.0, 20.0, 5.0)
coolant_pressure = st.sidebar.number_input("Coolant Pressure (bar)", 0.0, 20.0, 3.0)
lub_oil_temp = st.sidebar.number_input("Lub Oil Temperature (°C)", 0.0, 150.0, 80.0)
coolant_temp = st.sidebar.number_input("Coolant Temperature (°C)", 0.0, 150.0, 90.0)

# ----- Emission Inputs -----
st.sidebar.subheader("🌍 Emission Parameters")
engine_size = st.sidebar.number_input("Engine Size (L)", 0.0, 15.0, 3.0, step=0.1)
mileage = st.sidebar.number_input("Mileage (km)", 0, 1000000, 150000, step=1000)
speed = st.sidebar.number_input("Speed (km/h)", 0, 200, 60)
acceleration = st.sidebar.number_input("Acceleration (m/s²)", 0.0, 20.0, 3.0)
temperature = st.sidebar.number_input("Ambient Temperature (°C)", -50, 60, 25)
humidity_em = st.sidebar.number_input("Humidity (%)", 0, 100, 50)
fuel_type = st.sidebar.selectbox("Fuel Type", ("Petrol", "Diesel", "Hybrid", "Electric"))
road_type = st.sidebar.selectbox("Road Type", ("Highway", "City", "Rural"))
vehicle_type = st.sidebar.selectbox("Vehicle Type", ("Car", "Truck", "Bus", "Bike"))
traffic_condition = st.sidebar.selectbox("Traffic Condition", ("Light", "Moderate", "Heavy"))

predict_btn = st.sidebar.button("🔍 Run Predictions")

# ------------------------------
# INPUT PREPARATION
# ------------------------------
fault_input = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp,
}])

emission_input = pd.DataFrame([{
    'Vehicle Type': vehicle_type,
    'Fuel Type': fuel_type,
    'Engine Size': engine_size,
    'Age of Vehicle': np.random.randint(1, 10),
    'Mileage': mileage,
    'Speed': speed,
    'Acceleration': acceleration,
    'Road Type': road_type,
    'Traffic Conditions': traffic_condition,
    'Temperature': temperature,
    'Humidity': humidity_em,
    'Wind Speed': np.random.uniform(0, 20),
    'Air Pressure': np.random.uniform(950, 1050),
    'Emission_Index': 0
}])

# ------------------------------
# PREDICTION PIPELINE
# ------------------------------
if predict_btn:

    # ---- (1) Fault Detection ----
    X_fault_scaled = fault_scaler.transform(fault_input)
    fault_pred = fault_model.predict(X_fault_scaled)[0]
    fault_label = "⚠️ Fault Detected" if fault_pred == 1 else "✅ No Fault Detected"

    # ---- (2) Multi-Emission Prediction ----
    X_emission = emission_preprocessor.transform(
        emission_input.drop(columns=['Vehicle Type', 'Traffic Conditions', 'Wind Speed', 'Air Pressure', 'Emission_Index'],
                            errors='ignore')
    )
    emission_preds = emission_model.predict(X_emission)[0]
    emission_names = ['CO₂ Emissions', 'NOx Emissions', 'PM2.5 Emissions', 'VOC Emissions', 'SO₂ Emissions']

    # ---- (3) Emission Index Prediction (XGBoost)
    numeric_cols = ["CO2 Emissions", "NOx Emissions", "PM2.5 Emissions", "VOC Emissions", "SO2 Emissions"]
    pred_df = pd.DataFrame([emission_preds], columns=numeric_cols)
    combined_input = pd.concat([
        emission_input.reset_index(drop=True).drop(columns=['Emission_Index'], errors='ignore'),
        pred_df.reset_index(drop=True)
    ], axis=1)

    X_index = index_preprocessor.transform(combined_input)
    emission_index_pred = float(index_model.predict(X_index)[0])

    # ---- (4) Determine Level by Quantile-Based Threshold ----
    low_thr = 0.15073993760975574
    med_thr = 0.36767003665036824

    if emission_index_pred < low_thr:
        level_label = "Low"
    elif emission_index_pred < med_thr:
        level_label = "Medium"
    else:
        level_label = "High"

    # ------------------------------
    # DISPLAY RESULTS
    # ------------------------------
    st.markdown("### 🔧 Fault Detection Result")
    if fault_pred == 0:
        st.success(fault_label)
        st.image("https://cdn.pixabay.com/photo/2013/07/13/12/38/green-159075_1280.png", width=180)
    else:
        st.error(fault_label)
        st.image("https://cdn.pixabay.com/photo/2013/07/12/18/39/warning-153601_1280.png", width=180)

    st.markdown("---")
    st.markdown("### 💨 Emission Predictions")
    colA, colB = st.columns([1, 3])
    with colA:
        st.image("https://cdn.pixabay.com/photo/2014/04/02/10/54/car-304642_1280.png", use_container_width=True)
    with colB:
        cols = st.columns(2)
        for i, name in enumerate(emission_names):
            cols[i % 2].metric(label=name, value=f"{emission_preds[i]:.2f} g/km")

    st.markdown("---")
    st.markdown("### 📈 Emission Index & Level Classification")
    st.metric(label="Predicted Emission Index", value=f"{emission_index_pred:.3f}")

    if level_label == "High":
        st.warning(f"⚠️ Emission Level: **{level_label}** — Above Safe Threshold")
    elif level_label == "Medium":
        st.info(f"ℹ️ Emission Level: **{level_label}** — Moderate Emission Detected")
    else:
        st.success(f"✅ Emission Level: **{level_label}** — Within Safe Range")

    # 📊 Optional Emission Index Visualization
    st.markdown("### 📊 Emission Index Visualization")
    fig, ax = plt.subplots(figsize=(5, 0.5))
    color = "limegreen" if level_label == "Low" else "gold" if level_label == "Medium" else "red"
    ax.barh([0], emission_index_pred, color=color)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Emission Index (0 = Low | 1 = High)")
    st.pyplot(fig)

    st.markdown("---")
    st.caption("📘 Models: Fault Detection | Multi-Emission RF | Emission Index XGBoost (Quantile Thresholds)")

else:
    st.info("👈 Enter parameters and click **Run Predictions** to view results.")
    st.image("https://cdn.pixabay.com/photo/2017/03/02/20/28/car-2112300_1280.png", use_container_width=True)
