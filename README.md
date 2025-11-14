# 🚗 Vehicle Emission & Fault Detection System  
### Cascaded Machine Learning Pipeline for Real-Time Vehicle Health & Emissions Monitoring

---

## 🔥 Overview  
This project implements a complete **three-stage machine learning pipeline** for monitoring vehicle health, predicting pollutant emissions, and classifying emission levels.  
The system includes:

1. **Fault Detection Model** – Detects engine anomalies from sensor data  
2. **Multi-Emission Prediction Model** – Predicts 5 major pollutants  
3. **Emission Level Classifier** – Categorizes emissions as Low / Medium / High  
4. **Streamlit Dashboard** – Complete real-time visualization interface  

---

## 🧠 Cascaded ML Architecture

ENGINE SENSOR DATA ─► MODEL 1: Fault Detection (Random Forest)
│
▼
VEHICLE PARAMETERS ─► MODEL 2: Multi-Emission Prediction (Multi-Output RF)
│
▼
PREDICTED EMISSIONS + VEHICLE FEATURES ─► MODEL 3: Emission Level Classifier


Each model works independently AND sequentially, forming a powerful **cascaded decision system**.

---

## 🏗️ System Components  

### **1️⃣ Fault Detection Model**
- Dataset: `engine_data.csv`
- Inputs: RPM, oil pressure, fuel pressure, coolant temp, etc.
- Output:
  - **0 → No Fault**
  - **1 → Fault Detected**
- Model: `RandomForestClassifier`

---

### **2️⃣ Multi-Emission Prediction Model**
- Dataset: `vehicle_emission_dataset_synthetic_v3_labeled.csv`
- Predicts 5 pollutant emission levels:
  - CO₂  
  - NOₓ  
  - PM2.5  
  - VOC  
  - SO₂  
- Model: `MultiOutputRegressor(RandomForestRegressor)`
- Synthetic dataset includes **real-world pollutant correlations**.

---

### **3️⃣ Emission Level Classifier**
- Labels: **Low**, **Medium**, **High**
- Inputs:
  - Vehicle features  
  - **Predicted emissions from Model 2**  
- Best-performing model: `RandomForestClassifier`

---

## 🖥️ Streamlit Dashboard  

A clean and responsive dashboard that allows users to:

✔ Detect vehicle engine faults  
✔ Predict pollutant emissions in real-time  
✔ Classify emission level  
✔ Visualize outputs with graphs & icons  

📊 **Model Performance Summary**

### 🔧 Fault Detection Model
- Accuracy: **~99%**
- Strong generalization performance
- Dataset balanced using **SMOTE**
- Robust Random Forest–based classifier

### 💨 Multi-Emission Regression (Synthetic Dataset)
Model trained on highly realistic synthetic emission correlations.

**R² Scores:**
- **CO₂:** ~0.98  
- **NOₓ:** ~0.94  
- **PM2.5:** ~0.70  
- **VOC:** ~0.91  
- **SO₂:** ~0.68  

### 🌫 Emission Level Classification
- Accuracy: **95%+**
- High macro F1-scores
- Balanced class distribution (**Low / Medium / High**)
- Models compared → best estimator saved automatically

---

## 🎯 Key Features

- 🔗 **Cascaded ML system** combining three independent models  
- 🌍 Predicts **five pollutants simultaneously**  
- ⚙ Highly accurate **engine fault detection**  
- 📊 Classifies vehicles into **Low / Medium / High** emission categories  
- 💻 Fully interactive **Streamlit dashboard**  
- 🔬 Synthetic but **realistic environment-based emission modeling**  
- 🧩 Modular architecture — each model is **plug-and-play**  

---

## 🚀 Future Enhancements
- Deep learning for time-series emission prediction  
- Cloud deployment (Streamlit Cloud / AWS / Azure)  
- SHAP explainability for model transparency  
- Real-time OBD-II sensor integration  
- Model monitoring + MLOps pipeline  

---

## 📜 License
This project is distributed under the **MIT License**.  
Feel free to use, modify, and distribute with attribution.

---

## 🤝 Contributing
Contributions are welcome!  
Open an Issue or Submit a Pull Request to improve models or add features.

👤 Author

Pranav Karande , Amit Mali ,Krishna patil . 
Machine Learning & Data Science
