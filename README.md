# 🚗 Driver Behavior & Route Anomaly Detection System

An AI-driven telemetry analytics platform designed to monitor driver behavior, quantify risk scores, and detect driving anomalies using vehicular sensor data.

---

## 📌 Project Overview

This project analyzes real-time telematics and sensor data collected from vehicles—including speed, acceleration, steering angle, and braking force. Using Machine Learning models, the system computes continuous risk scores, categorizes driving styles, and accurately detects hazardous events such as **Sudden Braking**, **Harsh Acceleration**, or **Unsafe Lane Changes**.

---

## 🛠️ Key Features

- **Real-Time Sensor Telemetry Analysis:** Processes high-frequency data streams (Speed, Acceleration, RPM, etc.).
- **Driving Style Profiling:** Classifies drivers into distinct behavioral categories (`Calm`, `Normal`, or `Aggressive`).
- **Dynamic Risk Score Calculation:** Calculates a continuous risk percentage score (0% to 100%) and overall severity level (`LOW`, `MEDIUM`, `HIGH`).
- **Hazardous Event Identification:** Rules-assisted Machine Learning logic to identify specific anomaly types.
- **Multi-Model Support:** Built to evaluate both linear risk regression models and ensemble classifiers (Linear Regression, Random Forest, XGBoost).

---

---

## 🤖 Models & Methodology

1. **Linear Regression:** Evaluates continuous risk intensity (`Risk Score %`) by measuring each sensor feature's exact coefficient weight.
2. **Ensemble Classifiers (Random Forest / XGBoost):** Classifies complex multi-feature driving anomalies while balancing class imbalances (`class_weight='balanced_subsample'`).

---

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have Python installed. Install all required dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn xgboost
