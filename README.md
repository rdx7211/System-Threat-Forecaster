# 🛡️ System Threat Forecaster  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  
![Flask](https://img.shields.io/badge/Flask-2.0+-black)  
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)  
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)  
![License: MIT](https://img.shields.io/badge/License-MIT-green)  

---

## 📌 Overview  
This project was developed as part of the [**Kaggle System Threat Forecaster Competition**](https://www.kaggle.com/competitions/System-Threat-Forecaster).  

The goal is to **predict the probability of a system being infected by malware** based on its hardware/software properties and telemetry data. The dataset is generated from threat reports collected by antivirus software.  

This project combines **data preprocessing, machine learning model training, and a Flask web application** to provide real-time threat predictions.  

---

## 🔹 Features  

### 1️⃣ **Data Preprocessing & Feature Engineering**  
- Handles **missing values, categorical encoding, and scaling**.  
- Feature selection from a **large telemetry dataset (~70+ features)**.  
- Generates a clean dataset for ML models.  

### 2️⃣ **Model Training & Evaluation**  
- Built multiple **classification models** (Logistic Regression, Random Forest, XGBoost,LightBgm etc..).  
- Finalized a **pipeline** and saved it as `final_pipeline.pkl`.  
- Evaluation metrics include **Accuracy, Precision, Recall, and ROC-AUC**.  

### 3️⃣ **Threat Prediction Web App (Flask)**  
- A user-friendly **form interface** where users input system properties.  
- Model outputs:  
  - ⚠️ **Threat Detected**  
  - ✅ **Safe**  
- Displays **prediction probability** for better interpretability.  

---

## 📂 Project Structure  

SYSTEM-THREAT-FORECASTER/

├── data/                        # Raw Kaggle dataset
│   ├── train.csv
│   ├── test.csv
│
├── models/                      # Saved ML models & defaults
│   ├── final_pipeline.pkl
│   ├── default_values.json
│
├── research/                    
│   ├── Training_notebook.ipynb  # Jupyter notebooks for experiments(Which includes)
|   ├── submission.csv           # Example Kaggle submission
│
├── src/                         # Modularized Python scripts
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data cleaning & feature engineering
│   ├── model_training.py        # Model training pipeline
│   ├── evaluation.py            # Evaluation functions
│   ├── utils.py                 # Helper functions
│
├── templates/                   # HTML frontend for Flask
│   ├── index.html
│
├── app.py                       # Flask web app
│
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment (optional)
├── .gitignore                   # Ignore data, venv, pycache, etc.
├── LICENSE                      # License file
├── README.md                    # Project documentation


## ⚙️ Tech Stack  

- **Python 3.9+**  
- **Pandas / NumPy** → Data preprocessing  
- **Scikit-learn / XGBoost** → ML modeling  
- **Flask** → Web app backend  
- **Bootstrap** → Frontend styling  
- **Joblib** → Model serialization  

---

## 🚀 How to Run  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/System-Threat-Forecaster.git

