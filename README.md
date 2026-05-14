# 🔧 Predictive Maintenance for Industrial Machines Using Machine Learning

## Overview

This project is a Machine Learning-based Predictive Maintenance System designed to analyze industrial machine sensor data and predict potential machine failures before breakdown occurs. The system helps reduce downtime, improve operational efficiency, and support proactive maintenance strategies in industrial environments.

The project includes:

* Data preprocessing and feature engineering
* Machine Learning model training
* Imbalanced data handling using SMOTE
* Model evaluation and performance analysis
* Streamlit web application deployment

---

## Live Demo

Streamlit Deployment:
[Live link](https://predictive-maintenance-ml-rsfp3ayzwdwxivyuanauy4.streamlit.app/)

---

## Features

* Predicts machine failure using sensor metrics
* Handles imbalanced datasets using SMOTE
* Trained using Random Forest Classifier
* Interactive Streamlit web interface
* End-to-end ML workflow implementation
* Deployment-ready project structure

---

## Tech Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest Classifier
* SMOTE (Imbalanced-learn)

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Deployment & UI

* Streamlit

---

## Project Structure

```bash
predictive-maintenance-ml/
│
├── dataset/
│   └── predictive_maintenance.csv
│
├── models/
│   └── model.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation & Setup

### 1️ Clone Repository

```bash
git clone https://github.com/devisri36/predictive-maintenance-ml.git
cd predictive-maintenance-ml
```

### 2️ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️ Train Model

```bash
python train_model.py
```

### 4️ Run Streamlit App

```bash
streamlit run app.py
```

---

## Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Scaling
4. Imbalanced Data Handling using SMOTE
5. Model Training using Random Forest
6. Model Evaluation
7. Deployment using Streamlit

---

## Model Performance

* Accuracy achieved: ~99%
* Implemented SMOTE oversampling to improve minority class prediction
* Evaluated using:

  * Confusion Matrix
  * Precision
  * Recall
  * F1-score

---

## Challenges Faced

* Highly imbalanced dataset with very few failure samples
* Improving failure detection capability
* Preventing model bias toward majority class

---

## Future Enhancements

* Real-time IoT sensor integration
* Advanced Deep Learning models (LSTM)
* Cloud deployment
* Real-time monitoring dashboard
* Alert and notification system

---

