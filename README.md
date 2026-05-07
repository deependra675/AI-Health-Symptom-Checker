---AI Health Symptom Checker---

An interactive AI-powered healthcare assistant built using Python, Machine Learning, and Streamlit.

This application predicts possible diseases based on user-selected symptoms and provides:

disease descriptions
precaution recommendations
confidence analysis
interactive medical dashboard UI

---Features---
Disease Prediction-

Predicts possible diseases using a trained Machine Learning model.

Symptom-Based Analysis-

Users can select multiple symptoms from an interactive sidebar.

Disease Description-

Displays medical information related to the predicted disease.

Precaution Suggestions-

Provides basic health precautions and safety recommendations.

Confidence Visualization-

Shows top disease probabilities using interactive charts.

Modern Interactive UI-

Built with Streamlit using:

tabs layout
animated loading
responsive dashboard
card-style interface
Technologies Used
Python
Streamlit
Scikit-learn
Pandas
Plotly
Joblib

Project Structure-

AI-Health-Assistant/
│
├── app.py
├── training_model.py
├── model.pkl
├── symptoms.pkl
├── description.pkl
├── precaution.pkl
│
├── dataset/
│   ├── dataset.csv
│   ├── symptom_Description.csv
│   └── symptom_precaution.csv
│
├── description.py
├── precaution.py
│
└── README.md

---Machine Learning Workflow---
Dataset-

The model was trained using a symptom-disease dataset where:

symptoms are input features
disease is the target label

Model Used-
Random Forest Classifier

Training Steps-

Data cleaning
Symptom encoding
Train-test split
Model training
Model serialization using Joblib

Preprocessing Files-

prepare_description.py
Processes disease descriptions into a fast dictionary lookup system.
prepare_precaution.py
Processes precaution data and stores it in serialized format.

---User Interface---

The UI includes:-

sidebar symptom selection
modern card layout
tab-based navigation
prediction confidence chart
smooth loading animation

---Disclaimer---

This project is developed for:

educational purposes
machine learning practice
healthcare AI experimentation

It is not intended for real medical diagnosis or professional healthcare advice.

Always consult a licensed medical professional for medical concerns.

Future Improvements-->>>

Planned upgrades include:

natural language symptom input
chatbot-style interaction
severity detection
PDF health report generation
multilingual support
doctor recommendation system

AUTHOR ---
DEEPENDRA SISODIA, MCA
