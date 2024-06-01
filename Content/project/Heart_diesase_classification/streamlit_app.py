import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('models/heart_disease_model.pkl')

# Title of the app
st.title('Heart Disease Prediction')

# Description of the app
st.write("""
This is a simple web app to predict heart disease based on clinical parameters.
""")

# Define inputs for the user to enter data
st.header('User Input Parameters')

def user_input_features():
    age = st.number_input('Age', min_value=1, max_value=120, value=30)
    sex = st.selectbox('Sex: male-1, Female-0', (0, 1))
    cp = st.selectbox('Chest Pain Type (cp)', (0, 1, 2, 3))
    trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
    chol = st.number_input('Serum Cholestoral in mg/dl (chol)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1))
    restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', (0, 1, 2))
    thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina (exang)', (0, 1))
    oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest (oldpeak)', min_value=0.0, max_value=6.0, value=1.0)
    slope = st.selectbox('The Slope of the Peak Exercise ST Segment (slope)', (0, 1, 2))
    ca = st.selectbox('Number of Major Vessels Colored by Flourosopy (ca)', (0, 1, 2, 3, 4))
    thal = st.selectbox('Thalium Stress Result (thal)', (1, 3, 6, 7))
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display the user input
st.subheader('User Input parameters')
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
heart_disease = np.array(['No', 'Yes'])
st.write(f'Heart Disease: {heart_disease[prediction][0]}')

st.subheader('Prediction Probability')
st.write(prediction_proba)
