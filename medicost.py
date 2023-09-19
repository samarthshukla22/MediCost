# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle

# Load the pre-trained model
model_file_path = 'medicost.pkl'
with open(model_file_path, 'rb') as file:
    regressor = pickle.load(file)

# Create a Streamlit app
st.title('MediCost Analyzer - Medical Insurance Cost Prediction App')

# Create input features for age, sex, bmi, children, smoker, and region
st.header('Please Enter Details')

# Age
age = st.slider('Age', 18, 100, 30)

# Sex
sex_options = ('Male', 'Female')
sex = st.radio('Sex', sex_options)

# BMI
bmi = st.slider('BMI', 15.0, 50.0, 25.0)

# Number of Children
children = st.slider('Number of Children', 0, 5, 0)

# Smoker
smoker_options = ('Yes', 'No')
smoker = st.radio('Smoker', smoker_options)

# Region
region_options = ('Southeast', 'Southwest', 'Northeast', 'Northwest')
region = st.selectbox('Region', region_options)

# Map sex and smoker values to 0 and 1
sex = 1 if sex == 'Female' else 0
smoker = 1 if smoker == 'Yes' else 0

# Map region values to encoded values
region_mapping = {'Southeast': 0, 'Southwest': 1, 'Northeast': 2, 'Northwest': 3}
region = region_mapping[region]

# Create a prediction button
if st.button('Predict Insurance Cost'):
    # Prepare the input data
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    
    # Make a prediction
    prediction = regressor.predict(input_data)
    
    # Display the prediction
    st.header('Predicted Insurance Cost')
    st.write(f'The estimated insurance cost is USD {prediction[0]:.2f}')
