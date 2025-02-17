import joblib
import streamlit as st
import numpy as np
import pandas as pd

@st.cache_resource

def load_model():
    return joblib.load('model.pkl')

st.cache_resource.clear()

st.title('Model to Predict Dry Eye Disease')
st.subheader('This model will help predict whether a patient has dry eye disease or not')

model = load_model()

if model:
    st.subheader('Please enter the following details')

Eye_Itchiness = st.selectbox('Do your eyes itch?', options=[(0,'No'), (1,'Yes')], format_func = lambda x: x[1])
Eye_Itchiness_Value = Eye_Itchiness[0]

Eye_Redness = st.selectbox('Are your eyes red?', options=[(0,'No'), (1,'Yes')], format_func=lambda x: x[1])
Eye_Redness_Value = Eye_Redness[0]

Eye_Strain = st.selectbox('Do your eyes strain?', options=[(0,'No'), (1,'Yes')], format_func=lambda x: x[1])
Eye_Strain_Value = Eye_Strain[0]

Screen_Time = st.number_input('What is your average screen time in minutes?', value = 0)

feature_names = ['Eye_Itchiness', 'Eye_Redness', 'Eye_Strain', 'Screen_Time']

user_input_df = pd.DataFrame([[Eye_Itchiness_Value, Eye_Redness_Value, Eye_Strain_Value, Screen_Time]],columns=feature_names)

if st.button('Click to predict'):
    prediction = model.predict(user_input_df)  
    condition_status = 'has Dry Eye Disease' if prediction[0] == 1 else 'does not have Dry Eye Disease' 
    st.subheader(f'The patient {condition_status}')