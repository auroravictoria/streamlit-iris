import streamlit as st
import plotly.express as px
import numpy as np
import pickle
from load_data import load_iris

st.title("My Growing Garden")
st.header("We plant plants")
st.subheader("orchids")


#load data

df_iris = load_iris()

show_df = st.checkbox("Do you want to see the plant data?")

if show_df:
    df_iris

st.plotly_chart(px.scatter(df_iris, 'sepal_width', 'sepal_length'))


#get user flower input
s_l = st.number_input('Input the sepal length')
s_w = st.number_input('Input the sepal width')
p_l = st.number_input('Input the petal length')
p_w = st.number_input('Input the petal width')


user_values = np.array([s_l, s_w, p_l, p_w])

#load model 

with open('saved-iris-model-2.pkl', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(user_values.reshape(1,-1))

st.header(f'The model predicts: {prediction[0]}')

st.balloons()

# col1, col2, col3 = st.beta_columns(3)
# col1.subheader('Columnisation')
# col2.subheader('Columnisation')
# col3.subheader('Columnisation')

# col1, col2, col3 = st.beta_columns(3)
# with col1:
#     "orchids"
# with col2:
#     "tulips"
# with col3:
#     "join us!"