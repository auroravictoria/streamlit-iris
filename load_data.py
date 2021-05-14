import streamlit as st
import plotly.express as px

@st.cache
def load_iris():
    return px.data.iris()
