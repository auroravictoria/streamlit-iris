
# Streamlit 

Streamlit is an opinionated library for serving models and visualizations in pure Python. It is interactive and quick to deploy.

## Learning objectives

- Create a streamlit app to serve models and visualizations
- Use plotly express to create interactive visualizations
- Deploy your streamlit on streamlit

## Install packages

In your terminal, list the packages in your virtual environments.

If you don't see streamlit and plotly, then install them.

`pip install streamlit`
`pip install plotly`

## Create a streamlit app

1. Arrange your screen(s) so you can see a browser window and your text editor. 
2. Opne your text editor
3. Navigate to the folder where you want to create your app
4. Make a new python file named `my_app.py`
5. In the file, add `import streamlit as st` and save the file
6. In the terminal, in the same directory as your `my_app.py` file execute `streamlit run my_app.py`
    - You are now serving your app loacally.
    - You should see a browser window pop up.
    - After changing your code, you can click a button at the top right of your browser to re-run the code. You can click a button in that area to tell your browser to rerun automatically after every change.
    - Your code runs from top to bottom, every time.
    - You can deploy your model to the internet by pushing it to your personal GitHub account and connecting it to Streamlit's servers.


