import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report



#web app title
st.markdown('''
# **Exploratory Data Analysis with Application**
This app is developed by Kanwar Hamza Shuja called **EDA**.
''')

#how to upload a file from computer
with st.sidebar.header('Upload your dataset (.csv)'):
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=['csv'])
    df = sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    
#load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("### **Dataframe Preview**")
    st.write(df.head())
    st.markdown("### **Dataframe Shape**")
    st.write(df.shape)
    st.markdown("### **Dataframe Description**")
    st.write(df.describe())
    st.markdown("### **Dataframe Columns**")
    st.write(df.columns)
    st.markdown("### **Dataframe Columns Data Types**")

    
    
#profile report
if uploaded_file is not None:
  def load_csv():
    csv = pd.read_csv(uploaded_file)
    return csv
  df_load = load_csv()
  pr = ProfileReport(df, explorative = True)
  st.header('**Input DF**')
  st.write(df)
  st.write('---')
  st.header('**Pandas Profiling Report**')
  st_profile_report(pr)
else:
   st.info('Awaiting for CSV file to be uploaded.')
   st.button('Press to use Example Dataset')
   
#example dataset
def load_data():
    a = pd.DataFrame(
        np.random.randn(100, 5),
        columns=('col %d' % i for i in range(5)))
    return a
df = load_data()
pr = ProfileReport(df, explorative = True)
st.header('**Input DF**')
st.write(df)
st.write('---')
st.header('**Pandas Profiling Report**')
st_profile_report(pr)