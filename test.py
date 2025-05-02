import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np

st.header("This video is brought to you by Streamlit")
st.text("This is my first attempt")

df = sns.load_dataset('iris')
st.write(df[['species', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']].head(10))


st.bar_chart(df)
st.line_chart(df)
