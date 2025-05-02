import plotly.express as px
import pandas as pd
import streamlit as st
import seaborn as sns


#import dataset
st.title("Plotly and Streamlit Combined App")
df = px.data.gapminder()
st.write(df)

st.subheader("Columns")
st.write(df.columns)


#summary statistics
st.subheader("Summary Statistics")
st.write(df.describe())


#managing data
year_option = df['year'].unique().tolist()
year = st.selectbox('Select Year', year_option, 0)
df = df[df['year'] == year]


#to see the result create plot
fig = px.scatter(df, x='gdpPercap', y='lifeExp', size='pop', color='continent', 
                 hover_name='country', log_x=True, size_max=55, range_x=[100,100000], range_y=[20,90])
fig.update_layout(width=800, height=600)
st.plotly_chart(fig)


