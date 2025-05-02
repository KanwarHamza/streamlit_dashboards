import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#making containers
header = st.container()
dataset = st.container()
features = st.container()
modeltraining = st.container()

with header:

    st.title("Welcome to my first app")
    st.text("In this project we will work with Titanic dataset and learn how to use streamlit")
    

with dataset:
    st.header("Titanic sinking")
    st.text("This is the dataset that we will be working with")
    #load dataset
    df = sns.load_dataset("titanic")
    df = df.dropna()
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/95/Titanic_sinking%2C_painting_by_Willy_St%C3%B6wer.jpg")
    st.write(df.head(10))
   
    
    #lets write markdown for the graphs
    st.markdown("1. **Graphs 1:** Graphs of the dataset" )
    
    #lets make a plot
    st.subheader("Gender of the passengers")
    st.bar_chart(df["sex"].value_counts())
    
    #another plot will be nice
    st.subheader("Different classes of passengers")
    st.bar_chart(df["class"].value_counts())
    
    #another important plot
    st.subheader("number of survivors")
    st.bar_chart(df["survived"].value_counts())
    
    #lets make a plot of age
    st.subheader("Age of the passengers")
    st.bar_chart(df["age"].value_counts())
    
    #lets make a plot of fare
    st.subheader("Fare of the passengers")
    st.bar_chart(df["fare"].value_counts())
    
    
    


with features:
    st.header("These are the features of the dataset")
    st.text("These are the features of the dataset that will be explored in this project")
    
    
    
    
with modeltraining:
    st.header("What happen to the passengers?")
    st.text("This is where we will train the model and see how accurate it is")
    
    #making columns
    input, display = st.columns(2)
    
    #making a slider for the age
    max_depth = input.slider("How many people survived", min_value=0, max_value=100, value=20, step=5)
    

    #n-estimator function
    n_estimator = input.selectbox("How many tree should be in RF", options=[10, 50, 100, 200, 500, 1000])
    
    
    #list of features
    input.write(df.columns)


    #input features from users
    input_features = input.text_input("Which features you want to use?")
    
    
    #machine learning model
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator)
    
    #fit model
    x = df[[input_features]]
    y = df[["fare"]]
    model.fit(x, y)
    
    #predict
    pred = model.predict(x)
    
    #display model predictions
    display.subheader("mean absolute error is: ")
    display.write(mean_absolute_error(y, pred))
    display.subheader("mean squared error is: ")
    display.write(mean_squared_error(y, pred))
    display.subheader("R2 score is: ")
    display.write(r2_score(y, pred))
    
   
