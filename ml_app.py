import pandas as pd
from sklearn.base import is_classifier
import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.decomposition import PCA

#app title and headings
st.title("Machine Learning App")
st.write('This is a machine learning app applied on different datasets.')

#a sideb box for the user to select the dataset
dataset = st.sidebar.selectbox(
    'Select a Dataset',
    ('Iris Dataset', 'Breast Cancer Dataset', 'Wine Dataset')
)

#a side box for the user to select the model
model = st.sidebar.selectbox(
    'Select a Model',
    ('KNN', 'SVM', 'Random Forest')
)

#import the three datasets
def get_data(dataset):
    if dataset == 'Iris Dataset':
        data = datasets.load_iris()
    elif dataset == 'Breast Cancer Dataset':
        data = datasets.load_breast_cancer()
    else:
        dataset == 'Wine Dataset' #though this line is not required since else means whatever is left in the else statement will be executed
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

#setting the parameters for the models
X, y = get_data(dataset)

#printing the data shape
st.write('Shape of X: ', X.shape)
st.write('Number of classes: ', len(np.unique(y)))

#setting the parameters for the models
def add_parameter_ui(classifier):
    param = dict() #create a dictionary
    if classifier == 'KNN':
        K = st.sidebar.slider('K', 1, 15) #number of neighbors
        param['K'] = K
    elif classifier == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0) #regularization parameter
        param['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15) #depth of the tree
        param['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100) #number of trees in the forest
        param['n_estimators'] = n_estimators 
    return param

#calling the function to add parameters
param = add_parameter_ui(model)

#setting the parameters based on the classifier
def get_classifier(classifier, param):
    clf = None
    if classifier == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=param['K'])
    elif classifier == 'SVM':
        clf = SVC(C=param['C'])
    else:
        clf = RandomForestClassifier(n_estimators=param['n_estimators'], 
            max_depth=param['max_depth'], random_state=1234)
    return clf
    
#calling on to the function and naming it clf
clf = get_classifier(model, param)

#now we are going to train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#train the model
try:
    clf.fit(X_train, y_train)
except ValueError as e:
    st.error(f"⚠️ Training failed: {e}")
    st.info("Try adjusting hyperparameters")
#predict the model
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {clf.__class__.__name__}')
st.write(f'Accuracy = {acc}')

# After accuracy_score:
st.write("**Detailed Performance:**")
report = classification_report(y_test, y_pred, output_dict=True)
st.table(pd.DataFrame(report).transpose())  # Displays as a nice table

#plotting the plot based on PCA
pca = PCA(2)
X_projected = pca.fit_transform(X)

#we will slice our data into two variables 0 and 1 as PCA will give us two dimensions
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.5, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt show the plot
st.pyplot(fig)

fig = px.scatter(
    x=x1, y=x2, 
    color=y, 
    title="PCA Projection",
    labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
    hover_name=y  # Shows class label on hover
)
st.plotly_chart(fig)

# Add this after training:
if st.button("💾 Save Model"):
    with open("saved_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    st.success("Model saved to 'saved_model.pkl'!")

# To load later:
if st.button("📂 Load Model"):
    try:
        with open("saved_model.pkl", "rb") as f:
            clf = pickle.load(f)
        st.success("Model loaded!")
    except FileNotFoundError:
        st.error("No saved model found!")
        

