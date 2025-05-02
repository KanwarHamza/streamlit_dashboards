import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import pickle

# App configuration
st.set_page_config(page_title="ML Playground", layout="wide")

# App title and headings
st.title("🎯 Machine Learning Playground")
st.write("Explore different datasets and classifiers with this interactive app!")

# ========================================
# Sidebar Controls
# ========================================
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Dataset selection
    dataset = st.selectbox(
        'Select a Dataset',
        ('Upload file', 'Iris Dataset', 'Breast Cancer Dataset', 'Wine Dataset')
    )
    
    # File uploader
    uploaded_file = None
    if dataset == 'Upload file':
        uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=['csv'])
        if uploaded_file:
            st.success("File uploaded successfully!")
    
    # Model selection
    model = st.selectbox(
        'Select a Model',
        ('KNN', 'SVM', 'Random Forest')
    )
    
    # Hyperparameters
    st.markdown("### Model Hyperparameters")
    param = {}
    if model == 'KNN':
        param['K'] = st.slider('Number of neighbors (K)', 1, 15, 5)
    elif model == 'SVM':
        param['C'] = st.slider('Regularization (C)', 0.01, 10.0, 1.0)
    else:
        param['max_depth'] = st.slider('Max depth', 2, 15, 5)
        param['n_estimators'] = st.slider('Number of trees', 1, 100, 10)

# ========================================
# Data Loading and Preparation
# ========================================
@st.cache_data
def load_data(dataset, uploaded_file):
    if dataset == 'Upload file':
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Let user select target column
            target_col = st.selectbox("Select target column", df.columns)
            
            # Separate features and target
            X = df.drop(columns=[target_col]).values
            y = df[target_col].values
            
            # Encode text labels to numbers
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            return X, y, df
        else:
            st.warning("Please upload a CSV file")
            st.stop()
    else:
        # Load sklearn built-in datasets
        if dataset == 'Iris Dataset':
            data = datasets.load_iris()
        elif dataset == 'Breast Cancer Dataset':
            data = datasets.load_breast_cancer()
        else:
            data = datasets.load_wine()
        
        X = data.data
        y = data.target
        df = pd.DataFrame(X, columns=data.feature_names if hasattr(data, 'feature_names') else range(X.shape[1]))
        df['target'] = y
        
        return X, y, df

# Load data
X, y, df = load_data(dataset, uploaded_file)

# ========================================
# Data Exploration Section
# ========================================
st.header("🔍 Data Exploration")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(), height=200)

with col2:
    st.markdown("### Basic Statistics")
    st.dataframe(df.describe(), height=200)

# Visualization tabs
tab1, tab2, tab3 = st.tabs(["📊 Pair Plot", "📈 Correlation", "🔢 Distribution"])

with tab1:
    st.write("Feature relationships (first 5 columns)")
    fig = px.scatter_matrix(df.iloc[:, :5], color=df['target'])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("Feature correlation heatmap")
    fig = px.imshow(df.corr(), text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    selected_col = st.selectbox("Select column to visualize", df.columns[:-1])
    fig = px.histogram(df, x=selected_col, color='target', marginal="box")
    st.plotly_chart(fig, use_container_width=True)

# ========================================
# Model Training and Evaluation
# ========================================
st.header("🤖 Model Training")

# Get classifier
def get_classifier(model, param):
    if model == 'KNN':
        return KNeighborsClassifier(n_neighbors=param['K'])
    elif model == 'SVM':
        return SVC(C=param['C'])
    else:
        return RandomForestClassifier(
            n_estimators=param['n_estimators'], 
            max_depth=param['max_depth'], 
            random_state=42
        )

clf = get_classifier(model, param)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training
try:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Accuracy", f"{acc:.2%}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    
    with col2:
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)
    
except Exception as e:
    st.error(f"Training failed: {str(e)}")

# ========================================
# Visualization Section
# ========================================
st.header("📊 Visualizations")

# PCA Projection
pca = PCA(2)
X_projected = pca.fit_transform(X)

col1, col2 = st.columns(2)
with col1:
    st.write("Matplotlib Version")
    fig1, ax = plt.subplots()
    scatter = ax.scatter(X_projected[:, 0], X_projected[:, 1], c=y, alpha=0.6, cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    plt.colorbar(scatter)
    st.pyplot(fig1)

with col2:
    st.write("Interactive Plotly Version")
    fig2 = px.scatter(
        x=X_projected[:, 0], 
        y=X_projected[:, 1], 
        color=y,
        title="PCA Projection",
        labels={'x': 'PC1', 'y': 'PC2'},
        hover_name=y
    )
    st.plotly_chart(fig2, use_container_width=True)

# ========================================
# Model Persistence
# ========================================
st.header("💾 Save/Load Model")

col1, col2 = st.columns(2)
with col1:
    if st.button("Save Model"):
        with open("saved_model.pkl", "wb") as f:
            pickle.dump(clf, f)
        st.success("Model saved to 'saved_model.pkl'")

with col2:
    if st.button("Load Model"):
        try:
            with open("saved_model.pkl", "rb") as f:
                clf = pickle.load(f)
            st.success("Model loaded successfully!")
        except FileNotFoundError:
            st.error("No saved model found!")

# ========================================
# Feature Importance (for tree-based models)
# ========================================
if hasattr(clf, "feature_importances_"):
    st.write("### Feature Importance")
    importance = pd.DataFrame({
        'Feature': range(len(clf.feature_importances_)),
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance, x='Feature', y='Importance')
    st.plotly_chart(fig, use_container_width=True)