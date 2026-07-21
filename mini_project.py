import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
st.title("🏥 Diabetes Risk Prediction & Model Analysis System")

@st.cache_resource
def load_and_train_model():
    # 1. Handle file paths for both local and Streamlit Cloud environments
    file_path = os.path.join("dataset", "diabetes_risk_dataset.csv")
    
    if not os.path.exists(file_path):
        file_path = "diabetes_risk_dataset.csv"
        if not os.path.exists(file_path):
            st.error("Dataset file not found! Please place 'diabetes_risk_dataset.csv' in your repository.")
            st.stop()
        
    data = pd.read_csv(file_path)
    
    # Identify target column dynamically
    target_col = [col for col in data.columns if "out" in col.lower() or "diag" in col.lower() or "diabet" in col.lower()]
    if not target_col:
        st.error("Could not find a target outcome column in dataset.")
        st.stop()
        
    selected_features = ['Age', 'BMI', 'Physical activity level', 'Blood pressure', 'Cholesterol', 'Glucose level']
    
    # 2. Fix target column processing to ensure clean binary integer targets (0 and 1)
    y_raw = data[target_col[0]].astype(str).str.strip().str.lower()
    mapping = {
        'no': 0, 'false': 0, '0': 0, '0.0': 0,
        'yes': 1, 'true': 1, '1': 1, '1.0': 1, 'diabetes': 1
    }
    y = y_raw.map(mapping)
    
    # Fallback to numeric conversion if mapping resulted in NaNs
    if y.isna().all():
        y = pd.to_numeric(data[target_col[0]], errors='coerce')
        
    valid_idx = y.notna()
    X = data.loc[valid_idx, selected_features].copy()
    y = y[valid_idx].astype(int)
    
    activity_categories = X['Physical activity level'].dropna().unique().tolist()
    
    # Fill missing feature values
    X['BMI'] = X['BMI'].fillna(X['BMI'].median())
    X['Glucose level'] = X['Glucose level'].fillna(X['Glucose level'].median())
    
    # Encode categorical variable
    X = pd.get_dummies(X, columns=['Physical activity level'], drop_first=True)
    processed_feature_names = X.columns.tolist()
    
    numerical_cols = ['Age', 'BMI', 'Blood pressure', 'Cholesterol', 'Glucose level']
    
    # Scale numerical columns using StandardScaler
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate Model
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    return rf_model, scaler, processed_feature_names, numerical_cols, activity_categories, metrics, fpr, tpr

# Load and train
rf_model, scaler, processed_feature_names, numerical_cols, activity_categories, metrics, fpr, tpr = load_and_train_model()

# UI Layout
col_left, col_right = st.columns(2)

with col_left:
    st.header("📊 Model Performance Metrics")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['Precision']:.2%}")
    c3.metric("Recall", f"{metrics['Recall']:.2%}")
    
    c4, c5 = st.columns(2)
    c4.metric("F1-Score", f"{metrics['F1-Score']:.2%}")
    c5.metric("ROC-AUC Score", f"{metrics['ROC-AUC']:.2%}")
    
    st.write("")
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {metrics['ROC-AUC']:.2f}")
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    ax.grid(True)
    
    st.pyplot(fig)
    plt.close(fig) 

with col_right:
    st.header("👤 Patient Risk Predictor Interface")
    st.write("Adjust the features below to view instantaneous clinical risk calculations.")
    
    age = st.slider("Age", 1, 100, 45)
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 24.5, step=0.1)
    bp = st.slider("Blood Pressure (Systolic)", 60, 200, 120)
    chol = st.slider("Cholesterol level (mg/dL)", 100, 400, 190)
    glucose = st.slider("Glucose Level (mg/dL)", 50, 300, 95)
    activity = st.selectbox("Physical Activity Level", options=activity_categories)
    
    user_input_df = pd.DataFrame([{
        'Age': age, 'BMI': bmi, 'Physical activity level': activity,
        'Blood pressure': bp, 'Cholesterol': chol, 'Glucose level': glucose
    }])
    
    input_encoded = pd.get_dummies(user_input_df, columns=['Physical activity level'])
    input_encoded = input_encoded.reindex(columns=processed_feature_names, fill_value=0)
    
    # Transform user inputs using StandardScaler
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])
    
    if st.button("Calculate Patient Risk Status", type="primary"):
        prediction = rf_model.predict(input_encoded)
        probability = rf_model.predict_proba(input_encoded)[0][1]
        
        st.write("---")
        if prediction == 1:
            st.error(f"🔴 **Prediction**: Patient has **High Risk of Diabetes**")
        else:
            st.success(f"🟢 **Prediction**: Patient has **Low Risk of Diabetes**")
            
        st.info(f"📊 **Calculated Clinical Risk Probability**: **{probability * 100:.1f}%**")
