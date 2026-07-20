# 🏥 Diabetes Risk Prediction System

An interactive machine learning web application built with **Streamlit**, **TensorFlow**, and **Scikit-Learn** to analyze clinical datasets and predict patient diabetes risk profiles instantaneously.

---

## 🚀 How to Run the App

Follow these simple steps to install the prerequisites and launch the application on your local machine.

### Prerequisites
Make sure you have Python installed (Python 3.9 to 3.13 supported).

### 1. Install Dependencies
Open your terminal or command prompt inside the project folder and install the required libraries from your `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Run the Streamlit web server by executing your main script:



```bash
streamlit run mini_project.py
```

### 3. Access the Web App
Once running, the terminal will provide a local URL. Open your web browser and navigate to:


http://localhost:8501
---

## 📊 Dataset Description

The application processes a clinical dataset (`diabetes_risk_dataset.csv`) containing physiological and behavioral health metrics of patients. The system utilizes **6 core predictive features** to determine diabetes risk:

1. **Age**: The chronological age of the patient (Numerical).
2. **BMI (Body Mass Index)**: A measure of body fat based on height and weight (Numerical).
3. **Physical Activity Level**: Categorical classification of the patient's lifestyle movement (Categorical, e.g., 'Low', 'Medium', 'High').
4. **Blood Pressure**: Systolic blood pressure measurements (Numerical).
5. **Cholesterol**: Total cholesterol levels measured in mg/dL (Numerical).
6. **Glucose Level**: Blood sugar concentration levels measured in mg/dL (Numerical).
7. **Target Output (e.g., Outcome/Diagnosis)**: Binary indicator representing presence ('Yes') or absence ('No') of diabetes risk.

---

## ⚙️ Preprocessing Techniques

To ensure high mathematical stability and peak model performance, the pipeline implements an automated data preprocessing layer:

* **Missing Value Imputation**: Missing values within critical fields (`BMI` and `Glucose level`) are automatically filled using the column **median**. This handles data gaps safely without skewing results via outliers.
* **Target Label Conversion**: The target text variables (`'No'` / `'Yes'`) are encoded into machine-readable binary values (`0` / `1`).
* **Categorical Feature Encoding**: The `Physical activity level` textual groupings are transformed into numeric structures using **One-Hot Encoding** (`pd.get_dummies`), avoiding any artificial rank bias.
* **Feature Scaling via TensorFlow**: Numerical vectors are standardized using a **TensorFlow/Keras `Normalization` Layer**. This centers features around a mean of `0` with a standard deviation of `1`. Crucially, this normalization layer adapts to the training data and processes user inputs dynamically to prevent data leakage.

---

## 🤖 Model Training & Results

* **Algorithm**: The core classifier uses a **Random Forest Classifier** configured with 100 decision tree estimators.
* **Data Splitting**: The dataset is divided using a stratified **80% training** and **20% testing** split. Stratification ensures that the balance of diabetic vs. non-diabetic records remains identical across both sets.
* **Inference Pipeline**: The model operates in tandem with the saved Keras scale parameters, mapping live user input settings from the web interface to match the training data format seamlessly.

---

## 📈 Evaluation Metrics

The system measures classifier reliability across 5 distinct clinical evaluation standards, visible instantly on the app's dashboard alongside a live-rendered **ROC-AUC Curve**:

* **Accuracy**: The overall percentage of correctly identified patient states (both diabetic and non-diabetic).
* **Precision**: The proportion of predicted high-risk patients who actually have diabetes (minimizing false alarms).
* **Recall (Sensitivity)**: The proportion of actual diabetic patients correctly flagged by the system (minimizing missed clinical cases).
* **F1-Score**: The harmonic mean of Precision and Recall, providing a balanced metric for uneven datasets.
* **ROC-AUC Score**: Measures the model's capability to distinguish between positive risk states and healthy states across all thresholds. Higher values indicate sharper diagnostic accuracy.
