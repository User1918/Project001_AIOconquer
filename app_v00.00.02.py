import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
import joblib
import sklearn


# Model paths
RF_MODEL_PATH = "random_forest_model.joblib"
LR_MODEL_PATH = "logistic_regression_model.joblib"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "model_features.pkl"

@st.cache_resource
def load_artifacts():
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        lr_model = joblib.load(LR_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        model_features = joblib.load(FEATURES_PATH)
        return rf_model, lr_model, scaler, model_features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please make sure scikit-learn version is compatible. Try: pip install --upgrade scikit-learn")
        return None, None, None, None

def preprocess_input(input_dict, scaler, model_features):
    """
    Biến đổi dữ liệu nhập vào y hệt như lúc train.
    """
    # 1. Tạo DataFrame từ input dictionary
    df = pd.DataFrame([input_dict])
    
    # 2. Binary Encoding (Thủ công cho chắc ăn)
    df['OverTime'] = 1 if df['OverTime'].iloc[0] == 'Yes' else 0
    
    # 3. One-Hot Encoding
    if 'MaritalStatus' in df.columns:
        df = pd.get_dummies(df, columns=['MaritalStatus'], drop_first=True)
    
    # 4. ALIGN COLUMNS (Bước QUAN TRỌNG NHẤT)
    df = df.reindex(columns=model_features, fill_value=0)
    
    # 5. Scaling
    numeric_cols = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

def predict_employee(input_data, model, scaler, model_features, model_name="Model"):
    """
    Predict for a single model with error handling
    """
    try:
        # Xử lý dữ liệu
        X_new = preprocess_input(input_data, scaler, model_features)
        
        # Predict
        prediction = model.predict(X_new)[0]
        
        # Try predict_proba with error handling
        try:
            probability = model.predict_proba(X_new)
        except Exception as proba_error:
            st.warning(f"{model_name}: predict_proba failed, using binary prediction only")
            # Fallback: create pseudo-probabilities
            if prediction == 1:
                probability = np.array([[0.3, 0.7]])
            else:
                probability = np.array([[0.7, 0.3]])
        
        # Trả về kết quả
        result = "Leave" if prediction == 1 else "Stay"
        confidence = probability[0][prediction] * 100
        
        return result, confidence, prediction, probability[0]
    
    except Exception as e:
        st.error(f"Error in {model_name} prediction: {e}")
        return "Error", 0, 0, np.array([0.5, 0.5])

# Streamlit UI
st.title("Employee Attrition Prediction - Model Comparison")
st.markdown("Compare predictions from **Random Forest** and **Logistic Regression** models")

# Display scikit-learn version

st.sidebar.info(f"scikit-learn version: {sklearn.__version__}")

st.header("Employee Info")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)

with col2:
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
    overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
    job_satisfaction = st.slider(
        "Job Satisfaction",
        min_value=1,
        max_value=4,
        value=3,
        help="1: Low, 2: Medium, 3: High, 4: Very High"
    )

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

st.markdown("---")
predict_button = st.button("Compare Model Predictions", use_container_width=True)

if predict_button:
    input_data = {
        'Age': age,
        'MonthlyIncome': monthly_income,
        'TotalWorkingYears': total_working_years,
        'YearsAtCompany': years_at_company,
        'OverTime': overtime,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status 
    }
    
    with st.spinner('Analyzing employee data with both models...'):
        rf_model, lr_model, scaler, model_features = load_artifacts()
        
        if rf_model is None or lr_model is None:
            st.stop()
        
        # Get predictions from both models
        rf_result, rf_conf, rf_prediction, rf_probabilities = predict_employee(
            input_data, rf_model, scaler, model_features, "Random Forest"
        )
        lr_result, lr_conf, lr_prediction, lr_probabilities = predict_employee(
            input_data, lr_model, scaler, model_features, "Logistic Regression"
        )
    
    st.markdown("---")
    st.header("Model Comparison Results")
    
    # Side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Random Forest")
        if rf_prediction == 1:
            st.error(f"{rf_result}")
        else:
            st.success(f"### {rf_result}")
        st.markdown(f"**Confidence:** {rf_conf:.2f}%")
        
        st.markdown("#### Probabilities")
        st.metric("Stay", f"{rf_probabilities[0]*100:.2f}%")
        st.metric("Leave", f"{rf_probabilities[1]*100:.2f}%")
        
    with col2:
        st.subheader(" Logistic Regression")
        if lr_prediction == 1:
            st.error(f"###  {lr_result}")
        else:
            st.success(f"###  {lr_result}")
        st.markdown(f"**Confidence:** {lr_conf:.2f}%")
        
        st.markdown("#### Probabilities")
        st.metric("Stay", f"{lr_probabilities[0]*100:.2f}%")
        st.metric("Leave", f"{lr_probabilities[1]*100:.2f}%")

         
