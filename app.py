import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Intelligent Loan Assessor", page_icon="🏦", layout="wide")

# 2. Load the trained artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    accuracy = joblib.load('accuracy.pkl')
    return model, scaler, features, accuracy

model, scaler, features, accuracy = load_artifacts()

# Extract the algorithm's actual name dynamically
model_name = type(model).__name__

# 3. Header Section
st.title("🏦 Intelligent Loan Approval System")
st.markdown("Predict loan outcomes based on income, credit history, and key household expense drivers (Dependents, Marital Status, Education).")

# 4. Champion Model Overview (NEW SECTION)
st.markdown("---")
st.markdown("### 🏆 Champion Model Overview")
met1, met2 = st.columns(2)
met1.metric(label="Best Algorithm Deployed", value=model_name)
met2.metric(label="Model ROC-AUC Accuracy", value=f"{accuracy:.2%}")

# 5. Show Data & Global Feature Importance Option (NEW SECTION)
with st.expander("📊 View Training Data & Global Feature Importance"):
    col_data, col_feat = st.columns(2)
    
    with col_data:
        st.markdown("Raw Training Data Preview")
        try:
            # Load and show the first 100 rows of the training data
            raw_data = pd.read_csv('loan_data.csv')
            st.dataframe(raw_data.head(100), use_container_width=True)
        except FileNotFoundError:
            st.warning("⚠️ 'loan_data.csv' not found in the current folder. Please ensure the file is present to view data.")

    with col_feat:
        st.markdown("#### Overall Feature Importance")
        fig_global, ax_global = plt.subplots(figsize=(6, 4))
        
        # Calculate feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.ones(len(features))
            
        indices = np.argsort(importances)
        clean_labels = [features[i].replace('_', ' ') for i in indices]
        
        ax_global.barh(range(len(indices)), importances[indices], color='#1f77b4')
        ax_global.set_yticks(range(len(indices)))
        ax_global.set_yticklabels(clean_labels)
        ax_global.set_xlabel("Relative Importance / Weight")
        
        st.pyplot(fig_global)

st.markdown("---")

# 6. Sidebar: Display Model Metrics
st.sidebar.header("🧠 Model Architecture")
st.sidebar.metric("Champion Model ROC-AUC", f"{accuracy:.2%}")
st.sidebar.info(
    f"This pipeline automatically deployed the most confident algorithm (**{model_name}**) "
    "using GridSearchCV."
)

# 7. Input Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Assets & Requests")
    app_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
    co_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Requested Loan Amount (in thousands)", min_value=1.0, value=150.0, step=10.0)
    credit_history = st.selectbox("Credit History", ["Good (No past defaults)", "Bad (Past defaults)"])

with col2:
    st.subheader("Household Expense Drivers")
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    married = st.selectbox("Marital Status", ["Married", "Single"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])

st.markdown("---")

# 8. Prediction Logic
if st.button("Predict Eligibility", type="primary"):
    
    # A. Map text inputs back to the numbers the LabelEncoder used
    cred_map = 1.0 if "Good" in credit_history else 0.0
    edu_map = 0 if education == "Graduate" else 1
    mar_map = 1 if married == "Married" else 0
    dep_map = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
    
    # B. Feature Engineering
    total_income = app_income + co_income
    loan_log = np.log(loan_amount)
    
    # C. Create DataFrame ensuring exact column names and order
    input_df = pd.DataFrame([[total_income, loan_log, cred_map, edu_map, mar_map, dep_map]], columns=features)
    
    # D. Scale the inputs
    input_scaled = scaler.transform(input_df)
    
    # E. Get Predictions and Confidence Scores
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)[0]
    
    # 9. Display Results
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("Evaluation Result")
        if prediction[0] == 1:
            st.success("🎉 **APPROVED**")
            st.write(f"The AI is **{probabilities[1]:.1%} confident** in approving this loan.")
            st.progress(float(probabilities[1]))
        else:
            st.error("❌ **DENIED**")
            st.write(f"The AI is **{probabilities[0]:.1%} confident** in denying this loan.")
            st.progress(float(probabilities[0]))
            
    with res_col2:
        st.subheader("Decision Drivers")
        st.write("Feature Importance for this decision:")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # We reuse the importances calculated earlier
        ax.barh(range(len(indices)), importances[indices], color='#4CAF50')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(clean_labels)
        ax.set_xlabel("Relative Importance / Weight")
        
        st.pyplot(fig)