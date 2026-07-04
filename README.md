# 🏦 Intelligent Loan Approval System

An enterprise-grade machine learning application designed to assess credit risk and predict loan defaults using real-world banking metrics. 

This project goes beyond simple predictive modeling by addressing real-world financial data challenges, such as class imbalance and spurious correlations, utilizing data structures heavily inspired by the Lending Club loan dataset.

## 🚀 Features

* **Interactive Risk Dashboard:** A frontend interface built with Streamlit that allows users to input applicant financial data and instantly receive an approval/denial evaluation.
* **AutoML Pipeline:** The backend training script automatically evaluates multiple algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM) using `GridSearchCV` and deploys the highest-performing model.
* **Class Imbalance Handling:** Implements mathematically balanced class weights to prevent the AI from defaulting to "lazy" auto-approvals due to skewed historical payment data.
* **Dynamic Model Recognition:** The UI automatically detects and displays the specific algorithm crowned as the "Champion Model."
* **Global & Local Feature Importance:** Visualizes exactly which financial metrics (e.g., Debt-to-Income, Interest Rate, FICO Score) drove the AI's decision, providing crucial transparency for financial compliance.

## 🛠️ Technology Stack

* **Language:** Python
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn, LightGBM, XGBoost
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib

## 📂 Project Structure

```text
├── app.py                # The Streamlit web application frontend
├── train_model.py        # The backend AutoML training pipeline
├── requirements.txt      # Python dependencies required to run the app
├── .gitignore            # Git exclusion rules (ignores raw datasets)
├── model.pkl             # The serialized champion ML model
├── scaler.pkl            # The saved standard scaler for data normalization
├── features.pkl          # The ordered list of model features
└── accuracy.pkl          # The saved ROC-AUC score of the champion model