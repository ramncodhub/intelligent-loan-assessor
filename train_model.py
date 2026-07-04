import pandas as pd
import numpy as np
import joblib
import warnings

# Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Preprocessing & Evaluation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# 1. Load Original Data
print("Loading 'loan_data.csv'...")
df = pd.read_csv('loan_data.csv')

# 2. Preprocessing & Cleaning
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Fill categorical missing values
for col in ['Married', 'Dependents', 'Education']:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Feature Engineering

df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
# Log transform to handle skewed data
df['LoanAmount_Log'] = np.log(df['LoanAmount'])

# 4. Encode Categorical Variables
le = LabelEncoder()

for col in ['Married', 'Education', 'Dependents']:
    df[col] = le.fit_transform(df[col].astype(str))

# Convert Target to binary numbers (1 for Y, 0 for N)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# 5. Feature Selection (Practical Financial & Expense Features)
features = ['Total_Income', 'LoanAmount_Log', 'Credit_History', 'Education', 'Married', 'Dependents']
X = df[features]
y = df['Loan_Status']

# 6. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Define Models and Hyperparameter Grids
models = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42),
        "params": {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, eval_metric='logloss'),
        "params": {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=42, objective='binary', verbose=-1),
        "params": {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [15, 31]
        }
    }
}

# 9. Execute Grid Search for Every Model
best_overall_model = None
best_overall_score = 0
best_model_name = ""

print("\nStarting AutoML Pipeline: Tuning algorithms using ROC-AUC...\n")

for model_name, model_info in models.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(
        model_info["model"], 
        model_info["params"], 
        cv=5, 
        scoring='roc_auc', 
        n_jobs=-1 
    )
    
    grid_search.fit(X_train, y_train)
    current_best_model = grid_search.best_estimator_
    
    # Score using Probabilities
    y_pred_proba = current_best_model.predict_proba(X_test)[:, 1]
    roc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f" - Best Params: {grid_search.best_params_}")
    print(f" - ROC-AUC Score: {roc_score * 100:.2f}%\n")
    
    if roc_score > best_overall_score:
        best_overall_score = roc_score
        best_overall_model = current_best_model
        best_model_name = model_name

print("="*40)
print(f" CHAMPION MODEL: {best_model_name}")
print(f" CHAMPION ROC-AUC: {best_overall_score * 100:.2f}%")
print("="*40)

# 10. Save Only the Champion Model's Artifacts
joblib.dump(best_overall_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features, 'features.pkl')
joblib.dump(best_overall_score, 'accuracy.pkl')

print("\nBest model saved !")