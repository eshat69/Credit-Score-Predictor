https://credit-score-predictor-eshat.streamlit.app/

# 💳 Credit Score Predictor

An end-to-end Machine Learning web application that predicts customer credit score categories using financial and banking information. The project covers data preprocessing, feature engineering, model training, evaluation, and deployment through an interactive **Streamlit** web application.

---

## 📖 Project Overview

Credit score prediction is a crucial task in the financial industry, helping banks and financial institutions assess customer creditworthiness. This project leverages Machine Learning models to classify customers into different credit score categories based on their financial behavior and banking history.

### Workflow

- Data Collection
- Data Preprocessing
- Feature Engineering
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Feature Importance Analysis
- Streamlit Web App Deployment

---

## 🚀 Features

- ✅ Data Cleaning & Preprocessing
- ✅ Missing Value Handling
- ✅ Label Encoding for Categorical Features
- ✅ Feature Engineering
- ✅ Multi-Model Training & Comparison
- ✅ Hyperparameter Optimization
- ✅ Feature Importance Visualization
- ✅ Real-Time Credit Score Prediction
- ✅ Interactive Streamlit Web Application
- ✅ Saved Models using Joblib

---

## 🌐 Live Demo

🔗 **Try the App Here:**

https://your-app-name.streamlit.app

> Replace the URL above with your deployed Streamlit application.

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

- 🌲 Random Forest Classifier
- 🚀 XGBoost Classifier
- 🐱 CatBoost Classifier

After comparing model performance, the **CatBoost model** was selected for deployment due to its superior predictive performance.

---

## 📊 Dataset Features

The model uses customer financial information such as:

- Age
- Occupation
- Annual Income
- Monthly Inhand Salary
- Number of Bank Accounts
- Number of Credit Cards
- Interest Rate
- Number of Loans
- Delay from Due Date
- Number of Delayed Payments
- Changed Credit Limit
- Number of Credit Inquiries
- Credit Mix
- Outstanding Debt
- Credit Utilization Ratio
- Credit History Age
- Payment Behaviour
- Monthly Balance
- Total EMI per Month
- Payment of Minimum Amount
- Type of Loan

---

## 📈 Model Performance

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Feature Importance analysis was performed to interpret model predictions.

### Top Influential Features

- Outstanding Debt
- Credit Mix
- Interest Rate
- Delay from Due Date
- Changed Credit Limit

These features contributed most significantly to predicting customer credit scores.

---

## 🛠️ Tech Stack

### Programming Language

- Python 3.x

### Libraries & Frameworks

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib
- Seaborn
- Joblib

### Tools

- Jupyter Notebook
- Git
- Git LFS
- VS Code

---

## 📂 Repository Structure

```text
Credit-Score-Predictor/
│
├── bank.ipynb                 # EDA, preprocessing & model training
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── label_encoders.joblib      # Saved label encoders
├── XGBoost_model.joblib       # Trained XGBoost model
├── catboost_model.joblib      # Trained CatBoost model
├── assets/
│   └── feature_importance.png # Feature importance visualization
├── .gitattributes             # Git LFS configuration
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/eshat9/Credit-Score-Predictor.git
```

### 2. Navigate to the Project Directory

```bash
cd Credit-Score-Predictor
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Launch the Streamlit app locally:

```bash
streamlit run app.py
```

### 🌐 Live Application

Access the deployed application here:
https://credit-score-predictor-eshat.streamlit.app/

---

## 📷 Sample Output

The application predicts a customer's credit score category based on financial information entered by the user.

Example Prediction:

```text
Predicted Credit Score

🟢 Good
```

---

## 📊 Feature Importance

The CatBoost model provides feature importance scores that improve model interpretability.

Top predictors include:

- Outstanding Debt
- Credit Mix
- Interest Rate
- Delay from Due Date
- Changed Credit Limit

This analysis helps explain which customer attributes have the greatest influence on credit score prediction.

---

## 📌 Future Improvements

- 🔐 User Authentication
- 📊 Interactive Analytics Dashboard
- ⚡ FastAPI REST API
- 🐳 Docker Containerization
- ☁️ AWS/Azure Deployment
- 📈 Automated Model Retraining
- 🤖 Explainable AI (SHAP/LIME)
- 🔄 CI/CD Pipeline Integration

---

## 👨‍💻 Author

**Eshat Rahman**



- GitHub: https://github.com/eshat69
- LinkedIn: https://linkedin.com/in/eshat7

