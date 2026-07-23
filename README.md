https://credit-score-predictor-eshat.streamlit.app/
# 💳 Credit Score Predictor

An end-to-end Machine Learning web application designed to classify and predict credit score categories based on financial and banking customer profiles.

---

## 📌 Features

* **Data Preprocessing & Encoding:** Automated handling of categorical features using custom label encoders.
* **Multi-Model Evaluation:** Experiments conducted with multiple algorithms including **Random Forest**, **XGBoost**, and **CatBoost**.
* **High-Performance Inference:** Employs optimized model artifacts saved via `joblib` for rapid predictions.
* **Interactive Web UI:** Simple, user-friendly interface built with Python for instant predictions.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python 3.x
* **Core Libraries:** `pandas`, `numpy`, `scikit-learn`
* **ML Frameworks:** `xgboost`, `catboost`
* **Version Control & LFS:** Git, Git LFS (Large File Storage)

---

## 📂 Repository Structure

```text
├── bank.ipynb              # Exploratory Data Analysis & Model Training Notebook
├── app.py                  # Main application entry point
├── requirements.txt        # Python package dependencies
├── label_encoders.joblib   # Categorical feature encoders
├── XGBoost_model.joblib    # Trained XGBoost classifier
├── catboost_model.joblib   # Trained CatBoost classifier (managed via Git LFS)
├── .gitattributes          # Git LFS configuration file
└── .gitignore              # Files ignored by version control
