# 💳 Credit Risk Prediction

A machine learning project that predicts whether a loan applicant is a **good** or **bad** credit risk, served through an interactive **Streamlit** web app. The model is an **XGBoost** classifier trained on the German Credit dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

---

## 📌 Overview

Lenders need a fast way to judge how risky a loan application is. This project covers the full workflow:

1. **Exploratory data analysis** of 1,000 loan applicants
2. **Data cleaning and encoding** of categorical features
3. **Training and tuning** four tree-based models with cross-validated grid search
4. **Deploying** the best model in a simple web app where a user enters applicant details and gets an instant prediction with a confidence score

---

## 🖥️ App Features

- Sidebar form for applicant details (age, sex, job level, housing, savings, checking account, credit amount, duration)
- Live **application summary** table
- One-click **risk prediction** with a green (GOOD) or red (BAD) result card
- **Confidence score** and progress bar based on the model's predicted probability
- Friendly error message if an input value is not one the model was trained on

> 📸 *Add a screenshot of the app here:* `![App Screenshot](screenshots/app.png)`

---

## 📊 Dataset

The [German Credit dataset](https://www.kaggle.com/datasets/uciml/german-credit) contains 1,000 applicants.

| Column | Description |
|---|---|
| Age | Applicant age |
| Sex | male / female |
| Job | 0 = unskilled non-resident, 1 = unskilled resident, 2 = skilled, 3 = highly skilled |
| Housing | own / rent / free |
| Saving accounts | little / moderate / quite rich / rich |
| Checking account | little / moderate / rich |
| Credit amount | Loan amount requested |
| Duration | Loan duration in months |
| Purpose | Reason for the loan (not used in the model) |
| **Risk** | **Target:** good / bad |

Saving accounts and Checking account contain missing values. Rows with missing values were dropped, leaving roughly 525 rows for modeling.

---

## 🧠 Methodology

1. **Cleaning:** removed the unused index column and rows with missing values
2. **EDA:** distributions, boxplots, count plots, correlation heatmap, and feature comparisons by risk class
3. **Encoding:** `LabelEncoder` for each categorical column (encoders saved with `joblib` so the app reuses the exact same mapping)
4. **Split:** 80% train / 20% test, stratified by the target
5. **Model selection:** `GridSearchCV` with 5-fold cross-validation, comparing four models
6. **Class imbalance:** handled with `class_weight="balanced"` (tree models) and `scale_pos_weight` (XGBoost)

### Results

| Model | Test Accuracy |
|---|---|
| Decision Tree | 58.1% |
| Random Forest | 66.7% |
| Extra Trees | 64.8% |
| **XGBoost** | **68.6%** |

**Best XGBoost parameters:** `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `subsample=0.7`, `colsample_bytree=0.7`

---

## 📁 Project Structure

```
credit-risk-prediction/
├── app.py                          # Streamlit web app
├── analysis_model.ipynb            # EDA + model training notebook
├── german_credit_data.csv          # Dataset
├── XGB_credit_model.pkl            # Trained XGBoost model
├── Sex_encoder.pkl                 # Label encoders
├── Housing_encoder.pkl
├── Saving accounts_encoder.pkl
├── Checking account_encoder.pkl
├── target_encoder.pkl              # Maps bad = 0, good = 1
├── requirements.txt
└── README.md
```

> **Note:** `app.py` loads the encoder files by column name, so `Saving accounts_encoder.pkl` and `Checking account_encoder.pkl` must keep the space in their file names.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/credit-risk-prediction.git
cd credit-risk-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
seaborn
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### 4. (Optional) Re-train the model

Open `analysis_model.ipynb` in Jupyter and run all cells. This regenerates the model and encoder `.pkl` files.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn, XGBoost
- **Web app:** Streamlit
- **Model persistence:** Joblib

---

## ⚠️ Limitations and Future Improvements

- Accuracy is modest (about 68.6%) and measured on a small test set of 105 rows, so the estimate is noisy. Precision, recall and ROC-AUC should be reported alongside accuracy.
- Dropping rows with missing values removes nearly half the data. Treating "missing" as its own category (for example, "no account") would keep more rows.
- Savings and checking levels are ordered categories, so an explicit `OrdinalEncoder` would be more appropriate than alphabetical label encoding.
- The `Purpose` feature is not used yet and could be added.
- The model uses `Sex` as an input. This is fine for a learning project, but real lending systems must follow fairness and anti-discrimination rules.
- Possible next steps: SHAP explanations in the app, ROC and confusion-matrix plots, and deployment on Streamlit Community Cloud.

---

## 👤 Author

**Tuhin Roy**
BCA Student, JIS University, Kolkata
Interested in Data Analytics, SQL, and AI/ML

- GitHub: [TuhinRoy07](https://github.com/your-username)

---

## 📄 License

This project is for educational purposes. Add a license of your choice (for example, MIT) before sharing publicly.
