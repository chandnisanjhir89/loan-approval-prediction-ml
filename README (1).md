# 🏦 Loan Approval Prediction System

An end-to-end machine learning project: raw applicant details go in, a loan approval prediction (with probability) comes out through an interactive **Streamlit** web app.

Built for **Devixo Solutions – AI/ML Internship – Task 04 (End-to-End ML Application with Deployment)**.

---

## 🎯 What the app does

| Page | Content |
|---|---|
| 🔍 **Predict** | Enter applicant info → click *Predict* → approval/rejection, approval probability, EMI & income summary, and what every other model predicts |
| 📊 **Model performance** | Metrics table for all 5 models, before-vs-after tuning chart, confusion matrix and ROC curve |
| ⭐ **Feature importance** | Permutation importance for the selected model |
| 📈 **Data visualization** | Interactive charts of the dataset |
| ⚙️ **Sidebar** | Choose any of the 5 trained models |

---

## 📂 Dataset

`LoanApprovalPrediction.csv` — 614 loan applications, 12 input columns + target `Loan_Status` (Y = approved, N = rejected; about 69% approved).

`Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount (in thousands), Loan_Amount_Term (months), Credit_History, Property_Area`

`Loan_ID` is only a label, so it is dropped.

---

## 🔧 What was done (notebook: `Loan_Approval_Prediction.ipynb`)

1. **Data processing**
   - Missing values: median (numeric) / most frequent (categorical), done *inside* the pipeline to avoid data leakage
   - Duplicate check and removal
   - **Outlier detection** with the IQR rule; values are capped (bounds learned on training data only)
   - **Feature engineering**: `TotalIncome`, `EMI`, `Loan_to_Income`, `Balance_Income`, `Log_TotalIncome`
   - One-hot encoding and standard scaling
2. **Model development** — 5 models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM
   - Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC (+ 5-fold cross-validated ROC-AUC)
   - `class_weight="balanced"` is used to handle the imbalance between approved and rejected loans
3. **Hyperparameter tuning** — `GridSearchCV` (5-fold stratified, scored on ROC-AUC) for every model, with a before/after comparison
4. **Model selection** — best model chosen by cross-validated ROC-AUC on the training data (not on the test set)
5. **Model saving** — final model with `joblib`

### Results (test set = 123 applications)

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (final, tuned)** | 0.870 | 0.879 | 0.941 | 0.909 | 0.871 |

Cross-validated ROC-AUC: 0.765 → 0.770 after tuning. The most important feature by far is **Credit_History**, followed by income and loan amount.

> ⚠️ The test set is small (123 rows), so differences of 1–2% between models are within noise. Treat this as a learning project, not a production credit-scoring system.

---

## 🚀 Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model, open and run `Loan_Approval_Prediction.ipynb` from top to bottom — it regenerates all `.joblib`, `.csv` and `.json` files used by the app.

## ☁️ Deploy (Streamlit Community Cloud — free)

1. Push this whole folder to a GitHub repository
2. Go to <https://share.streamlit.io> and sign in with GitHub
3. Click **New app**, choose the repository, branch `main`, main file `app.py`
4. Click **Deploy**

Add your live link here after deployment: `https://<your-app-name>.streamlit.app`

---

## 📁 Project structure

```
├── app.py                          # Streamlit application
├── loan_utils.py                   # Custom transformer (outliers + feature engineering) and data helpers
├── Loan_Approval_Prediction.ipynb  # Full ML workflow
├── LoanApprovalPrediction.csv      # Dataset
├── loan_approval_model.joblib      # Final (best) model
├── all_models.joblib               # All 5 tuned models (for model selection in the app)
├── model_results.csv               # Metrics before/after tuning
├── feature_importance.csv          # Permutation importance per model
├── model_meta.json                 # Best model name and parameters
└── requirements.txt
```

## 🛠️ Tech stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Joblib · Streamlit · Git & GitHub
