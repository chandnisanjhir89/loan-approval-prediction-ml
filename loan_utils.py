"""Shared helpers for the Loan Approval project.

The custom transformer lives in a module (not inside the notebook) so that the
saved .joblib pipeline can be loaded later by app.py without errors.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_FEATURES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
]

# Columns capped for outliers (IQR rule, bounds learned on training data only)
OUTLIER_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]

# Columns that exist after feature engineering
NUMERIC_FEATURES = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "TotalIncome", "EMI", "Loan_to_Income", "Balance_Income", "Log_TotalIncome",
]
CATEGORICAL_FEATURES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "Credit_History", "Property_Area",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """1) Caps outliers with the IQR rule (bounds learned in fit -> no leakage)
    2) Creates new features: TotalIncome, EMI, Loan_to_Income,
       Balance_Income, Log_TotalIncome.
    Missing values are left as NaN here; the next pipeline step imputes them.
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.bounds_ = {}
        for col in OUTLIER_COLS:
            q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, (low, high) in self.bounds_.items():
            X[col] = X[col].astype(float).clip(lower=max(low, 0), upper=high)

        X["TotalIncome"] = X["ApplicantIncome"] + X["CoapplicantIncome"]
        # LoanAmount is in thousands, term is in months
        term = X["Loan_Amount_Term"].replace(0, np.nan)
        X["EMI"] = X["LoanAmount"] / term
        X["Loan_to_Income"] = X["LoanAmount"] * 1000 / X["TotalIncome"].replace(0, np.nan)
        X["Balance_Income"] = X["TotalIncome"] - X["EMI"] * 1000
        X["Log_TotalIncome"] = np.log1p(X["TotalIncome"])
        return X[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


# ---------------------------------------------------------------------------
# Data helpers (used by both the notebook and the Streamlit app so that the
# train/test split is always identical)
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

TARGET = "Loan_Status"


def load_clean_data(path="LoanApprovalPrediction.csv"):
    """Read CSV, drop duplicates and the useless Loan_ID column.
    Returns the cleaned dataframe (target still 'Y'/'N')."""
    df = pd.read_csv(path)
    df = df.drop_duplicates().drop(columns=["Loan_ID"])
    return df


def get_xy(df):
    X = df[RAW_FEATURES].copy()
    y = (df[TARGET] == "Y").astype(int)  # 1 = Approved, 0 = Rejected
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
