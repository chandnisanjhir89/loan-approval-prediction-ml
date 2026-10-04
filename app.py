import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Page Settings
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

BASE = Path(__file__).parent

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(BASE / "loan_approval_model.joblib")

model = load_model()

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏦 Loan Approval Prediction System")
st.write(
    "Enter applicant information below to predict whether "
    "the loan is likely to be approved or rejected."
)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏦 Loan Prediction",
        "📊 Model Performance",
        "📈 Data Visualization"
    ]
)

# ==================================================
# PAGE 1 - LOAN PREDICTION
# ==================================================

if page == "🏦 Loan Prediction":

    st.header("Loan Application Details")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000,
            step=500
        )

    with col2:

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=150.0,
            step=10.0
        )

        loan_term = st.selectbox(
            "Loan Amount Term",
            [360, 180, 120, 240, 60, 480]
        )

        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    st.divider()

    if st.button("🔮 Predict Loan Approval", type="primary"):

        # Create input dataframe
        input_data = pd.DataFrame({
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area]
        })

        try:

            prediction = model.predict(input_data)[0]

            st.subheader("Prediction Result")

            # Handle Y/N output
            if str(prediction).upper() == "Y":

                st.success("✅ Loan Approved")

            elif str(prediction).upper() == "N":

                st.error("❌ Loan Not Approved")

            else:

                st.info(f"Prediction: {prediction}")

            # Probability
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]

                classes = model.classes_

                probability_df = pd.DataFrame({
                    "Class": classes,
                    "Probability": probabilities
                })

                st.write("### Prediction Probability")

                st.dataframe(
                    probability_df,
                    use_container_width=True
                )

        except Exception as e:

            st.error("Prediction Error")
            st.code(str(e))


# ==================================================
# PAGE 2 - MODEL PERFORMANCE
# ==================================================

elif page == "📊 Model Performance":

    st.header("📊 Model Performance")

    st.write(
        "Performance metrics of the trained Loan Approval Prediction model."
    )

    # Try to load results file
    results_file = BASE / "model_results.csv"

    if results_file.exists():

        results = pd.read_csv(results_file)

        st.dataframe(
            results,
            use_container_width=True
        )

        # Find numeric metric columns
        metric_columns = [
            col for col in
            ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
            if col in results.columns
        ]

        if metric_columns:

            st.subheader("Performance Comparison")

            chart_data = results.set_index(
                results.columns[0]
            )[metric_columns]

            st.bar_chart(chart_data)

    else:

        st.warning(
            "model_results.csv was not found."
        )


# ==================================================
# PAGE 3 - DATA VISUALIZATION
# ==================================================

elif page == "📈 Data Visualization":

    st.header("📈 Loan Dataset Visualization")

    data_file = BASE / "LoanApprovalPrediction.csv"

    if data_file.exists():

        df = pd.read_csv(data_file)

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.write(
            f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns"
        )

        st.subheader("Loan Approval Distribution")

        if "Loan_Status" in df.columns:

            fig, ax = plt.subplots()

            sns.countplot(
                data=df,
                x="Loan_Status",
                ax=ax
            )

            ax.set_title("Loan Approval Distribution")

            st.pyplot(fig)

        elif "LoanAmount" in df.columns:

            fig, ax = plt.subplots()

            ax.hist(
                df["LoanAmount"].dropna(),
                bins=30
            )

            ax.set_title("Loan Amount Distribution")
            ax.set_xlabel("Loan Amount")
            ax.set_ylabel("Frequency")

            st.pyplot(fig)

        st.subheader("Applicant Income Distribution")

        if "ApplicantIncome" in df.columns:

            fig, ax = plt.subplots()

            ax.hist(
                df["ApplicantIncome"].dropna(),
                bins=30
            )

            ax.set_title("Applicant Income Distribution")
            ax.set_xlabel("Applicant Income")
            ax.set_ylabel("Frequency")

            st.pyplot(fig)

    else:

        st.error(
            "LoanApprovalPrediction.csv was not found."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Loan Approval Prediction System | Machine Learning Project"
)