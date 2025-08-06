import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder

# Set Streamlit page configuration
st.set_page_config(page_title="Colorectal Cancer Prediction App", layout="wide")

# --- Helper Functions ---


@st.cache_data
def load_data():
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv("colorectal_cancer_dataset.csv")
        return df
    except FileNotFoundError:
        st.error(
            "Error: 'colorectal_cancer_dataset.csv' not found. Please ensure the file is in the same directory."
        )
        return pd.DataFrame()  # Return empty DataFrame on error


@st.cache_data
def preprocess_data(df):
    """
    Preprocesses the data by encoding categorical features and splitting into
    training and testing sets.
    """
    if df.empty:
        return None, None, None, None, None

    # Drop Patient_ID as it's not a predictive feature
    df = df.drop("Patient_ID", axis=1)

    # Convert 'Survival_Prediction' to numerical (Yes=1, No=0)
    df["Survival_Prediction"] = df["Survival_Prediction"].map({"Yes": 1, "No": 0})

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include="object").columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Apply Label Encoding to categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Define features (X) and target (y)
    X = df.drop("Survival_Prediction", axis=1)
    y = df["Survival_Prediction"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        df.columns.drop("Survival_Prediction").tolist(),
    )


@st.cache_resource
def train_model(model_name, X_train, y_train):
    """Trains the specified machine learning model."""
    if model_name == "AdaBoost":
        model = AdaBoostClassifier(random_state=42)
    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        return None  # Should not happen with current options
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns performance metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC AUC Score": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


# --- Streamlit App Layout ---

st.title("Colorectal Cancer Survival Prediction 📊")
st.markdown(
    "This application predicts the 5-year survival of colorectal cancer patients using machine learning models."
)

# --- Data Loading and Overview ---
st.header("1. Data Overview")
st.write("Loading and displaying the first few rows of the dataset.")

df = load_data()

if not df.empty:
    st.dataframe(df.head())
    st.write(f"Dataset Shape: {df.shape}")
    st.write(f"Number of records: {df.shape[0]}")
    st.write(f"Number of features: {df.shape[1]}")
    st.write("---")

    st.header("2. Exploratory Data Analysis (EDA)")

    # Age Distribution
    st.subheader("Age Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Age"], kde=True, ax=ax1, color="skyblue")
    ax1.set_title("Age Distribution of Patients")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)
    st.write("---")

    # Gender Distribution
    st.subheader("Gender Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Gender", data=df, ax=ax2, palette="viridis")
    ax2.set_title("Gender Distribution")
    ax2.set_xlabel("Gender")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)
    st.write("---")

    # Cancer Stage Distribution
    st.subheader("Cancer Stage Distribution")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.countplot(
        x="Cancer_Stage",
        data=df,
        ax=ax3,
        palette="magma",
        order=df["Cancer_Stage"].value_counts().index,
    )
    ax3.set_title("Distribution of Cancer Stages")
    ax3.set_xlabel("Cancer Stage")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)
    st.write("---")

    # Tumor Size Distribution
    st.subheader("Tumor Size Distribution")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Tumor_Size_mm"], kde=True, ax=ax4, color="lightcoral")
    ax4.set_title("Tumor Size Distribution (mm)")
    ax4.set_xlabel("Tumor Size (mm)")
    ax4.set_ylabel("Frequency")
    st.pyplot(fig4)
    st.write("---")

    # Survival Prediction Distribution
    st.subheader("Survival Prediction Distribution")
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Survival_Prediction", data=df, ax=ax5, palette="cividis")
    ax5.set_title("Distribution of 5-year Survival Prediction")
    ax5.set_xlabel("Survival Prediction")
    ax5.set_ylabel("Count")
    st.pyplot(fig5)
    st.write("---")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap of Numerical Features")
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numerical_df = df[numerical_cols]
    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    ax_corr.set_title("Correlation Matrix of Numerical Features")
    st.pyplot(fig_corr)
    st.write("---")

    # --- Model Training and Evaluation ---
    st.header("3. Model Training and Evaluation")

    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        df.copy()
    )  # Use .copy() to avoid modifying original df

    if X_train is not None:
        model_choice = st.selectbox(
            "Choose a Classification Model:", ("AdaBoost", "GradientBoosting")
        )

        if st.button("Train and Evaluate Model"):
            with st.spinner(f"Training {model_choice} model..."):
                model = train_model(model_choice, X_train, y_train)

            if model:
                metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)

                st.subheader(f"Model Performance: {model_choice}")
                metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Score"])
                st.dataframe(metrics_df)

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=ax_cm,
                    xticklabels=["Predicted No", "Predicted Yes"],
                    yticklabels=["Actual No", "Actual Yes"],
                )
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                ax_cm.set_title("Confusion Matrix")
                st.pyplot(fig_cm)

                # ROC Curve
                st.subheader("ROC Curve")
                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                ax_roc.plot(
                    fpr,
                    tpr,
                    label=f'{model_choice} (AUC = {metrics["ROC AUC Score"]:.2f})',
                )
                ax_roc.plot([0, 1], [0, 1], "k--", label="Random Classifier")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("ROC Curve")
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)

                st.write("---")

                # --- Make a Prediction ---
                st.header("4. Make a Prediction")
                st.write("Enter patient details to get a survival prediction.")

                # Create input fields for user
                input_data = {}
                for feature in feature_names:
                    if (
                        feature
                        in df.select_dtypes(include=["int64", "float64"]).columns
                    ):
                        min_val = df[feature].min()
                        max_val = df[feature].max()
                        if feature == "Age":
                            input_data[feature] = st.slider(
                                f"Enter {feature}",
                                int(min_val),
                                int(max_val),
                                int(df[feature].mean()),
                            )
                        elif feature == "Tumor_Size_mm":
                            input_data[feature] = st.slider(
                                f"Enter {feature} (mm)",
                                int(min_val),
                                int(max_val),
                                int(df[feature].mean()),
                            )
                        elif feature == "Healthcare_Costs":
                            input_data[feature] = st.number_input(
                                f"Enter {feature}",
                                min_value=int(min_val),
                                max_value=int(max_val),
                                value=int(df[feature].mean()),
                            )
                        elif (
                            feature == "Incidence_Rate_per_100K"
                            or feature == "Mortality_Rate_per_100K"
                        ):
                            input_data[feature] = st.slider(
                                f"Enter {feature}",
                                int(min_val),
                                int(max_val),
                                int(df[feature].mean()),
                            )
                        else:
                            input_data[feature] = st.number_input(
                                f"Enter {feature}",
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=float(df[feature].mean()),
                            )
                    else:  # Categorical features
                        unique_values = df[feature].unique().tolist()
                        # Convert numerical labels back to original for display in selectbox
                        # This requires mapping back to original string labels, which is not directly available here.
                        # For simplicity, we'll use the encoded numerical values.
                        input_data[feature] = st.selectbox(
                            f"Select {feature}", options=unique_values
                        )

                if st.button("Predict Survival"):
                    # Create a DataFrame from input data
                    input_df = pd.DataFrame([input_data])
                    # Ensure columns are in the same order as training data
                    input_df = input_df[X_train.columns]

                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    prediction_proba = model.predict_proba(input_df)[0][1]

                    st.subheader("Prediction Result:")
                    if prediction == 1:
                        st.success(f"The model predicts a **YES** for 5-year survival.")
                    else:
                        st.error(f"The model predicts a **NO** for 5-year survival.")
                    st.write(
                        f"Confidence (Probability of Survival): {prediction_proba:.2f}"
                    )
