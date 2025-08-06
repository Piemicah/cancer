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
    try:
        df = pd.read_csv("colorectal_cancer_dataset.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'colorectal_cancer_dataset.csv' not found.")
        return pd.DataFrame()


@st.cache_data
def preprocess_data(df):
    if df.empty:
        return None, None, None, None, None

    df = df.drop(["Patient_ID", "Country"], axis=1)
    df["Survival_Prediction"] = df["Survival_Prediction"].map({"Yes": 1, "No": 0})

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop("Survival_Prediction", axis=1)
    y = df["Survival_Prediction"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, df.columns.drop("Survival_Prediction").tolist()


@st.cache_resource
def train_model(model_name, X_train, y_train):
    if model_name == "AdaBoost":
        model = AdaBoostClassifier(random_state=42)
    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        return None
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
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

# --- Streamlit App ---
st.title("Colorectal Cancer Survival Prediction 📊")
st.markdown("This app predicts 5-year survival of colorectal cancer patients using ML.")

st.header("1. Data Overview")
df = load_data()

if not df.empty:
    st.dataframe(df.head())
    st.write(f"Dataset Shape: {df.shape}")
    st.write("---")

    st.header("2. Exploratory Data Analysis (EDA)")

    # EDA Charts
    with st.expander("📈 Show EDA Visualizations"):
        fig1, ax1 = plt.subplots()
        sns.histplot(df["Age"], kde=True, ax=ax1, color="skyblue")
        ax1.set_title("Age Distribution")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        sns.countplot(x="Gender", data=df, ax=ax2)
        ax2.set_title("Gender Distribution")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        sns.countplot(x="Cancer_Stage", data=df, order=df["Cancer_Stage"].value_counts().index, ax=ax3)
        ax3.set_title("Cancer Stage Distribution")
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots()
        sns.histplot(df["Tumor_Size_mm"], kde=True, ax=ax4, color="coral")
        ax4.set_title("Tumor Size Distribution")
        st.pyplot(fig4)

        fig5, ax5 = plt.subplots()
        sns.countplot(x="Survival_Prediction", data=df, ax=ax5)
        ax5.set_title("5-Year Survival Distribution")
        st.pyplot(fig5)

        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.select_dtypes(include=["int64", "float64"]).corr(), annot=True, cmap="coolwarm", ax=ax_corr)
        ax_corr.set_title("Correlation Matrix")
        st.pyplot(fig_corr)

    st.write("---")

    # --- Model Training ---
    st.header("3. Model Training and Evaluation")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df.copy())

    if X_train is not None:
        model_choice = st.selectbox("Choose Model:", ("AdaBoost", "GradientBoosting"))

        if st.button("Train and Evaluate Model"):
            with st.spinner("Training model..."):
                model = train_model(model_choice, X_train, y_train)
                st.session_state.model = model  # Save model

            metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
            st.subheader(f"{model_choice} Performance")
            st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Score"]))

            # Confusion matrix
            fig_cm, ax_cm = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_title("Confusion Matrix")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

            # ROC Curve
            fig_roc, ax_roc = plt.subplots()
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax_roc.plot(fpr, tpr, label=f"AUC = {metrics['ROC AUC Score']:.2f}")
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)

        # --- Prediction Section ---
        st.header("4. Make a Prediction")
        st.write("Enter patient details to get survival prediction:")

        input_data = {}
        for feature in feature_names:
            if feature in df.select_dtypes(include=["int64", "float64"]).columns:
                min_val = df[feature].min()
                max_val = df[feature].max()
                mean_val = df[feature].mean()

                if feature == "Age":
                    input_data[feature] = st.slider("Enter Age", int(min_val), int(max_val), int(mean_val), help="Patient's age in years")
                elif feature == "Tumor_Size_mm":
                    input_data[feature] = st.slider("Tumor Size (mm)", int(min_val), int(max_val), int(mean_val), help="Tumor size in millimeters")
                elif feature == "Healthcare_Costs":
                    input_data[feature] = st.number_input("Healthcare Costs", int(min_val), int(max_val), int(mean_val), help="Estimated healthcare cost")
                elif feature in ["Incidence_Rate_per_100K", "Mortality_Rate_per_100K"]:
                    input_data[feature] = st.slider(feature, int(min_val), int(max_val), int(mean_val))
                else:
                    input_data[feature] = st.number_input(f"Enter {feature}", float(min_val), float(max_val), float(mean_val))
            else:
                # Categorical inputs
                labels = df[feature].unique().tolist()
                selected_label = st.selectbox(f"Select {feature}", options=labels)
                le = LabelEncoder()
                le.fit(labels)
                input_data[feature] = le.transform([selected_label])[0]

        if st.button("Predict Survival"):
            if "model" not in st.session_state:
                st.warning("Please train the model first before predicting.")
            else:
                input_df = pd.DataFrame([input_data])
                input_df = input_df[X_train.columns]

                model = st.session_state.model
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0][1]

                st.subheader("Prediction Result:")
                if prediction == 1:
                    st.success("✅ The model predicts a **YES** for 5-year survival.")
                else:
                    st.error("❌ The model predicts a **NO** for 5-year survival.")

                st.write(f"🔍 Confidence Score: **{prediction_proba:.2f}**")
