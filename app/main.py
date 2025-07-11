import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import xgboost as xgb
from scipy.stats import zscore

st.set_page_config(layout="wide")
st.title("🧠 Explain My Model – Interactive ML Dashboard")
st.markdown("Upload a CSV, select a target, train a model, and get SHAP-based explanations.")

# Preprocessing function
def clean_and_preprocess(df, target_column, z_thresh=3):
    df = df.dropna(thresh=0.8 * len(df), axis=1)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.loc[:, df.nunique() > 1]
    df = df[df[target_column].notna()]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.select_dtypes(include=np.number).shape[1] > 0:
        mask = (np.abs(zscore(X.select_dtypes(include=np.number), nan_policy='omit')) < z_thresh).all(axis=1)
        X = X[mask]
        y = y[mask]

    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

import base64

def get_csv_download_link(file_path, label="📥  Download Sample CSV"):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="heart.csv">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# Show link
try:
    get_csv_download_link("data/heart (1).csv")
except FileNotFoundError:
    st.info("Sample dataset will appear here when available.")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    target = st.selectbox("🌟 Select your target column", df.columns)

    with st.expander("❓ What is a target column?"):
        st.markdown("""
        A **target column** is what you want the model to predict.
        - If it's a category or label (e.g., `0` for No Disease, `1` for Disease), it's a **classification** task.
        - If it's a continuous number (e.g., `age`, `price`, `cholesterol level`), it's a **regression** task.

        Always ask yourself: *What am I trying to predict?*
        """)

    if target:
        unique_labels = np.unique(df[target].dropna())
        is_regression = df[target].dtype in [np.float32, np.float64] or len(unique_labels) > 20

        X, y = clean_and_preprocess(df, target)

        if not is_regression:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            st.info(f"Target classes mapped as: {mapping}")
        else:
            y = pd.to_numeric(y, errors="coerce")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_type = st.radio("🤖 Choose model type", ["Logistic Regression", "XGBoost"])

        if st.button("🚀 Train Model"):
            if model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("📊 Model Performance")
            if is_regression:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"**MSE:** {mse:.2f}")
                st.write(f"**R² Score:** {r2:.2f}")
            else:
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().style.highlight_max(axis=0))

            st.subheader("🔍 SHAP Feature Importance")
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)

            st.markdown("#### 🌍 SHAP Summary Plot")
            fig_summary = plt.figure()
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig_summary)
            plt.clf()

            with st.expander("📘 What is a SHAP Summary Plot?"):
                st.markdown(f"""
                **Dataset**: `{uploaded_file.name}`  
                **Target column**: `{target}`  
                **Problem type**: {"Regression" if is_regression else "Classification"}

                - **Y-axis**: Features ranked by importance
                - **X-axis**: SHAP values showing impact on model output
                - **Dots**: Each dot = one row in dataset
                - **Color**: 🔴 High value, 🔵 Low value
                """)

            st.markdown("#### 🕵️ SHAP Waterfall Plot")
            try:
                instance_idx = 0

                if len(shap_values.shape) == 3:
                    class_index = 0
                    values = shap_values.values[instance_idx, class_index]
                    base = shap_values.base_values[instance_idx, class_index]
                else:
                    values = shap_values.values[instance_idx]
                    base = shap_values.base_values[instance_idx]

                data_row = shap_values.data[instance_idx]
                if hasattr(data_row, 'flatten'):
                    data_row = data_row.flatten()
                else:
                    data_row = np.array(data_row)
                data_row = data_row.astype(float)

                values = np.array(values).flatten()
                base = float(base)
                min_len = min(len(values), len(data_row), len(shap_values.feature_names))

                explanation = shap.Explanation(
                    values=values[:min_len],
                    base_values=base,
                    data=data_row[:min_len],
                    feature_names=shap_values.feature_names[:min_len]
                )

                shap.plots.waterfall(explanation, show=False)
                st.pyplot(plt.gcf())
                plt.clf()

            except Exception as e:
                st.error(f"SHAP Waterfall plot failed: {e}")

            with st.expander("📘 What is a SHAP Waterfall Plot?"):
                st.markdown("""
                - Starts at the **baseline** prediction (average prediction)
                - Each bar represents one feature’s contribution
                - Red bars push prediction up, Blue bars push it down
                - Final value = predicted output for that row

                Great for understanding **why** a single prediction was made.
                """)
