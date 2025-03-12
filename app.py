import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import pymysql
import requests
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# Streamlit Page Config
st.set_page_config(page_title="AutoML Trainer", page_icon="🤖", layout="wide")
st.title("🤖 No-Code ML Model Trainer")

# Upload CSV File
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Dataset Preview:")
    st.dataframe(df.head())

    # Select Target Column
    target_column = st.selectbox("🎯 Select Target Variable", df.columns)

    if target_column:
        unique_values = df[target_column].nunique()
        task_type = "Classification" if unique_values < 10 else "Regression"
        st.write(f"**Task Type: {task_type}**")

        # Model Selection
        model_options = {
            "Classification": {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "SVM": SVC()
            },
            "Regression": {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor()
            }
        }
        selected_model_name = st.selectbox("🛠 Select Model", list(model_options[task_type].keys()))
        model = model_options[task_type][selected_model_name]

        # Feature Selection
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column]

        # Identify Numeric & Categorical Columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X.select_dtypes(include=['object']).columns

        # **Detect Ordinal Columns Automatically**
        ordinal_cols = []
        categorical_cols = []

        for col in cat_cols:
            unique_vals = list(df[col].dropna().unique())
            try:
                sorted_vals = sorted(unique_vals, key=lambda x: str(x))
                ordinal_cols.append(col)
            except:
                categorical_cols.append(col)

        st.write(f"📌 **Ordinal Columns Detected:** {ordinal_cols}")
        st.write(f"📌 **Categorical Columns (One-Hot Encoded):** {categorical_cols}")

        # Feature Scaling Selection
        scaling_option = st.radio("⚖ Select Feature Scaling Method", ["StandardScaler", "MinMaxScaler"])
        scaler = StandardScaler() if scaling_option == "StandardScaler" else MinMaxScaler()

        # Preprocessing Pipeline
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', scaler)
            ]), num_cols),
            ('ordinal', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
            ]), ordinal_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols)
        ])

        # Hyperparameter Tuning
        if "n_estimators" in model.get_params():
            n_estimators = st.slider("🌲 Number of Trees (n_estimators)", 10, 500, 100, 10)
            model.set_params(n_estimators=n_estimators)
        if "max_depth" in model.get_params():
            max_depth = st.slider("📏 Max Depth of Trees", 1, 50, 10, 1)
            model.set_params(max_depth=max_depth)

        # Training Pipeline
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', model)
        ])

        # Train Model Button
        if st.button("🚀 Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scores = cross_val_score(pipeline, X_train, y_train, cv=5)
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, "model.pkl")

            st.success(f"✅ Model Trained Successfully!")
            st.write(f"📊 **Model Used:** {selected_model_name}")
            st.write(f"🎯 **Cross-Validation Score:** {np.mean(scores):.2f}")

            # Model Evaluation
            y_pred = pipeline.predict(X_test)
            if task_type == "Classification":
                st.write("📊 **Classification Report:**")
                st.text(classification_report(y_test, y_pred))
            else:
                st.write(f"📉 **RMSE:** {mean_squared_error(y_test, y_pred, squared=False):.2f}")
                st.write(f"📊 **R² Score:** {r2_score(y_test, y_pred):.2f}")

        # Model Deployment
        if os.path.exists("model.pkl"):
            st.sidebar.header("🔮 Live Prediction")
            st.sidebar.write("Fill in values to make a prediction:")
            model = joblib.load("model.pkl")
            input_data = {}
            for col in feature_columns:
                input_data[col] = st.sidebar.text_input(f"Enter value for {col}")

            if st.sidebar.button("Predict"):
                df_input = pd.DataFrame([input_data])
                prediction = model.predict(df_input)[0]
                st.sidebar.write(f"**Prediction: {prediction}**")
