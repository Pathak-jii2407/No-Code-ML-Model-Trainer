import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# Streamlit Page Config
st.set_page_config(page_title="AutoML Trainer", page_icon="🤖", layout="wide")
st.title("🤖 No-Code ML Model Trainer")

# Folder path for pre-uploaded datasets
UPLOADS_FOLDER = "uploads"

# Select dataset from "uploads" folder
st.sidebar.subheader("📂 Choose Preloaded Dataset")
preloaded_files = [f for f in os.listdir(UPLOADS_FOLDER) if f.endswith(".csv")]
selected_file = st.sidebar.selectbox("Select a dataset", ["Upload your own"] + preloaded_files)

# Load dataset
df = None
if selected_file != "Upload your own":
    df = pd.read_csv(os.path.join(UPLOADS_FOLDER, selected_file))
    st.success(f"✅ Loaded `{selected_file}` from preloaded datasets!")

# Upload CSV File
uploaded_file = st.file_uploader("📂 Or Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

# Show dataset if available
if df is not None:
    st.write("### 🔍 Dataset Preview:")
    st.dataframe(df.head())

    # Generate Report
    if st.button("🚀 Generate Report"):
        with st.spinner("Generating Report... Please Wait ⏳"):
            profile = ProfileReport(df, explorative=True)
            st_profile_report(profile)

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

        st.write(f"📌 **Categorical Columns (One-Hot Encoded):** {cat_cols}")

        # Feature Scaling Selection
        scaling_option = st.radio("⚖ Select Feature Scaling Method", ["StandardScaler", "MinMaxScaler"])
        scaler = StandardScaler() if scaling_option == "StandardScaler" else MinMaxScaler()

        # Preprocessing Pipeline
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', scaler)
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ])

        # Hyperparameter Tuning
        if hasattr(model, "n_estimators"):
            n_estimators = st.slider("🌳 Number of Trees (n_estimators)", 10, 500, 100, 10)
            model.set_params(n_estimators=n_estimators)
        if hasattr(model, "max_depth"):
            max_depth = st.slider("📍 Max Depth of Trees", 1, 50, 10, 1)
            model.set_params(max_depth=max_depth)

        # Training Pipeline
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', model)
        ])

        # **Auto Train or Manual Train**
        train_option = st.radio("🛠 Choose Training Mode", ["Manual Train", "Auto Train"])

        # Train Model
        if train_option == "Auto Train" or st.button("🚀 Train Model"):
            df.fillna(method='ffill', inplace=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            try:
                scores = cross_val_score(pipeline, X_train, y_train, cv=5, error_score='raise')
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
                    st.write(f"📉 **RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                    st.write(f"📊 **R² Score:** {r2_score(y_test, y_pred):.2f}")

            except Exception as e:
                st.error(f"❌ Training Failed: {str(e)}")

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
                try:
                    prediction = model.predict(df_input)[0]
                    st.sidebar.write(f"**Prediction: {prediction}**")
                except Exception as e:
                    st.sidebar.error(f"❌ Prediction Failed: {str(e)}")
