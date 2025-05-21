import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import logging
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
from sklearn.metrics import classification_report, mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor

# Configure logging
logging.basicConfig(filename='automl_errors.log', level=logging.ERROR)

# Streamlit Page Config
st.set_page_config(page_title="AutoML Trainer", page_icon="🤖", layout="wide")
st.title("🤖 No-Code ML Model Trainer")

# Folder path for pre-uploaded datasets
UPLOADS_FOLDER = "uploads"

# Ensure uploads folder exists
if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER)

# Sidebar: Dataset Selection
st.sidebar.subheader("📂 Choose Preloaded Dataset")
preloaded_files = [f for f in os.listdir(UPLOADS_FOLDER) if f.endswith((".csv", ".xlsx", ".json"))]
selected_file = st.sidebar.selectbox("Select a dataset", ["Upload your own"] + preloaded_files)

# Load dataset
df = None
if selected_file != "Upload your own":
    file_path = os.path.join(UPLOADS_FOLDER, selected_file)
    try:
        if selected_file.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif selected_file.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif selected_file.endswith(".json"):
            df = pd.read_json(file_path)
        st.success(f"✅ Loaded `{selected_file}` from preloaded datasets!")
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {str(e)}")
        logging.error(f"Dataset loading failed: {str(e)}")

# Upload File
uploaded_file = st.file_uploader("📂 Or Upload a File", type=["csv", "xlsx", "json"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to upload file: {str(e)}")
        logging.error(f"File upload failed: {str(e)}")

# Dataset Validation and Preview
if df is not None:
    if df.empty or len(df.columns) == 0:
        st.error("❌ Dataset is empty or invalid!")
        st.stop()
    st.write("### 🔍 Dataset Preview:")
    st.dataframe(df.head())

    # Dataset Cleaning
    st.write("### 🧹 Dataset Cleaning")
    columns_to_drop = st.multiselect("Select Columns to Drop", df.columns)
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        st.success("Selected columns dropped!")

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
                "SVM": SVC(probability=True),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            },
            "Regression": {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor()
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

        # Imputation and Scaling Options
        imputation_strategy = st.selectbox("Select Imputation Strategy for Numeric Columns", 
                                          ["Mean", "Median", "Most Frequent", "Constant"])
        num_imputer = SimpleImputer(strategy=imputation_strategy.lower() if imputation_strategy != "Constant" else "constant", 
                                   fill_value=0 if imputation_strategy == "Constant" else None)
        scaling_option = st.radio("⚖ Select Feature Scaling Method", ["StandardScaler", "MinMaxScaler"])
        scaler = StandardScaler() if scaling_option == "StandardScaler" else MinMaxScaler()

        # Preprocessing Pipeline
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', num_imputer),
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
        if selected_model_name == "SVM":
            C = st.slider("SVM C Parameter", 0.1, 10.0, 1.0, 0.1)
            model.set_params(C=C)

        # Training Pipeline
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', model)
        ])

        # Cross-Validation Customization
        cv_folds = st.slider("Number of Cross-Validation Folds", 2, 10, 5)

        # Training Mode
        train_option = st.radio("🛠 Choose Training Mode", ["Manual Train", "Auto Train"])

        # Train Model
        if train_option == "Auto Train" or st.button("🚀 Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            try:
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, error_score='raise')
                pipeline.fit(X_train, y_train)
                
                # Save Model with Custom Name
                model_name = st.text_input("Enter Model Name for Saving", "model")
                joblib.dump(pipeline, f"{model_name}.pkl")
                st.download_button("Download Model", data=open(f"{model_name}.pkl", "rb"), file_name=f"{model_name}.pkl")

                st.success(f"✅ Model Trained Successfully!")
                st.write(f"📊 **Model Used:** {selected_model_name}")
                st.write(f"🎯 **Cross-Validation Score (Mean):** {np.mean(scores):.2f} (±{np.std(scores):.2f})")

                # Model Evaluation
                y_pred = pipeline.predict(X_test)
                if task_type == "Classification":
                    st.write("### 📊 Classification Report:")
                    st.text(classification_report(y_test, y_pred))
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    st.write("### Confusion Matrix")
                    st.write(cm)
                    # Feature Importance for Tree-Based Models
                    if selected_model_name in ["Random Forest", "XGBoost"]:
                        feature_importance = pipeline.named_steps['model'].feature_importances_
                        feature_names = num_cols.tolist() + list(pipeline.named_steps['preprocess'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols))
                        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
                        st.write("### Feature Importance")
                        st.dataframe(importance_df.sort_values("Importance", ascending=False))
                else:
                    st.write(f"📉 **RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                    st.write(f"📊 **R² Score:** {r2_score(y_test, y_pred):.2f}")
                    st.write(f"📈 **MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
                    # Predicted vs Actual Plot
                    st.write("### Predicted vs Actual Plot")
                    chart_data = {
                        "type": "scatter",
                        "data": {
                            "datasets": [
                                {
                                    "label": "Predicted vs Actual",
                                    "data": [{"x": actual, "y": pred} for actual, pred in zip(y_test, y_pred)],
                                    "backgroundColor": "rgba(75, 192, 192, 0.6)",
                                    "borderColor": "rgba(75, 192, 192, 1)",
                                    "borderWidth": 1
                                }
                            ]
                        },
                        "options": {
                            "scales": {
                                "x": {"title": {"display": True, "text": "Actual Values"}},
                                "y": {"title": {"display": True, "text": "Predicted Values"}}
                            },
                            "plugins": {"title": {"display": True, "text": "Predicted vs Actual"}}
                        }
                    }
                    ```chartjs
                    chart_data
