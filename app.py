```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import logging
import json
import base64
import zipfile
import io
try:
    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    profiling_available = True
except ImportError:
    profiling_available = False
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, roc_curve, auc
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from scikeras.wrappers import KerasClassifier, KerasRegressor
    tensorflow_available = True
except ImportError:
    Sequential = Dense = KerasClassifier = KerasRegressor = None
    tensorflow_available = False

# Configure logging
logging.basicConfig(filename='automl_errors.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit Page Config
st.set_page_config(page_title="AutoML Pro", page_icon="🤖", layout="wide")

# Dependency warnings
if not tensorflow_available:
    st.warning("TensorFlow not installed. Neural Network models unavailable. Install with 'pip install tensorflow'.")
if not profiling_available:
    st.warning("ydata-profiling not installed. Dataset reports unavailable. Install with 'pip install ydata-profiling streamlit-pandas-profiling'.")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stTextInput>input, .stNumberInput>input {border: 1px solid #ddd; border-radius: 5px;}
    .stSidebar .stButton>button {background-color: #2196F3;}
    .st-expander {background-color: white; border: 1px solid #ddd; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

# Session state
if 'model_name' not in st.session_state:
    st.session_state.model_name = "model"
if 'dataset_key' not in st.session_state:
    st.session_state.dataset_key = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Clear old model files on startup
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
for file in os.listdir(MODEL_DIR):
    if file.endswith(".pkl"):
        os.remove(os.path.join(MODEL_DIR, file))
        logging.debug(f"Cleared old model file: {file}")

# Title
st.title("🤖 AutoML Pro: No-Code Machine Learning")

# Debug mode
debug_mode = st.checkbox("Enable Debug Mode (logs detailed info)", value=False)

# Dataset folder
UPLOADS_FOLDER = "Uploads"
if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER)

# Sidebar: Dataset Selection
with st.sidebar:
    st.subheader("📂 Dataset Selection")
    preloaded_files = [f for f in os.listdir(UPLOADS_FOLDER) if f.endswith((".csv", ".xlsx", ".json"))]
    selected_file = st.selectbox("Select a dataset", ["Upload your own"] + preloaded_files, key="dataset_select")

# Load dataset
df = None
dataset_key = selected_file if selected_file != "Upload your own" else (uploaded_file.name if uploaded_file else None)
if dataset_key != st.session_state.dataset_key:
    st.session_state.dataset_key = dataset_key
    st.session_state.model_trained = False
    st.session_state.model_name = f"model_{dataset_key.split('.')[0]}" if dataset_key else "model"

if selected_file != "Upload your own":
    file_path = os.path.join(UPLOADS_FOLDER, selected_file)
    try:
        if selected_file.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif selected_file.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif selected_file.endswith(".json"):
            df = pd.read_json(file_path)
        st.success(f"✅ Loaded `{selected_file}`!")
        if debug_mode:
            logging.debug(f"Loaded dataset: {selected_file}, shape: {df.shape}")
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {str(e)}")
        logging.error(f"Dataset loading failed: {str(e)}")

# Upload File
uploaded_file = st.file_uploader("📂 Or Upload a File", type=["csv", "xlsx", "json"], key="file_uploader")
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        st.success("✅ File uploaded successfully!")
        if debug_mode:
            logging.debug(f"Uploaded dataset: {uploaded_file.name}, shape: {df.shape}")
    except Exception as e:
        st.error(f"❌ Failed to upload file: {str(e)}")
        logging.error(f"File upload failed: {str(e)}")

# Dataset Processing
if df is not None:
    if df.empty or len(df.columns) == 0:
        st.error("❌ Dataset is empty or invalid!")
        logging.error("Empty or invalid dataset")
        st.stop()

    with st.expander("🔍 Dataset Options", expanded=True):
        # Preview
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Sampling
        sample_data = st.checkbox("Sample dataset (for large datasets)", value=False)
        if sample_data:
            sample_size = st.slider("Sample Size", 100, len(df), min(1000, len(df)))
            df = df.sample(n=sample_size, random_state=42)
            st.success(f"Dataset sampled to {sample_size} rows.")
            if debug_mode:
                logging.debug(f"Sampled dataset to {sample_size} rows")

        # Cleaning
        st.write("### 🧹 Dataset Cleaning")
        columns_to_drop = st.multiselect("Select Columns to Drop", df.columns, key="drop_columns")
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            st.success("Selected columns dropped!")
            if debug_mode:
                logging.debug(f"Dropped columns: {columns_to_drop}")

        # Missing Values
        st.write("### 🛠 Handle Missing Values")
        missing_cols = df.columns[df.isna().any()].tolist()
        if missing_cols:
            st.warning(f"⚠ Missing values in columns: {missing_cols}")
            missing_action = st.radio("Handle missing values:", 
                                     ["Impute automatically", "Drop rows", "Drop columns"],
                                     key="missing_action")
            if missing_action == "Impute automatically":
                st.info("Imputing: mean for numeric, most frequent for categorical.")
            elif missing_action == "Drop rows":
                df = df.dropna()
                st.success("Rows with missing values dropped!")
                if debug_mode:
                    logging.debug(f"Dropped rows, new shape: {df.shape}")
            elif missing_action == "Drop columns":
                df = df.drop(columns=missing_cols)
                st.success(f"Dropped columns: {missing_cols}")
                if debug_mode:
                    logging.debug(f"Dropped columns: {missing_cols}")

    # Dataset Report
    if Draghi_profiling_available:
        if st.checkbox("Generate Dataset Report", value=False):
            if st.button("🚀 Generate Report"):
                with st.spinner("Generating Report..."):
                    try:
                        profile = ProfileReport(df, explorative=True)
                        st_profile_report(profile)
                        if debug_mode:
                            logging.debug("Generated dataset report")
                    except Exception as e:
                        st.error(f"❌ Failed to generate report: {str(e)}")
                        logging.error(f"Report generation failed: {str(e)}")
    else:
        st.info("Dataset reports disabled due to missing dependencies.")

    # Model Configuration
    with st.expander("🛠 Model Configuration", expanded=True):
        target_column = st.selectbox("🎯 Select Target Variable", df.columns, key="target_column")
        
        if target_column:
            unique_values = df[target_column].nunique()
            task_type = "Classification" if unique_values < 10 else "Regression"
            st.write(f"**Task Type: {task_type}**")
            if debug_mode:
                logging.debug(f"Task type: {task_type}, target: {target_column}, unique values: {unique_values}")

            # Model Selection
            model_options = {
                "Classification": {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "SVM": SVC(probability=True, random_state=42),
                    "KNN": KNeighborsClassifier()
                },
                "Regression": {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "KNN": KNeighborsRegressor()
                }
            }
            if tensorflow_available:
                model_options["Classification"]["Neural Network"] = None
                model_options["Regression"]["Neural Network"] = None

            selected_model_name = st.selectbox("🛠 Select Model", list(model_options[task_type].keys()), key="model_select")
            
            # Features
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns]
            y = df[target_column]

            # Numeric & Categorical Columns
            num_cols = X.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns
            st.write(f"📌 **Categorical Columns (One-Hot Encoded):** {cat_cols.tolist()}")
            if debug_mode:
                logging.debug(f"Features: numeric={num_cols.tolist()}, categorical={cat_cols.tolist()}")

            # Data Validation
            if not isinstance(y, pd.Series):
                st.error("❌ Target variable must be a single column. Select a valid target column.")
                logging.error("Target variable is not a pandas Series")
                st.stop()
            if task_type == "Classification" and not np.issubdtype(y.dtype, np.number):
                st.error("❌ Classification requires numeric target labels (e.g., 0, 1). Encode the target column.")
                logging.error("Non-numeric target labels for classification")
                st.stop()
            if task_type == "Regression":
                if not np.issubdtype(y.dtype, np.number):
                    st.error("❌ Regression target 'Price' must be numeric. Check for non-numeric values.")
                    logging.error("Non-numeric values in regression target")
                    st.stop()
                if y.isna().any():
                    st.error("❌ Missing values in target 'Price'. Impute or drop rows with missing target values.")
                    logging.error("Missing values in regression target")
                    st.stop()
                if not np.isfinite(y).all():
                    st.error("❌ Invalid values (e.g., infinite) in target 'Price'. Clean the target column.")
                    logging.error("Infinite values in regression target")
                    st.stop()
            if X.isna().any().any() and missing_action != "Impute automatically":
                st.error("❌ Missing values in features. Select 'Impute automatically' or drop rows/columns.")
                logging.error("Missing values in features")
                st.stop()
            for col in num_cols:
                if not np.isfinite(X[col]).all():
                    st.error(f"❌ Invalid values (e.g., infinite) in numeric column '{col}'. Clean the data.")
                    logging.error(f"Infinite values in numeric column: {col}")
                    st.stop()

            # Preprocessing Pipeline
            preprocessor = ColumnTransformer([
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy="mean")),
                    ('scaler', StandardScaler())
                ]), num_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), cat_cols)
            ])

            # Neural Network
            if selected_model_name == "Neural Network" and tensorflow_available:
                try:
                    X_sample = X.iloc[:min(100, len(X))]
                    preprocessor.fit(X_sample)
                    num_features = preprocessor.transform(X_sample).shape[1]
                except Exception as e:
                    st.error(f"❌ Preprocessing failed: {str(e)}. Check categorical columns for invalid data.")
                    logging.error(f"Preprocessing failed: {str(e)}", exc_info=True)
                    st.stop()
                
                hidden_layers = st.slider("Hidden Layers", 1, 3, 1, key="nn_layers")
                nodes_per_layer = st.slider("Nodes per Layer", 8, 64, 32, 8, key="nn_nodes")
                
                def build_nn_model():
                    model = Sequential()
                    model.add(Dense(nodes_per_layer, input_dim=num_features, activation="relu"))
                    for _ in range(hidden_layers - 1):
                        model.add(Dense(nodes_per_layer, activation="relu"))
                    model.add(Dense(1 if task_type == "Regression" else len(np.unique(y)), 
                                   activation="linear" if task_type == "Regression" else "softmax"))
                    model.compile(loss="mse" if task_type == "Regression" else "sparse_categorical_crossentropy",
                                 optimizer="adam", metrics=["mse" if task_type == "Regression" else "accuracy"])
                    return model
                
                model = KerasRegressor(model=build_nn_model, epochs=50, batch_size=32, verbose=0) if task_type == "Regression" else \
                        KerasClassifier(model=build_nn_model, epochs=50, batch_size=32, verbose=0)
                if debug_mode:
                    logging.debug(f"Neural Network: input_dim={num_features}, layers={hidden_layers}, nodes={nodes_per_layer}")
            else:
                model = model_options[task_type][selected_model_name]

            # Pipeline
            pipeline = Pipeline([
                ('preprocess', preprocessor),
                ('model', model)
            ])

            # Cross-Validation
            cv_folds = st.slider("Cross-Validation Folds", 3, 5, 3, key="cv_f Ascolta ora (Playback Speed) 🔊

            # Train Model
            if st.button("🚀 Train Model"):
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    with st.spinner("Training Model..."):
                        scoring = "accuracy" if task_type == "Classification" else "neg_mean_squared_error"
                        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring=scoring, error_score='raise')
                        pipeline.fit(X_train, y_train)

                    # Save Model
                    st.session_state.model_name = st.text_input("Model Name", st.session_state.model_name, key="model_name")
                    model_path = os.path.join(MODEL_DIR, f"{st.session_state.model_name}.pkl")
                    joblib.dump(pipeline, model_path)
                    st.session_state.model_trained = True
                    
                    # ZIP Outputs
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        with open(model_path, "rb") as f:
                            zip_file.writestr(f"{st.session_state.model_name}.pkl", f.read())
                        y_pred = pipeline.predict(X_test)
                        metrics = {}
                        if task_type == "Classification":
                            metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
                            cm = confusion_matrix(y_test, y_pred)
                            metrics["confusion_matrix"] = cm.tolist()
                        else:
                            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                            metrics["r2_score"] = float(r2_score(y_test, y_pred))
                            metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
                        with io.StringIO() as f:
                            json.dump(metrics, f)
                            zip_file.writestr("metrics.json", f.getvalue())
                    
                    zip_buffer.seek(0)
                    b64 = base64.b64encode(zip_buffer.getvalue()).decode()
                    href = f'<a href="data:application/zip;base64,{b64}" download="{st.session_state.model_name}_outputs.zip">Download Model + Metrics</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    st.success(f"✅ Model Trained Successfully!")
                    st.write(f"📊 **Model:** {selected_model_name}")
                    st.write(f"🎯 **CV Score (Mean):** {np.mean(scores):.2f} (±{np.std(scores):.2f})")
                    if debug_mode:
                        logging.debug(f"Model trained: {selected_model_name}, CV score: {np.mean(scores):.2f}")

                    # Evaluation
                    if task_type == "Classification":
                        st.write("### 📊 Classification Report:")
                        st.text(classification_report(y_test, y_pred))
                        st.write("### Confusion Matrix")
                        st.write(cm)
                        if hasattr(pipeline.named_steps['model'], 'predict_proba'):
                            y_prob = pipeline.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
                            if y_prob is not None:
                                fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=np.unique(y_test)[1])
                                roc_auc = auc(fpr, tpr)
                                st.write("### ROC Curve")
                                chart_data = {
                                    "type": "line",
                                    "data": {
                                        "datasets": [{
                                            "label": f"ROC Curve (AUC = {roc_auc:.2f})",
                                            "data": [{"x": float(f), "y": float(t)} for f, t in zip(fpr, tpr)],
                                            "fill": False,
                                            "borderColor": "rgba(255, 99, 132, 1)",
                                            "borderWidth": 2
                                        }]
                                    },
                                    "options": {
                                        "scales": {
                                            "x": {"title": {"display": True, "text": "False Positive Rate"}},
                                            "y": {"title": {"display": True, "text": "True Positive Rate"}}
                                        },
                                        "plugins": {"title": {"display": True, "text": "ROC Curve"}}
                                    }
                                }
                                st.components.v1.html(f"""
                                    <div style='width: 100%; height: 400px;'>
                                        <canvas id='roc_chart'></canvas>
                                        <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
                                        <script>
                                            const ctx = document.getElementById('roc_chart').getContext('2d');
                                            new Chart(ctx, {json.dumps(chart_data)});
                                        </script>
                                    </div>
                                """, height=400)
                        if selected_model_name == "Random Forest":
                            feature_importance = pipeline.named_steps['model'].feature_importances_
                            feature_names = num_cols.tolist() + list(pipeline.named_steps['preprocess'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols))
                            importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
                            st.write("### Feature Importance")
                            st.dataframe(importance_df.sort_values("Importance", ascending=False))
                    else:
                        st.write(f"📉 **RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                        st.write(f"📊 **R² Score:** {r2_score(y_test, y_pred):.2f}")
                        st.write(f"📈 **MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
                        st.write("### Predicted vs Actual")
                        chart_data = {
                            "type": "scatter",
                            "data": {
                                "datasets": [{
                                    "label": "Predicted vs Actual",
                                    "data": [{"x": float(actual), "y": float(pred)} for actual, pred in zip(y_test, y_pred)],
                                    "backgroundColor": "rgba(75, 192, 192, 0.6)",
                                    "borderColor": "rgba(75, 192, 192, 1)",
                                    "borderWidth": 1
                                }]
                            },
                            "options": {
                                "scales": {
                                    "x": {"title": {"display": True, "text": "Actual Values"}},
                                    "y": {"title": {"display": True, "text": "Predicted Values"}}
                                },
                                "plugins": {"title": {"display": True, "text": "Predicted vs Actual"}}
                            }
                        }
                        st.components.v1.html(f"""
                            <div style='width: 100%; height: 400px;'>
                                <canvas id='chart'></canvas>
                                <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
                                <script>
                                    const ctx = document.getElementById('chart').getContext('2d');
                                    new Chart(ctx, {json.dumps(chart_data)});
                                </script>
                            </div>
                        """, height=400)

                except Exception as e:
                    st.error(f"❌ Training Failed: {str(e)}. Check 'Price' for invalid values, ensure sufficient data, or try another model (e.g., Random Forest).")
                    logging.error(f"Training failed: {str(e)}", exc_info=True)

    # Live Prediction
    with st.sidebar:
        st.header("🔮 Live Prediction")
        model_path = os.path.join(MODEL_DIR, f"{st.session_state.model_name}.pkl")
        if not st.session_state.model_trained or not os.path.exists(model_path):
            st.warning("⚠ No trained model available. Please train a model first.")
        else:
            try:
                model = joblib.load(model_path)
                input_data = {}
                for col in feature_columns:
                    if col in cat_cols:
                        unique_vals = df[col].dropna().unique()
                        input_data[col] = st.selectbox(f"Select {col}", unique_vals, key=f"pred_{col}")
                    else:
                        min_val, max_val = float(df[col].min()), float(df[col].max())
                        input_data[col] = st.slider(f"Enter {col}", min_val, max_val, float(df[col].mean()), key=f"pred_{col}")
                
                if st.button("Predict"):
                    try:
                        df_input = pd.DataFrame([input_data])
                        prediction = model.predict(df_input)[0]
                        st.write(f"**Prediction: {prediction}**")
                        if debug_mode:
                            logging.debug(f"Prediction: {prediction}, input: {input_data}")
                    except Exception as e:
                        st.error(f"❌ Prediction Failed: Ensure valid inputs. Error: {str(e)}")
                        logging.error(f"Prediction failed: {str(e)}", exc_info=True)
            except Exception as e:
                st.error(f"❌ Failed to load model: {str(e)}. Please train a new model.")
                logging.error(f"Model loading failed: {str(e)}", exc_info=True)
```
