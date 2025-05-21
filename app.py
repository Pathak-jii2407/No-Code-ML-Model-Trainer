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
import itertools
try:
    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    profiling_available = True
except ImportError:
    ProfileReport = st_profile_report = None
    profiling_available = False
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, roc_curve, auc

# Optional TensorFlow import for neural networks
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

# Streamlit Page Config (must be first Streamlit command)
st.set_page_config(page_title="AutoML Pro", page_icon="🤖", layout="wide")

# Display warnings for missing optional dependencies
if not tensorflow_available:
    st.warning("TensorFlow not installed. Neural Network models will be unavailable. Install with 'pip install tensorflow'.")
if not profiling_available:
    st.warning("ydata-profiling not installed. Dataset reports will be unavailable. Install with 'pip install ydata-profiling streamlit-pandas-profiling'.")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stTextInput>input, .stNumberInput>input {border: 1px solid #ddd; border-radius: 5px;}
    .stSidebar .stButton>button {background-color: #2196F3;}
    .st-expander {background-color: white; border: 1px solid #ddd; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_name' not in st.session_state:
    st.session_state.model_name = "model"

# Title
st.title("🤖 AutoML Pro: No-Code Machine Learning")

# Debug mode toggle
debug_mode = st.checkbox("Enable Debug Mode (logs detailed info)", value=False)

# Folder path for pre-uploaded datasets
UPLOADS_FOLDER = "Uploads"

# Ensure uploads folder exists
if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER)

# Sidebar: Dataset Selection
with st.sidebar:
    st.subheader("📂 Dataset Selection")
    preloaded_files = [f for f in os.listdir(UPLOADS_FOLDER) if f.endswith((".csv", ".xlsx", ".json"))]
    selected_file = st.selectbox("Select a dataset", ["Upload your own"] + preloaded_files, key="dataset_select")

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

    # Dataset Sampling
    with st.expander("🔍 Dataset Options", expanded=True):
        sample_data = st.checkbox("Sample dataset (for large datasets)", value=False)
        if sample_data:
            sample_size = st.slider("Sample Size", 100, len(df), min(1000, len(df)))
            df = df.sample(n=sample_size, random_state=42)
            st.success(f"Dataset sampled to {sample_size} rows.")
            if debug_mode:
                logging.debug(f"Sampled dataset to {sample_size} rows")

        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Dataset Cleaning
        st.write("### 🧹 Dataset Cleaning")
        columns_to_drop = st.multiselect("Select Columns to Drop", df.columns, key="drop_columns")
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            st.success("Selected columns dropped!")
            if debug_mode:
                logging.debug(f"Dropped columns: {columns_to_drop}")

    # Generate Report
    if profiling_available:
        generate_report = st.checkbox("Generate Dataset Report", value=False)
        if generate_report and st.button("🚀 Generate Report"):
            with st.spinner("Generating Report... Please Wait ⏳"):
                try:
                    progress_bar = st.progress(0)
                    profile = ProfileReport(df, explorative=True)
                    progress_bar.progress(100)
                    st_profile_report(profile)
                    if debug_mode:
                        logging.debug("Generated dataset report")
                except Exception as e:
                    st.error(f"❌ Failed to generate report: {str(e)}")
                    logging.error(f"Report generation failed: {str(e)}")
    else:
        st.info("Dataset report generation is disabled due to missing dependencies.")

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
                    "Logistic Regression": LogisticRegression(),
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "SVM": SVC(probability=True),
                    "KNN": KNeighborsClassifier()
                },
                "Regression": {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "KNN": KNeighborsRegressor()
                }
            }
            if tensorflow_available:
                model_options["Classification"]["Neural Network"] = None
                model_options["Regression"]["Neural Network"] = None

            selected_model_name = st.selectbox("🛠 Select Model", list(model_options[task_type].keys()), key="model_select")
            
            # Neural Network Configuration
            if selected_model_name == "Neural Network" and tensorflow_available:
                num_features = len([col for col in df.columns if col != target_column])
                hidden_layers = st.slider("Number of Hidden Layers", 1, 5, 1, key="nn_layers")
                nodes_per_layer = st.slider("Nodes per Hidden Layer", 8, 128, 32, 8, key="nn_nodes")
                activation = st.selectbox("Activation Function", ["relu", "tanh", "sigmoid"], key="nn_activation")
                
                def build_nn_classifier():
                    model = Sequential()
                    model.add(Dense(nodes_per_layer, input_dim=num_features, activation=activation))
                    for _ in range(hidden_layers - 1):
                        model.add(Dense(nodes_per_layer, activation=activation))
                    output_units = len(np.unique(df[target_column])) if task_type == "Classification" else 1
                    output_activation = "softmax" if task_type == "Classification" and output_units > 2 else "sigmoid" if task_type == "Classification" else "linear"
                    model.add(Dense(output_units, activation=output_activation))
                    loss = "sparse_categorical_crossentropy" if task_type == "Classification" and output_units > 2 else "binary_crossentropy" if task_type == "Classification" else "mse"
                    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"] if task_type == "Classification" else ["mse"])
                    return model
                
                model = KerasClassifier(model=build_nn_classifier, epochs=50, batch_size=32, verbose=0) if task_type == "Classification" else \
                        KerasRegressor(model=build_nn_classifier, epochs=50, batch_size=32, verbose=0)
                if debug_mode:
                    logging.debug(f"Neural Network configured: layers={hidden_layers}, nodes={nodes_per_layer}, activation={activation}")
            else:
                model = model_options[task_type][selected_model_name]

            # Feature Selection
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns]
            y = df[target_column]

            # Identify Numeric & Categorical Columns
            num_cols = X.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = X.select_dtypes(include=['object']).columns
            st.write(f"📌 **Categorical Columns (One-Hot Encoded):** {cat_cols}")
            if debug_mode:
                logging.debug(f"Features: numeric={num_cols.tolist()}, categorical={cat_cols.tolist()}")

            # Imputation and Scaling
            imputation_strategy = st.selectbox("Select Imputation Strategy for Numeric Columns", 
                                              ["Mean", "Median", "Most Frequent", "Constant"], key="imputation")
            num_imputer = SimpleImputer(strategy=imputation_strategy.lower() if imputation_strategy != "Constant" else "constant", 
                                       fill_value=0 if imputation_strategy == "Constant" else None)
            scaling_option = st.radio("⚖ Select Feature Scaling Method", ["StandardScaler", "MinMaxScaler"], key="scaling")
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

            # Hyperparameter Tuning Options
            hyperparams = {}
            if hasattr(model, "n_estimators"):
                hyperparams["n_estimators"] = st.slider("🌳 Number of Trees (n_estimators)", 10, 500, 100, 10, key="n_estimators")
                model.set_params(n_estimators=hyperparams["n_estimators"])
            if hasattr(model, "max_depth"):
                hyperparams["max_depth"] = st.slider("📍 Max Depth of Trees", 1, 50, 10, 1, key="max_depth")
                model.set_params(max_depth=hyperparams["max_depth"])
            if selected_model_name == "SVM":
                hyperparams["C"] = st.slider("SVM C Parameter", 0.1, 10.0, 1.0, 0.1, key="svm_c")
                model.set_params(C=hyperparams["C"])
            if selected_model_name == "KNN":
                hyperparams["n_neighbors"] = st.slider("Number of Neighbors", 3, 15, 5, key="knn_neighbors")
                model.set_params(n_neighbors=hyperparams["n_neighbors"])
            if selected_model_name == "Neural Network" and tensorflow_available:
                hyperparams["epochs"] = st.slider("Epochs", 10, 100, 50, 10, key="nn_epochs")
                model.set_params(epochs=hyperparams["epochs"])
            if debug_mode:
                logging.debug(f"Hyperparameters: {hyperparams}")

            # Training Pipeline
            pipeline = Pipeline([
                ('preprocess', preprocessor),
                ('model', model)
            ])

            # Cross-Validation Customization
            cv_folds = st.slider("Number of Cross-Validation Folds", 2, 10, 5, key="cv_folds")

            # Auto Hyperparameter Tuning
            auto_tune = st.checkbox("Enable Auto Hyperparameter Tuning", value=False)
            tuning_results = []
            if auto_tune and selected_model_name != "Neural Network":
                st.write("### Auto Hyperparameter Tuning")
                progress_bar = st.progress(0)
                param_grid = {
                    "Random Forest": {"model__n_estimators": [50, 100, 200], "model__max_depth": [5, 10, 20]},
                    "Gradient Boosting": {"model__n_estimators": [50, 100, 200], "model__max_depth": [3, 5, 7]},
                    "SVM": {"model__C": [0.1, 1.0, 10.0]},
                    "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
                    "Linear Regression": {},
                    "KNN": {"model__n_neighbors": [3, 5, 7]}
                }.get(selected_model_name, {})
                if param_grid:
                    try:
                        param_combinations = list(itertools.product(*[param_grid[p] for p in param_grid]))
                        total_iterations = len(param_combinations)
                        for idx, params in enumerate(param_combinations):
                            param_dict = {f"model__{k.split('__')[-1]}": v for k, v in zip(param_grid.keys(), params)}
                            model.set_params(**param_dict)
                            scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring="accuracy" if task_type == "Classification" else "neg_mean_squared_error")
                            tuning_results.append({
                                **{k.split('__')[-1]: v for k, v in param_dict.items()},
                                "score": float(np.mean(scores))
                            })
                            progress_bar.progress((idx + 1) / total_iterations)
                        tuning_df = pd.DataFrame(tuning_results)
                        st.write("### Tuning Results")
                        st.dataframe(tuning_df)
                        # Plot tuning results (single parameter for simplicity)
                        if tuning_results and "n_estimators" in tuning_results[0]:
                            chart_data = {
                                "type": "line",
                                "data": {
                                    "datasets": [
                                        {
                                            "label": "Cross-Validation Score",
                                            "data": [{"x": float(row["n_estimators"]), "y": float(row["score"])} for row in tuning_results],
                                            "fill": False,
                                            "borderColor": "rgba(75, 192, 192, 1)",
                                            "borderWidth": 2
                                        }
                                    ]
                                },
                                "options": {
                                    "scales": {
                                        "x": {"title": {"display": True, "text": "Number of Trees"}},
                                        "y": {"title": {"display": True, "text": "Cross-Validation Score"}}
                                    },
                                    "plugins": {"title": {"display": True, "text": "Hyperparameter Tuning Results"}}
                                }
                            }
                            st.components.v1.html(f"""
                                <div style='width: 100%; height: 400px;'>
                                    <canvas id='tuning_chart'></canvas>
                                    <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
                                    <script>
                                        const ctx = document.getElementById('tuning_chart').getContext('2d');
                                        new Chart(ctx, {json.dumps(chart_data)});
                                    </script>
                                </div>
                            """, height=400)
                        if debug_mode:
                            logging.debug(f"Tuning results: {tuning_results}")
                    except Exception as e:
                        st.error(f"❌ Hyperparameter tuning failed: {str(e)}. Try a different model or adjust parameters.")
                        logging.error(f"Tuning failed: {str(e)}", exc_info=True)
                else:
                    st.info("No tunable hyperparameters for this model.")

            # Training Mode
            train_option = st.radio("🛠 Choose Training Mode", ["Manual Train", "Auto Train"], key="train_mode")

            # Train Model
            if train_option == "Auto Train" or st.button("🚀 Train Model"):
                try:
                    # Validate data
                    if X.isna().any().any() or y.isna().any():
                        st.error("❌ Data contains missing values. Please use imputation or clean the dataset.")
                        logging.error("Missing values detected in data")
                        st.stop()
                    if not np.all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))) and not cat_cols.any():
                        st.error("❌ Non-numeric features detected without categorical encoding. Please encode categorical columns.")
                        logging.error("Non-numeric features detected")
                        st.stop()
                    if selected_model_name == "Neural Network" and tensorflow_available:
                        if task_type == "Classification" and not np.all(y.dtypes.apply(lambda x: np.issubdtype(x, np.integer))):
                            st.error("❌ Neural network classification requires integer-encoded target labels.")
                            logging.error("Invalid target labels for neural network classification")
                            st.stop()

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    with st.spinner("Training Model..."):
                        progress_bar = st.progress(0)
                        scoring = "accuracy" if task_type == "Classification" else "neg_mean_squared_error"
                        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring=scoring)
                        pipeline.fit(X_train, y_train)
                        progress_bar.progress(100)

                    # Save Model and Outputs
                    st.session_state.model_name = st.text_input("Enter Model Name for Saving", st.session_state.model_name, key="model_name")
                    joblib.dump(pipeline, f"{st.session_state.model_name}.pkl")
                    
                    # Create ZIP archive for downloads
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        with open(f"{st.session_state.model_name}.pkl", "rb") as f:
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
                    href = f'<a href="data:application/zip;base64,{b64}" download="{st.session_state.model_name}_outputs.zip">Download All Outputs (Model + Metrics)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    st.success(f"✅ Model Trained Successfully!")
                    st.write(f"📊 **Model Used:** {selected_model_name}")
                    st.write(f"🎯 **Cross-Validation Score (Mean):** {np.mean(scores):.2f} (±{np.std(scores):.2f})")
                    if debug_mode:
                        logging.debug(f"Model trained: {selected_model_name}, CV score: {np.mean(scores):.2f}")

                    # Model Evaluation
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
                                roc_chart_data = {
                                    "type": "line",
                                    "data": {
                                        "datasets": [
                                            {
                                                "label": f"ROC Curve (AUC = {roc_auc:.2f})",
                                                "data": [{"x": float(f), "y": float(t)} for f, t in zip(fpr, tpr)],
                                                "fill": False,
                                                "borderColor": "rgba(255, 99, 132, 1)",
                                                "borderWidth": 2
                                            }
                                        ]
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
                                            new Chart(ctx, {json.dumps(roc_chart_data)});
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
                        st.write("### Predicted vs Actual Plot")
                        chart_data = {
                            "type": "scatter",
                            "data": {
                                "datasets": [
                                    {
                                        "label": "Predicted vs Actual",
                                        "data": [{"x": float(actual), "y": float(pred)} for actual, pred in zip(y_test, y_pred)],
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
                    st.error(f"❌ Training Failed: {str(e)}. Try a different model (e.g., Random Forest) or check data for non-numeric values or missing data.")
                    logging.error(f"Training failed: {str(e)}", exc_info=True)

            # Model Comparison
            compare_models = st.checkbox("Compare Multiple Models", value=False)
            if compare_models and st.button("Compare Models"):
                with st.spinner("Comparing Models..."):
                    try:
                        progress_bar = st.progress(0)
                        comparison_results = []
                        for idx, (name, model) in enumerate(model_options[task_type].items()):
                            if name == "Neural Network" and not tensorflow_available:
                                continue
                            pipeline = Pipeline([('preprocess', preprocessor), ('model', model)])
                            scoring = "accuracy" if task_type == "Classification" else "neg_mean_squared_error"
                            scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring=scoring)
                            comparison_results.append({"Model": name, "Mean CV Score": float(np.mean(scores)), "Std CV Score": float(np.std(scores))})
                            progress_bar.progress((idx + 1) / len(model_options[task_type]))
                        comparison_df = pd.DataFrame(comparison_results)
                        st.write("### Model Comparison")
                        st.dataframe(comparison_df)
                        if debug_mode:
                            logging.debug(f"Model comparison results: {comparison_results}")
                    except Exception as e:
                        st.error(f"❌ Model comparison failed: {str(e)}. Try cleaning the dataset or using fewer models.")
                        logging.error(f"Model comparison failed: {str(e)}", exc_info=True)

    # Live Prediction
    with st.sidebar:
        st.header("🔮 Live Prediction")
        st.write("Fill in values to make a prediction:")
        if os.path.exists(f"{st.session_state.model_name}.pkl"):
            model = joblib.load(f"{st.session_state.model_name}.pkl")
            input_data = {}
            for col in feature_columns:
                if col in cat_cols:
                    unique_vals = df[col].dropna().unique()
                    input_data[col] = st.selectbox(f"Select value for {col}", unique_vals, key=f"pred_{col}")
                else:
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    try:
                        input_data[col] = st.slider(f"Enter value for {col}", min_val, max_val, float(df[col].mean()), key=f"pred_{col}")
                    except ValueError:
                        st.error(f"Invalid range for {col}. Please check data types.")
                        logging.error(f"Invalid slider range for {col}: min={min_val}, max={max_val}")
            
            if st.button("Predict"):
                try:
                    df_input = pd.DataFrame([input_data])
                    prediction = model.predict(df_input)[0]
                    st.write(f"**Prediction: {prediction}**")
                    if debug_mode:
                        logging.debug(f"Prediction made: {prediction}, input: {input_data}")
                except Exception as e:
                    st.error(f"❌ Prediction Failed: Ensure all inputs are valid (e.g., correct data types). Error: {str(e)}")
                    logging.error(f"Prediction failed: {str(e)}", exc_info=True)
