# # Streamlit app for your regression model
# # Save this file as streamlit_app.py and run:
# # streamlit run streamlit_app.py

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import io
# from pathlib import Path

# MODEL_PATH = 'modeldeploy.pkl'  # the uploaded model file

# st.set_page_config(page_title='Regression Model Demo', layout='wide')
# st.title('Regression Model UI')
# st.markdown(
#     """
#     Upload a CSV/XLSX with input features or enter single-sample values manually.

#     The app will try to read feature names from the saved model (if available).
#     If feature names are not present in the model, upload a sample file and select which
#     columns to use as features.
#     """
# )

# @st.cache_resource
# def load_model(path: str):
#     try:
#         m = joblib.load(path)
#         return m
#     except Exception as e:
#         st.error(f"Could not load model from {path}: {e}")
#         return None

# model = load_model(MODEL_PATH)

# if model is None:
#     st.warning('Model not loaded. Please check MODEL_PATH or upload a different model file.')

# # Try to get expected feature names from the model
# feature_names = None
# if model is not None:
#     feature_names = getattr(model, 'feature_names_in_', None)
#     if feature_names is not None:
#         try:
#             feature_names = list(feature_names)
#         except Exception:
#             feature_names = None

# st.sidebar.header('Options')
# mode = st.sidebar.selectbox('Mode', ['Predict single sample', 'Batch predict (upload file)'])

# uploaded_model = st.sidebar.file_uploader('(Optional) Upload a different model (.pkl)', type=['pkl','joblib'])
# if uploaded_model is not None:
#     try:
#         model = joblib.load(uploaded_model)
#         st.sidebar.success('Uploaded model loaded')
#         feature_names = getattr(model, 'feature_names_in_', None)
#         if feature_names is not None:
#             try:
#                 feature_names = list(feature_names)
#             except Exception:
#                 feature_names = None
#     except Exception as e:
#         st.sidebar.error(f'Failed to load uploaded model: {e}')

# # Helper: sanitize dataframe
# def to_dataframe(obj):
#     if isinstance(obj, pd.DataFrame):
#         return obj
#     try:
#         return pd.read_csv(obj)
#     except Exception:
#         try:
#             return pd.read_excel(obj)
#         except Exception:
#             return None

# if mode == 'Predict single sample':
#     st.subheader('Single sample prediction')
#     if feature_names:
#         st.info('Detected feature names from model: ' + ', '.join(feature_names))
#         values = {}
#         cols = st.columns(2)
#         for i, fname in enumerate(feature_names):
#             with cols[i % 2]:
#                 values[fname] = st.number_input(fname, value=0.0, format='%.6f')
#         if st.button('Predict'):
#             X = np.array([list(values.values())])
#             try:
#                 pred = model.predict(X)
#                 st.success(f'Prediction: {pred[0]}')
#             except Exception as e:
#                 st.error(f'Prediction failed: {e}')
#     else:
#         st.info('Model does not expose feature names. Provide a comma-separated list of feature names or upload a sample file to detect columns.')
#         fn_input = st.text_input('Enter feature names (comma-separated)', '')
#         if fn_input.strip():
#             feature_names = [f.strip() for f in fn_input.split(',') if f.strip()]
#         if feature_names:
#             st.write('Provide values for: ', feature_names)
#             values = {}
#             cols = st.columns(2)
#             for i, fname in enumerate(feature_names):
#                 with cols[i % 2]:
#                     values[fname] = st.number_input(fname, value=0.0, format='%.6f')
#             if st.button('Predict'):
#                 X = np.array([list(values.values())])
#                 try:
#                     pred = model.predict(X)
#                     st.success(f'Prediction: {pred[0]}')
#                 except Exception as e:
#                     st.error(f'Prediction failed: {e}')
#         else:
#             st.info('Or upload a small CSV/XLSX sample to choose columns.')

# elif mode == 'Batch predict (upload file)':
#     st.subheader('Batch prediction (CSV or XLSX)')
#     uploaded_file = st.file_uploader('Upload CSV or XLSX file with features', type=['csv','xlsx'])
#     if uploaded_file is not None:
#         df = to_dataframe(uploaded_file)
#         if df is None:
#             st.error('Could not read uploaded file. Make sure it is a valid CSV or XLSX.')
#         else:
#             st.write('Preview of uploaded data (first 5 rows):')
#             st.dataframe(df.head())

#             if feature_names:
#                 st.info('Model expects features: ' + ', '.join(feature_names))
#                 missing = [f for f in feature_names if f not in df.columns]
#                 if missing:
#                     st.error('Uploaded data is missing required columns: ' + ', '.join(missing))
#                     st.warning('If your file has different column names, select the columns to use below.')
#                 else:
#                     if st.button('Run predictions'):
#                         X = df[feature_names]
#                         try:
#                             preds = model.predict(X)
#                             out = df.copy()
#                             out['prediction'] = preds
#                             st.success('Predictions done')
#                             st.dataframe(out.head())
#                             csv = out.to_csv(index=False).encode('utf-8')
#                             st.download_button('Download predictions as CSV', csv, 'predictions.csv', 'text/csv')
#                         except Exception as e:
#                             st.error(f'Prediction failed: {e}')
#             else:
#                 st.info('Model has no stored feature names. Select which columns to use as features:')
#                 cols = st.multiselect('Select feature columns in order', options=list(df.columns))
#                 if cols:
#                     st.write('Columns selected (order matters):', cols)
#                     if st.button('Run predictions'):
#                         X = df[cols]
#                         try:
#                             preds = model.predict(X)
#                             out = df.copy()
#                             out['prediction'] = preds
#                             st.success('Predictions done')
#                             st.dataframe(out.head())
#                             csv = out.to_csv(index=False).encode('utf-8')
#                             st.download_button('Download predictions as CSV', csv, 'predictions.csv', 'text/csv')
#                         except Exception as e:
#                             st.error(f'Prediction failed: {e}')

# # Utility: let user download an example input file
# st.sidebar.markdown('---')
# if st.sidebar.button('Download example input file'):
#     # create small example depending on detected features
#     if feature_names:
#         ex = pd.DataFrame([ {f:0 for f in feature_names} ])
#         st.sidebar.download_button('Download example', ex.to_csv(index=False).encode('utf-8'), 'example_input.csv', 'text/csv')
#     else:
#         ex = pd.DataFrame({'feature_1':[0], 'feature_2':[0]})
#         st.sidebar.download_button('Download example', ex.to_csv(index=False).encode('utf-8'), 'example_input.csv', 'text/csv')

# st.sidebar.markdown('\n')
# st.sidebar.write('Model path:')
# st.sidebar.code(MODEL_PATH)

# st.markdown('---')
# st.caption('Built with Streamlit. If prediction fails, check model compatibility and that feature columns are numeric.')
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="💼 Salary Prediction App",
    layout="wide",
    page_icon="💰"
)

# ==========================================================
# CUSTOM STYLING
# ==========================================================
st.markdown(
    """
    <style>
        /* Background gradient */
        .main {
            background: linear-gradient(145deg, #141E30 0%, #243B55 100%);
        }

        /* Text */
        h1, h2, h3, h4, h5, h6, div, p {
            color: #f1f1f1 !important;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Card design */
        .block-container {
            padding-top: 2rem;
        }

        .card {
            background: rgba(255, 255, 255, 0.07);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }

        /* Prediction text */
        .prediction-box {
            background: rgba(0, 255, 150, 0.15);
            border-left: 5px solid #00ff9d;
            padding: 15px;
            border-radius: 10px;
            font-size: 1.3rem;
        }

        /* Button styling */
        .stButton>button {
            background-color: #00BFFF;
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            border: none;
        }

        .stButton>button:hover {
            background-color: #009acd;
            transform: scale(1.03);
        }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    """
    <h1 style='text-align:center; font-size: 3rem; margin-bottom: 0;'>
    💼 Salary Prediction App
    </h1>
    <p style='text-align:center; font-size: 1.2rem;'>
        Powered by D05 Batch ML Team
    </p>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("salaries.csv")
    return df

try:
    df = load_data()
except:
    st.error("❌ salaries.csv not found. Place it next to app.py and restart.")
    st.stop()

# ==========================================================
# TRAIN MODEL (FROM .ipynb LOGIC)
# ==========================================================
def train_model(df):
    X = df.drop(columns=['salary_in_usd'])
    y = df['salary_in_usd']

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler, X

model, scaler, X_matrix = train_model(df)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("⚙ Controls")
st.sidebar.info("Adjust inputs and press *Predict Salary*")

# ==========================================================
# FEATURE INPUT UI
# ==========================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔧 Enter Employee Details")

input_values = {}
original_columns = df.drop(columns=['salary_in_usd']).columns

cols = st.columns(2)

for i, col in enumerate(original_columns):
    with cols[i % 2]:
        if df[col].dtype in [np.float64, np.int64]:
            input_values[col] = st.number_input(
                f"{col}",
                min_value=float(df[col].min()),
                max_value=float(df[col].max()),
                value=float(df[col].median())
            )
        else:
            choices = sorted(df[col].unique().tolist())
            input_values[col] = st.selectbox(f"{col}", choices)

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# PREDICT
# ==========================================================
if st.button("🚀 Predict Salary"):
    input_df = pd.DataFrame([input_values])
    input_df = pd.get_dummies(input_df)

    for col in X_matrix.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X_matrix.columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.balloons()
    
    st.markdown(
        f"""
        <div class='prediction-box'>
        <strong>Estimated Salary:</strong> ₹ {prediction:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size: 0.9rem;'>Created with ❤ in Streamlit</p>
    """,
    unsafe_allow_html=True
)