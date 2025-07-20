import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn

# --- Page Configuration ---
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom Modern and Neutral CSS ---
st.markdown("""
<style>
html, body, .stApp {
    background-color: #f4f6f9 !important; /* Overall light background */
    color: #2d2d2d !important; /* Default text color */
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
.stSidebar {
    background-color: #222428 !important; /* Dark sidebar background */
    padding: 20px;
    color: #f0f0f0; /* Light text for sidebar */
}
.stSidebar h2, .stSidebar h3, .stSidebar label, .stSidebar span {
    color: #ffffff !important; /* Ensure all text in sidebar is white */
}

/* Main container (Card-like) */
.main .block-container {
    background-color: #ffffff; /* White background for main content card */
    padding: 2.5rem 3rem;
    border-radius: 12px;
    box-shadow: 0 0 20px rgba(0,0,0,0.08); /* Soft shadow */
    margin-top: 30px;
}

/* Headers */
h1, h2, h3 {
    font-weight: 700;
    color: #1f2a40; /* Dark blue-gray for headers */
    margin-bottom: 20px;
}

/* Predict button */
.stButton>button {
    background-color: #4CAF50; /* Green button */
    color: white;
    padding: 12px 24px;
    font-size: 16px;
    border-radius: 8px;
    border: none;
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    transition: all 0.2s ease;
}
.stButton>button:hover {
    transform: scale(1.02);
     color: white;
}
/* General button styling outside sidebar */
 .stDownloadButton > button {
    background-color: #1f78b4;
    color: #ffffff;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    box-shadow: 0 4px 12px rgba(31, 120, 180, 0.3);
    transition: all 0.2s ease;
}

 .stDownloadButton > button:hover {
    transform: scale(1.03);
     color: white;
}


/* Input styling (Selectbox, NumberInput) */
div[data-baseweb="select"], input[type="number"] {
    border-radius: 8px;
    border: 1px solid #ccc;
    background-color: #fff;
    color: #2d2d2d; /* Ensure input text is dark */
}

/* File uploader */
.stFileUploader {
    border: 2px dashed #ccc;
    border-radius: 10px;
    padding: 20px;
    background-color: #f9f9f9;
}
.stFileUploader:hover {
    border-color: #4CAF50;
    background-color: #f0fff0;
}


/* Sliders - General Styling */
.stSlider > div[data-baseweb="slider"] {
    background-color: none; /* Neutral gray for the slider track */
}
.stSlider > div > div > div:nth-child(1) {
    background-color: none; /* Green for the filled portion of the slider */
}

/* Status Messages (Success, Info, Error) */
.stSuccess, .stInfo, .stError {
    padding: 15px;
    border-radius: 8px;
    font-size: 1em;
    font-weight: bold;
    margin-bottom: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.stSuccess {
    background-color: #d4edda;
    color: #155724; /* Dark green text */
    border-left: 5px solid #28a745;
}
.stInfo {
    background-color: #d1ecf1;
    color: #0c5460; /* Dark blue text */
    border-left: 5px solid #17a2b8;
}
.stError {
    background-color: #f8d7da;
    color: #721c24; /* Dark red text */
    border-left: 5px solid #dc3545;
}

/* Specific styling for prediction result text to be black */
.st.success p, .st.info p {
    color: rgb(0 7 12) !important; /* Force black text for prediction messages */
}
/* --- Custom Styles for st.success and st.info --- */


/* Adjust text inside the alert */
div.stAlert p {
    margin: 0;
    color: #000000 !important;
}



/* DataFrame */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #ddd;
}
.stDataFrame th {
    background-color: #f1f1f1;
    color: #333;
}
.stDataFrame td {
    background-color: #fff;
    color: #222;
}
/* Apply styles to success and error alerts in sidebar */
section[data-testid="stSidebar"] .stAlert {
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    background-color: white;
}


/* Success Alert in Sidebar */
section[data-testid="stSidebar"] .stAlert[data-testid="stAlert-success"] {
    background-color: #d4edda !important;
    border-left: 5px solid #28a745;
    color: #155724;
}

/* Error Alert in Sidebar */
section[data-testid="stSidebar"] .stAlert[data-testid="stAlert-error"] {
    background-color: #f8d7da !important;
    border-left: 5px solid #dc3545;
    color: #721c24;
}

</style>
""", unsafe_allow_html=True)


# --- Display Scikit-learn version ---
st.sidebar.write(f"Scikit-learn version: {sklearn.__version__}")

# --- Load Model ---
try:
    model_pipeline = joblib.load('employee_salary_prediction_model.joblib')
    st.sidebar.success("Model loaded successfully!")
except FileNotFoundError:
    st.sidebar.error("Model file not found.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Model loading error: {e}")
    st.stop()


# --- Feature Columns ---
expected_features = [
    'age', 'workclass', 'education',
    'marital-status', 'occupation', 'relationship',
    'race', 'gender', 'hours-per-week', 'native-country'
]

# --- Options ---
workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Without-pay', 'Never-worked']
education_options = ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', 'Assoc-acdm', 'Prof-school', 'Doctorate', '10th', '11th', '12th']
marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
gender_options = ['Male', 'Female']
native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Columbia', 'Poland', 'Japan', 'Portugal', 'Taiwan', 'Haiti', 'Iran', 'Ecuador', 'France', 'Nicaragua', 'Peru', 'Greece', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Thailand', 'Cambodia', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands']


# --- App Title ---
st.title("💼 Employee Salary Prediction")
st.markdown("Predict if an employee's income is **<=50K** or **>50K** per year.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Input Employee Details")
    age = st.slider("Age", 17, 90, 30)
    education = st.selectbox("Education Level", education_options)
    occupation = st.selectbox("Job Role", occupation_options)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    workclass = st.selectbox("Workclass", workclass_options)
    marital_status = st.selectbox("Marital Status", marital_status_options)
    relationship = st.selectbox("Relationship", relationship_options)
    race = st.selectbox("Race", race_options)
    gender = st.selectbox("Gender", gender_options)
    native_country = st.selectbox("Native Country", native_country_options)

    fnlwgt = 200000
    educational_num = 9
    capital_gain = 0
    capital_loss = 0

    st.markdown("---")

    if st.button("Predict Salary Class"):
        input_data = pd.DataFrame([[
            age, workclass, fnlwgt, education, educational_num,
            marital_status, occupation, relationship, race, gender,
            capital_gain, capital_loss, hours_per_week, native_country
        ]], columns=[
            'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            'marital-status', 'occupation', 'relationship', 'race', 'gender',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
        ])

        try:
            prediction = model_pipeline.predict(input_data)[0]
            proba = model_pipeline.predict_proba(input_data)[0] if hasattr(model_pipeline.named_steps['classifier'], 'predict_proba') else None

            st.session_state['prediction_result'] = prediction
            st.session_state['prediction_proba'] = proba
            st.session_state['input_data_display'] = input_data

        except Exception as e:
            st.session_state['error'] = str(e)

# --- Display Prediction ---
st.header("🎯 Prediction Result")
if 'prediction_result' in st.session_state:
    pred = st.session_state['prediction_result']
    proba = st.session_state['prediction_proba']

    if pred == 1:
        st.success("The model predicts the employee's income is **>50K** per Month! 🎉")
        if proba is not None:
            st.write(f"Confidence: {proba[1]*100:.2f}%")
    else:
        st.info("The model predicts the employee's income is **<=50K** per Month. 📉")
        if proba is not None:
            st.write(f"Confidence: {proba[0]*100:.2f}%")

    st.markdown("---")
    st.write("Input Data Used for Prediction:")
    st.dataframe(st.session_state['input_data_display'][expected_features])

if 'error' in st.session_state:
    st.error(f"Prediction Error: {st.session_state['error']}")
    del st.session_state['error']

# --- Batch Prediction ---
st.markdown("---")
st.subheader("📄 Batch Prediction")
st.write("Upload a CSV file for batch prediction.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)

        for feature in ['fnlwgt', 'educational-num', 'capital-gain', 'capital-loss']:
            if feature not in batch_df.columns:
                batch_df[feature] = {'fnlwgt': 200000, 'educational-num': 9, 'capital-gain': 0, 'capital-loss': 0}[feature]

        full_df = batch_df[[
            'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            'marital-status', 'occupation', 'relationship', 'race', 'gender',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
        ]]

        batch_df['Predicted_Income'] = np.where(model_pipeline.predict(full_df) == 1, '>50K', '<=50K')

        with st.expander("📄 Uploaded CSV Preview"):
            st.dataframe(batch_df.head())

        st.subheader("✅ Batch Prediction Results")
        st.dataframe(batch_df[['age', 'education', 'occupation', 'hours-per-week', 'Predicted_Income']])

        csv_out = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", data=csv_out, file_name="batch_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error during batch prediction: {e}")
