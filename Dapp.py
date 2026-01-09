import streamlit as st 
from streamlit_option_menu import option_menu 
import pandas as pd
import base64 
import plotly.express as px
import xgboost
import pickle

with open("xgb_pipeline.pkl", "rb") as f:
    parkinsons_pipeline = pickle.load(f) 
with open("rf_pipeline.pkl", "rb") as f: 
    kidney_pipeline = pickle.load(f) 
with open("xgb1_pipeline.pkl", "rb") as f:
    liver_pipeline = pickle.load(f) 


# Page Configuration
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")

# Setting background image
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Title
st.title("PredictMyHealth")

# Sidebar Menu
menu = st.sidebar.selectbox("🩺 Mult-Disease Prediction",
        [
            "🧬 Dashboard",
            "🧠 Parkinson's Prediction",
            "💊 Kidney Prediction",
            "🫁 Liver Prediction"
        ])
if menu == "🧬 Dashboard":

    set_bg("C:/Users/harip/Downloads/disease1.png")

    st.markdown("Welcome to the Multiple Disease Prediction System!")
    st.markdown("""
        <h2 style='font-weight: 900'>
        Welcome
        </h2>""", unsafe_allow_html=True)
    
    # How it Works
    st.subheader("🔍 How it Works")

    st.markdown("""
    1. Select a disease from the sidebar  
    2. Enter the required medical parameters  
    3. Click **Predict**  
    4. Get instant results powered by ML models  
    """)

    # Disease Metrics
    st.subheader("📊 Model Performance Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🧠 Parkinson’s F1", "85.62%")

    with col2:
        st.metric("💊 Kidney F1", "98.60%")

    with col3:
        st.metric("🫁 Liver F1", "82.43%")

    data = pd.DataFrame({
    "Disease": ["Parkinson’s", "Kidney", "Liver"],
    "F1 Score (%)": [85.62, 98.60, 82.43]
})
    
elif menu == "🧠 Parkinson's Prediction":
    st.title("Parkinson's Prediction")
    st.write("Provide your data to check for Parkinson's disease.")
    set_bg("C:/Users/harip/Downloads/brain.png")

    col1, col2, col3, col4 = st.columns(4) 
    
    with col1:
        fo = st.text_input('Fundamental Frequency (Hz)', value=119.5)
        
    with col2:
        fhi = st.text_input('Max Frequency (Hz)',value=157.3)

    with col3:
        Jitter_percent = st.text_input('MDVP: Jitter(%)',value=0.0078)
        
    with col4:
        Shimmer = st.text_input('MDVP: Shimmer',value=0.043)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        HNR = st.text_input('HNR',value=21.3) 
        
    with col2:
        RPDE = st.text_input('RPDE',value=0.35)
        
    with col3:
        DFA = st.text_input('DFA',value=0.61)
        
    with col4:
        PPE = st.text_input('PPE',value=0.28)
        
    
    # Prediction
    parkinsons_diagnosis = ''
    
    # Creating a button for Prediction 

    feature_names = parkinsons_pipeline.feature_names_in_
    
    if st.button("Parkinson's Test Result"):

    #  Create DataFrame for input
        input_data = pd.DataFrame(
            [[fo, fhi, Jitter_percent, Shimmer, 
              HNR, RPDE, DFA, PPE]],
            columns=feature_names
        )

        # prediction
        parkinsons_prediction = parkinsons_pipeline.predict(input_data)[0]
        parkinsons_probability = parkinsons_pipeline.predict_proba(input_data)[0][1]

        # display result
        if parkinsons_prediction == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

        st.success(parkinsons_diagnosis)
        st.info(f"Risk Probability: {parkinsons_probability*100:.2f}%")

elif menu == "💊 Kidney Prediction":
    st.title("Kidney Prediction")
    st.write("Provide your data to check for Kidney disease.")
    set_bg("C:/Users/harip/Downloads/kidney.png")

    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        age = st.text_input('Age',value=45)
        
    with col2:
        bp = st.text_input('Blood Pressure(BP)',value=80)
        
    with col3:
        sg = st.number_input("Specific Gravity", value=1.02)
        
    with col4:
        rbc = st.selectbox('Red Blood Cells(RBC)', ['normal', 'abnormal'],index=0)
        rbc = 0 if rbc == 'normal' else 1
        
    with col5:
        pc = st.selectbox('PC', ['normal', 'abnormal'],index=0)
        pc = 0 if pc == 'normal' else 1
        
    with col1:
        bgr = st.text_input('Blood Glucose Random(BGR)',value=120)
        
    with col2:
        sc = st.number_input("Serum Creatinine (mg/dL)", value=1.2)
        
    with col3:
        hemo = st.text_input('Hemoglobin',value=11.5)
        
    with col4:
        pcv = st.text_input('PCV',value=36)
        
    with col5:
        wc = st.number_input('White Blood Cells (WC)',value=8000)
        
    with col1:
        rc = st.text_input('Red Blood Cells(RC)',value=4.5)
        
    with col2:
        htn = st.selectbox('Hypertension', ['yes', 'no'],index=1)
        htn = 1 if htn == 'yes' else 0
        
    with col3:
        dm = st.selectbox('Diabetes Mellitus(DM)', ['yes', 'no'],index=1)
        dm = 1 if dm == 'yes' else 0
        
    # Prediction
    kidney_diagnosis = ''
    
    # Creating a button for Prediction 

    feature_names = [
    'age', 'bp', 'sg', 'rbc', 'pc', 'bgr', 'sc', 'hemo', 
    'pcv', 'wc', 'rc', 'htn', 'dm'
]
    
    if st.button("Kidney Disease Test Result"):

    #  Create DataFrame for input
        input_data = pd.DataFrame(
            [[age, bp, sg, rbc, pc, bgr, sc, hemo, pcv, wc, rc, htn, dm]],
            columns=feature_names
        )

        # prediction
        kidney_prediction = kidney_pipeline.predict(input_data)[0]
        kidney_prob = kidney_pipeline.predict_proba(input_data)[0][1]

        # display result
        if kidney_prediction == 1:
            kidney_diagnosis = "The person has Kidney disease"
        else:
            kidney_diagnosis = "The person does not have Kidney disease"

        st.success(kidney_diagnosis)
        st.info(f"Risk Probability: {kidney_prob*100:.2f}%")


elif menu == "🫁 Liver Prediction":
    st.title("Liver Prediction")
    st.write("Provide your data to check for Liver disease.")
    set_bg("C:/Users/harip/Downloads/liver.png")

    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        age = st.text_input('Age',value=45)
        
    with col2:
        gender = st.selectbox('Gender', ['Female', 'Male'],index=0)
        gender = 0 if gender == 'Male' else 1
        
    with col3:
        total_bilirubin = st.text_input('Total Bilirubin',value=1.0)
        
    with col4:
        direct_bilirubin = st.text_input('Direct Bilirubin',value=0.3)
        
    with col5:
        alkaline_phosphotase = st.text_input('Alkaline Phosphotase',value=250)
        
    with col1:
        alamine_aminotransferase = st.text_input('Alamine Aminotransferase',value=30)
        
    with col2:
        aspartate_aminotransferase = st.text_input('Aspartate Aminotransferase',value=35)
        
    with col3:
        total_protiens = st.text_input('Total Protiens',value=6.5)
        
    with col4:
        albumin = st.text_input('Albumin',value=3.5)
        
    with col5:
        albumin_and_globulin_ratio = st.text_input('Albumin/Globulin_Ratio',value=1.2)
        
    
    # Prediction
    liver_diagnosis = ''
    
    # Creating a button for Prediction 

    feature_names = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio'
]
    
    if st.button("Liver Disease Test Result"):

    #  Create DataFrame for input
        input_data = pd.DataFrame(
            [[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase, alamine_aminotransferase,
               aspartate_aminotransferase, total_protiens, albumin, albumin_and_globulin_ratio
            
            ]],columns=feature_names
        )

        # prediction
        liver_prediction = liver_pipeline.predict(input_data)[0]
        liver_probabilities = liver_pipeline.predict_proba(input_data)
        liver_risk = liver_probabilities[0][1]

        # display result
        if liver_prediction == 1:
            liver_diagnosis = "The person has Liver disease"
        else:
            liver_diagnosis = "The person does not have Liver disease"

        st.success(liver_diagnosis)

        # Show probability
        st.info(f"Risk Probability: {liver_risk*100:.2f}%")