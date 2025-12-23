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

    # Plotting-Bar
    st.subheader("Diseases F1 Score Comparison")
    fig = px.bar(
        data,
        x="Disease",
        y="F1 Score (%)",
        color="Disease",
        color_discrete_map={
            "Parkinson’s": "#1f77b4",
            "Kidney": "#2ca02c",
            "Liver": "#ff7f0e",
        },
        title="📊 F1 Score Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    
elif menu == "🧠 Parkinson's Prediction":
    st.title("Parkinson's Prediction")
    st.write("Provide your data to check for Parkinson's disease.")
    set_bg("C:/Users/harip/Downloads/brain.png")

    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP: Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP: Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP: Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP: Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP: Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP: RAP')
        
    with col2:
        PPQ = st.text_input('MDVP: PPQ')
        
    with col3:
        DDP = st.text_input('Jitter: DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP: Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP: Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer: APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer: APQ5')
        
    with col3:
        APQ = st.text_input('MDVP: APQ')
        
    with col4:
        DDA = st.text_input('Shimmer: DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR') 
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    # Prediction
    parkinsons_diagnosis = ''
    
    # Creating a button for Prediction 

    feature_names = parkinsons_pipeline.feature_names_in_
    
    if st.button("Parkinson's Test Result"):

    #  Create DataFrame for input
        input_data = pd.DataFrame(
            [[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
            Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA,
            NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]],
            columns=feature_names
        )

        # prediction
        parkinsons_prediction = parkinsons_pipeline.predict(input_data)[0]

        # display result
        if parkinsons_prediction == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)


elif menu == "💊 Kidney Prediction":
    st.title("Kidney Prediction")
    st.write("Provide your data to check for Kidney disease.")
    set_bg("C:/Users/harip/Downloads/kidney.png")

    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        bp = st.text_input('BP')
        
    with col3:
        sg = st.selectbox('SG', ['normal', 'abnormal'])
        sg = 0 if sg == 'normal' else 1
        
    with col4:
        rbc = st.selectbox('RBC', ['normal', 'abnormal'])
        rbc = 0 if rbc == 'normal' else 1
        
    with col5:
        pc = st.selectbox('PC', ['normal', 'abnormal'])
        pc = 0 if pc == 'normal' else 1
        
    with col1:
        bgr = st.text_input('BGR')
        
    with col2:
        sc = st.selectbox('SC', ['normal', 'abnormal'])
        sc = 0 if sc == 'normal' else 1
        
    with col3:
        hemo = st.text_input('HEMO')
        
    with col4:
        pcv = st.text_input('PCV')
        
    with col5:
        wc = st.selectbox('WC', ['yes', 'no'])
        wc = 1 if wc == 'yes' else 0
        
    with col1:
        rc = st.text_input('RC')
        
    with col2:
        htn = st.selectbox('HTN', ['yes', 'no'])
        htn = 1 if htn == 'yes' else 0
        
    with col3:
        dm = st.selectbox('Diabetes Mellitus (DM)', ['yes', 'no'])
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

        # display result
        if kidney_prediction == 1:
            kidney_diagnosis = "The person has Kidney disease"
        else:
            kidney_diagnosis = "The person does not have Kidney disease"

    st.success(kidney_diagnosis)


elif menu == "🫁 Liver Prediction":
    st.title("Liver Prediction")
    st.write("Provide your data to check for Liver disease.")
    set_bg("C:/Users/harip/Downloads/liver.png")

    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        gender = st.selectbox('Gender', ['Female', 'Male'])
        gender = 0 if gender == 'Male' else 1
        
    with col3:
        total_bilirubin = st.text_input('Total_Bilirubin')
        
    with col4:
        direct_bilirubin = st.text_input('Direct_Bilirubin')
        
    with col5:
        alkaline_phosphotase = st.text_input('Alkaline_Phosphotase')
        
    with col1:
        alamine_aminotransferase = st.text_input('Alamine_Aminotransferase')
        
    with col2:
        aspartate_aminotransferase = st.text_input('Aspartate_Aminotransferase')
        
    with col3:
        total_protiens = st.text_input('Total_Protiens')
        
    with col4:
        albumin = st.text_input('Albumin')
        
    with col5:
        albumin_and_globulin_ratio = st.text_input('Albumin_and_Globulin_Ratio')
        
    
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

        # display result
        if liver_prediction == 1:
            liver_diagnosis = "The person has Liver disease"
        else:
            liver_diagnosis = "The person does not have Liver disease"

    st.success(liver_diagnosis)