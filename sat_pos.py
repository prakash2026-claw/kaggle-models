import pandas as pd
from pycaret.regression import load_model, predict_model
import streamlit as st

# Set page configurations
st.set_page_config(page_title="Satellite Position Predictor", layout="centered")

# 1. Load the trained PyCaret model (cached so it only loads once)
@st.cache_resource
def get_model():
    # Make sure model is in the same directory as this script
    return load_model("sat_pos")

model = get_model()

st.title("🏥 Satellite Position Prediction Dashboard")
st.write("Fill out the product,client and worker realted details below to check the Satellite Position prediction.")

# 2. Build the Form Interface
with st.form("prediction_form"):
    st.subheader("Decimal number Inputs")
    col1,col2 = st.columns(2)
    
    with col1:
        X_Position = st.number_input("X_Position", min_value=0.0, value=98.58666575005762)
        Velocity = st.number_input("Velocity", min_value=0.0, value=7.5511267713932995)
        Altitude = st.number_input("Altitude", min_value=0.0, value=-28.0)
        Fuel_Level = st.number_input("Fuel_Level", min_value=0.0, value=-20.15839350086838)
     
    with col2:        
        Signal_Strength = st.number_input("Signal_Strength", min_value=0.0, value=14.817947157134617)
        Battery_Temp = st.number_input("Battery_Temp", min_value=0.0, value=1.0384929611720055)
        Solar_Exposure = st.number_input("Solar_Exposure", min_value=0.0, value=-5.760837892713907)
        
    # st.subheader("Integer number Inputs")
    # col1,col2 = st.columns(2)
    
    # with col1:
        # JumlahNasabahAktif = st.number_input("JumlahNasabahAktif", min_value=0, value=5943)
        # JumlahTeller = st.number_input("JumlahTeller", min_value=0, value=2)
        # JumlahPegawai = st.number_input("JumlahPegawai", min_value=0, value=8)
        # JumlahATM = st.number_input("JumlahATM", min_value=0, value=4)

     
    # with col2:        
        # UmurCabang_tahun = st.number_input("UmurCabang_tahun", min_value=0, value=5)
        # TransaksiHarian = st.number_input("TransaksiHarian", min_value=0, value=366)
        # TransaksiBulanan = st.number_input("TransaksiBulanan", min_value=0, value=8059)
        # KodeInternal = st.number_input("KodeInternal", min_value=0, value=3457)        

    # st.subheader("Categorical Inputs")
    # col3,col4 = st.columns(2)
    
    # with col3:
        # KategoriKota = st.text_input("KategoriKota", "Kota Besar")
        
    # with col4:
        # Wilayah = st.text_input("Wilayah", "Jawa Timur")

    # Submit button for the form
    submit_button = st.form_submit_button("Predict Satellite Position")

# 3. Handle Prediction Logic upon form submission
if submit_button:
    # Compile the form inputs into a dictionary matching your PyCaret model's features
    input_data = {'X_Position': X_Position,'Velocity': Velocity,'Altitude': Altitude,'Fuel_Level': Fuel_Level,'Signal_Strength': Signal_Strength,'Battery_Temp': Battery_Temp,'Solar_Exposure': Solar_Exposure}    
    
    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])
    
    with st.spinner("Calculating risk..."):
        # Make the prediction using PyCaret
        #predictions = predict_model(model, data=df,round=2,raw_score=True)
        predictions = predict_model(model, data=df,round=2)
        prediction_label = predictions["prediction_label"].iloc[0]
        # prediction_score_1 = predictions["prediction_score_1"].iloc[0]
        # prediction_score_0 = predictions["prediction_score_0"].iloc[0]
        
        # Display the result to the user
        st.success("### Prediction Complete!")
        st.metric(label="Satellite Position Result", value=f"Position: {prediction_label}")
        # st.metric(label="Risk Status Result", value=f"Class: {prediction_label}")
        # st.metric(label="Prediction Confidence Scores", value=f"Class 0 Score:{prediction_score_0}, Class 1 Score:{prediction_score_1}")
