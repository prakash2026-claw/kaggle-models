import pandas as pd
from pycaret.regression import load_model, predict_model
import streamlit as st

# Set page configurations
st.set_page_config(page_title="Bank Operational Cost Predictor", layout="centered")

# 1. Load the trained PyCaret model (cached so it only loads once)
@st.cache_resource
def get_model():
    # Make sure model is in the same directory as this script
    return load_model("bank_op_cost")

model = get_model()

st.title("🏥 Bank Operational Cost Prediction Dashboard")
st.write("Fill out the product,client and worker realted details below to check the Bank Operational Cost prediction.")

# 2. Build the Form Interface
with st.form("prediction_form"):
    st.subheader("Categorical Inputs")
    col1,col2 = st.columns(2)
    
    with col1:
        JumlahNasabahAktif = st.number_input("JumlahNasabahAktif", min_value=0, value=5943)
        JumlahTeller = st.number_input("JumlahTeller", min_value=0, value=2)
        JumlahPegawai = st.number_input("JumlahPegawai", min_value=0, value=8)
        JumlahATM = st.number_input("JumlahATM", min_value=0, value=4)
        LuasCabang_m2 = st.number_input("LuasCabang_m2", min_value=0.0, value=466.0)
     
    with col2:        
        UmurCabang_tahun = st.number_input("UmurCabang_tahun", min_value=0, value=5)
        JamOperasionalPerHari = st.number_input("JamOperasionalPerHari", min_value=0.0, value=9.6)
        TransaksiHarian = st.number_input("TransaksiHarian", min_value=0, value=366)
        TransaksiBulanan = st.number_input("TransaksiBulanan", min_value=0, value=8059)
        SkorKepuasanNasabah = st.number_input("SkorKepuasanNasabah", min_value=0.0, value=4.48)
        KodeInternal = st.number_input("KodeInternal", min_value=0, value=3457)

    st.subheader("Numerical Inputs")
    col3,col4 = st.columns(2)
    
    with col3:
        KategoriKota = st.text_input("KategoriKota", "Kota Besar")
        
    with col4:
        Wilayah = st.text_input("Wilayah", "Jawa Timur")

    # Submit button for the form
    submit_button = st.form_submit_button("Predict Bank Operational Cost")

# 3. Handle Prediction Logic upon form submission
if submit_button:
    # Compile the form inputs into a dictionary matching your PyCaret model's features
    input_data = {'IDCabang': IDCabang,'KategoriKota': KategoriKota,'Wilayah': Wilayah,'JumlahNasabahAktif': JumlahNasabahAktif,'JumlahTeller': JumlahTeller,'JumlahPegawai': JumlahPegawai,'JumlahATM': JumlahATM,'LuasCabang_m2': LuasCabang_m2,'UmurCabang_tahun': UmurCabang_tahun,'JamOperasionalPerHari': JamOperasionalPerHari,'TransaksiHarian': TransaksiHarian,'TransaksiBulanan': TransaksiBulanan,'SkorKepuasanNasabah': SkorKepuasanNasabah,'KodeInternal': KodeInternal}    
    
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
        st.metric(label="Bank Operational Cost Result", value=f"Cost: {prediction_label}")
        # st.metric(label="Risk Status Result", value=f"Class: {prediction_label}")
        # st.metric(label="Prediction Confidence Scores", value=f"Class 0 Score:{prediction_score_0}, Class 1 Score:{prediction_score_1}")
