import pandas as pd
from pycaret.regression import load_model, predict_model
import streamlit as st

# Set page configurations
st.set_page_config(page_title="Insurance Cross Selling Predictor", layout="centered")

# 1. Load the trained PyCaret model (cached so it only loads once)
@st.cache_resource
def get_model():
    # Make sure model is in the same directory as this script
    # ins_cross_sell
    return load_model("ins_cross_sell")

model = get_model()

st.title("🏥 Insurance Cross Selling Prediction Dashboard")
#st.link_button("Go to kaggle completion page", "https://www.kaggle.com/competitions/prediksi-biaya-operasional-cabang-odp-b-2-2026")
st.write("Visit [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s4e7) for more details.")
st.write("Fill out the client realted details below to check the Insurance Cross Selling prediction.")

# 2. Build the Form Interface
with st.form("prediction_form"):
    st.subheader("Numerical Inputs")
    col1,col2 = st.columns(2)
    
    with col1:
        Annual_Premium = st.number_input("Annual_Premium", min_value=0.0, value=2630.0)

     
    with col2:        
        Age = st.number_input("Age", min_value=0, value=20)


    st.subheader("Categorical Inputs")
    col3,col4 = st.columns(2)
    
    with col3:
        Gender = st.text_input("Gender", "Female")
        Driving_License = st.text_input("Driving_License", "1")
        Region_Code = st.text_input("Region_Code", "47.0")
        Previously_Insured = st.text_input("Previously_Insured", "0")        
        
    with col4:
        Vehicle_Age = st.text_input("Vehicle_Age", "< 1 Year")
        Vehicle_Damage = st.text_input("Vehicle_Damage", "No")
        Policy_Sales_Channel = st.text_input("Policy_Sales_Channel", "160.0")
        Vintage = st.text_input("Vintage", "228")        

    # Submit button for the form
    submit_button = st.form_submit_button("Predict Insurance Cross Selling")

# 3. Handle Prediction Logic upon form submission
if submit_button:
    # Compile the form inputs into a dictionary matching your PyCaret model's features
    input_data = {'Gender': Gender,'Driving_License': Driving_License,'Region_Code': Region_Code,'Previously_Insured': Previously_Insured,'Vehicle_Age': Vehicle_Age,'Vehicle_Damage': Vehicle_Damage,'Policy_Sales_Channel': Policy_Sales_Channel,'Vintage': Vintage,'Age': Age,'Annual_Premium': Annual_Premium}    
    
    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])
    
    with st.spinner("Calculating risk..."):
        # Make the prediction using PyCaret
        predictions = predict_model(model, data=df,round=2,raw_score=True)
        # predictions = predict_model(model, data=df,round=2)
        prediction_label = predictions["prediction_label"].iloc[0]
        prediction_score_1 = predictions["prediction_score_1"].iloc[0]
        prediction_score_0 = predictions["prediction_score_0"].iloc[0]
        
        # Display the result to the user
        st.success("### Prediction Complete!")
        st.metric(label="Insurance Cross Selling Result", value=f"Response: {prediction_label}")
        # st.metric(label="Risk Status Result", value=f"Class: {prediction_label}")
        st.metric(label="Prediction Confidence Scores", value=f"Class 0 Score:{prediction_score_0}, Class 1 Score:{prediction_score_1}")
      
