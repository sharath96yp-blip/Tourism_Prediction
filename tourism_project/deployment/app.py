
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# =========================
# LOAD MODEL
# =========================
model_path = hf_hub_download(
    repo_id="sharath96yp/tourism-prediction-model",
    filename="tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# =========================
# UI
# =========================
st.title("Tourism Package Purchase Prediction")
st.write("""
This app predicts whether a customer is likely to purchase a tourism package.
Fill in the customer details below.
""")

# =========================
# USER INPUT
# =========================

# Numeric Inputs
age = st.number_input("Age", 18, 100, 30)
city_tier = st.selectbox("City Tier", [1, 2, 3])
num_people = st.number_input("Number of People Visiting", 1, 10, 2)
property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
num_trips = st.number_input("Number of Trips per Year", 0, 20, 2)
passport = st.selectbox("Has Passport?", [0, 1])
own_car = st.selectbox("Owns Car?", [0, 1])
num_children = st.number_input("Children Visiting (<5 yrs)", 0, 5, 0)
income = st.number_input("Monthly Income", 1000, 1000000, 30000)
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
followups = st.number_input("Number of Follow-ups", 0, 10, 2)
pitch_duration = st.number_input("Duration of Pitch (minutes)", 0, 120, 20)

# Categorical Inputs
contact_type = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Manager", "Senior Manager", "Executive", "AVP", "VP"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"])

# =========================
# CREATE INPUT DATAFRAME
# =========================
input_data = pd.DataFrame([{
    'Age': age,
    'CityTier': city_tier,
    'NumberOfPersonVisiting': num_people,
    'PreferredPropertyStar': property_star,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children,
    'MonthlyIncome': income,
    'PitchSatisfactionScore': pitch_score,
    'NumberOfFollowups': followups,
    'DurationOfPitch': pitch_duration,
    'TypeofContact': contact_type,
    'Occupation': occupation,
    'Gender': gender,
    'MaritalStatus': marital_status,
    'Designation': designation,
    'ProductPitched': product_pitched
}])

# =========================
# PREDICTION
# =========================
if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"Customer is likely to PURCHASE the package ✅")
    else:
        st.error(f"Customer is NOT likely to purchase ❌")

    st.write(f"**Probability of Purchase:** {probability:.2f}")
