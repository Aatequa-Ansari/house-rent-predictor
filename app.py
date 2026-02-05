# ================================
# Import Libraries
# ================================

import streamlit as st
import pandas as pd
import joblib


# ================================
# Page Config
# ================================

st.set_page_config(
    page_title="House Rent Predictor",
    layout="centered"
)


# ================================
# Load Trained Model
# ================================

@st.cache_resource
def load_model():
    return joblib.load("linear_model.pkl")


model = load_model()


# ================================
# Load Dataset (For City-Locality Mapping)
# ================================

@st.cache_data
def load_data():
    return pd.read_csv("cities_magicbricks_rental_prices.csv")


df = load_data()

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")


# ================================
# Create City → Locality Mapping
# ================================

city_locality_map = (
    df.groupby("city")["locality"]
    .unique()
    .apply(list)
    .to_dict()
)


# ================================
# Title
# ================================

st.title("House Rent Prediction App")
st.write("Predict Monthly Rent using Machine Learning")


# ================================
# Input Section
# ================================

st.header("Enter Property Details")


# ================================
# Area Input
# ================================

area = st.number_input(
    "Area (sqft)",
    min_value=200,
    max_value=10000,
    value=1000,
    step=50,
    key="area_input"
)


# ================================
# Auto Rules Based On Area
# ================================

if area < 600:
    max_bhk = 1
    max_bath = 1

elif area < 900:
    max_bhk = 2
    max_bath = 2

elif area < 1300:
    max_bhk = 3
    max_bath = 2

elif area < 1800:
    max_bhk = 4
    max_bath = 3

else:
    max_bhk = 5
    max_bath = 4


# ================================
# Balconies (Slider)
# ================================

balconies = st.slider(
    "Number of Balconies",
    0, 5, 1,
    key="balcony_slider"
)


# ================================
# BHK (Dynamic Dropdown)
# ================================

bhk = st.selectbox(
    "BHK",
    list(range(1, max_bhk + 1)),
    key="bhk_select"
)


# ================================
# Bathrooms (Dynamic Dropdown)
# ================================

bathrooms = st.selectbox(
    "Number of Bathrooms",
    list(range(1, max_bath + 1)),
    key="bath_select"
)


# ================================
# Furnishing
# ================================

furnishing = st.selectbox(
    "Furnishing Type",
    ["Unfurnished", "Semi-Furnished", "Furnished"],
    key="furnish_select"
)


# ================================
# City Dropdown
# ================================

cities = sorted(df["city"].unique())

city = st.selectbox(
    "City",
    cities,
    key="city_select"
)


# ================================
# Locality Dropdown (Dynamic)
# ================================

localities = sorted(city_locality_map.get(city, []))

locality = st.selectbox(
    "Locality",
    localities,
    key="locality_select"
)


# ================================
# Furnishing Encoding
# ================================

furnish_map = {
    "Furnished": 2,
    "Semi-Furnished": 1,
    "Unfurnished": 0
}

furnishing_score = furnish_map[furnishing]


# ================================
# Prediction
# ================================

import numpy as np


if st.button("Predict Rent", key="predict_btn"):

    input_data = pd.DataFrame({
        "area": [area],
        "balconies": [balconies],
        "bhk": [bhk],
        "bathrooms": [bathrooms],
        "furnishing_score": [furnishing_score],
        "city": [city],
        "locality": [locality]
    })


    log_pred = model.predict(input_data)[0]

    prediction = np.expm1(log_pred)


    st.success(f"Estimated Rent: ₹ {int(prediction):,} per month")



# ================================
# Footer
# ================================

st.markdown("---")

