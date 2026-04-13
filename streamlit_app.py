import streamlit as st
import pandas as pd
import joblib

import utils.feature_engineering as fe
import models.model_training as mt
from utils.llm_explainer import explain_location

import os
print(os.environ.get("GROQ_API_KEY"))
from dotenv import load_dotenv
load_dotenv()

# load pipeline
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/best_pipeline.pkl")
    except:
        return None

pipeline = load_model()

st.title("Solar Suitability Demo")

# input
name = st.text_input("Location Name", "My Location")

col1, col2 = st.columns(2)
lat = col1.number_input("Latitude", value=26.9)
lon = col2.number_input("Longitude", value=70.9)

# simple features (for demo)
sample = {
    "ghi": 5.5, "dni": 5.0, "temperature": 30,
    "cloud_pct": 25, "clearness": 0.6,
    "elevation": 200, "slope": 5, "aspect": 180,
    "ndvi": 0.3, "road_km": 2, "grid_km": 5,
    "wind_speed": 10, "precipitation": 1,
    "humidity": 40, "land_score": 0.7,
    "lat": lat, "lon": lon
}

if st.button("Analyse"):

    if pipeline is not None:
        score = mt.predict_location(pipeline, sample)
        method = "ML"
    else:
        score = fe.compute_score(sample)
        method = "Formula"

    rank = fe.get_rank(score)

    st.subheader(f"{name}")
    st.write(f"Score: {score:.1f} / 100")
    st.write(f"Rank: {rank}")
    st.write(f"Method: {method}")

    st.progress(score / 100)

    st.subheader("AI Explanation")

    explanation = explain_location(
        score=score,
        rank=rank,
        features=sample,
        location_name=name
    )

    st.write(explanation)