import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import pandas as pd
import joblib

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Load model and scaler
model = joblib.load("gaming_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Gaming Performance Analyzer",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Gaming Performance Analyzer")
st.markdown("Analyze player performance and predict match outcomes using Machine Learning.")

st.write("Enter player statistics and click Analyze")

# Inputs

player_id = st.number_input("Player ID", min_value=0)
map_id = st.number_input("Map", min_value=0)
tier = st.number_input("Tier", min_value=0)

kills = st.number_input("Kills", min_value=0)
damage = st.number_input("Damage Dealt", min_value=0.0)

survival = st.number_input("Survival Time (Minutes)", min_value=0.0)

headshot = st.number_input("Headshot Percentage", min_value=0.0)

assists = st.number_input("Assists", min_value=0)

revives = st.number_input("Revives", min_value=0)

distance = st.number_input("Distance Travelled (km)", min_value=0.0)

longest_kill = st.number_input("Longest Kill (meters)", min_value=0.0)

heals = st.number_input("Heals Used", min_value=0)

boosts = st.number_input("Boosts Used", min_value=0)

favorite_weapon = st.number_input("Favorite Weapon", min_value=0)
if st.button("Analyze"):

    data = pd.DataFrame({
        "Player_ID":[player_id],
        "Map":[map_id],
        "Tier":[tier],
        "Kills":[kills],
        "Damage_Dealt":[damage],
        "Survival_Time_Minutes":[survival],
        "Headshot_Percentage":[headshot],
        "Assists":[assists],
        "Revives":[revives],
        "Distance_Travelled_km":[distance],
        "Longest_Kill_meters":[longest_kill],
        "Heals_Used":[heals],
        "Boosts_Used":[boosts],
        "Favorite_Weapon":[favorite_weapon]
    })

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

if prediction[0] == 1:
    result_text = "High Chance of Winning"
else:
    result_text = "Low Chance of Winning"

# 👇 THIS MUST COME BEFORE generate_content
prompt = f"""
You are a gaming performance AI coach.

Player Stats:
Kills: {kills}
Damage: {damage}
Survival Time: {survival}
Headshot Percentage: {headshot}
Assists: {assists}
Revives: {revives}

Prediction: {result_text}

Give:
1. Summary
2. Strengths
3. Weaknesses
4. Improvement tips
Keep it short.
"""

response = gemini_model.generate_content(prompt)

ai_analysis = response.text

st.write(ai_analysis)