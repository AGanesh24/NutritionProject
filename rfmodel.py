import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, hamming_loss

# Configure Gemini API (Replace with your own API key)
genai.configure(api_key="AIzaSyDFVCvSDTtQxjOcH2rTrHI0Ab9LhnDp_wg")  # Replace with actual key
MODEL_NAME = "gemini-2.5-flash-lite"  # Updated to a valid model name; gemini-2.5-flash-lite may not exist yet



@st.cache_resource
def load_and_train_model():

    df = pd.read_csv("Data\\user_nutrition_dataset.csv")

    # Features
    features = ['age', 'gender', 'bmi', 'region', 'diet', 'religion', 'bmi_category',
                'is_athlete', 'sport_type', 'training_intensity', 'health_condition', 'dietary_restrictions']
    cat_cols = ['gender', 'region', 'diet', 'religion', 'bmi_category', 'is_athlete',
                'sport_type', 'training_intensity', 'health_condition', 'dietary_restrictions']
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    df_encoded = pd.get_dummies(df[features], columns=cat_cols)
    X = df_encoded

    # Binary target
    y_bin = df['is_recommended']

    # Multi-label
    y_ml_raw = df['recommended_foods'].str.split(',').apply(lambda x: [item.strip() for item in x if item.strip()])
    mlb = MultiLabelBinarizer()
    y_ml = mlb.fit_transform(y_ml_raw)

    # Split
    X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.3, random_state=42)
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X, y_ml, test_size=0.3, random_state=42)

    # Train Binary RF
    rf_bin = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1, class_weight='balanced', max_depth=3,
                                    min_samples_split=4)
    rf_bin.fit(X_train, y_train_bin)

    # Train Multi-Label RF
    rf_multi = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1, max_depth=3, min_samples_split=4))
    rf_multi.fit(X_train_ml, y_train_ml)

    # For prediction alignment
    feature_columns = X.columns.tolist()
    mlb_classes = mlb.classes_.tolist()

    # Optional: Print metrics (for debugging, can remove in prod)
    y_pred_bin = rf_bin.predict(X_test)
    #st.info({accuracy_score(y_test_bin, y_pred_bin):.4f}

    return rf_bin, rf_multi, mlb, feature_columns, mlb_classes


# Function to preprocess user input for ML prediction
def preprocess_user_input(age, gender, bmi, region, diet, religion, bmi_category, is_athlete, sport_type,
                          training_intensity, health_condition, dietary_restrictions):
    user_df = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'gender': [gender],
        'region': [region],
        'diet': [diet],
        'religion': [religion],
        'bmi_category': [bmi_category],
        'is_athlete': [is_athlete],
        'sport_type': [sport_type],
        'training_intensity': [training_intensity],
        'health_condition': [health_condition],
        'dietary_restrictions': [', '.join(dietary_restrictions) if dietary_restrictions else 'None']
    })

    cat_cols = ['gender', 'region', 'diet', 'religion', 'bmi_category', 'is_athlete',
                'sport_type', 'training_intensity', 'health_condition', 'dietary_restrictions']
    user_df[cat_cols] = user_df[cat_cols].fillna('Unknown')
    user_encoded = pd.get_dummies(user_df, columns=cat_cols)

    # Align columns with training features
    for col in feature_columns:
        if col not in user_encoded.columns:
            user_encoded[col] = 0
    user_encoded = user_encoded[feature_columns]

    return user_encoded


# Function to get BMI category
def get_bmi_category(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif 18.5 <= bmi < 25:
        return 'normal'
    elif 25 <= bmi < 30:
        return 'overweight'
    else:
        return 'obese'


# Gemini response function
def get_gemini_response(input_prompt):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(input_prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"


# Load model
rf_bin, rf_multi, mlb, feature_columns, mlb_classes = load_and_train_model()

# Streamlit UI
st.set_page_config(page_title="Health & Athlete Nutrition Checker")
st.title("🏋️ Health, Athlete, and Nutrients Checker")

# Basic user details
name = st.text_input("Name:")
age = st.number_input("Age:", min_value=1, max_value=120, step=1)
gender = st.selectbox("Gender:", ["Male", "Female", "Other"])
height = st.number_input("Height (cm):", min_value=50.0, step=0.1)
weight = st.number_input("Weight (kg):", min_value=1.0, step=0.1)
religion = st.selectbox("Religion:", ["Hindu", "Muslim", "Christian", "Other"])
region = st.selectbox("Region:", ["North", "South", "East", "West"])
diet = st.selectbox("Primary Diet:", ["Vegetarian", "Non-Vegetarian", "Vegan"])
restrictions = st.multiselect("Dietary Restrictions:",
                              ["Jain", "Gluten-Free", "Dairy-Free", "Egg-Free", "No Onion/Garlic"])

# Calculate BMI
bmi = 0
if height > 0:
    bmi = weight / ((height / 100) ** 2)
bmi_category = get_bmi_category(bmi)

#st.info(f"Calculated BMI: {bmi:.2f} ({bmi_category})")

# Athlete-specific input
is_athlete = st.selectbox("Are you an athlete?", ["No", "Yes"])
sport_type, training_intensity = "Unknown", "Unknown"
if is_athlete == "Yes":
    sport_type = st.selectbox("Type of Sport:", [
        "Endurance (e.g., Marathon, Cycling)",
        "Strength (e.g., Weightlifting, Wrestling)",
        "Mixed (e.g., Football, Basketball)"
    ])
    training_intensity = st.selectbox("Training Intensity:", ["Light", "Moderate", "Intense"])

# Health condition
health_condition = st.text_input("Health Condition (e.g., Diabetes, High BP, None):")

# Button to analyze
if st.button("Analyze and Suggest Food"):
    # Preprocess for ML
    user_input = preprocess_user_input(
        age, gender, bmi, region, diet, religion, bmi_category, is_athlete,
        sport_type, training_intensity, health_condition or 'None', restrictions
    )

    # ML Predictions
    is_recommended = rf_bin.predict(user_input)[0]
    predicted_foods_binary = mlb.inverse_transform(rf_multi.predict(user_input))[0]
    predicted_foods_str = ', '.join(
        predicted_foods_binary) if predicted_foods_binary else "No specific foods recommended based on profile."

    # st.success(f"ML Decision: {'Recommended' if is_recommended else 'Not Recommended'}")
    # st.info(f"ML-Predicted Foods: {predicted_foods_str}")

    # Athlete info for prompt
    athlete_info = ""
    if is_athlete == "Yes":
        athlete_info = (
            f"The user is an athlete in '{sport_type}' with {training_intensity} training. "
            "Focus on performance and recovery."
        )

    # Gemini Prompt using ML outputs
    input_prompt = f"""Based on user details: Name: {name}, Age: {age}, Gender: {gender}, BMI: {bmi:.2f}, Religion: {religion}, Region: {region}, Diet: {diet}, Restrictions: {', '.join(restrictions) or 'None'}, Health: {health_condition or 'None'}. {athlete_info}

    

    Create a concise, easy-to-follow meal plan (breakfast, lunch, dinner, snacks) using ONLY these foods. Include:
    - Assignment to meals.
    - Nutrient breakdown (proteins, carbs, fats, vitamins, minerals).
    - How it supports health, BMI, and athlete needs if applicable.
    - be much concise, must be easy to read.
    - Give amount of each food to take accordingly in respective metric for that, like grams, cup...    .
    Format as bullet points for readability. No paragraphs.
    """

    response = get_gemini_response(input_prompt)

    if response and not response.startswith("Error:"):
        #st.subheader("ML-Driven Nutrition & Meal Suggestions")
        st.write(response)
    else:
        st.error(response or "Failed to generate suggestions.")

# Food nutrient checker (unchanged)
st.subheader("Food Nutrient Analyzer")
food_item = st.text_input("Food Item:")
food_weight = st.number_input("Weight (grams):", min_value=1.0, step=0.1)
if st.button("Analyze Nutrient Content"):
    input_prompt2 = f"""Analyze '{food_item}' ({food_weight}g). Include:
    - Proteins, carbs, fats, vitamins, minerals.
    - Health contributions.
    - Athlete benefits if relevant.
    - be much concise, must be easy to read.
    Concise bullet points only."""
    response = get_gemini_response(input_prompt2)
    if response and not response.startswith("Error:"):
        st.write(response)
    else:
        st.error(response or "Failed to analyze.")