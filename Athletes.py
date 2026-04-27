import streamlit as st
import google.generativeai as genai
import os

# Configure Gemini API (Replace with your own API key)
genai.configure(api_key="AIzaSyDFVCvSDTtQxjOcH2rTrHI0Ab9LhnDp_wg")

# Use gemini-2.0-flash for higher quota & faster responses
MODEL_NAME = "gemini-2.5-flash-lite"

def get_gemini_response(input_prompt):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(input_prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

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
diet = st.selectbox(
    "Primary Diet:",
    ["Vegetarian", "Non-Vegetarian", "Vegan"]
)

restrictions = st.multiselect(
    "Dietary Restrictions:",
    ["Jain", "Gluten-Free", "Dairy-Free", "Egg-Free", "No Onion/Garlic"]
)


# Calculate BMI
if height > 0:
    bmi = weight / ((height / 100) ** 2)
else:
    bmi = 0

# Athlete-specific input
is_athlete = st.selectbox("Are you an athlete?", ["No", "Yes"])
sport_type, training_intensity = "", ""
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
    athlete_info = ""
    if is_athlete == "Yes":
        athlete_info = (
            f"The user is an athlete in the category '{sport_type}' "
            f"with {training_intensity} training intensity. "
            "Recommendations should focus on optimal performance, recovery, and sport-specific nutrition."
        )

    if health_condition.strip().lower() != "none" and health_condition.strip():
        input_prompt = f"""
        Based on the user details:
        Name: {name}, Age: {age}, Gender: {gender}, Height: {height} cm, Weight: {weight} kg,
        Religion: {religion}, Region: {region}, BMI: {bmi:.2f}, Health Condition: {health_condition}.
        {athlete_info}

        Suggest a balanced meal plan for breakfast, lunch, dinner, and snacks that:
        - Helps manage {health_condition}.
        - Promotes overall health.
        - Considers gender, regional food availability, and religious preferences.
        - Provides nutrient breakdown (proteins, carbs, fats, vitamins, minerals).
        - Includes athlete-specific performance and recovery recommendations if applicable.
        """
    else:
        input_prompt = f"""
        Based on the user details:
        Name: {name}, Age: {age}, Gender: {gender}, Height: {height} cm, Weight: {weight} kg,
        Religion: {religion}, Region: {region}, BMI: {bmi:.2f}.
        {athlete_info}

        Suggest a balanced meal plan for breakfast, lunch, dinner, and snacks that:
        - Supports healthy BMI.
        - Considers gender, regional food availability, and religious preferences.
        - Provides nutrient breakdown (proteins, carbs, fats, vitamins, minerals).
        - Includes athlete-specific performance and recovery recommendations if applicable.
        - All needs to be in very concise and sharp and effective.
        - Don't give paragraphs, it should be easy to read and follow immediately
        """

    response = get_gemini_response(input_prompt)
    if response and not response.startswith("Error:"):
        st.subheader("Nutrition & Meal Suggestions")
        st.write(response)
    else:
        st.error(response or "Failed to generate suggestions.")

# Food nutrient checker
st.subheader("Food Nutrient Analyzer")
food_item = st.text_input("Food Item:")
food_weight = st.number_input("Weight (grams):", min_value=1.0, step=0.1)

if st.button("Analyze Nutrient Content"):
    input_prompt2 = f"""
    Analyze the nutrient content of '{food_item}' weighing {food_weight} grams.
    Include:
    - Proteins, carbs, fats, vitamins, minerals.
    - How they contribute to overall health.
    - Any benefits for athletic performance and recovery (if relevant).
    - All needs to be in very concise and sharp and effective.
    - Don't give paragraphs, it should be easy to read and follow immediately
    """
    response = get_gemini_response(input_prompt2)
    if response and not response.startswith("Error:"):
        st.write(response)
    else:
        st.error(response or "Failed to analyze nutrient content.")
