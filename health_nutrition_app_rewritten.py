import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sqlalchemy import text

st.set_page_config(page_title="Health & Athlete Nutrition Checker", page_icon="🏋️")
st.title("🏋️ Health, Athlete, and Nutrients Checker")

# --- AUTHENTICATION ---
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter the app password to access:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter the app password to access:", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

if not check_password():
    st.stop()

# --- DATABASE SETUP ---
try:
    conn = st.connection('my_database', type='sql')
except Exception:
    conn = None

# --- GEMINI SETUP ---
MODEL_NAME = "gemini-2.5-flash" # Updated to standard flash model name for better compatibility

def get_api_key() -> str:
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    env_key = os.getenv("GEMINI_API_KEY", "")
    return (secret_key or env_key or "").strip()

def get_gemini_response(input_prompt: str) -> str:
    api_key = get_api_key()
    if not api_key:
        return "Error: Missing GEMINI_API_KEY in Streamlit secrets or environment variables."
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=input_prompt,
        )
        return getattr(response, "text", None) or "Error: Empty response from Gemini."
    except Exception as e:
        return f"Error: {e}"

def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5: return "underweight"
    if bmi < 25: return "normal"
    if bmi < 30: return "overweight"
    return "obese"

# --- UI INPUTS ---
name = st.text_input("Name:")
age = st.number_input("Age:", min_value=1, max_value=120, step=1)
gender = st.selectbox("Gender:", ["Male", "Female", "Other"])
height = st.number_input("Height (cm):", min_value=50.0, step=0.1)
weight = st.number_input("Weight (kg):", min_value=1.0, step=0.1)
religion = st.selectbox("Religion:", ["Hindu", "Muslim", "Christian", "Other"])
region = st.selectbox("Region:", ["North", "South", "East", "West"])
diet = st.selectbox("Primary Diet:", ["Vegetarian", "Non-Vegetarian", "Vegan"])
restrictions = st.multiselect(
    "Dietary Restrictions:",
    ["Jain", "Gluten-Free", "Dairy-Free", "Egg-Free", "No Onion/Garlic"],
)

bmi = 0.0
if height > 0:
    bmi = weight / ((height / 100) ** 2)
bmi_category = get_bmi_category(bmi)

is_athlete = st.selectbox("Are you an athlete?", ["No", "Yes"])
sport_type, training_intensity = "Unknown", "Unknown"
if is_athlete == "Yes":
    sport_type = st.selectbox(
        "Type of Sport:",
        [
            "Endurance (e.g., Marathon, Cycling)",
            "Strength (e.g., Weightlifting, Wrestling)",
            "Mixed (e.g., Football, Basketball)",
        ],
    )
    training_intensity = st.selectbox("Training Intensity:", ["Light", "Moderate", "Intense"])

health_condition = st.text_input("Health Condition (e.g., Diabetes, High BP, None):")

# --- AI ANALYSIS & DATABASE SAVE ---
if st.button("Analyze and Suggest Food"):
    
    athlete_info = ""
    if is_athlete == "Yes":
        athlete_info = (
            f"The user is an athlete in '{sport_type}' with {training_intensity} training. "
            "Prioritize performance, recovery, hydration, and protein quality."
        )

    # We now ask Gemini to act as the ML model and recommend the foods itself
    input_prompt = f"""Act as an expert sports nutritionist. Based on these user details:
Name: {name}
Age: {age}
Gender: {gender}
BMI: {bmi:.2f} ({bmi_category})
Religion: {religion}
Region: {region}
Diet: {diet}
Restrictions: {', '.join(restrictions) or 'None'}
Health: {health_condition or 'None'}
{athlete_info}

First, list 5-7 highly recommended specific whole foods for this exact profile.
Then, create a concise meal plan with exactly these sections:
- Breakfast
- Lunch
- Dinner
- Snacks

Rules:
- Use only foods that strictly fit the user's diet, religion, and restrictions.
- Include rough portions in grams, cups, pieces, or tablespoons.
- Add a compact nutrient note for each meal: protein, carbs, fats.
- Mention how the plan supports their BMI goals and athlete needs if relevant.
- Keep it easy to read and brief.
- Use bullet points only. No paragraphs.
"""

    with st.spinner("Gemini is analyzing your profile..."):
        response = get_gemini_response(input_prompt)

    if response and not response.startswith("Error:"):
        st.write(response)
        
        # --- DATABASE SAVE LOGIC ---
        if 'conn' in locals() and conn is not None:
            try:
                with conn.session as s:
                    s.execute(text("""
                        CREATE TABLE IF NOT EXISTS user_nutrition_logs (
                            name VARCHAR(255),
                            age INT,
                            bmi FLOAT,
                            diet VARCHAR(100),
                            is_athlete VARCHAR(10),
                            ai_response TEXT
                        );
                    """))
                    
                    s.execute(text("""
                        INSERT INTO user_nutrition_logs (name, age, bmi, diet, is_athlete, ai_response) 
                        VALUES (:name, :age, :bmi, :diet, :is_athlete, :ai_response);
                    """), 
                    {
                        "name": name, 
                        "age": age, 
                        "bmi": bmi, 
                        "diet": diet,
                        "is_athlete": is_athlete, 
                        "ai_response": response
                    })
                    
                    s.commit()
                st.success("✅ Results securely logged to the database!")
            except Exception as e:
                st.warning(f"⚠️ App ran successfully, but could not save to database: {e}")

    else:
        st.error(response or "Failed to generate suggestions.")

# --- FOOD NUTRIENT ANALYZER ---
st.subheader("Food Nutrient Analyzer")
food_item = st.text_input("Food Item:")
food_weight = st.number_input("Weight (grams):", min_value=1.0, step=0.1)
if st.button("Analyze Nutrient Content"):
    input_prompt2 = f"""Analyze '{food_item}' ({food_weight}g).
Return concise bullet points only:
- proteins
- carbs
- fats
- vitamins
- minerals
- health contributions
- athlete benefits if relevant
"""
    with st.spinner("Analyzing nutrients..."):
        response = get_gemini_response(input_prompt2)
    if response and not response.startswith("Error:"):
        st.write(response)
    else:
        st.error(response or "Failed to analyze.")
























# st.set_page_config(page_title="Health & Athlete Nutrition Checker", page_icon="🏋️")
# st.title("🏋️ Health, Athlete, and Nutrients Checker")
# # --- AUTHENTICATION ---
# def check_password():
#     def password_entered():
#         if st.session_state["password"] == st.secrets["app_password"]:
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # don't store password
#         else:
#             st.session_state["password_correct"] = False

#     if "password_correct" not in st.session_state:
#         st.text_input("Enter the app password to access:", type="password", on_change=password_entered, key="password")
#         return False
#     elif not st.session_state["password_correct"]:
#         st.text_input("Enter the app password to access:", type="password", on_change=password_entered, key="password")
#         st.error("😕 Password incorrect")
#         return False
#     return True

# if not check_password():
#     st.stop()

# # --- DATABASE SETUP ---
# # This initializes the connection using Streamlit's Secrets
# try:
#     conn = st.connection('my_database', type='sql')
# except Exception:
#     conn = None
# DATA_PATH = Path("Data") / "user_nutrition_dataset.csv"
# MODEL_NAME = "gemini-2.5-flash-lite"


# def get_bmi_category(bmi: float) -> str:
#     if bmi < 18.5:
#         return "underweight"
#     if bmi < 25:
#         return "normal"
#     if bmi < 30:
#         return "overweight"
#     return "obese"


# def get_api_key() -> str:
#     try:
#         secret_key = st.secrets.get("GEMINI_API_KEY", "")
#     except Exception:
#         secret_key = ""
#     env_key = os.getenv("GEMINI_API_KEY", "")
#     api_key = (secret_key or env_key or "").strip()
#     return api_key


# @st.cache_resource
# def load_and_train_model():
#     if not DATA_PATH.exists():
#         raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

#     df = pd.read_csv(DATA_PATH)

#     required_cols = {
#         "age",
#         "gender",
#         "bmi",
#         "region",
#         "diet",
#         "religion",
#         "bmi_category",
#         "is_athlete",
#         "sport_type",
#         "training_intensity",
#         "health_condition",
#         "dietary_restrictions",
#         "is_recommended",
#         "recommended_foods",
#     }
#     missing = required_cols - set(df.columns)
#     if missing:
#         raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

#     features = [
#         "age",
#         "gender",
#         "bmi",
#         "region",
#         "diet",
#         "religion",
#         "bmi_category",
#         "is_athlete",
#         "sport_type",
#         "training_intensity",
#         "health_condition",
#         "dietary_restrictions",
#     ]
#     categorical_features = [
#         "gender",
#         "region",
#         "diet",
#         "religion",
#         "bmi_category",
#         "is_athlete",
#         "sport_type",
#         "training_intensity",
#         "health_condition",
#         "dietary_restrictions",
#     ]
#     numeric_features = ["age", "bmi"]

#     df = df.copy()
#     df[categorical_features] = df[categorical_features].fillna("Unknown").astype(str)
#     df["recommended_foods"] = df["recommended_foods"].fillna("").astype(str)

#     X = df[features].copy()
#     y_bin = df["is_recommended"].astype(int)

#     y_ml_raw = (
#         df["recommended_foods"]
#         .str.split(",")
#         .apply(lambda items: [item.strip() for item in items if item and item.strip()])
#     )
#     mlb = MultiLabelBinarizer()
#     y_ml = mlb.fit_transform(y_ml_raw)

#     idx = np.arange(len(df))
#     train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=42, shuffle=True)

#     X_train = X.iloc[train_idx].reset_index(drop=True)
#     X_test = X.iloc[test_idx].reset_index(drop=True)
#     y_train_bin = y_bin.iloc[train_idx].reset_index(drop=True)
#     y_test_bin = y_bin.iloc[test_idx].reset_index(drop=True)
#     y_train_ml = y_ml[train_idx]
#     y_test_ml = y_ml[test_idx]

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", "passthrough", numeric_features),
#             ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
#         ],
#         remainder="drop",
#     )

#     rf_params = dict(
#         n_estimators=150,
#         random_state=42,
#         n_jobs=-1,
#         class_weight="balanced",
#         max_depth=6,
#         min_samples_split=4,
#     )

#     rf_bin = Pipeline(
#         steps=[
#             ("preprocessor", preprocessor),
#             ("model", RandomForestClassifier(**rf_params)),
#         ]
#     )
#     rf_bin.fit(X_train, y_train_bin)

#     rf_multi = Pipeline(
#         steps=[
#             ("preprocessor", preprocessor),
#             ("model", MultiOutputClassifier(RandomForestClassifier(**rf_params))),
#         ]
#     )
#     rf_multi.fit(X_train, y_train_ml)

#     # Optional metrics for validation in the app logs / console.
#     y_pred_bin = rf_bin.predict(X_test)
#     y_pred_ml = rf_multi.predict(X_test)
#     _ = {
#         "binary_accuracy": accuracy_score(y_test_bin, y_pred_bin),
#         "binary_f1": f1_score(y_test_bin, y_pred_bin, zero_division=0),
#         "multi_hamming_loss": hamming_loss(y_test_ml, y_pred_ml),
#     }

#     return rf_bin, rf_multi, mlb


# rf_bin, rf_multi, mlb = None, None, None
# try:
#     rf_bin, rf_multi, mlb = load_and_train_model()
# except Exception as exc:
#     st.error(f"Model loading/training failed: {exc}")


# def get_gemini_response(input_prompt: str) -> str:
#     api_key = get_api_key()
#     if not api_key:
#         return "Error: Missing GEMINI_API_KEY in Streamlit secrets or environment variables."
#     try:
#         client = genai.Client(api_key=api_key)
#         response = client.models.generate_content(
#             model=MODEL_NAME,
#             contents=input_prompt,
#         )
#         return getattr(response, "text", None) or "Error: Empty response from Gemini."
#     except Exception as e:
#         return f"Error: {e}"


# # Basic user details
# name = st.text_input("Name:")
# age = st.number_input("Age:", min_value=1, max_value=120, step=1)
# gender = st.selectbox("Gender:", ["Male", "Female", "Other"])
# height = st.number_input("Height (cm):", min_value=50.0, step=0.1)
# weight = st.number_input("Weight (kg):", min_value=1.0, step=0.1)
# religion = st.selectbox("Religion:", ["Hindu", "Muslim", "Christian", "Other"])
# region = st.selectbox("Region:", ["North", "South", "East", "West"])
# diet = st.selectbox("Primary Diet:", ["Vegetarian", "Non-Vegetarian", "Vegan"])
# restrictions = st.multiselect(
#     "Dietary Restrictions:",
#     ["Jain", "Gluten-Free", "Dairy-Free", "Egg-Free", "No Onion/Garlic"],
# )

# bmi = 0.0
# if height > 0:
#     bmi = weight / ((height / 100) ** 2)
# bmi_category = get_bmi_category(bmi)

# is_athlete = st.selectbox("Are you an athlete?", ["No", "Yes"])
# sport_type, training_intensity = "Unknown", "Unknown"
# if is_athlete == "Yes":
#     sport_type = st.selectbox(
#         "Type of Sport:",
#         [
#             "Endurance (e.g., Marathon, Cycling)",
#             "Strength (e.g., Weightlifting, Wrestling)",
#             "Mixed (e.g., Football, Basketball)",
#         ],
#     )
#     training_intensity = st.selectbox("Training Intensity:", ["Light", "Moderate", "Intense"])

# health_condition = st.text_input("Health Condition (e.g., Diabetes, High BP, None):")


# def preprocess_user_input(
#     age,
#     gender,
#     bmi,
#     region,
#     diet,
#     religion,
#     bmi_category,
#     is_athlete,
#     sport_type,
#     training_intensity,
#     health_condition,
#     dietary_restrictions,
# ):
#     return pd.DataFrame(
#         {
#             "age": [age],
#             "bmi": [bmi],
#             "gender": [str(gender)],
#             "region": [str(region)],
#             "diet": [str(diet)],
#             "religion": [str(religion)],
#             "bmi_category": [str(bmi_category)],
#             "is_athlete": [str(is_athlete)],
#             "sport_type": [str(sport_type)],
#             "training_intensity": [str(training_intensity)],
#             "health_condition": [str(health_condition or "None")],
#             "dietary_restrictions": [", ".join(dietary_restrictions) if dietary_restrictions else "None"],
#         }
#     )


# if st.button("Analyze and Suggest Food"):
#     if rf_bin is None or rf_multi is None or mlb is None:
#         st.stop()

#     user_input = preprocess_user_input(
#         age,
#         gender,
#         bmi,
#         region,
#         diet,
#         religion,
#         bmi_category,
#         is_athlete,
#         sport_type,
#         training_intensity,
#         health_condition,
#         restrictions,
#     )

#     is_recommended = int(rf_bin.predict(user_input)[0])
#     ml_pred = rf_multi.predict(user_input)
#     predicted_foods_binary = mlb.inverse_transform(ml_pred)[0]
#     predicted_foods_str = ", ".join(predicted_foods_binary) if predicted_foods_binary else ""

#     athlete_info = ""
#     if is_athlete == "Yes":
#         athlete_info = (
#             f"The user is an athlete in '{sport_type}' with {training_intensity} training. "
#             "Prioritize performance, recovery, hydration, and protein quality."
#         )

#     food_guidance = (
#         f"Recommended foods from ML model: {predicted_foods_str}."
#         if predicted_foods_str
#         else "No specific foods were predicted by the ML model."
#     )

#     input_prompt = f"""Based on these user details:
# Name: {name}
# Age: {age}
# Gender: {gender}
# BMI: {bmi:.2f}
# BMI Category: {bmi_category}
# Religion: {religion}
# Region: {region}
# Diet: {diet}
# Restrictions: {', '.join(restrictions) or 'None'}
# Health: {health_condition or 'None'}
# {athlete_info}
# {food_guidance}

# Create a concise meal plan with exactly these sections:
# - Breakfast
# - Lunch
# - Dinner
# - Snacks

# Rules:
# - Prefer the ML-predicted foods when available.
# - Use only foods that fit the user's diet, religion, and restrictions.
# - Include rough portions in grams, cups, pieces, or tablespoons.
# - Add a compact nutrient note for each meal: protein, carbs, fats, vitamins, minerals.
# - Mention how the plan supports BMI goals and athlete needs if relevant.
# - Keep it easy to read and brief.
# - Use bullet points only. No paragraphs.
# """
# response = get_gemini_response(input_prompt)

# if response and not response.startswith("Error:"):
#     st.write(response)

#     # --- DATABASE SAVE LOGIC ---
#     if 'conn' in locals() and conn is not None:
#         try:
#             from sqlalchemy import text

#             with conn.session as s:
#                 # 1. Auto-create the table if it doesn't exist yet
#                 s.execute(text("""
#                         CREATE TABLE IF NOT EXISTS user_nutrition_logs (
#                             name VARCHAR(255),
#                             age INT,
#                             bmi FLOAT,
#                             diet VARCHAR(100),
#                             is_athlete VARCHAR(10),
#                             ai_response TEXT
#                         );
#                     """))

#                 # 2. Insert the user's inputs and the AI's response safely
#                 s.execute(text("""
#                         INSERT INTO user_nutrition_logs (name, age, bmi, diet, is_athlete, ai_response) 
#                         VALUES (:name, :age, :bmi, :diet, :is_athlete, :ai_response);
#                     """),
#                           {
#                               "name": name,
#                               "age": age,
#                               "bmi": bmi,
#                               "diet": diet,
#                               "is_athlete": is_athlete,
#                               "ai_response": response
#                           })

#                 s.commit()
#             st.success("✅ Results securely logged to the database!")
#         except Exception as e:
#             st.warning(f"⚠️ App ran successfully, but could not save to database: {e}")

# else:
#     st.error(response or "Failed to generate suggestions.")
#     response = get_gemini_response(input_prompt)
#
#     if response and not response.startswith("Error:"):
#         st.write(response)
#     else:
#         st.error(response or "Failed to generate suggestions.")
#
#
# st.subheader("Food Nutrient Analyzer")
# food_item = st.text_input("Food Item:")
# food_weight = st.number_input("Weight (grams):", min_value=1.0, step=0.1)
# if st.button("Analyze Nutrient Content"):
#     input_prompt2 = f"""Analyze '{food_item}' ({food_weight}g).
# Return concise bullet points only:
# - proteins
# - carbs
# - fats
# - vitamins
# - minerals
# - health contributions
# - athlete benefits if relevant
# """
#     response = get_gemini_response(input_prompt2)
#     if response and not response.startswith("Error:"):
#         st.write(response)
#     else:
#         st.error(response or "Failed to analyze.")
