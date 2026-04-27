from __future__ import annotations

import math
import os
from textwrap import dedent

import streamlit as st
from google import genai
from supabase import Client, create_client


APP_TITLE = "Athlete Food Recommendation System"
USERS_TABLE = "app_users"
PLANS_TABLE = "athlete_recommendations"
MODEL_CANDIDATES = ("gemini-2.5-flash", "gemini-2.0-flash")

SPORT_OPTIONS = [
    "Endurance (e.g., Marathon, Cycling)",
    "Strength (e.g., Weightlifting, Wrestling)",
    "Mixed (e.g., Football, Basketball)",
    "General Fitness / Unsure",
]
GOAL_OPTIONS = ["Performance", "Muscle Gain", "Fat Loss", "Maintenance", "Recovery"]
PLAN_OPTIONS = ["Training Day", "Competition Day", "Rest Day"]
INTENSITY_OPTIONS = ["Light", "Moderate", "Intense"]
REGION_OPTIONS = ["North", "South", "East", "West", "International"]
RELIGION_OPTIONS = ["None", "Hindu", "Muslim", "Christian", "Other"]
DIET_OPTIONS = ["Vegetarian", "Non-Vegetarian", "Vegan", "Eggetarian"]
RESTRICTION_OPTIONS = [
    "Jain",
    "Gluten-Free",
    "Dairy-Free",
    "Egg-Free",
    "No Onion/Garlic",
    "Nut-Free",
    "Soy-Free",
]
HEALTH_OPTIONS = [
    "None",
    "Diabetes",
    "High BP",
    "High Cholesterol",
    "Iron Deficiency",
    "Low Energy",
    "Dehydration",
    "Muscle Fatigue",
    "Muscle Strain",
    "Joint Pain",
    "Inflammation",
    "Other",
]


class LightweightMealSignalModel:
    def __init__(self) -> None:
        self.weights: dict[str, float] = {}
        self.bias = 0.0

    def fit(self, rows: list[dict[str, float]]) -> "LightweightMealSignalModel":
        totals: dict[str, float] = {}
        for row in rows:
            target = float(row.get("target", 0.0))
            self.bias += target
            for feature_name, feature_value in row.items():
                if feature_name == "target":
                    continue
                totals[feature_name] = totals.get(feature_name, 0.0) + (float(feature_value) * target)
        count = max(len(rows), 1)
        self.bias /= count
        self.weights = {name: value / count for name, value in totals.items()}
        return self

    def predict_score(self, features: dict[str, float]) -> float:
        margin = self.bias
        for feature_name, feature_value in features.items():
            margin += self.weights.get(feature_name, 0.0) * float(feature_value)
        return 1.0 / (1.0 + math.exp(-margin))


def get_secret_value(name: str, env_name: str | None = None) -> str:
    try:
        secret_value = st.secrets.get(name, "")
    except Exception:
        secret_value = ""
    env_value = os.getenv(env_name or name, "")
    return str(secret_value or env_value or "").strip()


@st.cache_resource
def get_supabase_client() -> Client | None:
    url = get_secret_value("SUPABASE_URL")
    key = get_secret_value("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def get_api_key() -> str:
    return get_secret_value("GEMINI_API_KEY")


def get_current_user() -> dict | None:
    return st.session_state.get("auth_user")


def set_current_user(user_payload: dict | None) -> None:
    st.session_state["auth_user"] = user_payload


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    if height_cm <= 0:
        return 0.0
    return weight_kg / ((height_cm / 100) ** 2)


def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def estimate_daily_targets(
    age: int,
    gender: str,
    height_cm: float,
    weight_kg: float,
    sport_type: str,
    training_intensity: str,
    training_days: int,
    session_duration_mins: int,
    goal: str,
) -> dict[str, str | int | float]:
    gender_offset = {"Male": 5, "Female": -161}.get(gender, -78)
    bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + gender_offset

    intensity_multiplier = {"Light": 1.4, "Moderate": 1.6, "Intense": 1.8}.get(training_intensity, 1.5)
    weekly_adjustment = min(max((training_days - 3) * 0.03, -0.06), 0.12)
    duration_adjustment = min(max((session_duration_mins - 60) / 600, -0.05), 0.20)
    maintenance_calories = bmr * (intensity_multiplier + weekly_adjustment + duration_adjustment)

    calorie_adjustment = {
        "Performance": 200,
        "Muscle Gain": 250,
        "Fat Loss": -300,
        "Maintenance": 0,
        "Recovery": 100,
    }.get(goal, 0)
    target_calories = max(1200, round(maintenance_calories + calorie_adjustment))

    protein_per_kg = 1.4
    carb_range = "3.0-4.0 g/kg"
    if "Endurance" in sport_type:
        protein_per_kg = 1.6
        carb_range = "5.0-7.0 g/kg"
    elif "Strength" in sport_type:
        protein_per_kg = 1.8
        carb_range = "4.0-6.0 g/kg"
    elif "Mixed" in sport_type:
        protein_per_kg = 1.7
        carb_range = "4.0-6.0 g/kg"

    if goal == "Muscle Gain":
        protein_per_kg = max(protein_per_kg, 2.0)
    elif goal == "Fat Loss":
        protein_per_kg = max(protein_per_kg, 1.9)
    elif goal == "Recovery":
        protein_per_kg = max(protein_per_kg, 1.8)

    hydration_liters = round(((weight_kg * 35) + (training_days * 250) + (session_duration_mins * 8)) / 1000, 1)
    protein_target = round(weight_kg * protein_per_kg)
    fat_target = round(weight_kg * 0.9)

    return {
        "calories": target_calories,
        "protein_g": protein_target,
        "fat_g": fat_target,
        "carb_range": carb_range,
        "hydration_liters": hydration_liters,
    }


def fetch_user_profile(user_id: str) -> dict | None:
    client = get_supabase_client()
    if client is None:
        return None
    response = client.table(USERS_TABLE).select("*").eq("id", user_id).limit(1).execute()
    rows = response.data or []
    return rows[0] if rows else None


def upsert_user_profile(
    user_id: str,
    email: str,
    full_name: str,
    preferred_sport: str,
    preferred_goal: str,
    region: str,
) -> None:
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase is not configured.")

    payload = {
        "id": user_id,
        "email": email,
        "full_name": full_name,
        "preferred_sport": preferred_sport,
        "preferred_goal": preferred_goal,
        "region": region,
    }
    client.table(USERS_TABLE).upsert(payload).execute()


def sign_up_user(
    full_name: str,
    email: str,
    password: str,
    preferred_sport: str,
    preferred_goal: str,
    region: str,
) -> str:
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY first.")

    auth_response = client.auth.sign_up(
        {
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name,
                    "preferred_sport": preferred_sport,
                    "preferred_goal": preferred_goal,
                    "region": region,
                }
            },
        }
    )
    user = getattr(auth_response, "user", None)
    session = getattr(auth_response, "session", None)
    if user is None:
        raise RuntimeError("Supabase did not return a user for this sign-up request.")

    upsert_user_profile(user.id, email, full_name, preferred_sport, preferred_goal, region)

    if session is not None:
        set_current_user(
            {
                "id": user.id,
                "email": user.email,
                "full_name": full_name,
                "preferred_sport": preferred_sport,
                "preferred_goal": preferred_goal,
                "region": region,
            }
        )
        return "Account created and logged in."
    return "Account created, Log in from side panel"


def sign_in_user(email: str, password: str) -> None:
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY first.")

    auth_response = client.auth.sign_in_with_password({"email": email, "password": password})
    user = getattr(auth_response, "user", None)
    if user is None:
        raise RuntimeError("Login failed. Please check your email and password.")

    profile = fetch_user_profile(user.id) or {}
    metadata = getattr(user, "user_metadata", {}) or {}
    set_current_user(
        {
            "id": user.id,
            "email": user.email,
            "full_name": profile.get("full_name") or metadata.get("full_name", ""),
            "preferred_sport": profile.get("preferred_sport") or metadata.get("preferred_sport", SPORT_OPTIONS[0]),
            "preferred_goal": profile.get("preferred_goal") or metadata.get("preferred_goal", GOAL_OPTIONS[0]),
            "region": profile.get("region") or metadata.get("region", REGION_OPTIONS[0]),
        }
    )


def sign_out_user() -> None:
    client = get_supabase_client()
    if client is not None:
        try:
            client.auth.sign_out()
        except Exception:
            pass
    set_current_user(None)


def generate_gemini_response(prompt: str) -> tuple[str, str]:
    api_key = get_api_key()
    if not api_key:
        #raise RuntimeError("Missing GEMINI_API_KEY. Add it to Streamlit secrets before generating recommendations.")
        raise RuntimeError("Check API key...")

    configured_model = get_secret_value("GEMINI_MODEL")
    candidate_models = [model for model in [configured_model, *MODEL_CANDIDATES] if model]
    client = genai.Client(api_key=api_key)
    last_error: Exception | None = None

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            text = getattr(response, "text", "") or ""
            if text.strip():
                return text.strip(), model_name
            last_error = RuntimeError(f"Empty response from model '{model_name}'.")
        except Exception as exc:
            last_error = exc

    if last_error is None:
        last_error = RuntimeError("Gemini did not return any content.")
    raise RuntimeError(str(last_error))


def build_meal_plan_prompt(profile: dict[str, str | int | float], targets: dict[str, str | int | float]) -> str:
    restrictions = profile["restrictions"] or "None"
    include_foods = profile["include_foods"] or "None"
    avoid_foods = profile["avoid_foods"] or "None"
    cuisine_notes = profile["cuisine_notes"] or "No special cuisine preference"
    health_focus = profile["health_focus"] or "None"
    extra_notes = profile["extra_notes"] or "None"

    return dedent(
        f"""
        You are an expert sports nutritionist creating a practical athlete meal plan.

        Athlete profile:
        - Name: {profile["name"] or "Athlete"}
        - Age: {profile["age"]}
        - Gender: {profile["gender"]}
        - Height: {profile["height_cm"]} cm
        - Weight: {profile["weight_kg"]} kg
        - BMI: {profile["bmi"]:.1f} ({profile["bmi_category"]})
        - Region: {profile["region"]}
        - Religion: {profile["religion"]}
        - Diet style: {profile["diet"]}
        - Dietary restrictions: {restrictions}
        - Health focus: {health_focus}
        - Sport type: {profile["sport_type"]}
        - Training intensity: {profile["training_intensity"]}
        - Training days per week: {profile["training_days"]}
        - Session duration: {profile["session_duration_mins"]} minutes
        - Plan type: {profile["plan_type"]}
        - Primary goal: {profile["goal"]}
        - Preferred cuisines or foods: {cuisine_notes}
        - Foods to include: {include_foods}
        - Foods to avoid: {avoid_foods}
        - Extra notes: {extra_notes}

        Daily target context:
        - Calories: about {targets["calories"]} kcal
        - Protein: about {targets["protein_g"]} g
        - Fat: about {targets["fat_g"]} g
        - Carbohydrates: {targets["carb_range"]}
        - Hydration: about {targets["hydration_liters"]} liters

        Build a one-day athlete meal plan.

        Output rules:
        - Use markdown.
        - Be concise and easy to follow.
        - Use only foods that fit the diet style, religion, and restrictions.
        - Keep foods practical and commonly available in {profile["region"]} India when possible.
        - Include portion sizes in grams, cups, tablespoons, or pieces.
        - Include rough calories and a compact macro note for each meal.
        - Make the advice athlete-focused for training, performance, recovery, and hydration.
        - If the health focus is not None, adapt the meals to support that condition.
        - Do not mention datasets, machine learning, or that you are an AI.

        Use exactly these sections in this order:
        ## Athlete Summary
        ## Daily Targets
        ## Breakfast
        ## Pre-Workout
        ## Lunch
        ## Post-Workout
        ## Dinner
        ## Optional Snack
        ## Hydration
        ## Foods To Prioritize
        ## Foods To Limit
        ## Why This Plan Fits

        Inside each section, use short bullet points only.
        """
    ).strip()


def build_food_analyzer_prompt(food_item: str, serving_size: str, sport_type: str, goal: str) -> str:
    return dedent(
        f"""
        Analyze the food item below for athlete nutrition.

        Food: {food_item}
        Serving size: {serving_size}
        Athlete type: {sport_type}
        Goal: {goal}

        Output rules:
        - Use markdown.
        - Use short bullet points only.
        - Keep it concise and practical.

        Use exactly these sections:
        ## Estimated Nutrition
        ## Performance Benefits
        ## Best Timing
        ## Cautions
        """
    ).strip()


def save_plan_to_supabase(
    user_id: str,
    user_email: str,
    profile: dict[str, str | int | float],
    targets: dict[str, str | int | float],
    model_name: str,
    recommendation: str,
) -> str:
    client = get_supabase_client()
    if client is None:
        return "Supabase is not configured. Skipped database save."

    payload = {
        "user_id": user_id,
        "user_email": user_email,
        "name": profile["name"] or "Athlete",
        "age": int(profile["age"]),
        "gender": profile["gender"],
        "height_cm": float(profile["height_cm"]),
        "weight_kg": float(profile["weight_kg"]),
        "bmi": float(profile["bmi"]),
        "bmi_category": profile["bmi_category"],
        "region": profile["region"],
        "religion": profile["religion"],
        "diet": profile["diet"],
        "restrictions": profile["restrictions"] or "None",
        "health_focus": profile["health_focus"] or "None",
        "sport_type": profile["sport_type"],
        "goal": profile["goal"],
        "plan_type": profile["plan_type"],
        "training_intensity": profile["training_intensity"],
        "training_days": int(profile["training_days"]),
        "session_duration_mins": int(profile["session_duration_mins"]),
        "include_foods": profile["include_foods"] or "None",
        "avoid_foods": profile["avoid_foods"] or "None",
        "cuisine_notes": profile["cuisine_notes"] or "None",
        "extra_notes": profile["extra_notes"] or "None",
        "target_calories": int(targets["calories"]),
        "target_protein_g": int(targets["protein_g"]),
        "target_fat_g": int(targets["fat_g"]),
        "target_carb_range": str(targets["carb_range"]),
        "target_hydration_liters": float(targets["hydration_liters"]),
        "gemini_model": model_name,
        "recommendation": recommendation,
    }
    client.table(PLANS_TABLE).insert(payload).execute()
    return "Recommendation saved to Supabase."


def fetch_user_recommendations(user_id: str, limit: int = 8) -> list[dict]:
    client = get_supabase_client()
    if client is None:
        return []

    response = (
        client.table(PLANS_TABLE)
        .select("id, created_at, sport_type, goal, gemini_model, recommendation")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def render_sidebar() -> None:
    st.sidebar.title("Setup")
    if get_api_key():
        st.sidebar.success("Gemini API key detected.")
    else:
        st.sidebar.warning("Add `GEMINI_API_KEY` in Streamlit secrets to enable recommendations.")

    if get_supabase_client() is not None:
        st.sidebar.success("Supabase credentials detected.")
    else:
        st.sidebar.warning("Add `SUPABASE_URL` and `SUPABASE_KEY` in Streamlit secrets to enable login and saves.")

    current_user = get_current_user()
    if current_user:
        st.sidebar.markdown(f"**Signed in as:** {current_user.get('email', 'Unknown user')}")
        if st.sidebar.button("Log out"):
            sign_out_user()
            st.rerun()

    st.sidebar.markdown("**Streamlit Community Cloud secrets**")
    st.sidebar.code(
        'GEMINI_API_KEY = "your-api-key"\n'
        'SUPABASE_URL = "https://your-project.supabase.co"\n'
        'SUPABASE_KEY = "your-supabase-anon-key"\n'
        '# Optional\n'
        'GEMINI_MODEL = "gemini-2.5-flash"',
        language="toml",
    )

    with st.sidebar.expander("Suggested Supabase tables"):
        st.code(
            "create table app_users (\n"
            "  id uuid primary key references auth.users(id) on delete cascade,\n"
            "  email text unique,\n"
            "  full_name text,\n"
            "  preferred_sport text,\n"
            "  preferred_goal text,\n"
            "  region text,\n"
            "  created_at timestamptz default now()\n"
            ");\n\n"
            "create table athlete_recommendations (\n"
            "  id bigint generated always as identity primary key,\n"
            "  user_id uuid references app_users(id) on delete cascade,\n"
            "  user_email text,\n"
            "  created_at timestamptz default now(),\n"
            "  name text,\n"
            "  age int,\n"
            "  gender text,\n"
            "  height_cm numeric,\n"
            "  weight_kg numeric,\n"
            "  bmi numeric,\n"
            "  bmi_category text,\n"
            "  region text,\n"
            "  religion text,\n"
            "  diet text,\n"
            "  restrictions text,\n"
            "  health_focus text,\n"
            "  sport_type text,\n"
            "  goal text,\n"
            "  plan_type text,\n"
            "  training_intensity text,\n"
            "  training_days int,\n"
            "  session_duration_mins int,\n"
            "  include_foods text,\n"
            "  avoid_foods text,\n"
            "  cuisine_notes text,\n"
            "  extra_notes text,\n"
            "  target_calories int,\n"
            "  target_protein_g int,\n"
            "  target_fat_g int,\n"
            "  target_carb_range text,\n"
            "  target_hydration_liters numeric,\n"
            "  gemini_model text,\n"
            "  recommendation text\n"
            ");",
            language="sql",
        )

    st.sidebar.caption("This app gives athlete-focused food suggestions and is not a medical diagnosis tool.")


def render_auth_screen() -> bool:
    if get_current_user() is not None:
        return True

    st.subheader("Login or create your account")
    st.write("Use Supabase authentication so each athlete has a personal profile and saved recommendation history.")

    if get_supabase_client() is None:
        st.error("Supabase is not configured. Add `SUPABASE_URL` and `SUPABASE_KEY` in Streamlit secrets.")
        return False

    login_tab, signup_tab = st.tabs(["Login", "Create Account"])

    with login_tab:
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submitted = st.form_submit_button("Login")

        if login_submitted:
            try:
                sign_in_user(login_email.strip(), login_password)
                st.success("Logged in successfully.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with signup_tab:
        with st.form("signup_form"):
            full_name = st.text_input("Full name")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_password")
            preferred_sport = st.selectbox("Preferred sport type", SPORT_OPTIONS)
            preferred_goal = st.selectbox("Preferred goal", GOAL_OPTIONS)
            preferred_region = st.selectbox("Region", REGION_OPTIONS)
            signup_submitted = st.form_submit_button("Create account")

        if signup_submitted:
            if not full_name.strip() or not signup_email.strip() or not signup_password:
                st.error("Name, email, and password are required.")
            else:
                try:
                    message = sign_up_user(
                        full_name.strip(),
                        signup_email.strip(),
                        signup_password,
                        preferred_sport,
                        preferred_goal,
                        preferred_region,
                    )
                    st.success(message)
                    if get_current_user() is not None:
                        st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    return False


def render_planner_tab() -> None:
    current_user = get_current_user() or {}
    st.subheader("Build an athlete meal plan")
    #st.write("Fill in the athlete profile and Gemini will generate a performance-focused food recommendation plan.")

    default_sport = current_user.get("preferred_sport", SPORT_OPTIONS[0])
    default_goal = current_user.get("preferred_goal", GOAL_OPTIONS[0])
    default_region = current_user.get("region", REGION_OPTIONS[0])

    with st.form("athlete_plan_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value=current_user.get("full_name", ""))
            age = st.number_input("Age", min_value=12, max_value=80, value=25, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=170.0, step=0.5)
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
            region = st.selectbox(
                "Region",
                REGION_OPTIONS,
                index=REGION_OPTIONS.index(default_region) if default_region in REGION_OPTIONS else 0,
            )
            religion = st.selectbox("Religion", RELIGION_OPTIONS)
            diet = st.selectbox("Diet style", DIET_OPTIONS)
        with col2:
            sport_type = st.selectbox(
                "Sport type",
                SPORT_OPTIONS,
                index=SPORT_OPTIONS.index(default_sport) if default_sport in SPORT_OPTIONS else 0,
            )
            goal = st.selectbox(
                "Primary goal",
                GOAL_OPTIONS,
                index=GOAL_OPTIONS.index(default_goal) if default_goal in GOAL_OPTIONS else 0,
            )
            plan_type = st.selectbox("Plan type", PLAN_OPTIONS)
            training_intensity = st.selectbox("Training intensity", INTENSITY_OPTIONS)
            training_days = st.slider("Training days per week", min_value=1, max_value=7, value=5)
            session_duration_mins = st.slider(
                "Average session duration (mins)",
                min_value=30,
                max_value=240,
                value=90,
                step=15,
            )
            health_choice = st.selectbox("Health focus", HEALTH_OPTIONS)
            custom_health_focus = st.text_input("Custom health focus", placeholder="Optional")
            restrictions = st.multiselect("Dietary restrictions", RESTRICTION_OPTIONS)

        include_foods = st.text_input("Foods to include", placeholder="Example: paneer, oats, banana")
        avoid_foods = st.text_input("Foods to avoid", placeholder="Example: fried foods, soda")
        cuisine_notes = st.text_input("Cuisine preference", placeholder="Example: South Indian, simple home meals")
        extra_notes = st.text_area(
            "Extra notes",
            placeholder="Example: early-morning training, hostel food, budget-friendly meals",
            height=100,
        )
        submitted = st.form_submit_button("Generate athlete recommendation")

    if not submitted:
        return

    health_focus = custom_health_focus.strip() if health_choice == "Other" else health_choice
    if not health_focus:
        health_focus = "None"

    bmi = calculate_bmi(height_cm, weight_kg)
    bmi_category = get_bmi_category(bmi)
    targets = estimate_daily_targets(
        age=age,
        gender=gender,
        height_cm=height_cm,
        weight_kg=weight_kg,
        sport_type=sport_type,
        training_intensity=training_intensity,
        training_days=training_days,
        session_duration_mins=session_duration_mins,
        goal=goal,
    )

    profile = {
        "name": name,
        "age": age,
        "gender": gender,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "bmi_category": bmi_category,
        "region": region,
        "religion": religion,
        "diet": diet,
        "restrictions": ", ".join(restrictions),
        "sport_type": sport_type,
        "goal": goal,
        "plan_type": plan_type,
        "training_intensity": training_intensity,
        "training_days": training_days,
        "session_duration_mins": session_duration_mins,
        "health_focus": health_focus,
        "include_foods": include_foods,
        "avoid_foods": avoid_foods,
        "cuisine_notes": cuisine_notes,
        "extra_notes": extra_notes,
    }

    metric_cols = st.columns(4)
    metric_cols[0].metric("BMI", f"{bmi:.1f}")
    metric_cols[1].metric("BMI Band", bmi_category)
    metric_cols[2].metric("Target Calories", f"{targets['calories']} kcal")
    metric_cols[3].metric("Protein Target", f"{targets['protein_g']} g")

    prompt = build_meal_plan_prompt(profile, targets)

    try:
        with st.spinner("Classifying foods based on mentioned params..."):
            response, model_name = generate_gemini_response(prompt)
    except Exception as exc:
        st.error(str(exc))
        return

    #st.success(f"Recommendation generated with {model_name}.")
    st.markdown(response)
    st.download_button(
        "Download meal plan",
        data=response,
        file_name="athlete_meal_plan.md",
        mime="text/markdown",
    )

    try:
        save_message = save_plan_to_supabase(
            current_user["id"],
            current_user.get("email", ""),
            profile,
            targets,
            model_name,
            response,
        )
        if "saved" in save_message.lower():
            st.info(save_message)
        else:
            st.caption(save_message)
    except Exception as exc:
        st.warning(f"Gemini worked, but saving to Supabase failed: {exc}")


def render_food_analyzer_tab() -> None:
    current_user = get_current_user() or {}
    st.subheader("Analyze a food item")
    #st.write("Use Gemini to break down a food item for athlete performance, meal timing, and caution points.")

    default_sport = current_user.get("preferred_sport", SPORT_OPTIONS[0])
    default_goal = current_user.get("preferred_goal", GOAL_OPTIONS[0])

    with st.form("food_analyzer_form"):
        food_item = st.text_input("Food item", placeholder="Example: peanut butter banana smoothie")
        serving_size = st.text_input("Serving size", value="1 serving")
        sport_type = st.selectbox(
            "Athlete type for context",
            SPORT_OPTIONS,
            index=SPORT_OPTIONS.index(default_sport) if default_sport in SPORT_OPTIONS else 0,
            key="analyzer_sport_type",
        )
        goal = st.selectbox(
            "Goal for context",
            GOAL_OPTIONS,
            index=GOAL_OPTIONS.index(default_goal) if default_goal in GOAL_OPTIONS else 0,
            key="analyzer_goal",
        )
        submitted = st.form_submit_button("Analyze food")

    if not submitted:
        return
    if not food_item.strip():
        st.error("Enter a food item to analyze.")
        return

    prompt = build_food_analyzer_prompt(food_item.strip(), serving_size.strip() or "1 serving", sport_type, goal)

    try:
        with st.spinner("Analyzing food..."):
            response, model_name = generate_gemini_response(prompt)
    except Exception as exc:
        st.error(str(exc))
        return

    st.success(f"Analysis generated with {model_name}.")
    st.markdown(response)


def render_history_tab() -> None:
    current_user = get_current_user() or {}
    st.subheader("Saved recommendations")
    records = fetch_user_recommendations(current_user["id"])
    if not records:
        st.caption("No saved recommendations yet.")
        return

    for record in records:
        heading = (
            f"{record.get('created_at', 'Saved plan')} | "
            f"{record.get('sport_type', 'Unknown sport')} | "
            f"{record.get('goal', 'Goal not set')}"
        )
        with st.expander(heading):
            st.caption(f"Generated with {record.get('gemini_model', 'Gemini')}")
            st.markdown(record.get("recommendation", ""))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    #st.caption("Gemini-powered athlete meal planning with Supabase login, profile storage, and saved plan history.")

    render_sidebar()

    if not render_auth_screen():
        return

    planner_tab, analyzer_tab, history_tab = st.tabs(["Athlete Planner", "Food Analyzer", "Saved Plans"])
    with planner_tab:
        render_planner_tab()
    with analyzer_tab:
        render_food_analyzer_tab()
    with history_tab:
        render_history_tab()


if __name__ == "__main__":
    main()
















# import os
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import streamlit as st
# from google import genai
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score, hamming_loss
# from sklearn.model_selection import train_test_split
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sqlalchemy import text

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
# try:
#     conn = st.connection('my_database', type='sql')
# except Exception:
#     conn = None

# # --- GEMINI SETUP ---
# MODEL_NAME = "gemini-2.5-flash" # Updated to standard flash model name for better compatibility

# def get_api_key() -> str:
#     try:
#         secret_key = st.secrets.get("GEMINI_API_KEY", "")
#     except Exception:
#         secret_key = ""
#     env_key = os.getenv("GEMINI_API_KEY", "")
#     return (secret_key or env_key or "").strip()

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

# def get_bmi_category(bmi: float) -> str:
#     if bmi < 18.5: return "underweight"
#     if bmi < 25: return "normal"
#     if bmi < 30: return "overweight"
#     return "obese"

# # --- UI INPUTS ---
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

# # --- AI ANALYSIS & DATABASE SAVE ---
# if st.button("Analyze and Suggest Food"):
    
#     athlete_info = ""
#     if is_athlete == "Yes":
#         athlete_info = (
#             f"The user is an athlete in '{sport_type}' with {training_intensity} training. "
#             "Prioritize performance, recovery, hydration, and protein quality."
#         )

#     # We now ask Gemini to act as the ML model and recommend the foods itself
#     input_prompt = f"""Act as an expert sports nutritionist. Based on these user details:
# Name: {name}
# Age: {age}
# Gender: {gender}
# BMI: {bmi:.2f} ({bmi_category})
# Religion: {religion}
# Region: {region}
# Diet: {diet}
# Restrictions: {', '.join(restrictions) or 'None'}
# Health: {health_condition or 'None'}
# {athlete_info}

# First, list 5-7 highly recommended specific whole foods for this exact profile.
# Then, create a concise meal plan with exactly these sections:
# - Breakfast
# - Lunch
# - Dinner
# - Snacks

# Rules:
# - Use only foods that strictly fit the user's diet, religion, and restrictions.
# - Include rough portions in grams, cups, pieces, or tablespoons.
# - Add a compact nutrient note for each meal: protein, carbs, fats.
# - Mention how the plan supports their BMI goals and athlete needs if relevant.
# - Keep it easy to read and brief.
# - Use bullet points only. No paragraphs.
# """

#     with st.spinner("Gemini is analyzing your profile..."):
#         response = get_gemini_response(input_prompt)

#     if response and not response.startswith("Error:"):
#         st.write(response)
        
#         # --- DATABASE SAVE LOGIC ---
#         if 'conn' in locals() and conn is not None:
#             try:
#                 with conn.session as s:
#                     s.execute(text("""
#                         CREATE TABLE IF NOT EXISTS user_nutrition_logs (
#                             name VARCHAR(255),
#                             age INT,
#                             bmi FLOAT,
#                             diet VARCHAR(100),
#                             is_athlete VARCHAR(10),
#                             ai_response TEXT
#                         );
#                     """))
                    
#                     s.execute(text("""
#                         INSERT INTO user_nutrition_logs (name, age, bmi, diet, is_athlete, ai_response) 
#                         VALUES (:name, :age, :bmi, :diet, :is_athlete, :ai_response);
#                     """), 
#                     {
#                         "name": name, 
#                         "age": age, 
#                         "bmi": bmi, 
#                         "diet": diet,
#                         "is_athlete": is_athlete, 
#                         "ai_response": response
#                     })
                    
#                     s.commit()
#                 st.success("✅ Results securely logged to the database!")
#             except Exception as e:
#                 st.warning(f"⚠️ App ran successfully, but could not save to database: {e}")

#     else:
#         st.error(response or "Failed to generate suggestions.")

# # --- FOOD NUTRIENT ANALYZER ---
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
#     with st.spinner("Analyzing nutrients..."):
#         response = get_gemini_response(input_prompt2)
#     if response and not response.startswith("Error:"):
#         st.write(response)
#     else:
#         st.error(response or "Failed to analyze.")
























# # st.set_page_config(page_title="Health & Athlete Nutrition Checker", page_icon="🏋️")
# # st.title("🏋️ Health, Athlete, and Nutrients Checker")
# # # --- AUTHENTICATION ---
# # def check_password():
# #     def password_entered():
# #         if st.session_state["password"] == st.secrets["app_password"]:
# #             st.session_state["password_correct"] = True
# #             del st.session_state["password"]  # don't store password
# #         else:
# #             st.session_state["password_correct"] = False

# #     if "password_correct" not in st.session_state:
# #         st.text_input("Enter the app password to access:", type="password", on_change=password_entered, key="password")
# #         return False
# #     elif not st.session_state["password_correct"]:
# #         st.text_input("Enter the app password to access:", type="password", on_change=password_entered, key="password")
# #         st.error("😕 Password incorrect")
# #         return False
# #     return True

# # if not check_password():
# #     st.stop()

# # # --- DATABASE SETUP ---
# # # This initializes the connection using Streamlit's Secrets
# # try:
# #     conn = st.connection('my_database', type='sql')
# # except Exception:
# #     conn = None
# # DATA_PATH = Path("Data") / "user_nutrition_dataset.csv"
# # MODEL_NAME = "gemini-2.5-flash-lite"


# # def get_bmi_category(bmi: float) -> str:
# #     if bmi < 18.5:
# #         return "underweight"
# #     if bmi < 25:
# #         return "normal"
# #     if bmi < 30:
# #         return "overweight"
# #     return "obese"


# # def get_api_key() -> str:
# #     try:
# #         secret_key = st.secrets.get("GEMINI_API_KEY", "")
# #     except Exception:
# #         secret_key = ""
# #     env_key = os.getenv("GEMINI_API_KEY", "")
# #     api_key = (secret_key or env_key or "").strip()
# #     return api_key


# # @st.cache_resource
# # def load_and_train_model():
# #     if not DATA_PATH.exists():
# #         raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# #     df = pd.read_csv(DATA_PATH)

# #     required_cols = {
# #         "age",
# #         "gender",
# #         "bmi",
# #         "region",
# #         "diet",
# #         "religion",
# #         "bmi_category",
# #         "is_athlete",
# #         "sport_type",
# #         "training_intensity",
# #         "health_condition",
# #         "dietary_restrictions",
# #         "is_recommended",
# #         "recommended_foods",
# #     }
# #     missing = required_cols - set(df.columns)
# #     if missing:
# #         raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

# #     features = [
# #         "age",
# #         "gender",
# #         "bmi",
# #         "region",
# #         "diet",
# #         "religion",
# #         "bmi_category",
# #         "is_athlete",
# #         "sport_type",
# #         "training_intensity",
# #         "health_condition",
# #         "dietary_restrictions",
# #     ]
# #     categorical_features = [
# #         "gender",
# #         "region",
# #         "diet",
# #         "religion",
# #         "bmi_category",
# #         "is_athlete",
# #         "sport_type",
# #         "training_intensity",
# #         "health_condition",
# #         "dietary_restrictions",
# #     ]
# #     numeric_features = ["age", "bmi"]

# #     df = df.copy()
# #     df[categorical_features] = df[categorical_features].fillna("Unknown").astype(str)
# #     df["recommended_foods"] = df["recommended_foods"].fillna("").astype(str)

# #     X = df[features].copy()
# #     y_bin = df["is_recommended"].astype(int)

# #     y_ml_raw = (
# #         df["recommended_foods"]
# #         .str.split(",")
# #         .apply(lambda items: [item.strip() for item in items if item and item.strip()])
# #     )
# #     mlb = MultiLabelBinarizer()
# #     y_ml = mlb.fit_transform(y_ml_raw)

# #     idx = np.arange(len(df))
# #     train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=42, shuffle=True)

# #     X_train = X.iloc[train_idx].reset_index(drop=True)
# #     X_test = X.iloc[test_idx].reset_index(drop=True)
# #     y_train_bin = y_bin.iloc[train_idx].reset_index(drop=True)
# #     y_test_bin = y_bin.iloc[test_idx].reset_index(drop=True)
# #     y_train_ml = y_ml[train_idx]
# #     y_test_ml = y_ml[test_idx]

# #     preprocessor = ColumnTransformer(
# #         transformers=[
# #             ("num", "passthrough", numeric_features),
# #             ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
# #         ],
# #         remainder="drop",
# #     )

# #     rf_params = dict(
# #         n_estimators=150,
# #         random_state=42,
# #         n_jobs=-1,
# #         class_weight="balanced",
# #         max_depth=6,
# #         min_samples_split=4,
# #     )

# #     rf_bin = Pipeline(
# #         steps=[
# #             ("preprocessor", preprocessor),
# #             ("model", RandomForestClassifier(**rf_params)),
# #         ]
# #     )
# #     rf_bin.fit(X_train, y_train_bin)

# #     rf_multi = Pipeline(
# #         steps=[
# #             ("preprocessor", preprocessor),
# #             ("model", MultiOutputClassifier(RandomForestClassifier(**rf_params))),
# #         ]
# #     )
# #     rf_multi.fit(X_train, y_train_ml)

# #     # Optional metrics for validation in the app logs / console.
# #     y_pred_bin = rf_bin.predict(X_test)
# #     y_pred_ml = rf_multi.predict(X_test)
# #     _ = {
# #         "binary_accuracy": accuracy_score(y_test_bin, y_pred_bin),
# #         "binary_f1": f1_score(y_test_bin, y_pred_bin, zero_division=0),
# #         "multi_hamming_loss": hamming_loss(y_test_ml, y_pred_ml),
# #     }

# #     return rf_bin, rf_multi, mlb


# # rf_bin, rf_multi, mlb = None, None, None
# # try:
# #     rf_bin, rf_multi, mlb = load_and_train_model()
# # except Exception as exc:
# #     st.error(f"Model loading/training failed: {exc}")


# # def get_gemini_response(input_prompt: str) -> str:
# #     api_key = get_api_key()
# #     if not api_key:
# #         return "Error: Missing GEMINI_API_KEY in Streamlit secrets or environment variables."
# #     try:
# #         client = genai.Client(api_key=api_key)
# #         response = client.models.generate_content(
# #             model=MODEL_NAME,
# #             contents=input_prompt,
# #         )
# #         return getattr(response, "text", None) or "Error: Empty response from Gemini."
# #     except Exception as e:
# #         return f"Error: {e}"


# # # Basic user details
# # name = st.text_input("Name:")
# # age = st.number_input("Age:", min_value=1, max_value=120, step=1)
# # gender = st.selectbox("Gender:", ["Male", "Female", "Other"])
# # height = st.number_input("Height (cm):", min_value=50.0, step=0.1)
# # weight = st.number_input("Weight (kg):", min_value=1.0, step=0.1)
# # religion = st.selectbox("Religion:", ["Hindu", "Muslim", "Christian", "Other"])
# # region = st.selectbox("Region:", ["North", "South", "East", "West"])
# # diet = st.selectbox("Primary Diet:", ["Vegetarian", "Non-Vegetarian", "Vegan"])
# # restrictions = st.multiselect(
# #     "Dietary Restrictions:",
# #     ["Jain", "Gluten-Free", "Dairy-Free", "Egg-Free", "No Onion/Garlic"],
# # )

# # bmi = 0.0
# # if height > 0:
# #     bmi = weight / ((height / 100) ** 2)
# # bmi_category = get_bmi_category(bmi)

# # is_athlete = st.selectbox("Are you an athlete?", ["No", "Yes"])
# # sport_type, training_intensity = "Unknown", "Unknown"
# # if is_athlete == "Yes":
# #     sport_type = st.selectbox(
# #         "Type of Sport:",
# #         [
# #             "Endurance (e.g., Marathon, Cycling)",
# #             "Strength (e.g., Weightlifting, Wrestling)",
# #             "Mixed (e.g., Football, Basketball)",
# #         ],
# #     )
# #     training_intensity = st.selectbox("Training Intensity:", ["Light", "Moderate", "Intense"])

# # health_condition = st.text_input("Health Condition (e.g., Diabetes, High BP, None):")


# # def preprocess_user_input(
# #     age,
# #     gender,
# #     bmi,
# #     region,
# #     diet,
# #     religion,
# #     bmi_category,
# #     is_athlete,
# #     sport_type,
# #     training_intensity,
# #     health_condition,
# #     dietary_restrictions,
# # ):
# #     return pd.DataFrame(
# #         {
# #             "age": [age],
# #             "bmi": [bmi],
# #             "gender": [str(gender)],
# #             "region": [str(region)],
# #             "diet": [str(diet)],
# #             "religion": [str(religion)],
# #             "bmi_category": [str(bmi_category)],
# #             "is_athlete": [str(is_athlete)],
# #             "sport_type": [str(sport_type)],
# #             "training_intensity": [str(training_intensity)],
# #             "health_condition": [str(health_condition or "None")],
# #             "dietary_restrictions": [", ".join(dietary_restrictions) if dietary_restrictions else "None"],
# #         }
# #     )


# # if st.button("Analyze and Suggest Food"):
# #     if rf_bin is None or rf_multi is None or mlb is None:
# #         st.stop()

# #     user_input = preprocess_user_input(
# #         age,
# #         gender,
# #         bmi,
# #         region,
# #         diet,
# #         religion,
# #         bmi_category,
# #         is_athlete,
# #         sport_type,
# #         training_intensity,
# #         health_condition,
# #         restrictions,
# #     )

# #     is_recommended = int(rf_bin.predict(user_input)[0])
# #     ml_pred = rf_multi.predict(user_input)
# #     predicted_foods_binary = mlb.inverse_transform(ml_pred)[0]
# #     predicted_foods_str = ", ".join(predicted_foods_binary) if predicted_foods_binary else ""

# #     athlete_info = ""
# #     if is_athlete == "Yes":
# #         athlete_info = (
# #             f"The user is an athlete in '{sport_type}' with {training_intensity} training. "
# #             "Prioritize performance, recovery, hydration, and protein quality."
# #         )

# #     food_guidance = (
# #         f"Recommended foods from ML model: {predicted_foods_str}."
# #         if predicted_foods_str
# #         else "No specific foods were predicted by the ML model."
# #     )

# #     input_prompt = f"""Based on these user details:
# # Name: {name}
# # Age: {age}
# # Gender: {gender}
# # BMI: {bmi:.2f}
# # BMI Category: {bmi_category}
# # Religion: {religion}
# # Region: {region}
# # Diet: {diet}
# # Restrictions: {', '.join(restrictions) or 'None'}
# # Health: {health_condition or 'None'}
# # {athlete_info}
# # {food_guidance}

# # Create a concise meal plan with exactly these sections:
# # - Breakfast
# # - Lunch
# # - Dinner
# # - Snacks

# # Rules:
# # - Prefer the ML-predicted foods when available.
# # - Use only foods that fit the user's diet, religion, and restrictions.
# # - Include rough portions in grams, cups, pieces, or tablespoons.
# # - Add a compact nutrient note for each meal: protein, carbs, fats, vitamins, minerals.
# # - Mention how the plan supports BMI goals and athlete needs if relevant.
# # - Keep it easy to read and brief.
# # - Use bullet points only. No paragraphs.
# # """
# # response = get_gemini_response(input_prompt)

# # if response and not response.startswith("Error:"):
# #     st.write(response)

# #     # --- DATABASE SAVE LOGIC ---
# #     if 'conn' in locals() and conn is not None:
# #         try:
# #             from sqlalchemy import text

# #             with conn.session as s:
# #                 # 1. Auto-create the table if it doesn't exist yet
# #                 s.execute(text("""
# #                         CREATE TABLE IF NOT EXISTS user_nutrition_logs (
# #                             name VARCHAR(255),
# #                             age INT,
# #                             bmi FLOAT,
# #                             diet VARCHAR(100),
# #                             is_athlete VARCHAR(10),
# #                             ai_response TEXT
# #                         );
# #                     """))

# #                 # 2. Insert the user's inputs and the AI's response safely
# #                 s.execute(text("""
# #                         INSERT INTO user_nutrition_logs (name, age, bmi, diet, is_athlete, ai_response) 
# #                         VALUES (:name, :age, :bmi, :diet, :is_athlete, :ai_response);
# #                     """),
# #                           {
# #                               "name": name,
# #                               "age": age,
# #                               "bmi": bmi,
# #                               "diet": diet,
# #                               "is_athlete": is_athlete,
# #                               "ai_response": response
# #                           })

# #                 s.commit()
# #             st.success("✅ Results securely logged to the database!")
# #         except Exception as e:
# #             st.warning(f"⚠️ App ran successfully, but could not save to database: {e}")

# # else:
# #     st.error(response or "Failed to generate suggestions.")
# #     response = get_gemini_response(input_prompt)
# #
# #     if response and not response.startswith("Error:"):
# #         st.write(response)
# #     else:
# #         st.error(response or "Failed to generate suggestions.")
# #
# #
# # st.subheader("Food Nutrient Analyzer")
# # food_item = st.text_input("Food Item:")
# # food_weight = st.number_input("Weight (grams):", min_value=1.0, step=0.1)
# # if st.button("Analyze Nutrient Content"):
# #     input_prompt2 = f"""Analyze '{food_item}' ({food_weight}g).
# # Return concise bullet points only:
# # - proteins
# # - carbs
# # - fats
# # - vitamins
# # - minerals
# # - health contributions
# # - athlete benefits if relevant
# # """
# #     response = get_gemini_response(input_prompt2)
# #     if response and not response.startswith("Error:"):
# #         st.write(response)
# #     else:
# #         st.error(response or "Failed to analyze.")
