import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, f1_score
import joblib

# Create Data folder if it doesn't exist
os.makedirs('Data', exist_ok=True)

# Generate synthetic realistic dataset
np.random.seed(42)  # For reproducibility
n_samples = 300

data = {
    'age': np.random.randint(18, 65, n_samples),
    'bmi': np.round(np.random.uniform(17, 36, n_samples), 1),
    'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    'diet': np.random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan'], n_samples, p=[0.5, 0.4, 0.1]),
    'religion': np.random.choice(['Hindu', 'Muslim', 'Christian', 'Other'], n_samples, p=[0.6, 0.2, 0.1, 0.1]),
    'is_athlete': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75]),
    'health_condition': np.random.choice(['None', 'Diabetes', 'High BP'], n_samples, p=[0.7, 0.15, 0.15]),
    'dietary_restrictions': np.random.choice(
        ['None', 'Jain', 'Gluten-Free', 'Dairy-Free', 'Egg-Free', 'No Onion/Garlic'], n_samples,
        p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05]),
}

df = pd.DataFrame(data)


# BMI category
def get_bmi_cat(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    else:
        return 'obese'


df['bmi_category'] = df['bmi'].apply(get_bmi_cat)

# Athlete details
athlete_mask = df['is_athlete'] == 'Yes'
df.loc[~athlete_mask, 'sport_type'] = 'Unknown'
df.loc[~athlete_mask, 'training_intensity'] = 'Unknown'
df.loc[athlete_mask, 'sport_type'] = np.random.choice([
    'Endurance (e.g., Marathon, Cycling)',
    'Strength (e.g., Weightlifting, Wrestling)',
    'Mixed (e.g., Football, Basketball)'
], sum(athlete_mask))
df.loc[athlete_mask, 'training_intensity'] = np.random.choice(['Light', 'Moderate', 'Intense'], sum(athlete_mask))

# Synthetic is_recommended (rules + noise to avoid 100% accuracy)
df['is_recommended'] = 1
bad_conditions = (df['bmi_category'] == 'obese') & (df['is_athlete'] == 'No') & (df['health_condition'] != 'None')
df.loc[bad_conditions, 'is_recommended'] = 0
df.loc[(df['bmi_category'] == 'underweight') & (np.random.rand(n_samples) > 0.3), 'is_recommended'] = 0
# ~15% random noise flip
noise_mask = np.random.rand(n_samples) < 0.15
df.loc[noise_mask, 'is_recommended'] = 1 - df['is_recommended']

# Recommended foods (rule-based + variation)
all_foods = ['Apples', 'Bananas', 'Oats', 'Rice', 'Chicken', 'Fish', 'Salmon', 'Eggs', 'Lentils', 'Chickpeas', 'Tofu',
             'Quinoa', 'Almonds', 'Nuts', 'Spinach', 'Broccoli', 'Kale', 'Tomatoes', 'Carrots', 'Yogurt', 'Milk',
             'Cheese', 'Sweet Potatoes', 'Berries', 'Whole Grains', 'Avocado']


def generate_recommended_foods(row):
    possible = all_foods.copy()
    selected = set(np.random.choice(possible, np.random.randint(4, 8), replace=False))

    if row['diet'] in ['Vegetarian', 'Vegan']:
        selected -= {'Chicken', 'Fish', 'Salmon', 'Eggs', 'Milk', 'Cheese'}
    if row['diet'] == 'Vegan':
        selected -= {'Yogurt'}
    if 'Dairy-Free' in row['dietary_restrictions']:
        selected -= {'Milk', 'Cheese', 'Yogurt'}
    if 'Egg-Free' in row['dietary_restrictions']:
        selected -= {'Eggs'}

    if row['is_athlete'] == 'Yes':
        protein = {'Quinoa', 'Lentils', 'Chickpeas', 'Tofu', 'Almonds', 'Nuts', 'Yogurt', 'Chicken', 'Fish', 'Eggs'}
        selected.update(np.random.choice(list(protein), 2, replace=False))

    if row['health_condition'] == 'Diabetes':
        selected.update({'Oats', 'Whole Grains', 'Berries', 'Almonds'})

    return ', '.join(sorted(selected))


df['recommended_foods'] = df.apply(generate_recommended_foods, axis=1)

# Save dataset
df.to_csv("Data/user_nutrition_dataset.csv", index=False)
print("Synthetic dataset generated and saved.")

# Training starts here
features = ['age', 'gender', 'bmi', 'region', 'diet', 'religion', 'bmi_category',
            'is_athlete', 'sport_type', 'training_intensity', 'health_condition', 'dietary_restrictions']
cat_cols = ['gender', 'region', 'diet', 'religion', 'bmi_category', 'is_athlete',
            'sport_type', 'training_intensity', 'health_condition', 'dietary_restrictions']

df[cat_cols] = df[cat_cols].fillna('Unknown')
df_encoded = pd.get_dummies(df[features], columns=cat_cols)
X = df_encoded

y_bin = df['is_recommended'].astype(int)

y_ml_raw = df['recommended_foods'].str.split(', ').apply(lambda x: [item.strip() for item in x])
mlb = MultiLabelBinarizer()
y_ml = mlb.fit_transform(y_ml_raw)

X_train, X_test, y_train_bin, y_test_bin, y_train_ml, y_test_ml = train_test_split(
    X, y_bin, y_ml, test_size=0.3, random_state=42, stratify=y_bin)

# Binary classifier (tuned for ~78-85% accuracy)
rf_bin = RandomForestClassifier(
    n_estimators=80, max_depth=8, min_samples_split=5, min_samples_leaf=2,
    class_weight='balanced', random_state=42
)
rf_bin.fit(X_train, y_train_bin)

# Multi-label classifier
rf_multi = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_split=5, random_state=42)
)
rf_multi.fit(X_train, y_train_ml)

# Metrics
y_pred_bin = rf_bin.predict(X_test)
print("\nBinary Accuracy:", round(accuracy_score(y_test_bin, y_pred_bin), 4))
print(classification_report(y_test_bin, y_pred_bin))

y_pred_ml = rf_multi.predict(X_test)
print("\nMulti-Label Exact Match:", round(accuracy_score(y_test_ml, y_pred_ml), 4))
print("Hamming Loss:", round(hamming_loss(y_test_ml, y_pred_ml), 4))
print("F1 Micro:", round(f1_score(y_test_ml, y_pred_ml, average='micro'), 4))

# Save models and artifacts (optional, for use in Streamlit)
joblib.dump(rf_bin, 'rf_bin_model.joblib')
joblib.dump(rf_multi, 'rf_multi_model.joblib')
joblib.dump(mlb, 'mlb_encoder.joblib')
joblib.dump(X.columns.tolist(), 'feature_columns.joblib')
joblib.dump(mlb.classes_.tolist(), 'mlb_classes.joblib')
