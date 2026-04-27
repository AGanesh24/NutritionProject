"""
Project: Train model and Streamlit inference wiring
Files included below in this single document (copy each part into separate files):

1) train_model.py  -> trains a pipeline and saves /mnt/data/best_model.pkl
2) streamlit_app.py -> Streamlit app that loads best_model.pkl, predicts recommended_food, then calls Gemini to produce human-friendly meal plan.

USAGE:
- Run training first: python train_model.py
  This will read /mnt/data/athlete_health_food_dataset.csv (created earlier) and write /mnt/data/best_model.pkl

- Run Streamlit app: streamlit run streamlit_app.py

Note: Replace GEMINI API key via environment variable GEMINI_API_KEY or set it in Streamlit secrets.

"""

# -------------------------------
# train_model.py
# -------------------------------

# Save this part as: train_model.py

import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_and_save(dataset_path='athlete_health_food_dataset.csv', out_path='best_model.pkl'):
    # 1. Load dataset
    df = pd.read_csv(dataset_path)

    # 2. Features and target
    X = df[['athlete_type', 'health_problem']]
    y = df['recommended_food']

    # 3. Encode target labels
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y)

    # 4. Preprocessing for categorical features
    cat_features = ['athlete_type', 'health_problem']
    cat_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_features),
        ], remainder='drop'
    )

    # 5. Pipeline (preprocessing + classifier)
    pipeline = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # 7. Fit
    pipeline.fit(X_train, y_train)

    # 8. Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred, zero_division=0))

    # 9. Save the pipeline and the target label encoder together
    saved = {
        'model': pipeline,
        'target_label_encoder': target_le,
        'feature_columns': cat_features
    }

    with open(out_path, 'wb') as f:
        pickle.dump(saved, f)

    print(f"Saved model object to: {out_path}")


if __name__ == '__main__':
    dataset_path = 'Data/athlete_health_food_dataset.csv'
    out_path = 'best_model.pkl'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please place athlete_health_food_dataset.csv there.")
    train_and_save(dataset_path, out_path)



