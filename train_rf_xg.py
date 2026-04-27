"""
Train RandomForest and XGBoost classifiers on the generated meal dataset.
Produces tuned models, evaluation metrics, and saved joblib files.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
import xgboost as xgb
from scipy.stats import randint as sp_randint, uniform
import warnings
warnings.filterwarnings("ignore")
RND = 42

# ------------------- LOAD DATA -------------------

df = pd.read_csv("Data\generated_meal_profiles.csv")
print("Loaded generated_meal_profiles.csv")

# ------------------- BASIC FEATURE ENGINEERING -------------------
# Ensure target numeric
df['is_recommended'] = df['is_recommended'].astype(int)

# Derived features that help prediction
df['num_recommended_items'] = df['recommended_foods'].astype(str).apply(
    lambda s: 0 if s.strip().strip('"') in ("", "Standard Balanced Meal") else s.count(',')+1
)
df['primary_is_generic'] = (df['food_name'].astype(str).str.lower() == 'generic meal').astype(int)

# Columns to use
categorical_cols = [
    'food_name','food_region','gender','age_group','region','diet','religion',
    'bmi_category','is_athlete','sport_type','training_intensity','health_condition'
]
numeric_cols = ['num_recommended_items','primary_is_generic']

# Fill missing and ensure strings
for c in categorical_cols:
    df[c] = df[c].fillna('Missing').astype(str)
df[numeric_cols] = df[numeric_cols].fillna(0)

# Consistent is_athlete values
df['is_athlete'] = df['is_athlete'].astype(str)

X = df[categorical_cols + numeric_cols]
y = df['is_recommended']

# ------------------- SPLIT -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RND
)

# ------------------- PREPROCESSOR -------------------
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer([
    ('cat', cat_pipe, categorical_cols),
    ('num', num_pipe, numeric_cols)
], remainder='drop')

# Helper to extract feature names after preprocessing
def get_feature_names(preproc):
    cat_names = preproc.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(categorical_cols)
    num_names = numeric_cols
    return list(cat_names) + list(num_names)

# ------------------- RANDOM FOREST PIPELINE + TUNING -------------------
rf_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(random_state=RND, n_jobs=-1))
])

rf_param_dist = {
    'clf__n_estimators': [100, 200, 400],
    'clf__max_depth': [None, 8, 16, 24],
    'clf__max_features': ['sqrt', 'log2', 0.5],
    'clf__min_samples_split': sp_randint(2, 11),
    'clf__class_weight': [None, 'balanced']
}

rf_search = RandomizedSearchCV(
    rf_pipe, rf_param_dist, n_iter=20, scoring='roc_auc',
    cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=RND),
    n_jobs=-1, random_state=RND, verbose=1
)
print("Tuning Random Forest...")
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print("RF best params:", rf_search.best_params_)

# ------------------- XGBOOST PIPELINE + TUNING -------------------
# Compute scale_pos_weight for imbalance (neg/pos)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos if pos > 0 else 1.0

xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RND,
    n_jobs=-1
)

xgb_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', xgb_clf)
])

xgb_param_dist = {
    'clf__n_estimators': [100, 200, 400],
    'clf__max_depth': sp_randint(3, 10),
    'clf__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'clf__subsample': uniform(0.6, 0.4),
    'clf__colsample_bytree': uniform(0.6, 0.4),
    'clf__reg_alpha': [0, 0.5, 1.0],
    'clf__scale_pos_weight': [scale_pos_weight]
}

xgb_search = RandomizedSearchCV(
    xgb_pipe, xgb_param_dist, n_iter=24, scoring='roc_auc',
    cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=RND),
    n_jobs=-1, random_state=RND, verbose=1
)
print("Tuning XGBoost...")
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print("XGB best params:", xgb_search.best_params_)

# ------------------- EVALUATION -------------------
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1] if hasattr(model.named_steps['clf'], "predict_proba") else None
    print(f"\n=== {name} EVALUATION ===")
    print("Classification report:")
    print(classification_report(y_test, preds, digits=4))
    if probs is not None:
        print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

evaluate("Random Forest (best)", best_rf, X_test, y_test)
evaluate("XGBoost (best)", best_xgb, X_test, y_test)

# ------------------- FEATURE IMPORTANCES (top 20) -------------------
def show_top_importances(pipeline, top_n=20):
    pre = pipeline.named_steps['pre']
    clf = pipeline.named_steps['clf']
    feat_names = get_feature_names(pre)
    importances = None
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_).ravel()
    else:
        print("No importances available for this estimator.")
        return
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(top_n)
    print(fi)

print("\nTop RF feature importances:")
show_top_importances(best_rf)
print("\nTop XGB feature importances:")
show_top_importances(best_xgb)

# ------------------- SAVE MODELS -------------------
joblib.dump(best_rf, "model_random_forest_best.joblib")
joblib.dump(best_xgb, "model_xgboost_best.joblib")
print("\nSaved: model_random_forest_best.joblib, model_xgboost_best.joblib")
