import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "results/feature_table.csv"
)

MODEL_DIR = Path(
    "models"
)

MODEL_DIR.mkdir(
    exist_ok=True
)


RANDOM_STATE = 42


print("\n========================================")
print("TRAINING FINAL ML MODELS")
print("========================================")


# ============================================================
# 1. LOAD FEATURE TABLE
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


# Features
X = df.drop(
    columns=["subject", "label"]
)


# Labels
y = df["label"].map({
    "control": 0,
    "condition": 1
})


print(f"\nSubjects: {len(df)}")
print(f"Features: {X.shape[1]}")

print("\nClass distribution:")
print(
    y.value_counts().sort_index()
)


# ============================================================
# 2. RANDOM FOREST
# ============================================================

print("\n========================================")
print("TRAINING RANDOM FOREST")
print("========================================")


rf_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_features="sqrt"
        )
    )
])


rf_model.fit(
    X,
    y
)


rf_path = (
    MODEL_DIR /
    "random_forest_model.joblib"
)


joblib.dump(
    rf_model,
    rf_path
)


print(
    f"Random Forest saved to: {rf_path}"
)


# ============================================================
# 3. XGBOOST
# ============================================================

print("\n========================================")
print("TRAINING XGBOOST")
print("========================================")


xgb_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])


xgb_model.fit(
    X,
    y
)


xgb_path = (
    MODEL_DIR /
    "xgboost_model.joblib"
)


joblib.dump(
    xgb_model,
    xgb_path
)


print(
    f"XGBoost saved to: {xgb_path}"
)


# ============================================================
# 4. SAVE FEATURE ORDER
# ============================================================

feature_order_path = (
    MODEL_DIR /
    "feature_order.txt"
)


with open(
    feature_order_path,
    "w"
) as file:

    for feature in X.columns:

        file.write(
            feature + "\n"
        )


print(
    f"Feature order saved to: {feature_order_path}"
)


# ============================================================
# 5. VERIFY MODELS
# ============================================================

print("\n========================================")
print("MODEL VERIFICATION")
print("========================================")


rf_loaded = joblib.load(
    rf_path
)

xgb_loaded = joblib.load(
    xgb_path
)


rf_predictions = rf_loaded.predict(
    X.head(3)
)

xgb_predictions = xgb_loaded.predict(
    X.head(3)
)


print(
    "Random Forest test predictions:",
    rf_predictions
)

print(
    "XGBoost test predictions:",
    xgb_predictions
)


print("\n========================================")
print("FINAL MODELS COMPLETE")
print("========================================")

print("\nCreated files:")

print(
    f"- {rf_path}"
)

print(
    f"- {xgb_path}"
)

print(
    f"- {feature_order_path}"
)

print(
    "\nModels were trained using all 55 subjects "
    "for dashboard inference."
)