import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

INPUT_FILE = Path("results/feature_table.csv")

N_SPLITS = 5
RANDOM_STATE = 42


print("\n========================================")
print("RANDOM FOREST - GROUP K-FOLD")
print("========================================")


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["subject", "label"])
y = df["label"].map({
    "control": 0,
    "condition": 1
})

groups = df["subject"]


print(f"\nSubjects: {len(df)}")
print(f"Features: {X.shape[1]}")
print(f"Groups: {groups.nunique()}")


# --------------------------------------------------
# 2. Define model
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
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


# --------------------------------------------------
# 3. GroupKFold
# --------------------------------------------------

group_kfold = GroupKFold(
    n_splits=N_SPLITS
)


# Store predictions from every validation fold
all_true = []
all_pred = []
all_prob = []


fold_results = []


print("\n========================================")
print("CROSS-VALIDATION")
print("========================================")


for fold, (train_idx, test_idx) in enumerate(
    group_kfold.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]


    print(f"\nFold {fold}")
    print("-" * 40)

    print(f"Training subjects: {len(train_idx)}")
    print(f"Validation subjects: {len(test_idx)}")

    print(
        f"Training classes: "
        f"control={(y_train == 0).sum()}, "
        f"condition={(y_train == 1).sum()}"
    )

    print(
        f"Validation classes: "
        f"control={(y_test == 0).sum()}, "
        f"condition={(y_test == 1).sum()}"
    )

    # Verify there is no subject overlap
    overlap = set(groups_train) & set(groups_test)

    print(f"Subject overlap: {len(overlap)}")

    if len(overlap) != 0:
        raise ValueError(
            "Subject leakage detected!"
        )


    # Train only on this fold's training subjects
    model.fit(X_train, y_train)


    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]


    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # ROC-AUC requires both classes
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(
            y_test,
            y_prob
        )
    else:
        auc = np.nan


    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")


    fold_results.append({
        "fold": fold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    })


    all_true.extend(y_test)
    all_pred.extend(y_pred)
    all_prob.extend(y_prob)


# --------------------------------------------------
# 4. Overall out-of-fold metrics
# --------------------------------------------------

print("\n========================================")
print("OVERALL OUT-OF-FOLD RESULTS")
print("========================================")


overall_accuracy = accuracy_score(
    all_true,
    all_pred
)

overall_precision = precision_score(
    all_true,
    all_pred,
    zero_division=0
)

overall_recall = recall_score(
    all_true,
    all_pred,
    zero_division=0
)

overall_f1 = f1_score(
    all_true,
    all_pred,
    zero_division=0
)

overall_auc = roc_auc_score(
    all_true,
    all_prob
)


print(f"Accuracy : {overall_accuracy:.4f}")
print(f"Precision: {overall_precision:.4f}")
print(f"Recall   : {overall_recall:.4f}")
print(f"F1-score : {overall_f1:.4f}")
print(f"ROC-AUC  : {overall_auc:.4f}")


# --------------------------------------------------
# 5. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_true,
    all_pred
)

print("\nConfusion Matrix:")
print(cm)

print("\nFormat:")
print("[[True Control, False Condition],")
print(" [False Control, True Condition]]")


# --------------------------------------------------
# 6. Fold summary
# --------------------------------------------------

results_df = pd.DataFrame(
    fold_results
)

print("\n========================================")
print("FOLD SUMMARY")
print("========================================")

print(
    results_df.to_string(
        index=False
    )
)


print("\nMean fold metrics:")

print(
    results_df[
        [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    ].mean()
)


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

OUTPUT_FILE = Path(
    "results/random_forest_cv_results.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved fold results to: {OUTPUT_FILE}"
)

print("\n========================================")
print("RANDOM FOREST COMPLETE")
print("========================================")