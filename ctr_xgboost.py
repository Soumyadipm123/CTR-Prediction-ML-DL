# ============================================================
# CTR Prediction Project - XGBoost
# ============================================================

import os
import pandas as pd
import numpy as np

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

dataset = load_dataset(
    "reczoo/Criteo_x1",
    split="train",
    streaming=True
)

sample = dataset.take(100_000)

df = pd.DataFrame(list(sample))

print("Dataset shape:", df.shape)
print(df.head())
print()


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

numeric_cols = [f"I{i}" for i in range(1, 14)]
categorical_cols = [f"C{i}" for i in range(1, 27)]

print("Numerical features:", len(numeric_cols))
print("Categorical features:", len(categorical_cols))
print("Total features:", len(numeric_cols) + len(categorical_cols))
print()


# ============================================================
# 3. REMOVE C16 FREQUENCY FEATURE
# ============================================================

# This was explored during EDA but removed before modeling
if "C16_frequency" in df.columns:
    df.drop(columns=["C16_frequency"], inplace=True)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["label"])
y = df["label"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Click rate:", y.mean())
print()


# ============================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("=" * 60)
print("DATA SPLIT")
print("=" * 60)

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

print("\nTrain class distribution:")
print(y_train.value_counts(normalize=True))

print("\nValidation class distribution:")
print(y_val.value_counts(normalize=True))

print("\nTest class distribution:")
print(y_test.value_counts(normalize=True))

print()


# ============================================================
# 6. ONE-HOT ENCODING FOR XGBOOST
# ============================================================

print("=" * 60)
print("PREPROCESSING")
print("=" * 60)

preprocessor_xgb = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ]
)

print("Fitting encoder...")

X_train_xgb = preprocessor_xgb.fit_transform(X_train)

print("Transforming validation data...")
X_val_xgb = preprocessor_xgb.transform(X_val)

print("Transforming test data...")
X_test_xgb = preprocessor_xgb.transform(X_test)

print()
print("Train shape:", X_train_xgb.shape)
print("Validation shape:", X_val_xgb.shape)
print("Test shape:", X_test_xgb.shape)
print("Matrix type:", type(X_train_xgb))
print()


# ============================================================
# 7. XGBOOST MODEL
# ============================================================

print("=" * 60)
print("XGBOOST")
print("=" * 60)

import xgboost as xgb

print("XGBoost version:", xgb.__version__)

xgb_model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",

    # RTX 2050 GPU
    device="cuda",

    eval_metric="logloss",
    n_jobs=-1,
    random_state=42
)

print("\nStarting GPU training...")
print("GPU: NVIDIA GeForce RTX 2050")
print()


# ============================================================
# 8. TRAIN XGBOOST
# ============================================================

xgb_model.fit(
    X_train_xgb,
    y_train,
    eval_set=[
        (X_train_xgb, y_train),
        (X_val_xgb, y_val)
    ],
    verbose=50
)

print()
print("XGBoost training completed successfully!")
print()


# ============================================================
# 9. SAVE MODEL
# ============================================================

model_path = "xgb_ctr_model.json"

xgb_model.save_model(model_path)

print("=" * 60)
print("MODEL SAVED")
print("=" * 60)
print("Saved to:", os.path.abspath(model_path))
print()


# ============================================================
# 10. VALIDATION PREDICTIONS
# ============================================================

y_val_prob_xgb = xgb_model.predict_proba(
    X_val_xgb
)[:, 1]

y_val_pred_xgb = (
    y_val_prob_xgb >= 0.5
).astype(int)


# ============================================================
# 11. VALIDATION METRICS
# ============================================================

print("=" * 60)
print("XGBOOST VALIDATION RESULTS")
print("=" * 60)

accuracy = accuracy_score(
    y_val,
    y_val_pred_xgb
)

precision = precision_score(
    y_val,
    y_val_pred_xgb
)

recall = recall_score(
    y_val,
    y_val_pred_xgb
)

f1 = f1_score(
    y_val,
    y_val_pred_xgb
)

roc_auc = roc_auc_score(
    y_val,
    y_val_prob_xgb
)

logloss = log_loss(
    y_val,
    y_val_prob_xgb
)

print(f"Accuracy : {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1 Score : {f1:.6f}")
print(f"ROC-AUC  : {roc_auc:.6f}")
print(f"Log Loss : {logloss:.6f}")

print()


# ============================================================
# 12. TEST SET EVALUATION
# ============================================================

print("=" * 60)
print("XGBOOST TEST RESULTS")
print("=" * 60)

y_test_prob_xgb = xgb_model.predict_proba(
    X_test_xgb
)[:, 1]

y_test_pred_xgb = (
    y_test_prob_xgb >= 0.5
).astype(int)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred_xgb
)

test_precision = precision_score(
    y_test,
    y_test_pred_xgb
)

test_recall = recall_score(
    y_test,
    y_test_pred_xgb
)

test_f1 = f1_score(
    y_test,
    y_test_pred_xgb
)

test_roc_auc = roc_auc_score(
    y_test,
    y_test_prob_xgb
)

test_logloss = log_loss(
    y_test,
    y_test_prob_xgb
)

print(f"Accuracy : {test_accuracy:.6f}")
print(f"Precision: {test_precision:.6f}")
print(f"Recall   : {test_recall:.6f}")
print(f"F1 Score : {test_f1:.6f}")
print(f"ROC-AUC  : {test_roc_auc:.6f}")
print(f"Log Loss : {test_logloss:.6f}")

print()


# ============================================================
# 13. FINAL CHECK
# ============================================================

print("=" * 60)
print("FINAL CHECK")
print("=" * 60)

print("X_train_xgb:", X_train_xgb.shape)
print("X_val_xgb  :", X_val_xgb.shape)
print("X_test_xgb :", X_test_xgb.shape)
print("Matrix type:", type(X_train_xgb))
print("Model fitted:", hasattr(xgb_model, "_Booster"))
print("Model file exists:", os.path.exists(model_path))

print()
print("CTR XGBoost pipeline completed.")