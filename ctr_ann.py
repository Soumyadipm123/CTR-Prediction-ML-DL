# ============================================================
# CTR PREDICTION - DEEP LEARNING / ANN
# Criteo Dataset
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
SAMPLE_SIZE = 100_000

BATCH_SIZE = 1024
EPOCHS = 15
LEARNING_RATE = 1e-3

EMBED_DIM = 16
PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 60)
print("CTR PREDICTION - ANN / DEEP LEARNING")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

dataset = load_dataset(
    "reczoo/Criteo_x1",
    split="train",
    streaming=True
)

sample = dataset.take(SAMPLE_SIZE)

df = pd.DataFrame(list(sample))

print(f"Dataset shape: {df.shape}")
print(df.head())


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

numerical_features = [
    "I1", "I2", "I3", "I4", "I5", "I6",
    "I7", "I8", "I9", "I10", "I11", "I12", "I13"
]

categorical_features = [
    "C1", "C2", "C3", "C4", "C5", "C6",
    "C7", "C8", "C9", "C10", "C11", "C12",
    "C13", "C14", "C15", "C16", "C17", "C18",
    "C19", "C20", "C21", "C22", "C23", "C24",
    "C25", "C26"
]

target = "label"

print(f"\nNumerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Total features: {len(numerical_features) + len(categorical_features)}")


# ============================================================
# 3. PREPARE TARGET
# ============================================================

y = df[target].astype(np.float32).values

print(f"\nClick rate: {y.mean():.5f}")


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

indices = np.arange(len(df))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    stratify=y,
    random_state=SEED
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=y[temp_idx],
    random_state=SEED
)

print(f"Train:      {len(train_idx)}")
print(f"Validation: {len(val_idx)}")
print(f"Test:       {len(test_idx)}")

print("\nClass distribution:")
print(f"Train:      {y[train_idx].mean():.5f}")
print(f"Validation: {y[val_idx].mean():.5f}")
print(f"Test:       {y[test_idx].mean():.5f}")


# ============================================================
# 5. NUMERICAL FEATURE PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL PREPROCESSING")
print("=" * 60)

X_num = df[numerical_features].apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0.0).astype(np.float32).values

# Calculate statistics ONLY from training data
num_mean = X_num[train_idx].mean(axis=0)
num_std = X_num[train_idx].std(axis=0)

# Avoid division by zero
num_std[num_std < 1e-6] = 1.0

X_num = (X_num - num_mean) / num_std

X_num_train = X_num[train_idx]
X_num_val = X_num[val_idx]
X_num_test = X_num[test_idx]


# ============================================================
# 6. CATEGORICAL FEATURE ENCODING
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL ENCODING")
print("=" * 60)

"""
Each categorical feature gets its own integer encoding.

0 = unknown category

1, 2, 3, ... = known categories

This allows the neural network to use Embedding layers.
"""

X_cat = np.zeros(
    (len(df), len(categorical_features)),
    dtype=np.int64
)

cardinalities = []

for i, col in enumerate(categorical_features):

    print(f"Encoding {col}...", end=" ")

    # Convert to string so categorical IDs are handled consistently
    values = df[col].fillna("__MISSING__").astype(str)

    # Categories learned ONLY from training data
    train_values = values.iloc[train_idx]

    unique_values = pd.Index(train_values.unique())

    # category -> integer
    mapping = {
        value: idx + 1
        for idx, value in enumerate(unique_values)
    }

    encoded = values.map(mapping).fillna(0).astype(np.int64)

    X_cat[:, i] = encoded.values

    cardinality = len(unique_values) + 1
    cardinalities.append(cardinality)

    print(f"{cardinality} categories")


X_cat_train = X_cat[train_idx]
X_cat_val = X_cat[val_idx]
X_cat_test = X_cat[test_idx]

print("\nCategorical cardinalities:")
for col, card in zip(categorical_features, cardinalities):
    print(f"{col}: {card}")


# ============================================================
# 7. CREATE PYTORCH DATASETS
# ============================================================

print("\n" + "=" * 60)
print("CREATING PYTORCH DATASETS")
print("=" * 60)

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

train_dataset = TensorDataset(
    torch.tensor(X_num_train, dtype=torch.float32),
    torch.tensor(X_cat_train, dtype=torch.long),
    torch.tensor(y_train, dtype=torch.float32)
)

val_dataset = TensorDataset(
    torch.tensor(X_num_val, dtype=torch.float32),
    torch.tensor(X_cat_val, dtype=torch.long),
    torch.tensor(y_val, dtype=torch.float32)
)

test_dataset = TensorDataset(
    torch.tensor(X_num_test, dtype=torch.float32),
    torch.tensor(X_cat_test, dtype=torch.long),
    torch.tensor(y_test, dtype=torch.float32)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=torch.cuda.is_available()
)

print(f"Training batches:   {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches:       {len(test_loader)}")


# ============================================================
# 8. ANN MODEL
# ============================================================

class CTR_ANN(nn.Module):

    def __init__(
        self,
        cardinalities,
        num_numerical,
        embedding_dim=16
    ):
        super().__init__()

        # One embedding layer for every categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=cardinality,
                embedding_dim=embedding_dim
            )
            for cardinality in cardinalities
        ])

        total_embedding_dim = (
            len(cardinalities) * embedding_dim
        )

        input_dim = total_embedding_dim + num_numerical

        self.network = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 1)
        )

    def forward(self, numerical, categorical):

        embedded_features = []

        for i, embedding in enumerate(self.embeddings):
            embedded_features.append(
                embedding(categorical[:, i])
            )

        # Concatenate all categorical embeddings
        embedded = torch.cat(
            embedded_features,
            dim=1
        )

        # Add numerical features
        x = torch.cat(
            [numerical, embedded],
            dim=1
        )

        return self.network(x).squeeze(1)


# ============================================================
# 9. INITIALIZE MODEL
# ============================================================

print("\n" + "=" * 60)
print("INITIALIZING ANN")
print("=" * 60)

model = CTR_ANN(
    cardinalities=cardinalities,
    num_numerical=len(numerical_features),
    embedding_dim=EMBED_DIM
).to(DEVICE)

print(model)

total_parameters = sum(
    p.numel() for p in model.parameters()
)

print(f"\nTotal parameters: {total_parameters:,}")


# ============================================================
# 10. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)


# ============================================================
# 11. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for numerical, categorical, labels in loader:

        numerical = numerical.to(
            DEVICE,
            non_blocking=True
        )

        categorical = categorical.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(
            numerical,
            categorical
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# 12. VALIDATION FUNCTION
# ============================================================

def evaluate_loss(model, loader):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for numerical, categorical, labels in loader:

            numerical = numerical.to(
                DEVICE,
                non_blocking=True
            )

            categorical = categorical.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(
                numerical,
                categorical
            )

            loss = criterion(
                logits,
                labels
            )

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# 13. TRAIN ANN
# ============================================================

print("\n" + "=" * 60)
print("ANN TRAINING")
print("=" * 60)

best_val_loss = float("inf")
best_state = None
patience_counter = 0

for epoch in range(EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader
    )

    val_loss = evaluate_loss(
        model,
        val_loader
    )

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )

    # Save best model in memory
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        patience_counter = 0

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping triggered after "
                f"{epoch + 1} epochs."
            )

            break


# ============================================================
# 14. RESTORE BEST MODEL
# ============================================================

model.load_state_dict(best_state)
model.to(DEVICE)

print(
    f"\nBest validation log loss: "
    f"{best_val_loss:.6f}"
)


# ============================================================
# 15. PREDICTION FUNCTION
# ============================================================

def predict_probabilities(model, loader):

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for numerical, categorical, batch_labels in loader:

            numerical = numerical.to(
                DEVICE,
                non_blocking=True
            )

            categorical = categorical.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(
                numerical,
                categorical
            )

            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    return (
        np.array(labels),
        np.array(probabilities)
    )


# ============================================================
# 16. EVALUATION
# ============================================================

def evaluate_model(model, loader, dataset_name):

    labels, probabilities = predict_probabilities(
        model,
        loader
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    loss = log_loss(
        labels,
        probabilities
    )

    print("\n" + "=" * 60)
    print(f"ANN {dataset_name.upper()} RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1 Score : {f1:.6f}")
    print(f"ROC-AUC  : {roc_auc:.6f}")
    print(f"Log Loss : {loss:.6f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": loss
    }


# ============================================================
# 17. VALIDATION RESULTS
# ============================================================

val_results = evaluate_model(
    model,
    val_loader,
    "Validation"
)


# ============================================================
# 18. TEST RESULTS
# ============================================================

test_results = evaluate_model(
    model,
    test_loader,
    "Test"
)


# ============================================================
# 19. SAVE ANN MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING ANN MODEL")
print("=" * 60)

model_path = "ann_ctr_model.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "cardinalities": cardinalities,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "embedding_dim": EMBED_DIM,
        "num_numerical": len(numerical_features),
        "num_categorical": len(categorical_features),
        "num_parameters": total_parameters
    },
    model_path
)

print(f"Saved to: {os.path.abspath(model_path)}")


# ============================================================
# 20. FINAL CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL CHECK")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Model parameters: {total_parameters:,}")
print(f"Model file exists: {os.path.exists(model_path)}")

print("\nCTR ANN pipeline completed successfully.")