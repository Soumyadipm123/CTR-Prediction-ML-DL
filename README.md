# CTR Prediction System

An end-to-end Click-Through Rate (CTR) prediction project using the Criteo dataset. The project compares traditional machine learning, gradient-boosted trees, and deep learning approaches for predicting whether a user will click an advertisement.

## Project Overview

Click-Through Rate prediction is a binary classification problem widely used in online advertising and recommendation systems.

This project implements and compares three models:

1. Logistic Regression
2. XGBoost
3. Neural Network (ANN) with categorical feature embeddings

The objective is to investigate how different modeling approaches perform on a dataset containing both numerical and high-cardinality categorical features.

## Dataset

The project uses the **`reczoo/Criteo_x1`** dataset from Hugging Face.

For experimentation, 100,000 samples were used.

### Dataset Characteristics

- Samples: 100,000
- Numerical features: 13
- Categorical features: 26
- Total input features: 39
- Target: **`label`**
- Click-through rate: approximately 25.21%

### Data Split

The dataset was divided using stratified sampling:

- Training: 70,000 samples
- Validation: 15,000 samples
- Test: 15,000 samples

Stratification was used to preserve the class distribution across the three subsets.

## Preprocessing

### Numerical Features

Numerical features were converted to numeric values, missing values were handled, and the features were standardized using statistics calculated from the training set.

### Categorical Features

The dataset contains 26 categorical features with potentially high cardinality.

Different approaches were used for the models:

- XGBoost: categorical variables were transformed using one-hot encoding.
- ANN: categorical variables were integer encoded and represented using trainable embedding layers.

The one-hot encoded representation used by XGBoost resulted in:


```text
Training matrix: 70,000 × 122,659
Validation matrix: 15,000 × 122,659
Test matrix: 15,000 × 122,659
```
## Results

The three models were evaluated on the held-out test set using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Log Loss.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Log Loss |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 75.99% | 53.73% | **34.24%** | **41.83%** | 0.7402 | 0.5133 |
| XGBoost | **77.72%** | **64.47%** | 25.91% | 36.97% | **0.7543** | **0.4824** |
| ANN | 76.93% | 58.21% | 30.17% | 39.74% | 0.7342 | 0.4991 |

### Model Comparison

- **XGBoost** achieved the highest accuracy (77.72%), precision (64.47%), ROC-AUC (0.7543), and the lowest Log Loss (0.4824).

- **Logistic Regression** achieved the highest recall (34.24%) and F1-score (41.83%) among the three models.

- **ANN** achieved 76.93% accuracy, 58.21% precision, 30.17% recall, 39.74% F1-score, 0.7342 ROC-AUC, and 0.4991 Log Loss.

Overall, **XGBoost performed best in terms of accuracy, precision, ROC-AUC, and Log Loss**, while **Logistic Regression performed best in terms of recall and F1-score**.