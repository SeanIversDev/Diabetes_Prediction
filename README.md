# Diabetes Prediction

A machine learning pipeline that classifies patients as diabetic or non-diabetic using Logistic Regression on clinical lab data from Al-Kindy Teaching Hospital in Iraq.

## Overview

The dataset was sourced from Kaggle, provided by the Specialized Center for Endocrinology and Diabetes at Al-Kindy Teaching Hospital. It contains 1,000 patient records with 14 medical features. After cleaning, 946 records were used for training and evaluation.

## Dataset Features

| Feature | Description |
|---------|-------------|
| AGE | Patient age |
| Gender | Male / Female |
| Urea | Blood urea level |
| Cr | Creatinine ratio |
| HbA1c | Glycated hemoglobin |
| Chol | Total cholesterol |
| TG | Triglycerides |
| HDL | High-density lipoprotein |
| LDL | Low-density lipoprotein |
| VLDL | Very low-density lipoprotein |
| BMI | Body mass index |
| CLASS | Target — Diabetic (1) / Non-diabetic (0) |

## Pipeline

### 1. Data Cleaning
- Removed 53 pre-diabetic (`P`) records to create a clean binary classification problem
- Dropped 1 record with zero cholesterol (physiologically impossible)
- Encoded `Gender` (M=0, F=1) and `CLASS` (N=0, Y=1)

### 2. Train / Validation / Test Split
- **80%** training, **10%** validation, **10%** test
- Stratified sampling used throughout to handle class imbalance (~89% diabetic)

### 3. Feature Scaling
- `RobustScaler` applied — fit on training data only, then applied to validation and test sets

### 4. Model
- `sklearn` Logistic Regression with L2 regularization
- Hyperparameter tuned across multiple values of `C` (0.01, 0.1, 1.0) and `class_weight`
- Best model: `C=0.1`, L2 penalty, lbfgs solver

## Results

### Validation Set (Best Model — C=0.1)
| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| Sensitivity (TPR) | 100% |
| Specificity (TNR) | 100% |
| AUC-ROC | 1.00 |

### Test Set (Final Evaluation)
| Metric | Score |
|--------|-------|
| Accuracy | 92.6% |
| Sensitivity (TPR) | 97.6% |
| Specificity (TNR) | 50.0% |
| Precision | 94.3% |
| AUC-ROC | 0.984 |

The model is highly sensitive — it correctly identifies 97.6% of diabetic patients — but has lower specificity, reflecting the class imbalance in the dataset (844 diabetic vs. 103 non-diabetic).

## Project Structure

```
Diabetes_Prediction/
├── main.ipynb                   # Full pipeline notebook
└── Dataset of Diabetes .csv     # Raw dataset from Kaggle
```

## Dependencies

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
