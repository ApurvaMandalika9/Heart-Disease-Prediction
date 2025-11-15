# 🫀 Heart Disease Prediction API

##  Project Overview

This project builds an end-to-end **Heart Disease Risk Prediction System** using machine learning.  
Given a patient’s clinical attributes (age, cholesterol, chest pain type, blood pressure, etc.), the model predicts:

- **0 -> Low risk (no heart disease)**
- **1 -> High risk (heart disease present)**

The final model is deployed as a **Flask-based REST API**, containerized with **Docker**, and trained using the **Kaggle Heart Disease Dataset**.


## Problem Statement

Heart disease remains one of the leading causes of mortality worldwide. Early detection helps prevent severe outcomes and reduces healthcare burdens.

**Goal**: Build a model that predicts whether a patient is likely to have heart disease based on basic health measurements.

Who benefits?
- Clinicians & screening programs
- Health-tech startups
- Preventive-care systems
- Patients looking for early risk assessment


## Dataset

Dataset:  https://kaggle.com/datasets/johnsmith88/heart-disease-dataset

Size:
1. 918 rows
2. 12 clinical features
3. 1 binary target - `target`

Features include:
1. `age` 
2. `sex`
3. `cp` - chest pain type (4 values)
4. `trestbps` - resting blood pressure
5. `chol` - serum cholesterol in mg/dl
6. `fbs` - fasting blood sugar > 120 mg/dl
7. `restecg` - resting electrocardiographic results (values 0,1,2)
8. `thalach` - maximum heart rate achieved
9. `exang` - exercise-induced angina
10. `oldpeak` - ST depression induced by exercise relative to rest
11. `slope` - the slope of the peak exercise ST segment
12. `ca`- number of major vessels (0-3) colored by flourosopy
13. `thal` -  0 = normal; 1 = fixed defect; 2 = reversable defect

The dataset is balanced enough for standard classification metrics.


## EDA Summary

Key observations from exploratory data analysis:
1. Chest pain type (cp) strongly correlates with heart disease.
2. thalach (max heart rate) is typically higher in patients with disease.
3. oldpeak and exang show clear separation between classes.
4. No severe missing values; minor preprocessing required.
5. Features vary in scale → Standardization helps model performance.

EDA was performed in jupyter notebook and later exported to `heart-disease.py` which is added to this repo.


## Modeling Approach

Models evaluated
1. Logistic Regression
2. Random Forest (best model)
3. XGBoost (overfitted -> rejected)

Why Random Forest performed best?
- Handles non-linear interactions
- Robust to noise
- Works well with 12–20 features
- Less sensitive to scaling issues

Best Model Performance 

| **Metric**         | **Score**                                   |
|--------------------|---------------------------------------------|
| **ROC-AUC (Test)** | ≈ 0.90–0.95                                 |
| **PR-AUC**         | High                                        |
| **Accuracy**       | Consistent across cross-validation folds    |

**ROC-AUC** was used as the primary evaluation metric because the dataset is relatively balanced.