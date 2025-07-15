# Groundwater Recharge Potential Classification

## Overview

This project demonstrates a complete workflow for simulating, preprocessing, analyzing, modeling, and interpreting groundwater recharge potential data. It aims to classify areas based on their potential to recharge groundwater into three categories: **High**, **Medium**, and **Low**.

The project includes:

- Synthetic data simulation mimicking realistic hydrogeological and environmental conditions.
- Data preprocessing and exploratory data analysis (EDA).
- Training and evaluation of multiple machine learning classification models.
- Feature importance analysis and a reusable prediction function.

---

## Project Structure

| File Name                 | Description                                                   |
|---------------------------|---------------------------------------------------------------|
| `part_1_data_simulation.py` | Simulates and generates synthetic groundwater recharge data and saves it as a CSV file. |
| `part_2_preprocessing_eda.py` | Loads data, performs encoding, scaling, and conducts exploratory data analysis including visualizations. |
| `part_3_model_training.py`    | Trains and evaluates multiple classification models (Random Forest, SVM, KNN, Logistic Regression) on the data. |
| `part_4_feature_importance.py` | Analyzes feature importances using Random Forest and provides a function for predicting recharge potential for new input data. |

---

## Detailed Description

### 1. Data Simulation (`part_1_data_simulation.py`)

- Generates a dataset with 500 samples.
- Features simulated include:
  - **Slope** (degrees)
  - **Elevation** (meters)
  - **Soil Permeability** (categorical: low, medium, high)
  - **Land Use** (categorical: urban, forest, agriculture, barren)
  - **Rainfall** (mm/year)
  - **Drainage Density** (km/km²)
  - **Depth to Water Table** (meters)
- The target variable `recharge_potential` is derived using domain-inspired rules based on the features.
- The resulting dataset is saved to `synthetic_groundwater_data.csv` for use in subsequent steps.

### 2. Preprocessing and EDA (`part_2_preprocessing_eda.py`)

- Loads the synthetic dataset.
- Encodes categorical variables (`land_use` and `recharge_potential`) using label encoding.
- Standardizes numerical features with `StandardScaler` for consistent scaling.
- Visualizes:
  - Class distribution of recharge potential.
  - Correlations between features and the target variable.
  - Boxplots of key features across recharge potential classes to understand their distributions.

### 3. Model Training and Evaluation (`part_3_model_training.py`)

- Loads and preprocesses data (encoding and scaling).
- Splits data into training and test sets with stratification.
- Trains four different classifiers:
  - Random Forest
  - Support Vector Machine (SVM) with RBF kernel
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
- Evaluates models using accuracy, classification reports, and confusion matrices.
- Compares model performances visually with a bar plot.

### 4. Feature Importance and Prediction (`part_4_feature_importance.py`)

- Trains a Random Forest classifier on the entire dataset.
- Extracts and plots feature importances to identify the most influential variables in predicting recharge potential.
- Defines a `predict_recharge()` function that accepts new input values, preprocesses them, and predicts the recharge potential category.
- Demonstrates example prediction usage.

---

## Usage Instructions

1. **Data Simulation**
   ```bash
   python part_1_data_simulation.py
   ```

   This creates the synthetic dataset CSV file.

2. **Preprocessing and EDA**

   ```bash
   python part_2_preprocessing_eda.py
   ```
   Loads the dataset and generates visualizations to explore the data.

3. **Model Training and Evaluation**

   ```bash
   python part_3_model_training.py
   ```
   Trains multiple classifiers and evaluates their performance on the test set.

4. **Feature Importance and Prediction**

   ```bash
   python part_4_feature_importance.py
   ```
   Displays feature importance plots and runs an example prediction using the trained Random Forest model.
