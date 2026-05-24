# AI Job Salary Prediction using Supervised Machine Learning

## Overview

This project presents a complete supervised machine learning pipeline for predicting AI job salaries using a structured tabular dataset. The work was developed as part of an IEEE-style supervised learning research assignment focused on model evaluation, preprocessing analysis, interpretability, robustness testing, and practical model recommendation.

The project follows a research-question-driven experimental structure consisting of seven separate notebooks, each addressing a different aspect of supervised learning experimentation.

---

# Problem Statement

The rapid growth of Artificial Intelligence and Data Science careers has created significant variations in salaries based on factors such as:

- Experience level
- Employment type
- Remote work ratio
- Company location
- Education level
- Industry domain
- Required skills

The objective of this project is to build and evaluate supervised learning regression models capable of predicting AI job salaries (salary_usd) from real-world job-related features.

---

# Dataset

Dataset: AI Job Salary Dataset

Type:
- Structured tabular dataset
- Supervised learning regression problem

Target Variable:
- salary_usd

Features Used:
- Experience level
- Employment type
- Remote ratio
- Company size
- Education required
- Years of experience
- Industry
- Skills
- Country/location related attributes

Dataset Characteristics:
- 15,000+ records
- Multiple categorical and numerical features
- Real-world AI/ML employment data

---

Dataset Source

Dataset Name:
Global AI Job Market and Salary Trends 2025

Kaggle Dataset Link:
https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025

---

# Project Objectives

The project was designed to answer the following research questions:

1. How effectively do baseline supervised learning models perform?
2. Which model achieves the best predictive performance?
3. How does preprocessing impact model accuracy?
4. Which features contribute most to salary prediction?
5. How sensitive are models to different evaluation metrics?
6. How robust are the models under different validation conditions?
7. Which model provides the best practical balance between performance, interpretability, and robustness?

---

# Project Structure

text AI-Job-Salary-Prediction/ │ ├── data/ │   └── ai_job_dataset1.csv │ ├── notebooks/ │   ├── RQ1_Baseline_Performance.ipynb │   ├── RQ2_Model_Comparison.ipynb │   ├── RQ3_Preprocessing_Effects.ipynb │   ├── RQ4_Feature_Importance.ipynb │   ├── RQ5_Metric_Sensitivity.ipynb │   ├── RQ6_Robustness.ipynb │   └── RQ7_Final_Recommendation.ipynb │ ├── figures/ │   ├── RQ1/ │   ├── RQ2/ │   ├── RQ3/ │   ├── RQ4/ │   ├── RQ5/ │   ├── RQ6/ │   └── RQ7/ │ ├── README.md ├── requirements.txt └── report.pdf 

---

# Machine Learning Workflow

The implementation follows a complete end-to-end supervised learning pipeline:

## 1. Data Preprocessing
- Handling missing values
- Feature engineering
- One-hot encoding
- Feature scaling
- Data cleaning

## 2. Baseline Model Training
Initial baseline regression models were trained to establish reference performance levels.

Models used:
- Linear Regression
- Decision Tree Regressor
- k-Nearest Neighbors

## 3. Advanced Model Comparison
Advanced supervised learning models were evaluated and compared.

Models evaluated:
- Random Forest Regressor
- XGBoost Regressor
- Support Vector Regressor (SVR)
- LightGBM Regressor

## 4. Feature Importance Analysis
Interpretability analysis was performed using feature importance techniques and SHAP analysis to identify the most influential variables affecting salary prediction.

## 5. Robustness Evaluation
The best-performing models were tested under:
- Cross-validation
- Noise injection
- Missing value perturbation
- Different train-test conditions

---

# Evaluation Metrics

Since this is a regression problem, the following metrics were used:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

These metrics were used consistently across all experimental notebooks.

---

# Research Question Breakdown

## RQ1 — Baseline Performance
Evaluates the performance of baseline regression models.

Notebook:
RQ1_Baseline_Performance.ipynb

Key Visualizations:
- Model comparison plots
- Residual analysis
- Correlation heatmaps
- Actual vs predicted salary plots

---

## RQ2 — Model Comparison
Compares advanced machine learning models using regression metrics.

Notebook:
RQ2_Model_Comparison.ipynb

Key Visualizations:
- Model ranking charts
- Comparative performance analysis
- Advanced regression comparison plots

---

## RQ3 — Effect of Preprocessing
Analyzes the impact of preprocessing techniques on model performance.

Notebook:
RQ3_Preprocessing_Effects.ipynb

Key Visualizations:
- Preprocessing impact comparison
- Scaling and encoding performance analysis

---

## RQ4 — Feature Importance & Interpretability
Identifies the most influential features contributing to salary prediction.

Notebook:
RQ4_Feature_Importance.ipynb

Key Visualizations:
- Feature importance plots
- SHAP summary plots
- Model interpretability analysis

---

## RQ5 — Sensitivity to Evaluation Metrics
Studies how model rankings vary across different evaluation metrics.

Notebook:
RQ5_Metric_Sensitivity.ipynb

Key Visualizations:
- Metric sensitivity plots
- Ranking comparison analysis

---

## RQ6 — Robustness & Generalization
Evaluates model stability under cross-validation and perturbed data conditions.

Notebook:
RQ6_Robustness.ipynb

Key Visualizations:
- Cross-validation analysis
- Noise robustness plots
- Stability comparison graphs

---

## RQ7 — Final Recommendation
Provides the final comparative analysis and practical recommendation of the best model.

Notebook:
RQ7_Final_Recommendation.ipynb

Key Visualizations:
- Final tradeoff analysis
- Model recommendation matrix
- Performance vs interpretability comparison

---

# Technologies Used

Programming Language:
- Python

Libraries:
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- Jupyter Notebook

---

# Key Outcomes

- Ensemble-based models outperformed baseline regression models.
- Proper preprocessing significantly improved predictive performance.
- Feature importance analysis identified critical salary-driving variables.
- Robustness experiments demonstrated stable model generalization.
- The final selected model achieved a strong balance between predictive accuracy and practical usability.

---

# How to Run the Project

## Clone Repository

bash git clone <repository-link> 

## Install Dependencies

bash pip install -r requirements.txt 

## Launch Jupyter Notebook

bash jupyter notebook 

Run the notebooks sequentially from RQ1 to RQ7.

---

# Academic Context

This project was developed as part of a supervised learning research assignment following an IEEE-style experimental structure focused on:
- supervised learning experimentation
- model comparison
- preprocessing analysis
- interpretability
- robustness evaluation
- practical deployment considerations

---

# Author

Sahith Ganta

Data Science & AI Student
