# AI Job Salary Prediction using Supervised Machine Learning

## Overview

This project presents a complete supervised machine learning pipeline for predicting AI job salaries using a structured tabular dataset. The work was developed as part of an IEEE-style supervised learning research assignment focused on model evaluation, preprocessing analysis, interpretability, robustness testing, and practical model recommendation.

The project follows a research-question-driven experimental structure consisting of seven separate notebooks, each addressing a different aspect of supervised learning experimentation.

---

## Problem Statement

The rapid growth of Artificial Intelligence and Data Science careers has created significant variations in salaries based on factors such as:

- Experience level
- Employment type
- Remote work ratio
- Company location
- Education level
- Industry domain
- Required skills

The objective of this project is to build and evaluate supervised learning regression models capable of predicting AI job salaries (`salary_usd`) from real-world job-related features.

---

## Dataset

**Dataset:** AI Job Salary Dataset

**Type:**
- Structured tabular dataset
- Supervised learning regression problem

**Target Variable:** `salary_usd`

**Features Used:**
- Experience level
- Employment type
- Remote ratio
- Company size
- Education required
- Years of experience
- Industry
- Required skills
- Country/location related attributes

**Dataset Characteristics:**
- 15,000 records
- Multiple categorical and numerical features
- Real-world AI/ML employment data

**Dataset Source:**
- Name: Global AI Job Market and Salary Trends 2025
- Kaggle link: https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025

---

## Project Objectives

The project was designed to answer the following research questions:

1. How effectively do baseline supervised learning models perform?
2. Which model achieves the best predictive performance?
3. How does preprocessing impact model accuracy?
4. Which features contribute most to salary prediction?
5. How sensitive are models to different evaluation metrics?
6. How robust are the models under different validation conditions?
7. Which model provides the best practical balance between performance, interpretability, and robustness?

---

## Project Structure

```
AI-Job-Salary-Prediction/
│
├── data/
│   └── ai_job_dataset1.csv
│
├── notebooks/
│   ├── RQ1_Baseline_Performance.ipynb
│   ├── RQ2_Model_Comparison.ipynb
│   ├── RQ3_Preprocessing_Effects.ipynb
│   ├── RQ4_Feature_Importance.ipynb
│   ├── RQ5_Metric_Sensitivity.ipynb
│   ├── RQ6_Robustness.ipynb
│   └── RQ7_Final_Recommendation.ipynb
│
├── figures/
│   ├── RQ1/
│   ├── RQ2/
│   ├── RQ3/
│   ├── RQ4/
│   ├── RQ5/
│   ├── RQ6/
│   └── RQ7/
│
├── README.md
├── requirements.txt
└── report.pdf
```

---

## Machine Learning Workflow

The implementation follows a complete end-to-end supervised learning pipeline:

### 1. Data Preprocessing
- Handling missing values
- Feature engineering (e.g. `num_skills` derived from `required_skills`)
- Label encoding and one-hot encoding (compared in RQ3)
- Feature scaling
- Outlier filtering (IQR)

### 2. Baseline Model Training (RQ1)
Initial baseline regression models were trained to establish reference performance levels.

Models used:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- k-Nearest Neighbors

### 3. Advanced Model Comparison (RQ2)
A wider set of supervised learning models was evaluated and compared.

Models evaluated:
- Linear Regression
- Ridge Regression
- Decision Tree
- k-Nearest Neighbors
- Support Vector Regressor (SVR)
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

### 4. Feature Importance Analysis (RQ4)
Interpretability analysis was performed using three complementary methods:
- Built-in Random Forest feature importance
- Permutation importance
- SHAP values

### 5. Robustness Evaluation (RQ6)
The selected model was tested under:
- Multiple train-test split ratios
- K-fold cross-validation (3, 5, 10 folds)
- Gaussian noise injection (0%–20%)

---

## Evaluation Metrics

Since this is a regression problem, the following metrics were used:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)
- MdAE (Median Absolute Error)

These metrics were used consistently across all experimental notebooks.

---

## Research Question Breakdown

### RQ1 — Baseline Performance
Evaluates the performance of baseline regression models.

**Notebook:** `RQ1_Baseline_Performance.ipynb`

**Key Visualizations:**
- Salary distribution and group breakdowns
- Correlation heatmap
- Actual vs predicted salary plots
- Residual analysis
- Feature importance and linear coefficients

---

### RQ2 — Model Comparison
Compares eight supervised learning models (linear, tree-based, kernel, and ensemble) using multiple regression metrics.

**Notebook:** `RQ2_Model_Comparison.ipynb`

**Key Visualizations:**
- Horizontal bar chart of R² rankings
- Radar plot across MAE, RMSE, R², MAPE, and training time

---

### RQ3 — Effect of Preprocessing
Analyzes the impact of six preprocessing strategies on model performance: no scaling, StandardScaler, MinMaxScaler, log-transformed target, IQR outlier removal, and one-hot encoding.

**Notebook:** `RQ3_Preprocessing_Effects.ipynb`

**Key Visualizations:**
- Preprocessing ablation bar chart (MAE / RMSE / R²)

---

### RQ4 — Feature Importance & Interpretability
Identifies the most influential features contributing to salary prediction using three independent methods.

**Notebook:** `RQ4_Feature_Importance.ipynb`

**Key Visualizations:**
- Random Forest feature importance plot
- Permutation importance plot
- SHAP summary plot

---

### RQ5 — Sensitivity to Evaluation Metrics
Studies how model rankings vary across R², MAE, RMSE, MAPE, and MdAE.

**Notebook:** `RQ5_Metric_Sensitivity.ipynb`

**Key Visualizations:**
- Grouped bar chart of ranks per metric
- Slope / bump chart of rank changes across metrics

---

### RQ6 — Robustness & Generalization
Evaluates model stability under varying validation regimes and noisy inputs.

**Notebook:** `RQ6_Robustness.ipynb`

**Key Visualizations:**
- Box plots across train-test splits
- Box plots across k-fold CV settings
- R² degradation curve under Gaussian noise injection

---

### RQ7 — Final Recommendation
Provides the final comparative analysis and practical recommendation by combining performance, robustness, interpretability, deployment ease, and training speed into a unified decision matrix.

**Notebook:** `RQ7_Final_Recommendation.ipynb`

**Key Visualizations:**
- Radar chart of top models across all dimensions
- Bubble chart of accuracy vs interpretability vs training time
- Final decision matrix table

---

## Results Summary

| Model | R² (Test) | MAE | RMSE | MAPE |
|---|---|---|---|---|
| Linear Regression | 0.6313 | $27,180 | $38,436 | 28.40% |
| Ridge Regression | 0.6313 | $27,180 | $38,436 | 28.40% |
| Decision Tree | 0.8729 | $16,028 | $22,572 | 12.94% |
| k-Nearest Neighbors | 0.6531 | $26,904 | $37,284 | 27.91% |
| SVR | 0.3067 | $36,576 | $52,710 | 35.62% |
| Random Forest | 0.8835 | $15,572 | $21,609 | — |
| Gradient Boosting | 0.8880 | $15,318 | $21,185 | 12.53% |
| XGBoost | 0.8839 | $15,492 | $21,565 | 12.60% |
| LightGBM | 0.8863 | $15,423 | $21,344 | 12.56% |

Ensemble methods (Gradient Boosting, LightGBM, XGBoost, Random Forest) and Decision Tree clearly outperform linear models and SVR on raw predictive metrics. The final model selection in RQ7 weighs these results against interpretability, deployment ease, and training speed.

---

## Technologies Used

**Programming Language:**
- Python

**Libraries:**
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

## Key Outcomes

- Ensemble and tree-based models outperformed linear regression, k-NN, and SVR baselines by a wide margin (R² ≈ 0.87–0.89 vs 0.31–0.65).
- Preprocessing strategies (scaling, log transform, one-hot encoding) had minimal effect on Random Forest performance, confirming its insensitivity to monotonic feature transformations.
- Feature importance analysis consistently identified `years_experience` and `company_location` as the dominant salary drivers across all three methods (built-in, permutation, SHAP).
- Robustness experiments showed stable generalization across train-test splits and k-fold CV, with graceful degradation under noise injection.
- The final recommendation in RQ7 selects **Decision Tree** as the best practical model — it sits within ~1.5 percentage points of the top ensemble models on R² (0.873 vs 0.888) while offering substantially higher interpretability, easier deployment, and faster training, giving it the highest overall score in the decision matrix.

---

## How to Run the Project

### Clone Repository

```bash
git clone https://github.com/<your-username>/AI-Job-Salary-Prediction-using-Supervised-Machine-Learning.git
cd AI-Job-Salary-Prediction-using-Supervised-Machine-Learning
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Run the notebooks sequentially from RQ1 to RQ7.

---

## Academic Context

This project was developed as part of a supervised learning research assignment following an IEEE-style experimental structure focused on:
- supervised learning experimentation
- model comparison
- preprocessing analysis
- interpretability
- robustness evaluation
- practical deployment considerations

---

## Author

**Sahith Ganta**
Data Science & AI Student
