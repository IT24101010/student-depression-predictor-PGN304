# IT2011 - AI/ML Project: Predicting Depression Risk in University Students

**Group ID:** PGN-304
**Date:** September 25, 2025

---

## 1. Project Overview

This project aims to develop a machine learning solution to address the critical real-world problem of student mental health. Our goal is to build a classification model that can identify university students at a higher risk of depression based on a combination of academic, lifestyle, and psychological factors. By leveraging machine learning, we hope to provide a tool that enables educational institutions to shift from reactive counseling to proactive outreach, offering early and targeted interventions to support student well-being.

---

## 2. Dataset Details

- **Dataset Name:** Student Depression and Mental Health Dataset 
- **Source:** Kaggle 
- **Link:** [https://www.kaggle.com/datasets/hopesb/student-depression-dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
- **Description:** This is a tabular dataset containing over 3,500 records and approximately 20 features. The data includes a rich mix of academic indicators (e.g., CGPA), lifestyle factors (e.g., Sleep Duration), and psychological metrics (e.g., Financial Stress). The target variable is a binary indicator, `Depression`, making this a supervised classification task. 

---

## 3. Group Members and Roles

This project was a collaborative effort, with each member responsible for a specific stage of the data preprocessing pipeline:

| Member Name              | IT Number    | Assigned Role                      |
| ------------------------ | ------------ | ---------------------------------- |
| PUNCHIHEWA P.K.N         | IT24101243   | 1. Handling Missing Values         |
| RANASINGHE.R.G.P.D       | IT24100910   | 2. Categorical Feature Encoding    |
| CHANDRASEKARA RMS        | IT24101383   | 3. Outlier Removal                 |
| DE SILVA THHD            | IT24101010   | 4. Feature Scaling (Standardization) |
| GUNASEKARA J.D.C.        | IT24101409   | 5. Feature Selection               |
| BHAGYA J.P.J.            | IT24101426   | 6. PCA (Dimensionality Reduction)  |

---

## 4. Repository Layout

<img width="558" height="514" alt="image" src="https://github.com/user-attachments/assets/898d904a-3eed-48b1-b52c-f1edbe7618fe" />


---

## 5. How to Run the Code

### Prerequisites
Ensure you have Python installed with the following libraries:
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `category_encoders`

### Instructions
1.  **Place the Dataset:** Download the dataset from the Kaggle link above and place `Student Depression Dataset.csv`.
2.  **Review Individual Work:** To see each member's specific contribution, EDA, and justifications, you can run the individual notebooks (`IT_Number_*.ipynb`) located in the `notebooks/` folder.
3.  **Run the Integrated Pipeline:** To execute the entire preprocessing workflow from raw data to the final analysis-ready datasets, run the **`group_pipeline.ipynb`** notebook from top to bottom. This script demonstrates the seamless integration of each member's work.

---

## 6. Preprocessing Pipeline Summary

The `group_pipeline.ipynb` demonstrates a logical flow of data preparation steps:

1.  **Handle Missing Values:** Impute missing data using the median.
2.  **Encode Categorical Variables:** Apply mapping, ordinal encoding, and target encoding based on feature characteristics.
3.  **Train-Test Split:** Divide the data to prevent data leakage in subsequent steps.
4.  **Remove Outliers:** Use the IQR method on the training data to remove extreme values.
5.  **Scale Numerical Features:** Standardize features to have a mean of 0 and a standard deviation of 1.
6.  **Select Features:** Use a Random Forest model to identify and keep the most important predictors.
7.  **Apply PCA:** Reduce the dimensionality of the data to 2 components for visualization.
