# IT2011 - AI/ML Project: Predicting Depression Risk in University Students

**Group ID:** PGN-304
**Date:** September 25, 2025

---

## 1. Project Overview

This project aims to develop a machine learning solution to address the critical real-world problem of student mental health. Our goal is to build a classification model that can identify university students at a higher risk of depression based on a combination of academic, lifestyle, and psychological factors. By leveraging machine learning, we hope to provide a tool that enables educational institutions to shift from reactive counseling to proactive outreach, offering early and targeted interventions to support student well-being.

---

## 2. Dataset Details

- [cite_start]**Dataset Name:** Student Depression and Mental Health Dataset [cite: 1084]
- [cite_start]**Source:** Kaggle [cite: 1085]
- **Link:** [https://www.kaggle.com/datasets/hopesb/student-depression-dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
- [cite_start]**Description:** This is a tabular dataset containing over 3,500 records and approximately 20 features. [cite: 1093, 1096] [cite_start]The data includes a rich mix of academic indicators (e.g., CGPA), lifestyle factors (e.g., Sleep Duration), and psychological metrics (e.g., Financial Stress). [cite: 1143, 1145] [cite_start]The target variable is a binary indicator, `Depression`, making this a supervised classification task. [cite: 1191, 1231]

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

The project follows the specified submission structure:Group_ID/
├── README.md
├── data/
│   └── raw/
│       └── Student Depression Dataset.csv
├── notebooks/
│   ├── IT24101243_Missing_Values.ipynb
│   ├── IT24100910_Encoding.ipynb
│   ├── ... (individual notebooks for each member)
│   └── group_pipeline.ipynb
└── results/
├── eda_visualizations/
│   └── (saved plots like heatmaps, boxplots, etc.)
└── outputs/
└── (final processed datasets)

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
