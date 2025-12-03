# IT2011 - AI/ML Project: Predicting Depression Risk in University Students

**Group ID:** PGN-304  
**Date:** December 2025

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Library](https://img.shields.io/badge/Library-Pandas-150458)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 1. Project Overview

Mental health in universities is a critical issue. This project aims to develop a machine learning solution to address the real-world problem of student depression. 

Our goal is to build a classification model that can identify university students at a higher risk of depression based on a combination of academic, lifestyle, and psychological factors. By leveraging machine learning, we facilitate a shift from reactive counseling to proactive outreach, offering targeted interventions to support student well-being.

**Key Objective:** Minimize **False Negatives** (missed diagnoses) to ensure at-risk students are identified. Therefore, our primary evaluation metric is **Recall**.

---

## 2. Dataset Details

* **Dataset Name:** Student Depression and Mental Health Dataset
* **Source:** [Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
* **Description:** A tabular dataset containing over 3,500 records and approximately 20 features. The data includes:
    * **Academic Indicators:** CGPA, Attendance, etc.
    * **Lifestyle Factors:** Sleep Duration, Dietary Habits, etc.
    * **Psychological Metrics:** Financial Stress, Study Pressure, etc.
* **Target Variable:** `Depression` (Binary Classification).

---

## 3. Group Members & Contributions

This project was a collaborative effort. Members managed specific stages of the preprocessing pipeline and developed individual model implementations.

| Member Name | IT Number | Assigned Preprocessing Role |
| :--- | :--- | :--- |
| GUNASEKARA J.D.C | IT24101409 | 1. Handling Missing Values |
| PUNCHIHEWA P.K.N | IT24101243 | 2. Categorical Feature Encoding |
| RANASINGHE R.G.P.D | IT24100910 | 3. Outlier Removal |
| CHANDRASEKARA R M S | IT24101383 | 4. Feature Scaling (Standardization) |
| DE SILVA THHD | IT24101010 | 5. Feature Selection |
| BHAGYA J.P.J. | IT24101426 | 6. PCA (Dimensionality Reduction) |

---

## 4. Repository Structure

The project is organized into modular folders for raw data, individual contributions, pipelines, and results.

```text
├── data/
│   └── raw/
│       └── Student Depression Dataset.csv       # Original dataset
│
├── models_notebooks/                            # Individual ML Algorithms
│   ├── AIML_Decisiontree.ipynb
│   ├── IT24100910_Logistic Regression.ipynb
│   ├── IT24101010_RandomForest_Analysis.ipynb
│   ├── IT24101243_SVM (1).ipynb
│   ├── IT24101409_MLP (1).ipynb
│   └── KNN (1).ipynb
│
├── preprocessing_notebooks/                     # Data Cleaning & Transformation
│   ├── IT24100910_remove_outliers.ipynb
│   ├── IT24101010_feature_selection.ipynb
│   ├── IT24101243_encoding.ipynb
│   ├── IT24101383_scaling.ipynb
│   ├── IT24101409_handling_missing_values.ipynb
│   ├── IT24101426_PCA.ipynb
│   └── group_pipeline.ipynb                     # INTEGRATED PIPELINE (Run this first)
│
├── preprocessing_output/                        # Generated Artifacts
│   ├── eda_visualizations/                      # Heatmaps, Boxplots, Distributions
│   └── preprocessed_datasets/                   # Final clean data for modeling
│       ├── X_train_final.csv
│       ├── X_test_final.csv
│       ├── y_train_final.csv
│       └── y_test_final.csv
│
└── results/
    ├── Group_Model_Comparison.ipynb             # Final Benchmarking & Conclusion
    └── download (2).png                         # Performance Visualizations
````
-----

## 5\. Methodology

### A. Preprocessing Pipeline

To prepare the raw data for machine learning, we implemented a strict pipeline (found in `preprocessing_notebooks/group_pipeline.ipynb`):

1.  **Imputation:** Missing values filled using median strategies.
2.  **Encoding:** Categorical variables transformed using Label and Target Encoding.
3.  **Outlier Handling:** Extreme values removed using the IQR method on training data.
4.  **Scaling:** Standardization (Z-score normalization) applied to numerical features.
5.  **Feature Selection:** Random Forest importance used to select the most predictive features.
6.  **Dimensionality Reduction:** PCA applied for visualization and noise reduction.

### B. Machine Learning Models

We trained and hyper-tuned six distinct algorithms to find the best fit:

  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * Support Vector Machine (SVM)
  * Decision Tree
  * Random Forest
  * Multi-Layer Perceptron (MLP/Neural Network)

-----

## 6\. Model Evaluation & Results

We evaluated models based on **Accuracy**, **F1-Score**, and **AUC**, with a specific focus on **Recall** for the "Depressed" class.

*Rationale: In mental health screening, a False Negative (missing a depressed student) is worse than a False Positive. High Recall ensures we identify as many at-risk students as possible.*

### Performance Table

| Model | Accuracy | Recall (Depressed) | F1-Score (Depressed) | AUC |
| :--- | :--- | :--- | :--- | :--- |
| **MLP (Neural Network)** | 0.82 | **0.89** | 0.85 | N/A |
| **Logistic Regression** | 0.84 | 0.88 | 0.86 | **0.913** |
| **SVM** | 0.84 | 0.88 | 0.86 | N/A |
| **Random Forest** | 0.84 | 0.87 | 0.86 | 0.830 |
| **KNN** | 0.81 | 0.84 | 0.83 | N/A |
| **Decision Tree** | 0.77 | 0.84 | 0.81 | N/A |

### Conclusion

The **MLP (Neural Network)** was selected as the optimal model for this specific use case, achieving the highest **Recall of 89%**.

*Note: Logistic Regression and SVM also performed exceptionally well, offering a good balance between precision and recall if model interpretability becomes a priority in the future.*

-----

## 7\. How to Run the Code

### Prerequisites

Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn category_encoders
```

### Execution Steps

1.  **Clone the Repository:** Clone this repo to your local machine.
2.  **Data Setup:** Ensure `Student Depression Dataset.csv` is located in `data/raw/`.
3.  **Run Preprocessing:** Open and run `preprocessing_notebooks/group_pipeline.ipynb`.
      * *This will generate the clean CSV files in `preprocessing_output/preprocessed_datasets/`.*
4.  **Run Models:** You can now run any notebook in the `models_notebooks/` folder.
5.  **View Comparison:** Open `results/Group_Model_Comparison.ipynb` to see the final comparative analysis and charts.

<!-- end list -->


```
