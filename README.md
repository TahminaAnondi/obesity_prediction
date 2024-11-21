# # **Obesity Prediction Using Machine Learning**

## **Table of Contents**
1. [Overview](#overview)  
2. [Objective](#objective)  
3. [Dataset Description](#dataset-description)  
4. [Methodology](#methodology)  
5. [Feature Engineering](#feature-engineering)  
6. [Model Selection and Justification](#model-selection-and-justification)  
7. [Handling Overfitting](#handling-overfitting)  
8. [Evaluation Metrics](#evaluation-metrics)  
9. [How to Run the Project](#how-to-run-the-project)  
10.[Formulas Explained with Examples](#formulas-explained-with-examples)  
11.[Output Highlights](#output-highlights)  

---

## **1. Overview**
This project predicts obesity status based on demographic and lifestyle factors using machine learning models. A Streamlit-based web application is provided for model selection and performance evaluation.



## **2. Objective**
The project aims to classify individuals as obese or not (`ob_BMI`) using various predictors to understand and predict obesity risk factors.



## **3. Dataset Description**
### **Files**
- `append_BMI_converted_training.csv` (Training Dataset)  
- `append_BMI_converted_testing.csv` (Testing Dataset)

### **Features**
The dataset includes:
- **Demographics:** Gender, age, income, education, race, marital status.  
- **Health Metrics:** BMI, waist circumference, cotinine levels.  
- **Lifestyle Indicators:** Physical activity, alcohol consumption, smoking status, sleep duration.

### **Target Variable**
`ob_BMI`: A binary column indicating whether an individual is obese.

---

## **4. Methodology**
1. **Data Loading:** Datasets are loaded using Streamlit’s `@st.cache_data` for efficiency.  
2. **Missing Values:** Imputed using mean values with `SimpleImputer`.  
3. **Feature Selection:** All columns except `ob_BMI` are used as predictors.  
4. **Model Training:** Users can select from three models:
   - Random Forest
   - Naive Bayes
   - Gradient Boosting Machine  
5. **Cross-Validation:** A 5-fold StratifiedKFold ensures robust performance evaluation.  
6. **Evaluation:** Accuracy, ROC-AUC, and classification reports are calculated.

---

## **5. Feature Engineering**
- **Renaming Features:** Columns are mapped to descriptive names for clarity.  
- **Imputation:** Missing values are replaced with the mean of respective columns.

---

## **6. Model Selection and Justification**
### **1. Random Forest**
- **Why:** Handles non-linear relationships and feature interactions.  
- **Parameters:**  
  - `n_estimators=30`: Uses 30 decision trees.  
  - `max_depth=2`: Limits tree depth to prevent overfitting.

### **2. Naive Bayes**
- **Why:** Computationally efficient for categorical predictors.  
- **Model Type:** GaussianNB assumes continuous variables follow a normal distribution.

### **3. Gradient Boosting Machine**
- **Why:** Builds sequential weak learners to improve predictions.  
- **Parameters:**  
  - `learning_rate=0.001`: Reduces each tree's contribution.  
  - `max_depth=2`, `min_samples_leaf=5`: Regularizes tree complexity.

---

## **7. Handling Overfitting**
1. **Cross-Validation:** Ensures generalization across different data splits.  
2. **Regularization:** Parameters like `max_depth` and `min_samples_leaf` reduce complexity.  
3. **Simpler Models:** Naive Bayes inherently avoids overfitting due to its assumptions.

---

## **8. Evaluation Metrics**
1. **Accuracy:** Measures the proportion of correct predictions.  
2. **ROC-AUC:** Evaluates the model's ability to distinguish between classes.  
3. **Classification Report:** Includes precision, recall, F1-score, and support.

---

## **9. How to Run the Project**
1. Clone the repository:
   ```
   git clone <repository_url>
   ```
2. Install dependencies:
    ```
    pip install streamlit pandas scikit-learn
    ```
3. Run the Streamlit application:
    ```
    streamlit run app.py
    ```
4. Upload the training and testing datasets when prompted.
5. Select a classifier in the sidebar to view performance metrics and prediction results.
