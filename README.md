# Heart Disease Prediction
Heart disease remains one of the leading causes of death globally, placing a significant burden on healthcare systems and affecting millions of people each year. Early detection and intervention can dramatically improve patient outcomes and reduce healthcare costs. However, identifying heart disease risk factors and making accurate predictions requires a thorough analysis of patient health data.

The dataset analyzed in this project contains various health indicators collected from 302 patients, including key attributes like age, cholesterol levels, blood pressure, chest pain type, and exercise-induced angina. The goal is to develop a machine learning model that accurately predicts whether a patient is likely to have heart disease based on these features.

We can explore the relationships between these health factors and the presence of heart disease with advanced statistics models such as GLM, t and Logistic Regression. By leveraging the power of data, we aim to assist healthcare providers in making more informed decisions, identifying at-risk patients earlier, and ultimately improving treatment outcomes.

In this analysis, we ethemultiple models to select the most accurate and reliable one for predicting heart disease, providing valuable insights into the key health factors that contribute to its onset.

## Project Objective
The main purpose of this project to do analysis of the dataset “[heart-disease.csv](heart-disease.csv)” taken from [kaggle.com](https://www.kaggle.com/datasets/krishujeniya/heart-diseae), select and build a model using the dataset and evaluate the model for accuracy.

### Contributors  
- Sai Ramanan, Prasanna G, John Kalaiselvan
  
### Technologies
- Python
- Jupyter lab
  
### Methods Used
- Data understanding
- Data cleaning and preparation
- Descriptive analysis
- Exploratory data analysis
- Model selection
- Model building
- Model evaluation

## Overview of the Dataset

The dataset used for this analysis was sourced from [Kaggle](https://www.kaggle.com/datasets/krishujeniya/heart-diseae), a platform known for its wide variety of high-quality datasets for data science projects. The dataset consists of **302 patient records** with 14 attributes related to health, all aimed at predicting the likelihood of heart disease.

## Key Features in the Dataset:
1. **Age**: The age of the patient in years.
2. **Sex**: The gender of the patient (1 = male, 0 = female).
3. **Chest Pain Type (cp)**: Four types of chest pain experienced by the patient (1-4 scale).
4. **Resting Blood Pressure (trestbps)**: The patient’s resting blood pressure (in mm Hg).
5. **Cholesterol (chol)**: The patient’s serum cholesterol level (in mg/dL).
6. **Fasting Blood Sugar (fbs)**: Whether the patient's fasting blood sugar is above 120 mg/dL (1 = true, 0 = false).
7. **Resting ECG (restecg)**: Results of the resting electrocardiographic measurement (0-2 scale).
8. **Maximum Heart Rate (thalach)**: The maximum heart rate achieved during testing.
9. **Exercise-Induced Angina (exang)**: Whether exercise induced angina (1 = yes, 0 = no).
10. **Oldpeak**: ST depression induced by exercise relative to rest.
11. **Slope**: The slope of the peak exercise ST segment (1-3 scale).
12. **Number of Major Vessels (ca)**: The number of major vessels colored by fluoroscopy (0-3).
13. **Thalassemia (thal)**: Thalassemia blood disorder variable (1-3 scale).
14. **Target**: The target variable that indicates the presence of heart disease (1 = heart disease, 0 = no heart disease).

## Purpose of the Dataset:
The goal of the dataset is to predict whether a patient is at risk of heart disease based on their medical attributes. This dataset is particularly useful for training machine learning models to predict heart disease and for exploring the relationships between various health metrics and the presence of heart disease.

With these features, we can apply machine learning techniques to identify patterns and predict whether a patient is likely to suffer from heart disease, potentially helping in early diagnosis and improving patient care.

### Why We Need Data Cleaning & Preparation

**Data Cleaning & Preparation** is a crucial step in any data analysis or machine learning project. Raw data often contains errors, inconsistencies, or missing values that can significantly affect the accuracy and reliability of the models. Cleaning and preparing the data ensures that we have a high-quality dataset that can produce valid, accurate, and meaningful results.

### Key Reasons for Data Cleaning & Preparation:

1. **Handling Missing Values**:
   - Datasets often have missing values, which can lead to inaccurate model predictions or errors during model training.
   - Techniques like imputing missing values or removing incomplete records ensure that the dataset is robust and can be processed by machine learning algorithms.

2. **Removing Duplicates**:
   - Duplicate entries in a dataset can skew the results by giving undue importance to certain records. Removing duplicates ensures each observation is treated equally.

3. **Dealing with Outliers**:
   - Outliers are extreme values that may distort the results of statistical analysis or model predictions.
   - Identifying and handling outliers is essential to ensure the model focuses on typical patterns rather than extreme anomalies.

4. **Standardizing Data Formats**:
   - Inconsistent data formats (e.g., different units, date formats, or case sensitivity) can create challenges during analysis.
   - Standardizing formats (such as converting all dates to a common format or all text to lowercase) ensures consistency across the dataset.

5. **Ensuring Data Types Are Correct**:
   - Features must have the correct data type (e.g., numerical values for continuous variables, categorical data for categories).
   - Converting and validating the correct data types ensures models handle the features appropriately.

6. **Improving Model Performance**:
   - A well-prepared dataset leads to better model training and more accurate predictions.
   - Cleaning the data removes noise, focuses on the essential features, and allows machine learning algorithms to identify true patterns and relationships.

7. **Enhancing Interpretability**:
   - A clean dataset is easier to interpret, visualize, and communicate with stakeholders.
   - Consistent and well-prepared data allows clearer insights and actionable business decisions.

## Model Analysis: Generalized Linear Model (GLM) with Logistic Regression Link

In this analysis, we used a **Generalized Linear Model (GLM)** with **Logistic Regression as the link function**. Since the dependent attribute, "Target," is binomial in nature (indicating whether heart disease is present or not), a binomial family of distribution was chosen for the model.

---

### Step-by-Step Process:

#### 1. Train-Test Split
The dataset was split into training and testing sets to evaluate the model's performance on unseen data. A **70/30 split** was used:
- **70%** for training.
- **30%** for testing.

This split allows the model to learn from a significant portion of the dataset while keeping enough data aside for testing to evaluate the model's performance.

#### 2. Model Initialization
- **Logistic Regression**:
  - Logistic regression is a simple and interpretable statistical model, which makes it easy to understand the relationship between the independent attributes and the target variable (heart disease).
  - We focused on identifying the independent attributes that have a significant impact on the target variable (i.e., attributes that are statistically significant).
  
- **Iterations**:
  1. **First Iteration (with outliers)**:
     - In the first iteration, the model was trained and tested using the entire dataset, without removing any outliers.
  2. **Second Iteration (without outliers)**:
     - In the second iteration, the model was trained and tested after removing the outliers from statistically significant attributes, such as **ca** (number of major vessels) and **thalach** (maximum heart rate). Removing these outliers helped improve the model's performance.

#### 3. Model Performance Metrics
To evaluate the model's performance, we used the following key metrics:
- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **F1-score**: The weighted average of precision and recall.

The confusion matrix provided a clear breakdown of true positives, true negatives, false positives, and false negatives, which helped evaluate the model's accuracy and error rate.

#### 4. Model Inference:
- After removing outliers from significant attributes (like **ca** and **thalach**), the **accuracy, precision, recall, and F1-score** increased significantly—by approximately **10%**.
- The final model accuracy currently stands at **89.16%**.

---

### Model Analysis Results:

#### Performance Metrics:
- **Accuracy**: **89.16%**
  - The model correctly classified 89.16% of the cases, indicating strong overall performance.
  
- **Precision**: **90.2%**
  - Of all patients predicted to have heart disease, 90.2% actually do have heart disease, which reflects high precision.

- **Recall**: **0.92**
  - The model correctly identified 92% of patients who truly have heart disease, indicating that it performs well in identifying positive cases.

- **F1-Score**: **0.91**
  - The F1-score balances precision and recall, yielding a strong performance measure of 0.91.

#### Confusion Matrix:
The confusion matrix helps in visualizing how well the model performs:

|              | Predicted No Disease | Predicted Disease |
|--------------|----------------------|-------------------|
| **Actual No Disease** | 28                       | 5                 |
| **Actual Disease**    | 4                        | 46                |

- **True Positives (TP)**: 46 patients were correctly identified as having heart disease.
- **True Negatives (TN)**: 28 patients were correctly identified as not having heart disease.
- **False Positives (FP)**: 5 patients were incorrectly predicted to have heart disease when they do not.
- **False Negatives (FN)**: 4 patients were incorrectly predicted to not have heart disease when they actually do.

### Conclusion:
The **Generalized Linear Model (GLM)** with **Logistic Regression link** provides a simple yet powerful approach for predicting heart disease. The removal of outliers from key attributes like **ca** (number of major vessels) and **thalach** (maximum heart rate) significantly improved the model's performance, boosting the accuracy to **89.16%**. This performance increase is reflected in higher precision, recall, and F1-scores, making this a robust and reliable model for heart disease prediction.




### License
[GNU General Public License (GPL)] (https://www.gnu.org/licenses/gpl-3.0.en.html)

