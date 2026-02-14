# Heart Disease Prediction System

## Problem Statement

Heart disease is one of the leading causes of death globally. Early detection and prediction of heart disease can significantly improve patient outcomes and reduce mortality rates. This project aims to develop a comprehensive machine learning system that can predict the likelihood of heart disease in patients based on various clinical and demographic features.

The system implements and compares six different machine learning algorithms to identify the most effective approach for heart disease prediction, providing healthcare professionals with a reliable tool for early diagnosis and risk assessment.

## Dataset Description

The heart disease dataset contains **1025 records** with **14 attributes** related to cardiovascular health indicators:

### Features:
- **age**: Age of the patient (years)
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol level (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-4)
- **thal**: Thalassemia (0-3)

### Target Variable:
- **target**: Presence of heart disease (1 = disease, 0 = no disease)

## Models Used

The system implements six different machine learning algorithms:

1. **K-Nearest Neighbors (KNN)**: Instance-based learning algorithm
2. **Logistic Regression**: Linear classification algorithm
3. **Decision Tree**: Tree-based classification algorithm
4. **Naive Bayes**: Probabilistic classification algorithm
5. **Random Forest**: Ensemble method using multiple decision trees
6. **XGBoost**: Gradient boosting ensemble method

## Model Comparison Table

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|-------|----------|-----------|-----------|---------|----------|-----------|
| K-Nearest Neighbors | 0.8852 | 0.9444 | 0.8906 | 0.8852 | 0.8869 | 0.7705 |
| Logistic Regression | 0.8852 | 0.9550 | 0.8906 | 0.8852 | 0.8869 | 0.7705 |
| Decision Tree | 0.8033 | 0.8056 | 0.8133 | 0.8033 | 0.8065 | 0.6067 |
| Naive Bayes | 0.8525 | 0.9283 | 0.8594 | 0.8525 | 0.8548 | 0.7051 |
| Random Forest | 0.8689 | 0.9394 | 0.8750 | 0.8689 | 0.8708 | 0.7378 |
| XGBoost | 0.8852 | 0.9550 | 0.8906 | 0.8852 | 0.8869 | 0.7705 |

## Model Performance Observations

### Top Performing Models:
1. **Logistic Regression & XGBoost** (tied): Both achieve the highest AUC score of 0.955, indicating excellent discrimination ability
2. **K-Nearest Neighbors**: Matches the top accuracy and provides robust performance across all metrics
3. **Random Forest**: Shows strong ensemble performance with good generalization

### Key Insights:

#### **Best Overall Performance:**
- **Logistic Regression**, **XGBoost**, and **KNN** all achieved the highest accuracy of 88.52%
- **Logistic Regression** and **XGBoost** demonstrated superior AUC scores (0.955), indicating excellent model discrimination

#### **Model-Specific Observations:**

**Logistic Regression:**
- Excellent performance across all metrics
- High AUC score indicates strong predictive capability
- Simple, interpretable model suitable for clinical use

**XGBoost:**
- Matches Logistic Regression performance
- Robust gradient boosting provides good generalization
- Handles feature interactions effectively

**K-Nearest Neighbors:**
- Strong performance with instance-based learning
- Good balance across all evaluation metrics
- Effective for this dataset size and complexity

**Random Forest:**
- Solid ensemble performance (86.89% accuracy)
- Good feature importance insights
- Reliable predictions with reduced overfitting

**Naive Bayes:**
- Respectable performance (85.25% accuracy) given its simplicity
- Fast training and prediction
- Good baseline model

**Decision Tree:**
- Lowest performance (80.33% accuracy)
- Prone to overfitting despite max_depth constraint
- Provides good interpretability but sacrifices accuracy

### **Recommendations:**

1. **For Clinical Deployment**: **Logistic Regression** - Best balance of performance, interpretability, and reliability
2. **For Complex Pattern Recognition**: **XGBoost** - Superior handling of non-linear relationships
3. **For Quick Prototyping**: **KNN** - Simple implementation with competitive performance

### **Clinical Significance:**
The high performance metrics (>85% accuracy, >0.92 AUC) across multiple models demonstrate the dataset's strong predictive potential for heart disease detection. The consistently high AUC scores indicate that these models can effectively distinguish between high-risk and low-risk patients, making them valuable tools for clinical decision support.

## Installation and Setup

1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
