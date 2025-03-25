# Project Overview: Banking Customer Churn Prediction

### ⚙️ Code: [Churn_Prediction_Model.ipynb](https://github.com/lekhakasinadhuni07/Churn-Prediction-Model/blob/main/Banking_Churn_Prediction_model.ipynb)


### 🎯 Goals:
- Predict whether a bank customer is likely to churn based on historical data.
- Analyze key factors influencing customer retention.
- Build an accurate machine learning model to classify customers as "churn" or "retain.


### 📖 Description:
This project focuses on customer churn prediction in the banking industry using machine learning algorithms. The dataset includes demographic, financial, and transaction-related features to identify patterns among customers likely to leave.

✔ **Data Cleaning & Preprocessing** (Handling missing values, encoding categorical variables)

✔ **Exploratory Data Analysis** (Visualizing correlations and trends)

✔ **Feature Engineering** (Transforming data for better predictions)

✔ **Model Training & Evaluation** (Random Forest Classifier)

✔ **Performance Metrics** (Confusion Matrix, Accuracy Score, Classification Report)


## 📊 Model Evaluation Summary

Tested the following machine learning models to predict customer churn, evaluating them using accuracy, precision, recall, and F1-score.

| Model                        | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|------------------------------|----------|----------------------|------------------|-------------------|
| **Random Forest**            | 86.65%   | 76%                  | 46%              | 58%               |
| **Logistic Regression**      | 81.1%    | 55%                  | 20%              | 29%               |
| **Support Vector Machine**   | 79.7%    | 48%                  | 32%              | 38%               |
| **K-Nearest Neighbors (KNN)**| 83%      | 61%                  | 37%              | 46%               |
| **Gradient Boosting**        | 86.75%   | 75%                  | 49%              | 59%               |

---

## 📌 Model Insights

### **1️⃣ Random Forest Classifier**
- **Accuracy:** 86.65%
- **Strengths:** High accuracy and good precision.
- **Weaknesses:** Recall for churned customers (46%) is relatively low, meaning the model misses a significant number of actual churn cases.

### **2️⃣ Logistic Regression**
- **Accuracy:** 81.1%
- **Strengths:** Simple and interpretable model.
- **Weaknesses:** Poor recall (20%) for churned customers, making it unreliable for identifying at-risk customers.

### **3️⃣ Support Vector Machine (SVM)**
- **Accuracy:** 79.7%
- **Strengths:** Effective in complex decision boundaries.
- **Weaknesses:** Low recall (32%) and F1-score (38%) for churned customers, making it less reliable for churn prediction.

### **4️⃣ K-Nearest Neighbors (KNN)**
- **Accuracy:** 83%
- **Strengths:** Higher precision (61%) than SVM and Logistic Regression.
- **Weaknesses:** Struggles with recall (37%) and may be sensitive to data size and distribution.

### **5️⃣ Gradient Boosting Classifier**
- **Accuracy:** **86.75% (Best Performance)**
- **Strengths:** Highest accuracy and a good balance of precision (75%) and recall (49%).
- **Weaknesses:** Computationally expensive compared to simpler models.

---

## ✅ **Conclusion & Recommendation**

Based on the evaluation, the **Gradient Boosting Classifier** emerges as the best model with the highest accuracy (**86.75%**) and a balanced trade-off between precision and recall.

### 🚀 **Next Steps:**
- **Feature Engineering:** Identify additional customer behavior indicators.
- **Data Balancing Techniques:** Improve recall by handling class imbalance.
- **Hyperparameter Tuning:** Optimize the model for better performance.

Implementing the Gradient Boosting model can enhance the bank’s ability to **identify and retain high-risk customers**, leading to improved customer satisfaction and reduced churn rates.

---

## 🛠 Skills Demonstrated:
✅ **Machine Learning** (Supervised Learning, Classification)

✅ **Data Preprocessing & Feature Engineering**

✅ **Data Visualization** (Matplotlib, Seaborn)

✅ **Model Evaluation** (Confusion Matrix, Precision-Recall, Accuracy)

✅ **SQL-like Data Handling** (Pandas, NumPy)

## 🖥 Technologies & Tools Used:
🔹 **Python** (Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn)

🔹 **Machine Learning** (Random Forest Classifier)

🔹 **Feature Encoding** (Label Encoding, One-Hot Encoding)

🔹 **Data Handling** (Pandas for CSV processing, handling missing values)

---

**📌 Note:** This report is based on a dataset of 2,000 samples. Further testing and model fine-tuning may be needed before deployment.
