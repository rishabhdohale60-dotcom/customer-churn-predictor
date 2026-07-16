# 📱 Customer Churn Predictor

A Machine Learning project that predicts whether a telecom customer will churn using ML Pipeline.

## 📊 Dataset
- 7043 customer records from Telco Customer Churn dataset
- Source: Kaggle

## 🔍 Features Used
- Tenure, MonthlyCharges, TotalCharges
- Contract Type, Payment Method
- Internet Service, Streaming Services
- Online Security, Tech Support etc.

## 🎯 Target
- Churn: 0 (Stay) / 1 (Leave)

## ⚙️ Tech Stack
- Python
- Pandas
- Scikit-learn
- NumPy

## 📈 Results
| Model | Accuracy | Churn Recall |
|-------|----------|--------------|
| Logistic Regression | 78.7% | 52% |
| LR + class_weight balanced | 73.1% | 79% 🔥 |

## 🔄 ML Pipeline
1. Data Loading & Exploration
2. TotalCharges string → float
3. Churn Yes/No → 1/0
4. One Hot Encoding
5. Train Test Split (80/20)
6. ML Pipeline (StandardScaler + LogisticRegression)
7. Class Weight Balancing

## 💡 Key Learnings
- ML Pipeline makes code cleaner and production-ready
- Imbalanced dataset → use class_weight='balanced'
- Business mein Accuracy se zyada Recall matter karta hai!
- Churn Recall 52% → 79% improved!
