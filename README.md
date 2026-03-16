# Customer Churn Prediction Dashboard

A Machine Learning web application that predicts whether a telecom customer is likely to churn using behavioral and service usage data.

The model was trained using XGBoost and deployed as an interactive Streamlit dashboard.

---

## Problem Statement

Customer churn is a major issue for telecom companies. Retaining existing customers is significantly cheaper than acquiring new ones.

This project predicts the probability of a customer leaving the service based on their profile and service usage.

---

## Machine Learning Workflow

1. Data Cleaning and Preprocessing
2. Feature Engineering
3. Encoding Categorical Variables
4. Model Training (XGBoost)
5. Cross Validation
6. Model Export using Joblib
7. Deployment using Streamlit

---

## Model Performance

Accuracy: **~80%**

Cross Validation Score: **~0.80**

---

## Features Used

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Payment Method
- Paperless Billing
- Tech Support
- Online Security
- Streaming Movies

---

## Application Preview

![App Screenshot](images/churn_app_dashboard.png)

---

## Tech Stack

- Python
- Pandas
- Scikit-Learn
- XGBoost
- Streamlit
- Joblib

---

## How to Run the Project

Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Author

Farhan Tanvir  
Machine Learning Enthusiast