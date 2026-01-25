# 📊 Multiple Linear Regression – Tip Prediction App

A **Machine Learning web application** built using **Multiple Linear Regression** to predict the **tip amount** based on the **total bill** and **group size**.  
The application is deployed using **Streamlit** with interactive visualizations and performance metrics.

---

## 📌 Project Overview

In restaurants, tipping behavior depends on multiple factors such as:
- Total bill amount
- Number of people in a group

This project uses **Multiple Linear Regression** to model the relationship between these variables and predict the expected tip amount.  
The goal is to demonstrate how regression models can be used for **real-world numerical predictions**.

---

## 🚀 Features

- 📈 Predict tip amount using **Multiple Linear Regression**
- 🧮 Uses **two independent variables**:
  - Total Bill
  - Group Size
- 📊 Interactive data visualization
- 📐 Model evaluation metrics:
  - MAE
  - RMSE
  - R² Score
  - Adjusted R²
- 🎛️ Slider-based user input for predictions
- 🎨 Custom UI styling using CSS

---

## 🧠 Machine Learning Workflow

1. **Dataset Loading**
   - Uses the `tips` dataset from Seaborn

2. **Data Preparation**
   - Selected features: `total_bill`, `size`
   - Target variable: `tip`

3. **Data Preprocessing**
   - Train-test split (80% training, 20% testing)
   - Feature scaling using `StandardScaler`

4. **Model Training**
   - Trained using `LinearRegression` from Scikit-learn

5. **Model Evaluation**
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - R² Score
   - Adjusted R² Score

6. **Prediction**
   - User inputs total bill and group size
   - Model predicts tip amount instantly

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Frontend:** Streamlit  
- **Machine Learning:** Scikit-learn  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Styling:** Custom CSS  

---

## 📂 Project Structure

📁 Multiple-Linear-Regression-Tip-Prediction
│
├── app.py # Streamlit application
├── style1.css # Custom styling
├── README.md # Project documentation
└── requirements.txt # Dependencies


---

## 📊 Dataset Information

The dataset contains restaurant billing information with the following features:

- `total_bill` – Total bill amount in dollars
- `size` – Number of people in the group
- `tip` – Tip amount (target variable)

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/multiple-linear-regression-tip-prediction.git
cd multiple-linear-regression-tip-prediction
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Streamlit App
streamlit run app.py
4️⃣ Open in Browser
http://localhost:8501
📈 Model Performance Metrics
MAE: Measures average prediction error

RMSE: Penalizes larger errors

R² Score: Explains variance in the data

Adjusted R²: Accounts for number of predictors

These metrics help evaluate the reliability and accuracy of the regression model.

🧠 Model Interpretation
Coefficient (Total Bill): Indicates how tip changes with bill amount

Coefficient (Group Size): Shows impact of group size on tip

Intercept: Base value when inputs are zero

📌 Tip amount depends on both the bill value and the number of people.

📌 Business Use Case
This model can be useful for:

Restaurant analytics

Customer behavior analysis

Educational demonstrations of regression models

🔮 Future Enhancements
📊 Add residual analysis plots

📈 Compare simple vs multiple linear regression

🌐 Deploy on cloud platforms

📱 Mobile-responsive UI

🧠 Add categorical variables
