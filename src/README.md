# Sales Forecasting

A machine learning project for analyzing daily sales and predicting future sales.

## What I did

* Cleaned and explored the sales data
* Looked at weekly, monthly, and yearly patterns
* Created lag features using previous sales
* Built a baseline forecast
* Trained a Random Forest Regressor
* Evaluated the model using MAE and RMSE
* Compared actual and predicted sales
* Checked which features were most important

## Features

* `lag_1` – sales from the previous day
* `lag_7` – sales from 7 days ago
* `lag_14` – sales from 14 days ago
* `month`
* `year`

## Tools

Python, Pandas, NumPy, Matplotlib, Scikit-learn, Joblib

## Model

**Random Forest Regressor**

The trained model is saved as `sales_forecasting_model.pkl`.
