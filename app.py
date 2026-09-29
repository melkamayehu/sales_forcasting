import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.forecasting import (
    load_daily_sales,
    add_features,
    train_and_evaluate,
)


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("📈 Sales Forecasting Dashboard")

st.write(
    "A Random Forest time-series forecasting model "
    "using historical sales data, lag features, "
    "and prediction intervals."
)


# -----------------------------
# Run forecasting
# -----------------------------

@st.cache_data
def run_forecast():

    daily_sales = load_daily_sales()

    forecast_df = add_features(
        daily_sales
    )

    (
        model,
        results,
        importance,
        metrics
    ) = train_and_evaluate(
        forecast_df
    )

    return results, importance, metrics


# -----------------------------
# Generate forecast
# -----------------------------

results, importance, metrics = run_forecast()


# -----------------------------
# Model Performance
# -----------------------------

st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Baseline MAE",
        f"{metrics['baseline_mae']:.2f}"
    )

with col2:
    st.metric(
        "Model MAE",
        f"{metrics['model_mae']:.2f}"
    )

with col3:
    st.metric(
        "Baseline RMSE",
        f"{metrics['baseline_rmse']:.2f}"
    )

with col4:
    st.metric(
        "Model RMSE",
        f"{metrics['model_rmse']:.2f}"
    )


st.metric(
    "95% Prediction Interval Margin",
    f"{metrics['prediction_interval_margin']:.2f}"
)


# -----------------------------
# Sales Forecast
# -----------------------------

st.header("Sales Forecast")

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    results["date"],
    results["actual_sales"],
    label="Actual Sales"
)

ax.plot(
    results["date"],
    results["predicted_sales"],
    label="Predicted Sales"
)

ax.fill_between(
    results["date"],
    results["lower_bound"],
    results["upper_bound"],
    alpha=0.2,
    label="95% Prediction Interval"
)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")

ax.set_title(
    "Sales Forecast with 95% Prediction Interval"
)

ax.legend()

ax.grid(alpha=0.3)

st.pyplot(fig)


# -----------------------------
# Feature Importance
# -----------------------------

st.header("Feature Importance")

fig2, ax2 = plt.subplots(
    figsize=(10, 5)
)

ax2.bar(
    importance["feature"],
    importance["importance"]
)

ax2.set_xlabel("Feature")
ax2.set_ylabel("Importance")

ax2.set_title(
    "Random Forest Feature Importance"
)

plt.xticks(rotation=45)

plt.tight_layout()

st.pyplot(fig2)


# -----------------------------
# Forecast Results
# -----------------------------

st.header("Forecast Results")

st.dataframe(
    results,
    use_container_width=True
)


# -----------------------------
# Download Results
# -----------------------------

csv = results.to_csv(
    index=False
)

st.download_button(
    label="Download Forecast Results",
    data=csv,
    file_name="forecast_results.csv",
    mime="text/csv"
)