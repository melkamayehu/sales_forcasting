from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "sales_forecasting_model.pkl"
RESULTS_PATH = PROJECT_ROOT / "forecast_results.csv"


FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "month",
    "year",
]


def load_daily_sales() -> pd.DataFrame:
    """Load the raw data and aggregate sales by day."""
    df = pd.read_csv(DATA_PATH)

    required_columns = {"date", "sales"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["sales"] = pd.to_numeric(
        df["sales"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "sales"]
    )

    daily_sales = (
        df.groupby(
            "date",
            as_index=False
        )["sales"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily_sales


def add_features(
    daily_sales: pd.DataFrame
) -> pd.DataFrame:

    """Create calendar and lag features."""
    forecast_df = daily_sales.copy()

    forecast_df["month"] = (
        forecast_df["date"].dt.month
    )

    forecast_df["year"] = (
        forecast_df["date"].dt.year
    )

    forecast_df["lag_1"] = (
        forecast_df["sales"].shift(1)
    )

    forecast_df["lag_7"] = (
        forecast_df["sales"].shift(7)
    )

    forecast_df["lag_14"] = (
        forecast_df["sales"].shift(14)
    )

    # Remove rows created by the lag features
    forecast_df = (
        forecast_df
        .dropna(subset=FEATURES)
        .reset_index(drop=True)
    )

    return forecast_df


def train_and_evaluate(
    forecast_df: pd.DataFrame
):
    """Train the model, evaluate it, and calculate prediction intervals."""

    X = forecast_df[FEATURES]
    y = forecast_df["sales"]

    # Keep chronological order
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # -----------------------------
    # Baseline
    # -----------------------------

    last_train_sales = y_train.iloc[-1]

    baseline_predictions = np.full(
        len(y_test),
        last_train_sales
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    baseline_rmse = (
        mean_squared_error(
            y_test,
            baseline_predictions
        ) ** 0.5
    )

    # -----------------------------
    # Calibration split
    # -----------------------------

    cal_split = int(len(X_train) * 0.8)

    X_model_train = X_train.iloc[:cal_split]
    X_cal = X_train.iloc[cal_split:]

    y_model_train = y_train.iloc[:cal_split]
    y_cal = y_train.iloc[cal_split:]

    # -----------------------------
    # Random Forest
    # -----------------------------

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_model_train,
        y_model_train
    )

    # -----------------------------
    # Calibration errors
    # -----------------------------

    cal_predictions = model.predict(
        X_cal
    )

    cal_errors = np.abs(
        y_cal.to_numpy()
        - cal_predictions
    )

    # 95th percentile error
    error_margin = float(
        np.quantile(
            cal_errors,
            0.95
        )
    )

    # -----------------------------
    # Final test predictions
    # -----------------------------

    predictions = model.predict(
        X_test
    )

    model_mae = mean_absolute_error(
        y_test,
        predictions
    )

    model_rmse = (
        mean_squared_error(
            y_test,
            predictions
        ) ** 0.5
    )

    # Prediction interval
    lower_bound = (
        predictions - error_margin
    )

    upper_bound = (
        predictions + error_margin
    )

    # -----------------------------
    # Results table
    # -----------------------------

    results = pd.DataFrame({
        "date": (
            forecast_df["date"]
            .iloc[split_index:]
            .to_numpy()
        ),

        "actual_sales": (
            y_test.to_numpy()
        ),

        "baseline_prediction": (
            baseline_predictions
        ),

        "predicted_sales": (
            predictions
        ),

        "lower_bound": (
            lower_bound
        ),

        "upper_bound": (
            upper_bound
        )
    })

    # -----------------------------
    # Feature importance
    # -----------------------------

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": (
            model.feature_importances_
        )
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    metrics = {
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "model_mae": model_mae,
        "model_rmse": model_rmse,
        "prediction_interval_margin": error_margin
    }

    return (
        model,
        results,
        importance,
        metrics
    )


def plot_results(
    results: pd.DataFrame
) -> None:

    """Plot actual sales, predictions, and prediction interval."""

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        results["date"],
        results["actual_sales"],
        label="Actual Sales"
    )

    plt.plot(
        results["date"],
        results["predicted_sales"],
        label="Predicted Sales"
    )

    plt.fill_between(
        results["date"],
        results["lower_bound"],
        results["upper_bound"],
        alpha=0.2,
        label="95% Prediction Interval"
    )

    plt.xlabel("Date")
    plt.ylabel("Sales")

    plt.title(
        "Sales Forecast with 95% Prediction Interval"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:

    # Load and prepare data
    daily_sales = load_daily_sales()

    # Create forecasting features
    forecast_df = add_features(
        daily_sales
    )

    # Train and evaluate
    (
        model,
        results,
        importance,
        metrics
    ) = train_and_evaluate(
        forecast_df
    )

    # Save model
    joblib.dump(
        model,
        MODEL_PATH
    )

    # Save forecast results
    results.to_csv(
        RESULTS_PATH,
        index=False
    )

    # Print results
    print(
        f"Baseline MAE: "
        f"{metrics['baseline_mae']:.2f}"
    )

    print(
        f"Model MAE: "
        f"{metrics['model_mae']:.2f}"
    )

    print(
        f"Baseline RMSE: "
        f"{metrics['baseline_rmse']:.2f}"
    )

    print(
        f"Model RMSE: "
        f"{metrics['model_rmse']:.2f}"
    )

    print(
        "95% prediction interval margin: "
        f"{metrics['prediction_interval_margin']:.2f}"
    )

    print("\nFeature importance:")

    print(
        importance.to_string(
            index=False
        )
    )

    print(
        f"\nSaved model: "
        f"{MODEL_PATH}"
    )

    print(
        f"Saved results: "
        f"{RESULTS_PATH}"
    )

    # Plot results
    plot_results(
        results
    )


if __name__ == "__main__":
    main()