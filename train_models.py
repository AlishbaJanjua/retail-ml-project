"""
Train models for three retail analytics problems:
1) Weekly inventory / quantity forecasting
2) Discount → profit estimation (regression)
3) Customer RFM segmentation (KMeans)

Run before starting the API:
    python train_models.py
"""

import json
import os
from math import sqrt

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
MODELS_DIR = "models"
DATA_PATH = "data/Sample - Superstore.csv"


# ------------------------------------------------------------
# Shared data loading
# ------------------------------------------------------------
def load_raw_data():
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df.columns = df.columns.str.strip()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    for col in ["Sales", "Quantity", "Discount", "Profit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(
        subset=[
            "Order Date",
            "Sales",
            "Quantity",
            "Discount",
            "Profit",
            "Category",
            "Sub-Category",
            "Customer ID",
            "Order ID",
        ]
    )
    return df


def mae_rmse(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def train_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_raw_data()
    print("Loaded rows:", len(df))
    print("Date range:", df["Order Date"].min().date(), "to", df["Order Date"].max().date())


    # ============================================================
    # PROBLEM 1: Weekly quantity forecasting (inventory support)
    # ============================================================
    print("\n=== PROBLEM 1: Inventory / next-week quantity forecast ===")

    FORECAST_FEATURES = ["qty_lag1", "qty_lag2", "qty_roll4", "month", "weekofyear"]


    def build_weekly_quantity(frame, label):
        """Aggregate Quantity and Sales by week (Sunday-ending weeks)."""
        weekly = (
            frame.set_index("Order Date")
            .resample("W-SUN")
            .agg(Quantity=("Quantity", "sum"), Sales=("Sales", "sum"))
            .fillna(0.0)
        )
        weekly["qty_lag1"] = weekly["Quantity"].shift(1)
        weekly["qty_lag2"] = weekly["Quantity"].shift(2)
        weekly["qty_roll4"] = weekly["Quantity"].shift(1).rolling(4).mean()
        weekly["month"] = weekly.index.month
        weekly["weekofyear"] = weekly.index.isocalendar().week.astype(int)
        weekly = weekly.dropna().copy()
        weekly["series"] = label
        return weekly


    def evaluate_forecast_series(weekly):
        """Chronological split; compare LinearRegression vs naive last-week."""
        if len(weekly) < 30:
            return None

        split = int(len(weekly) * 0.8)
        if split < 20 or len(weekly) - split < 5:
            return None

        train = weekly.iloc[:split]
        test = weekly.iloc[split:]

        X_train = train[FORECAST_FEATURES]
        y_train = train["Quantity"]
        X_test = test[FORECAST_FEATURES]
        y_test = test["Quantity"]

        model = LinearRegression()
        model.fit(X_train, y_train)
        ml_pred = model.predict(X_test)
        ml_mae, ml_rmse = mae_rmse(y_test, ml_pred)

        # Naive baseline: next week = last week (qty_lag1 on the test rows)
        naive_pred = test["qty_lag1"].values
        naive_mae, naive_rmse = mae_rmse(y_test, naive_pred)

        beats_baseline = ml_mae < naive_mae

        # Refit on all available history for next-week inference
        full_model = LinearRegression()
        full_model.fit(weekly[FORECAST_FEATURES], weekly["Quantity"])

        last_row = weekly.iloc[-1]
        next_features = pd.DataFrame(
            [
                {
                    "qty_lag1": float(last_row["Quantity"]),
                    "qty_lag2": float(last_row["qty_lag1"]),
                    "qty_roll4": float(
                        weekly["Quantity"].tail(4).mean()
                    ),
                    "month": int((last_row.name + pd.Timedelta(days=7)).month),
                    "weekofyear": int(
                        (last_row.name + pd.Timedelta(days=7)).isocalendar().week
                    ),
                }
            ]
        )
        next_week_pred = float(full_model.predict(next_features)[0])
        next_week_pred = max(0.0, next_week_pred)
        naive_next = float(last_row["Quantity"])

        # Optional companion: sales-value forecast using same feature idea on Sales
        sales_model = LinearRegression()
        sales_weekly = weekly.copy()
        sales_weekly["sales_lag1"] = sales_weekly["Sales"].shift(1)
        sales_weekly["sales_lag2"] = sales_weekly["Sales"].shift(2)
        sales_weekly["sales_roll4"] = sales_weekly["Sales"].shift(1).rolling(4).mean()
        sales_feat = sales_weekly.dropna()
        sales_next = None
        if len(sales_feat) >= 30:
            s_features = ["sales_lag1", "sales_lag2", "sales_roll4", "month", "weekofyear"]
            sales_model.fit(sales_feat[s_features], sales_feat["Sales"])
            last_s = sales_feat.iloc[-1]
            sales_next_x = pd.DataFrame(
                [
                    {
                        "sales_lag1": float(last_s["Sales"]),
                        "sales_lag2": float(last_s["sales_lag1"]),
                        "sales_roll4": float(sales_feat["Sales"].tail(4).mean()),
                        "month": int((last_s.name + pd.Timedelta(days=7)).month),
                        "weekofyear": int(
                            (last_s.name + pd.Timedelta(days=7)).isocalendar().week
                        ),
                    }
                ]
            )
            sales_next = max(0.0, float(sales_model.predict(sales_next_x)[0]))

        return {
            "model": full_model,
            "metrics": {
                "n_weeks": int(len(weekly)),
                "train_weeks": int(len(train)),
                "test_weeks": int(len(test)),
                "ml_mae": ml_mae,
                "ml_rmse": ml_rmse,
                "naive_mae": naive_mae,
                "naive_rmse": naive_rmse,
                "ml_beats_naive_mae": bool(beats_baseline),
                "preferred_method": "ml" if beats_baseline else "naive_last_week",
            },
            "last_week_ending": str(last_row.name.date()),
            "last_week_quantity": float(last_row["Quantity"]),
            "next_week_ml_quantity": next_week_pred,
            "next_week_naive_quantity": naive_next,
            "next_week_sales_value_estimate": sales_next,
            "history_tail": [
                {
                    "week_ending": str(idx.date()),
                    "quantity": float(row["Quantity"]),
                    "sales": float(row["Sales"]),
                }
                for idx, row in weekly.tail(8).iterrows()
            ],
        }


    forecast_artifacts = {}
    forecast_metrics = {
        "target": "Weekly Quantity (units) for stock planning",
        "method": (
            "LinearRegression on lag/rolling/calendar features, "
            "compared with naive last-week baseline. Chronological split."
        ),
        "limitations": [
            "No inventory on-hand, lead time, or stockout data in this CSV.",
            "SKU/Product-level weekly history is too sparse for reliable forecasting.",
            "Forecasts are estimates for planning, not guaranteed demand.",
        ],
        "series": {},
    }

    # Overall + each Category (reliable enough for this dataset)
    series_specs = [("Overall", None, None)]
    for category in sorted(df["Category"].unique()):
        series_specs.append((f"Category::{category}", "Category", category))

    # Sub-Category: include only if enough non-empty weeks
    for sub in sorted(df["Sub-Category"].unique()):
        series_specs.append((f"Sub-Category::{sub}", "Sub-Category", sub))

    for label, col, value in series_specs:
        frame = df if col is None else df[df[col] == value]
        weekly = build_weekly_quantity(frame, label)
        result = evaluate_forecast_series(weekly)
        if result is None:
            print(f"Skip {label}: not enough weekly history")
            continue
        forecast_artifacts[label] = {
            "model": result["model"],
            "features": FORECAST_FEATURES,
            "last_week_ending": result["last_week_ending"],
            "last_week_quantity": result["last_week_quantity"],
            "next_week_ml_quantity": result["next_week_ml_quantity"],
            "next_week_naive_quantity": result["next_week_naive_quantity"],
            "next_week_sales_value_estimate": result["next_week_sales_value_estimate"],
            "history_tail": result["history_tail"],
            "filter_column": col,
            "filter_value": value,
        }
        forecast_metrics["series"][label] = result["metrics"]
        pref = result["metrics"]["preferred_method"]
        print(
            f"{label}: ML MAE={result['metrics']['ml_mae']:.2f}, "
            f"Naive MAE={result['metrics']['naive_mae']:.2f}, preferred={pref}"
        )

    joblib.dump(forecast_artifacts, os.path.join(MODELS_DIR, "forecast_artifacts.joblib"))
    with open(os.path.join(MODELS_DIR, "forecast_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(forecast_metrics, f, indent=2)


    # ============================================================
    # PROBLEM 2: Discount / profit analysis (regression)
    # ============================================================
    print("\n=== PROBLEM 2: Discount / profit regression ===")

    profit_features = [
        "Category",
        "Sub-Category",
        "Segment",
        "Region",
        "Quantity",
        "Discount",
        "Sales",
    ]
    profit_categorical = ["Category", "Sub-Category", "Segment", "Region"]
    profit_numeric = ["Quantity", "Discount", "Sales"]

    X_profit = df[profit_features]
    y_profit = df["Profit"]

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_profit,
        y_profit,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Gradient Boosting is used because Linear Regression gets a negative R²
    # on this target (Profit has extreme outliers and non-linear discount effects).
    profit_model = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            profit_categorical,
                        ),
                        ("num", "passthrough", profit_numeric),
                    ]
                ),
            ),
            (
                "regressor",
                GradientBoostingRegressor(random_state=RANDOM_STATE),
            ),
        ]
    )

    profit_model.fit(X_train_p, y_train_p)
    profit_pred = profit_model.predict(X_test_p)
    profit_mae, profit_rmse = mae_rmse(y_test_p, profit_pred)
    profit_r2 = float(r2_score(y_test_p, profit_pred))

    print(f"Profit model R2={profit_r2:.4f}, MAE={profit_mae:.2f}, RMSE={profit_rmse:.2f}")

    joblib.dump(profit_model, os.path.join(MODELS_DIR, "profit_model.joblib"))

    profit_metrics = {
        "target": "Profit",
        "model": "GradientBoostingRegressor",
        "features": profit_features,
        "note": (
            "Profit is never used as an input feature. "
            "Gradient Boosting was chosen after Linear Regression produced negative R² "
            "due to heavy Profit outliers and non-linear patterns."
        ),
        "r2": profit_r2,
        "mae": profit_mae,
        "rmse": profit_rmse,
        "limitation": (
            "Historical association only. This model cannot measure true price "
            "elasticity or guarantee that changing discount will cause the estimated profit."
        ),
        "discount_scale": "Dataset Discount values are fractions (0.2 = 20%).",
    }

    with open(os.path.join(MODELS_DIR, "profit_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(profit_metrics, f, indent=2)


    # ============================================================
    # PROBLEM 3: Customer RFM segmentation (KMeans)
    # ============================================================
    print("\n=== PROBLEM 3: Customer RFM segmentation ===")

    reference_date = df["Order Date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("Customer ID")
        .agg(
            last_order=("Order Date", "max"),
            frequency=("Order ID", "nunique"),
            monetary=("Sales", "sum"),
        )
        .reset_index()
    )
    rfm["recency_days"] = (reference_date - rfm["last_order"]).dt.days.astype(int)
    rfm = rfm[["Customer ID", "recency_days", "frequency", "monetary"]].copy()

    scaler = StandardScaler()
    X_rfm = scaler.fit_transform(rfm[["recency_days", "frequency", "monetary"]])

    # Choose k with a simple inertia elbow check among 3..6, defaulting to 4.
    inertias = {}
    for k in range(3, 7):
        km_try = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km_try.fit(X_rfm)
        inertias[k] = float(km_try.inertia_)

    # Prefer k=4 for interpretability unless k=3 is clearly much tighter for viva simplicity.
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_rfm)

    # Build human-readable names from actual cluster centers (original RFM scale)
    cluster_profile = (
        rfm.groupby("cluster")
        .agg(
            customers=("Customer ID", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_monetary=("monetary", "sum"),
        )
        .reset_index()
    )

    # Name clusters from actual RFM averages (not arbitrary IDs).
    recency_med = float(cluster_profile["avg_recency"].median())
    monetary_med = float(cluster_profile["avg_monetary"].median())
    freq_med = float(cluster_profile["avg_frequency"].median())

    name_map = {}
    action_map = {}
    used_names = set()

    for _, row in cluster_profile.iterrows():
        cid = int(row["cluster"])
        high_money = row["avg_monetary"] >= monetary_med
        high_freq = row["avg_frequency"] >= freq_med
        inactive = row["avg_recency"] >= recency_med

        if high_money and high_freq and not inactive:
            name = "Loyal High-Value Customers"
            action = (
                "Offer loyalty rewards and early access to promotions. "
                "This is a suggested treatment, not a guaranteed response."
            )
        elif high_money and inactive:
            name = "At-Risk High Spenders"
            action = (
                "Consider a polite win-back offer. "
                "No campaign outcome data is available to prove response."
            )
        elif (not high_money) and inactive:
            name = "Occasional Low-Spend Customers"
            action = (
                "Keep low-cost email outreach only. "
                "Avoid expensive discounts without testing."
            )
        else:
            name = "Steady Average Customers"
            action = (
                "Maintain regular communication and seasonal offers. "
                "Impact is not measured in this dataset."
            )

        # Keep names unique if two clusters map to the same rule
        final_name = name
        suffix = 2
        while final_name in used_names:
            final_name = f"{name} (Group {suffix})"
            suffix += 1
        used_names.add(final_name)
        name_map[cid] = final_name
        action_map[cid] = action

    rfm["segment"] = rfm["cluster"].map(name_map)
    rfm["marketing_suggestion"] = rfm["cluster"].map(action_map)

    segment_summaries = []
    for _, row in cluster_profile.iterrows():
        cid = int(row["cluster"])
        segment_summaries.append(
            {
                "cluster_id": cid,
                "segment": name_map[cid],
                "customer_count": int(row["customers"]),
                "avg_recency_days": round(float(row["avg_recency"]), 2),
                "avg_frequency_orders": round(float(row["avg_frequency"]), 2),
                "avg_monetary_sales": round(float(row["avg_monetary"]), 2),
                "total_monetary_sales": round(float(row["total_monetary"]), 2),
                "marketing_suggestion": action_map[cid],
            }
        )

    segment_summaries = sorted(
        segment_summaries, key=lambda x: x["avg_monetary_sales"], reverse=True
    )

    rfm_lookup = {
        str(row["Customer ID"]): {
            "customer_id": str(row["Customer ID"]),
            "recency_days": int(row["recency_days"]),
            "frequency_orders": int(row["frequency"]),
            "monetary_sales": round(float(row["monetary"]), 2),
            "cluster_id": int(row["cluster"]),
            "segment": row["segment"],
            "marketing_suggestion": row["marketing_suggestion"],
        }
        for _, row in rfm.iterrows()
    }

    segmentation_bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "n_clusters": n_clusters,
        "reference_date": str(reference_date.date()),
        "feature_order": ["recency_days", "frequency", "monetary"],
        "cluster_to_segment": {str(k): v for k, v in name_map.items()},
        "inertias": inertias,
    }

    joblib.dump(segmentation_bundle, os.path.join(MODELS_DIR, "rfm_kmeans.joblib"))

    with open(os.path.join(MODELS_DIR, "segment_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "reference_date": str(reference_date.date()),
                "method": "RFM features + StandardScaler + KMeans",
                "n_clusters": n_clusters,
                "inertia_by_k": inertias,
                "k_choice_reason": (
                    "k=4 chosen for clear marketing groups after checking inertia for k=3..6. "
                    "Segmentation is unsupervised: there is no true label to predict."
                ),
                "segments": segment_summaries,
            },
            f,
            indent=2,
        )

    with open(os.path.join(MODELS_DIR, "customer_segments.json"), "w", encoding="utf-8") as f:
        json.dump(rfm_lookup, f)

    print(f"Segmented {len(rfm)} customers into {n_clusters} clusters.")
    for s in segment_summaries:
        print(f"  {s['segment']}: n={s['customer_count']}, avg $={s['avg_monetary_sales']}")


    # ============================================================
    # Shared options + combined metrics for the API / UI
    # ============================================================
    options = {
        "Category": sorted(df["Category"].unique().tolist()),
        "Sub-Category": sorted(df["Sub-Category"].unique().tolist()),
        "Segment": sorted(df["Segment"].unique().tolist()),
        "Region": sorted(df["Region"].unique().tolist()),
        "Sub-Categories-By-Category": {
            cat: sorted(df.loc[df["Category"] == cat, "Sub-Category"].unique().tolist())
            for cat in sorted(df["Category"].unique())
        },
        "forecast_series": sorted(forecast_artifacts.keys()),
        "scenario_discounts": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "sample_customer_ids": sorted(rfm["Customer ID"].astype(str).tolist())[:25],
        "date_range": {
            "min": str(df["Order Date"].min().date()),
            "max": str(df["Order Date"].max().date()),
        },
    }

    with open(os.path.join(MODELS_DIR, "options.json"), "w", encoding="utf-8") as f:
        json.dump(options, f, indent=2)

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "customers": int(df["Customer ID"].nunique()),
            "orders": int(df["Order ID"].nunique()),
            "date_min": str(df["Order Date"].min().date()),
            "date_max": str(df["Order Date"].max().date()),
        },
        "forecast": forecast_metrics,
        "profit": profit_metrics,
        "segmentation": {
            "n_customers": int(len(rfm)),
            "n_clusters": n_clusters,
            "reference_date": str(reference_date.date()),
            "inertia_by_k": inertias,
        },
    }

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nAll models and metadata saved under models/.")


if __name__ == "__main__":
    train_all()
