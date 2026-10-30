"""
Retail Machine Learning and Analytics API

Features:
1) Inventory / next-week quantity forecast
2) Discount & profit scenario analysis
3) Customer RFM segmentation

Train first:
    python train_models.py

Run:
    uvicorn main:app --reload
"""

import json
import math
import os
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


MODELS_DIR = "models"
INDEX_PATH = "static/index.html"

REQUIRED_FILES = [
    "forecast_artifacts.joblib",
    "forecast_metrics.json",
    "profit_model.joblib",
    "profit_metrics.json",
    "rfm_kmeans.joblib",
    "segment_summaries.json",
    "customer_segments.json",
    "options.json",
    "metrics.json",
]


def _path(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


missing = [name for name in REQUIRED_FILES if not os.path.exists(_path(name))]
if missing:
    raise RuntimeError(
        "Missing model files: "
        + ", ".join(missing)
        + ". Run: python train_models.py"
    )

forecast_artifacts = joblib.load(_path("forecast_artifacts.joblib"))
profit_model = joblib.load(_path("profit_model.joblib"))
rfm_bundle = joblib.load(_path("rfm_kmeans.joblib"))

with open(_path("forecast_metrics.json"), "r", encoding="utf-8") as f:
    forecast_metrics = json.load(f)
with open(_path("profit_metrics.json"), "r", encoding="utf-8") as f:
    profit_metrics = json.load(f)
with open(_path("segment_summaries.json"), "r", encoding="utf-8") as f:
    segment_summaries = json.load(f)
with open(_path("customer_segments.json"), "r", encoding="utf-8") as f:
    customer_segments = json.load(f)
with open(_path("options.json"), "r", encoding="utf-8") as f:
    options = json.load(f)
with open(_path("metrics.json"), "r", encoding="utf-8") as f:
    metrics = json.load(f)


app = FastAPI(
    title="Retail ML & Analytics",
    description=(
        "Inventory forecasting, discount/profit analysis, "
        "and customer RFM segmentation on the Superstore dataset."
    ),
    version="2.0.0",
)


# ------------------------------------------------------------
# Request models
# ------------------------------------------------------------
class ForecastRequest(BaseModel):
    scope: str = Field(
        "Overall",
        description="Overall, or Category::Furniture, or Sub-Category::Phones",
    )
    safety_buffer: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Extra stock fraction on top of forecast units (0.10 = 10%).",
    )


class DiscountScenarioRequest(BaseModel):
    category: str = Field(..., min_length=1)
    sub_category: str = Field(..., min_length=1)
    segment: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)
    quantity: int = Field(..., gt=0)
    sales: float = Field(..., gt=0)
    discounts: Optional[List[float]] = Field(
        default=None,
        description="Discount fractions, e.g. 0.2 means 20%.",
    )


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def resolve_forecast_key(scope: str) -> str:
    scope = scope.strip()
    if scope in forecast_artifacts:
        return scope
    # Allow plain category / sub-category names from the UI
    as_category = f"Category::{scope}"
    if as_category in forecast_artifacts:
        return as_category
    as_sub = f"Sub-Category::{scope}"
    if as_sub in forecast_artifacts:
        return as_sub
    raise HTTPException(
        status_code=404,
        detail=(
            f"Unknown forecast scope '{scope}'. "
            "Use values from GET /options -> forecast_series."
        ),
    )


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/")
def home():
    if not os.path.exists(INDEX_PATH):
        raise HTTPException(status_code=404, detail="static/index.html not found")
    return FileResponse(INDEX_PATH)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "forecast_series_loaded": len(forecast_artifacts),
        "profit_model_loaded": profit_model is not None,
        "customers_segmented": len(customer_segments),
    }


@app.get("/metrics")
def get_metrics():
    return metrics


@app.get("/options")
def get_options():
    return options


@app.post("/forecast/inventory")
def forecast_inventory(req: ForecastRequest):
    key = resolve_forecast_key(req.scope)
    art = forecast_artifacts[key]
    series_metrics = forecast_metrics["series"].get(key, {})

    preferred = series_metrics.get("preferred_method", "naive_last_week")
    if preferred == "ml":
        forecast_units = float(art["next_week_ml_quantity"])
        method_used = "ml_linear_regression"
    else:
        forecast_units = float(art["next_week_naive_quantity"])
        method_used = "naive_last_week"

    forecast_units = max(0.0, forecast_units)
    buffer = float(req.safety_buffer)
    suggested_stock = int(math.ceil(forecast_units * (1.0 + buffer)))

    ml_beats = bool(series_metrics.get("ml_beats_naive_mae", False))

    return {
        "scope": key,
        "last_week_ending": art["last_week_ending"],
        "last_week_quantity_units": art["last_week_quantity"],
        "forecast_next_week_quantity_units": round(forecast_units, 2),
        "method_used": method_used,
        "ml_forecast_units": round(float(art["next_week_ml_quantity"]), 2),
        "naive_forecast_units": round(float(art["next_week_naive_quantity"]), 2),
        "ml_beats_naive_on_test_mae": ml_beats,
        "safety_buffer": buffer,
        "suggested_stock_quantity_units": suggested_stock,
        "optional_next_week_sales_value_estimate": (
            None
            if art["next_week_sales_value_estimate"] is None
            else round(float(art["next_week_sales_value_estimate"]), 2)
        ),
        "recent_weekly_history": art["history_tail"],
        "test_metrics": series_metrics,
        "disclaimer": (
            "Suggested stock is a planning estimate only, not guaranteed demand. "
            "This dataset has no on-hand inventory or supplier lead-time fields. "
            "SKU-level forecasting is not supported reliably."
        ),
    }


@app.post("/discount/scenarios")
def discount_scenarios(req: DiscountScenarioRequest):
    discounts = req.discounts or options.get(
        "scenario_discounts", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    for d in discounts:
        if d < 0 or d > 1:
            raise HTTPException(
                status_code=422,
                detail="Discounts must be fractions between 0 and 1 (0.2 = 20%).",
            )

    rows = []
    for d in discounts:
        frame = pd.DataFrame(
            [
                {
                    "Category": req.category,
                    "Sub-Category": req.sub_category,
                    "Segment": req.segment,
                    "Region": req.region,
                    "Quantity": req.quantity,
                    "Discount": float(d),
                    "Sales": req.sales,
                }
            ]
        )
        try:
            est = float(profit_model.predict(frame)[0])
        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail=f"Could not estimate profit: {error}",
            ) from error
        rows.append(
            {
                "discount_fraction": float(d),
                "discount_percent": round(float(d) * 100.0, 2),
                "estimated_profit": round(est, 2),
            }
        )

    best = max(rows, key=lambda r: r["estimated_profit"])
    return {
        "inputs": {
            "category": req.category,
            "sub_category": req.sub_category,
            "segment": req.segment,
            "region": req.region,
            "quantity": req.quantity,
            "sales": req.sales,
        },
        "scenarios": rows,
        "exploratory_recommendation": {
            "discount_fraction": best["discount_fraction"],
            "discount_percent": best["discount_percent"],
            "estimated_profit": best["estimated_profit"],
            "note": (
                "Highest estimated profit among the tested discount values only. "
                "This is exploratory, not a proven optimal business discount."
            ),
        },
        "model_metrics": {
            "r2": profit_metrics.get("r2"),
            "mae": profit_metrics.get("mae"),
            "rmse": profit_metrics.get("rmse"),
        },
        "limitation": (
            "Historical association only. The model cannot know whether a higher "
            "discount will cause more people to buy or change sales volume. "
            "Sales is held fixed in these scenarios, so results are not true "
            "price-elasticity analysis."
        ),
        "discount_scale_note": "0.2 means 20% discount (same scale as the dataset).",
    }


@app.get("/segments/summary")
def segments_summary():
    return {
        "reference_date": segment_summaries.get("reference_date"),
        "method": segment_summaries.get("method"),
        "n_clusters": segment_summaries.get("n_clusters"),
        "k_choice_reason": segment_summaries.get("k_choice_reason"),
        "segments": segment_summaries.get("segments", []),
        "note": (
            "Unsupervised learning: customers are grouped by similar RFM behavior. "
            "There is no ground-truth segment label in the dataset."
        ),
    }


@app.get("/segments/customer/{customer_id}")
def segment_customer(customer_id: str):
    key = customer_id.strip()
    info = customer_segments.get(key)
    if info is None:
        # Try case-sensitive exact alternatives already stored as-is
        raise HTTPException(
            status_code=404,
            detail=(
                f"Customer ID '{customer_id}' was not found. "
                "Use an ID from the Superstore dataset (see GET /options)."
            ),
        )
    return {
        **info,
        "reference_date": rfm_bundle.get("reference_date"),
        "note": (
            "Marketing suggestions are cautious templates only. "
            "This project has no campaign-response data."
        ),
    }
