"""
Lightweight API checks using FastAPI TestClient.
Requires: pip install httpx
Run after training: python test_api.py
"""

import math
from pathlib import Path

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def main():
    r = client.get("/")
    assert r.status_code == 200
    assert "Retail ML" in r.text
    html = r.text
    assert "Sales and other inputs are held constant" in html
    assert 'id="forecastCategory"' in html
    assert 'id="forecastSubCategory"' in html
    assert "forecastScope" not in html
    assert "/forecast/inventory" in html
    assert "/discount/scenarios" in html
    assert "/segments/summary" in html
    print("PASS GET /")

    r = client.get("/health")
    assert r.status_code == 200
    print("PASS GET /health", r.json())

    r = client.get("/options")
    assert r.status_code == 200
    options = r.json()
    assert "forecast_series" in options
    print("PASS GET /options")

    r = client.get("/metrics")
    assert r.status_code == 200
    metrics = r.json()
    assert "forecast" in metrics and "profit" in metrics
    assert "Quantity" in metrics["forecast"]["target"]
    assert "Profit" not in metrics["profit"]["features"]
    print("PASS GET /metrics")

    for buffer in (0.0, 0.1, 0.25):
        r = client.post(
            "/forecast/inventory",
            json={"scope": "Overall", "safety_buffer": buffer},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        expected = int(
            math.ceil(body["forecast_next_week_quantity_units"] * (1.0 + buffer))
        )
        assert body["suggested_stock_quantity_units"] == expected
        assert "forecast_next_week_quantity_units" in body
    print(
        "PASS POST /forecast/inventory",
        body["suggested_stock_quantity_units"],
        "buffer_math_ok",
    )

    r = client.post(
        "/discount/scenarios",
        json={
            "category": "Furniture",
            "sub_category": "Bookcases",
            "segment": "Consumer",
            "region": "South",
            "quantity": 2,
            "sales": 250,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["scenarios"]) >= 5
    assert body["inputs"]["sales"] == 250
    assert "Sales is held fixed" in body["limitation"]
    print("PASS POST /discount/scenarios", body["exploratory_recommendation"])

    r = client.get("/segments/summary")
    assert r.status_code == 200
    body = r.json()
    assert len(body["segments"]) >= 3
    print("PASS GET /segments/summary", len(body["segments"]), "segments")

    sample_id = options["sample_customer_ids"][0]
    r = client.get(f"/segments/customer/{sample_id}")
    assert r.status_code == 200, r.text
    print("PASS GET /segments/customer", r.json()["segment"])

    r = client.get("/segments/customer/NOT-A-REAL-ID")
    assert r.status_code == 404
    print("PASS missing customer -> 404")

    r = client.post(
        "/discount/scenarios",
        json={
            "category": "Furniture",
            "sub_category": "Bookcases",
            "segment": "Consumer",
            "region": "South",
            "quantity": 0,
            "sales": 250,
        },
    )
    assert r.status_code == 422
    print("PASS invalid quantity -> 422")

    required = [
        "forecast_artifacts.joblib",
        "profit_model.joblib",
        "rfm_kmeans.joblib",
        "metrics.json",
        "options.json",
    ]
    for name in required:
        assert Path("models", name).exists(), name
    print("PASS model artifacts present")

    print("ALL_TESTS_PASSED")


if __name__ == "__main__":
    main()
