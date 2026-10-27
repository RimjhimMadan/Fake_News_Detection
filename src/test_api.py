"""
Automated Integration Test for TruthLens FastAPI Service
"""

import sys
import json
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    print(" [PASS] GET /health check succeeded")


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200, f"Metrics endpoint failed: {response.text}"
    data = response.json()
    assert "models" in data
    print(f" [PASS] GET /metrics check succeeded ({len(data['models'])} models listed)")


def test_predict_real():
    payload = {
        "text": (
            "WASHINGTON (Reuters) - The U.S. Federal Reserve decided on Wednesday to keep interest rates "
            "unchanged, citing stable economic growth and steady job gains across the country."
        ),
        "num_explanation_words": 6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Predict real failed: {response.text}"
    data = response.json()
    assert data["label"] in ["Real", "Fake"]
    assert "confidence_score" in data
    assert "top_influential_words" in data
    assert len(data["top_influential_words"]) > 0
    print(f" [PASS] POST /predict (Real news sample) -> Label: {data['label']} ({data['confidence_score']}%)")


def test_predict_fake():
    payload = {
        "text": (
            "BREAKING: Underground whistleblower reveals massive top secret conspiracy! "
            "Mainstream media caught in gigantic coverup of explosive leaked emails."
        ),
        "num_explanation_words": 6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Predict fake failed: {response.text}"
    data = response.json()
    assert data["label"] in ["Real", "Fake"]
    assert "confidence_score" in data
    assert len(data["top_influential_words"]) > 0
    print(f" [PASS] POST /predict (Fake news sample) -> Label: {data['label']} ({data['confidence_score']}%)")


def test_input_validation():
    # Test empty string
    res_empty = client.post("/predict", json={"text": ""})
    assert res_empty.status_code == 422, "Empty text did not return 422 validation error"
    
    # Test short text
    res_short = client.post("/predict", json={"text": "Hi"})
    assert res_short.status_code == 422, "Short text did not return 422 validation error"
    print(" [PASS] Input validation correctly rejected empty and short texts (422 Unprocessable Entity)")


if __name__ == "__main__":
    print("\n" + "="*50)
    print(" RUNNING FASTAPI ENDPOINT INTEGRATION TESTS")
    print("="*50)
    test_health()
    test_metrics()
    test_predict_real()
    test_predict_fake()
    test_input_validation()
    print("="*50)
    print(" ALL API TESTS PASSED SUCCESSFULLY!")
    print("="*50 + "\n")
