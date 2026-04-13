"""Integration tests for the FastAPI /predict endpoint."""
import statistics
import time

import pytest

from tests.conftest import VALID_TRANSACTION


class TestPredictHappyPath:
    def test_returns_200(self, client):
        response = client.post("/predict", json=[VALID_TRANSACTION])
        assert response.status_code == 200

    def test_response_is_list(self, client):
        response = client.post("/predict", json=[VALID_TRANSACTION])
        assert isinstance(response.json(), list)
        assert len(response.json()) == 1

    def test_response_has_fraud_probability(self, client):
        response = client.post("/predict", json=[VALID_TRANSACTION])
        item = response.json()[0]
        assert "fraud_probability" in item

    def test_fraud_probability_is_float_in_range(self, client):
        response = client.post("/predict", json=[VALID_TRANSACTION])
        prob = response.json()[0]["fraud_probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_response_has_prediction_field(self, client):
        response = client.post("/predict", json=[VALID_TRANSACTION])
        item = response.json()[0]
        assert "prediction" in item
        assert item["prediction"] in (0, 1)

    def test_batch_of_transactions(self, client):
        response = client.post("/predict", json=[VALID_TRANSACTION, VALID_TRANSACTION])
        assert response.status_code == 200
        assert len(response.json()) == 2


class TestPredictNegativeScenarios:
    def test_missing_required_field_returns_422(self, client):
        incomplete = {k: v for k, v in VALID_TRANSACTION.items() if k in ("Time", "Amount")}
        response = client.post("/predict", json=[incomplete])
        assert response.status_code == 422

    def test_wrong_type_amount_string_returns_422(self, client):
        bad_transaction = VALID_TRANSACTION.copy()
        bad_transaction["Amount"] = "abc"
        response = client.post("/predict", json=[bad_transaction])
        assert response.status_code == 422

    def test_wrong_type_time_string_returns_422(self, client):
        bad_transaction = VALID_TRANSACTION.copy()
        bad_transaction["Time"] = "not_a_float"
        response = client.post("/predict", json=[bad_transaction])
        assert response.status_code == 422

    def test_wrong_structure_returns_422(self, client):
        response = client.post("/predict", json=[{"features": [1.0, 2.0, 3.0]}])
        assert response.status_code == 422

    def test_empty_list_returns_200_empty_result(self, client):
        response = client.post("/predict", json=[])
        assert response.status_code == 200
        assert response.json() == []

    def test_extra_unknown_fields_are_ignored(self, client):
        with_extra = VALID_TRANSACTION.copy()
        with_extra["unknown_field"] = "garbage"
        response = client.post("/predict", json=[with_extra])
        assert response.status_code == 200

    def test_missing_single_v_field_returns_422(self, client):
        missing_v14 = {k: v for k, v in VALID_TRANSACTION.items() if k != "V14"}
        response = client.post("/predict", json=[missing_v14])
        assert response.status_code == 422





class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self, client):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}



class TestLoadPerformance:
    N_REQUESTS = 100
    MAX_MEDIAN_MS = 200.0

    def test_median_response_time_under_200ms(self, client):
        latencies_ms: list[float] = []
        for _ in range(self.N_REQUESTS):
            start = time.perf_counter()
            response = client.post("/predict", json=[VALID_TRANSACTION])
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert response.status_code == 200, "Unexpected non-200 during load test"
            latencies_ms.append(elapsed_ms)

        median_ms = statistics.median(latencies_ms)
        assert median_ms < self.MAX_MEDIAN_MS, (
            f"Median response time {median_ms:.1f} ms exceeds {self.MAX_MEDIAN_MS} ms limit"
        )
