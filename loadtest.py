"""Small Locust load test for the prediction endpoint.

Run: locust -f loadtest.py --host http://localhost:8000
"""

from locust import HttpUser, between, task


class FraudUser(HttpUser):
    """Simulate one integration client."""

    wait_time = between(0.1, 0.5)

    @task
    def predict(self) -> None:
        """Send a valid transaction; Locust records latency and failures."""
        self.client.post(
            "/predict",
            json={
                "amount": 2500,
                "merchant_risk": 0.8,
                "customer_age": 35,
                "velocity_24h": 7,
                "customer_id": "load-test-user",
            },
        )
