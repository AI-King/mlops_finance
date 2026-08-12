"""Continuous prediction traffic generator for live monitoring demos."""

import argparse
import random
import time
from typing import Any

import requests


def make_transaction(index: int, mode: str) -> dict[str, Any]:
    """Create one synthetic API payload.

    Complexity: O(1). DSA: dictionary stores one JSON-ready transaction.
    """
    amount = random.lognormvariate(3.5, 1.0)
    merchant_risk = random.betavariate(2, 8)
    velocity_24h = random.randint(0, 7)

    if mode == "feature_drift":
        # Feature drift changes input distributions while the fraud concept is unchanged.
        amount *= 8
        velocity_24h += random.randint(4, 12)
        merchant_risk = min(1.0, merchant_risk + random.uniform(0.2, 0.5))

    if mode == "concept_drift":
        # Concept drift is approximated here by sending combinations that the old
        # model may score differently than business rules would expect.
        amount *= random.choice([0.8, 1.2, 2.5])
        velocity_24h += random.randint(8, 18)
        merchant_risk = random.uniform(0.45, 0.95)

    return {
        "amount": round(amount, 2),
        "merchant_risk": round(merchant_risk, 4),
        "customer_age": random.randint(18, 90),
        "velocity_24h": velocity_24h,
        # Vary customer IDs so deterministic A/B routing hits both variants.
        "customer_id": f"live-{mode}-{index}-{random.randint(1, 100_000)}",
    }


def run(url: str, mode: str, interval: float, limit: int | None) -> None:
    """Send prediction requests until stopped or limit is reached."""
    sent = 0
    while limit is None or sent < limit:
        sent += 1
        payload = make_transaction(sent, mode)
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            print(
                f"{sent:05d} mode={mode} variant={result['variant']} "
                f"prob={result['fraud_probability']:.4f} "
                f"decision={result['decision']} fallback={result['fallback_used']}"
            )
        except requests.RequestException as exc:
            print(f"{sent:05d} request_failed error={exc}")
        time.sleep(interval)


def main() -> None:
    """Parse arguments and start live prediction traffic."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument(
        "--mode",
        choices=["normal", "feature_drift", "concept_drift"],
        default="normal",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.url, args.mode, args.interval, args.limit)


if __name__ == "__main__":
    main()
