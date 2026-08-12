"""Deterministic fallback rules used when model confidence is low."""


def rule_based_decision(
    amount: float, merchant_risk: float, velocity_24h: int
) -> tuple[str, str]:
    """Return a conservative decision and reason code.

    Complexity: O(1) time and O(1) space. DSA: a short ordered decision tree.
    """
    if merchant_risk >= 0.9 or velocity_24h >= 15:
        return "review", "high_merchant_or_velocity_risk"
    if amount >= 5_000 and velocity_24h >= 8:
        return "review", "high_amount_and_velocity"
    return "approve", "fallback_low_risk"
