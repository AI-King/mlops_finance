"""Run metadata helpers for experiment traceability."""

import subprocess


def current_git_sha() -> str:
    """Return the current Git commit SHA, or unknown outside a Git checkout.

    Complexity: O(1) from the Python side; Git resolves the current HEAD.
    DSA: no custom data structure is used, only a subprocess result string.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return "unknown"
    return result.stdout.strip() or "unknown"
