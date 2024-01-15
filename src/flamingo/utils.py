import os
from datetime import datetime

__all__ = ["get_default_run_name"]


def get_default_run_name(base_name: str = "dummy_run"):
    if user := os.getenv("USER", None):
        return f"{user}_{base_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    else:
        return f"{base_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
