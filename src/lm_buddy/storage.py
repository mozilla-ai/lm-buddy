import os
from pathlib import Path

DEFAULT_STORAGE_PATH: str = os.getenv(
    "LM_BUDDY_STORAGE",
    str(Path.home() / "lm_buddy_results"),
)
