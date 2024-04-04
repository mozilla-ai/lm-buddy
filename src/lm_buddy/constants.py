import os
from pathlib import Path

STORAGE_PATH_ENVIRONMENT_VARIABLE: str = "LM_BUDDY_STORAGE"

DEFAULT_STORAGE_PATH: str = os.getenv(
    STORAGE_PATH_ENVIRONMENT_VARIABLE,
    str(Path.home() / "lm_buddy_results"),
)
