import os
from pathlib import Path

LM_BUDDY_HOME_PATH: str = os.getenv(
    "LM_BUDDY_HOME",
    str(Path.home() / ".lm_buddy"),
)
