import logging
import os
from pathlib import Path

from ray.job_submission import JobSubmissionClient

import lm_buddy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LM Buddy")

# Head node, also where dashboard is
client = JobSubmissionClient("http://your.ray.cluster:8265")

# Setting local repo if you're working outside of LM Buddy
lm_buddy_repo = Path(lm_buddy.__file__).parents[1]
lm_buddy_module = lm_buddy_repo / "src" / "lm_buddy"
lm_buddy_requirements = lm_buddy_repo / "requirements.txt"

# Setting empty openAI key for eval harness
runtime_env = {
    "working_dir": "configs",
    "env_vars": {"WANDB_API_KEY": os.environ["WANDB_API_KEY"], "OPENAI_API_KEY": "EMPTY"},
    "py_modules": [str(lm_buddy_module)],
    "pip": str(lm_buddy_requirements),
}

# config file is in config dir relative to repo
client.submit_job(
    entrypoint="python -m lm_buddy evaluate lm-harness --config lm_harness.yaml",
    runtime_env=runtime_env,
)

logger.info(f"Job submitted to {client.get_address()}")
