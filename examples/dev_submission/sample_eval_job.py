import logging
import os
from pathlib import Path

from ray.job_submission import JobSubmissionClient

import flamingo  # noqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FLAMINGO")

# Head node, also where dashboard is
client = JobSubmissionClient("http://your.ray.cluster:8265")

# Setting local repo if you're working outside of Flamingo
flamingo_repo = Path(__file__).parents[1] / "flamingo"
flamingo_module = flamingo_repo / "src" / "flamingo"
flamingo_requirements = flamingo_repo / "requirements.txt"

# Setting empty openAI key for eval harness
runtime_env = {
    "working_dir": "configs",
    "env_vars": {"WANDB_API_KEY": os.environ["WANDB_API_KEY"], "OPENAI_API_KEY": "EMPTY"},
    "py_modules": [str(flamingo_module)],
    "pip": str(flamingo_requirements),
}

# config file is in config dir relative to repo
client.submit_job(
    entrypoint="python -m flamingo run lm-harness --config lm_harness.yaml", runtime_env=runtime_env
)

logger.info(f"Job submitted to {client.get_address()}")
