import ray

from lm_buddy.jobs.common import SimpleResult
from lm_buddy.jobs.configs import SimpleJobConfig


@ray.remote
def get_magic_number(config: SimpleJobConfig) -> int:
    return config.magic_number


def run_simple(config: SimpleJobConfig) -> SimpleResult:
    # Connect to the Ray cluster (if not already running)
    ray.init(ignore_reinit_error=True)
    # Run dummy remote task
    magic_number = ray.get(get_magic_number.remote(config))
    # Return output result
    print(f"The magic number is {magic_number}")
    return SimpleResult(magic_number=magic_number)
