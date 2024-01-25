import ray

from flamingo.jobs.simple import SimpleJobConfig


@ray.remote
def get_magic_number(config: SimpleJobConfig) -> int:
    return config.magic_number


def run_simple(config: SimpleJobConfig):
    """A simple entrypoint to demonstrate the Ray interface."""
    # Connect to the Ray cluster (if not already running)
    ray.init(ignore_reinit_error=True)
    # Run dummy remote task
    magic_number = ray.get(get_magic_number.remote(config))
    print(f"The magic number is {magic_number}")
