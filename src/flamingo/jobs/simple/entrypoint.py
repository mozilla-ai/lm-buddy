import ray

from flamingo.jobs.simple import SimpleJobConfig


@ray.remote
def do_magic(config: SimpleJobConfig):
    print(f"The magic number is {config.magic_number}")


def run_simple(config: SimpleJobConfig):
    """A simple job entrypoint to demonstrate the Ray interface."""
    # Connect to the Ray cluster (if not already running)
    ray.init(ignore_reinit_error=True)
    # Wait on remote task
    ray.get(do_magic.remote(config))
