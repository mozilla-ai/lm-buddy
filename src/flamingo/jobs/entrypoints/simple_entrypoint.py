from flamingo.jobs.configs import SimpleJobConfig


def run(config: SimpleJobConfig):
    """A simple job entrypoint to demonstrate the submission interface."""
    print(f"The magic number is {config.magic_number}")
