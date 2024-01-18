from flamingo.jobs import SimpleJobConfig


def run_simple(config: SimpleJobConfig):
    """A simple job entrypoint to demonstrate the submission interface."""
    print(f"The magic number is {config.magic_number}")
