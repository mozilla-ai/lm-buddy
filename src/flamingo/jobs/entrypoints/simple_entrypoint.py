from flamingo.jobs.configs import SimpleJobConfig


def run(config: SimpleJobConfig):
    print(f"The magic number is {config.magic_number}")
