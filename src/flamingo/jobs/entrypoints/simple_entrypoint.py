from flamingo.jobs.configs import SimpleJobConfig


def main(config: SimpleJobConfig):
    print(f"The magic number is {config.magic_number}")
