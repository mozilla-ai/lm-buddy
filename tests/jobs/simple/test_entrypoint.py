from flamingo.jobs.simple import SimpleJobConfig, run_simple


def test_run_simple(initialize_ray_cluster):
    config = SimpleJobConfig(magic_number=42)
    run_simple(config)
