from ray import train


def is_rank_zero_worker() -> bool:
    """Return whether the caller is on the rank zero worker within a Ray Train context.

    Reference: https://docs.ray.io/en/latest/train/user-guides/experiment-tracking.html
    """
    return train.get_context().get_world_rank() == 0
