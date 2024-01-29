import ray
from datasets import Dataset


def is_tracking_enabled() -> bool:
    """Return whether tracking is enabled on the current Ray train worker.

    Logging to a tracking platform should only be performed on the rank 0 worker.
    Reference: https://docs.ray.io/en/latest/train/user-guides/experiment-tracking.html
    """
    return ray.train.get_context().get_world_rank() == 0


def generate_huggingface_dataset(ray_dataset: ray.data.DatasetIterator) -> Dataset:
    """Generate a HuggingFace `Dataset` by iterating the rows of a Ray dataset/iterator."""

    def data_generator():
        yield from ray_dataset.iter_rows()

    return Dataset.from_generator(data_generator)
