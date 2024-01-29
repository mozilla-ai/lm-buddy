import ray

from flamingo.jobs.finetuning.utils import generate_huggingface_dataset


def test_huggingface_dataset_generation():
    ray_dataset = ray.data.from_items(
        [
            {"food": "spam", "price": 9.34},
            {"food": "ham", "price": 5.37},
            {"food": "eggs", "price": 0.94},
        ]
    )
    hf_dataset = generate_huggingface_dataset(ray_dataset)
    assert len(hf_dataset) == 3
    assert set(hf_dataset.column_names) == {"food", "price"}
