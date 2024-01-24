from pathlib import Path

from datasets import Dataset

if __name__ == "__main__":
    mappings = [{"x": i, "y": 2 * i, "z": 3 * i + 10} for i in range(100)]
    dataset = Dataset.from_list(mappings)
    dataset_dict = dataset.train_test_split(test_size=0.2, seed=12345)

    save_path = Path(__file__).parent / "xyz.hf"
    dataset_dict.save_to_disk(save_path)
