from datasets import Dataset

from lm_buddy.integrations.huggingface.dataset_config import TextDatasetConfig
from lm_buddy.preprocessing import format_dataset_with_prompt


def preprocess_text_dataset(dataset: Dataset, dataset_config: TextDatasetConfig) -> Dataset:
    """Prompt format a text dataset if a prompt template is specified on the config."""
    if dataset_config.prompt_template is not None:
        return format_dataset_with_prompt(
            dataset=dataset,
            template=dataset_config.prompt_template,
            output_field=dataset_config.text_field,
        )
    else:
        return dataset
