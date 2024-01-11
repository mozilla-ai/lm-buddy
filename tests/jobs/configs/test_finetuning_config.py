from flamingo.integrations.huggingface import QuantizationConfig
from flamingo.integrations.huggingface.trainer_config import TrainerConfig
from flamingo.jobs.configs import FinetuningJobConfig
from peft import LoraConfig
from ray.train import ScalingConfig


def test_serde_round_trip():
    trainer_config = TrainerConfig(torch_dtype="bfloat16")
    lora_config = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM")
    quantization_config = QuantizationConfig(load_in_8bit=True)
    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    config = FinetuningJobConfig(
        model="test-model",
        dataset="test-dataset",
        trainer=trainer_config,
        lora=lora_config,
        quantization=quantization_config,
        scaling=scaling_config,
        storage_path="/mnt/data/ray_results",
    )
    assert FinetuningJobConfig.parse_raw(config.json()) == config
