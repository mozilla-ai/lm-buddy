from flamingo.integrations.huggingface import QuantizationConfig
from flamingo.integrations.wandb import WandbEnvironment
from flamingo.jobs.configs import FinetuningJobConfig
from peft import LoraConfig
from ray.train import ScalingConfig


def test_serde_round_trip():
    wandb_env = WandbEnvironment(name="my-run", project="my-project")
    lora_config = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM")
    quantization_config = QuantizationConfig(load_in_8bit=True)
    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
    config = FinetuningJobConfig(
        model="test-model",
        dataset="test-dataset",
        torch_dtype="bfloat16",
        wandb_env=wandb_env,
        lora_config=lora_config,
        quantization_config=quantization_config,
        scaling_config=scaling_config,
        storage_path="/mnt/data/ray_results",
    )
    assert FinetuningJobConfig.parse_raw(config.json()) == config
