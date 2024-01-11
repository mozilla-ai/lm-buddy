import json

import torch
import wandb
from accelerate import Accelerator
from datasets import DatasetDict
from ray import train
from ray.train import CheckpointConfig, RunConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, TrainingArguments
from trl import SFTTrainer

from flamingo.integrations.wandb import update_wandb_summary
from flamingo.jobs.configs import FinetuningJobConfig


def is_wandb_enabled(config: FinetuningJobConfig):
    # Only report to WandB on the rank 0 worker
    # Reference: https://docs.ray.io/en/latest/train/user-guides/experiment-tracking.html
    return config.wandb_env and train.get_context().get_world_rank() == 0


def get_training_args(config: FinetuningJobConfig) -> TrainingArguments:
    """Get TrainingArguments appropriate for the worker rank and job config."""
    return TrainingArguments(
        output_dir="out",  # Local checkpoint path on a worker
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        weight_decay=config.weight_decay,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        logging_strategy=config.logging_strategy,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        run_name=config.wandb_name,
        report_to="wandb" if is_wandb_enabled(config) else "none",
        no_cuda=not config.scaling_config.use_gpu,
        push_to_hub=False,
        disable_tqdm=True,
        logging_dir=None,
    )


def get_datasets(config: FinetuningJobConfig) -> DatasetDict:
    # TODO: Refactor me somehow
    ...


def get_model(config: FinetuningJobConfig) -> PreTrainedModel:
    device_map, bnb_config = None, None
    if config.quantization_config:
        bnb_config = config.quantization_config.as_huggingface()
        # When quantization is enabled, model must all be on same GPU to work with DDP
        # If a device_map is not specified we will get accelerate errors downstream
        # Reference: https://github.com/huggingface/accelerate/issues/1840#issuecomment-1683105994
        current_device = Accelerator().local_process_index if torch.cuda.is_available() else "cpu"
        device_map = {"": current_device}
        print(f"Setting model device_map = {device_map} to enable quantization")

    return AutoModelForCausalLM.from_pretrained(
        config.model,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=config.torch_dtype,
        quantization_config=bnb_config,
        device_map=device_map,
    )


def get_tokenizer(config: FinetuningJobConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer or config.model,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if not tokenizer.pad_token_id:
        # Pad token required for generating consistent batch sizes
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def train_func(config_data: dict):
    config = FinetuningJobConfig(**config_data)
    model = get_model(config)
    tokenizer = get_tokenizer(config)

    datasets = get_datasets(config)
    training_args = get_training_args(config)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=config.lora_config,
        max_seq_length=config.max_seq_length,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        dataset_text_field="text",
    )
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()

    # Force WandB finish on rank 0 worker
    if is_wandb_enabled(config):
        wandb.finish()


def run(config: FinetuningJobConfig):
    print(f"Received job configuration: {config}")

    run_config = RunConfig(
        name=config.wandb_name,
        storage_path=config.storage_path,
        checkpoint_config=CheckpointConfig(num_to_keep=1),
    )
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=json.loads(config.json()),
        scaling_config=config.scaling_config,
        run_config=run_config,
    )
    result = trainer.fit()
    print(f"Training result: {result}")

    # Log additional training metrics to completed WandB run
    if config.wandb_env:
        result_paths = {"ray/result_path": result.path}
        if result.checkpoint:
            result_paths["ray/checkpoint_path"] = f"{result.checkpoint.path}/checkpoint"
        update_wandb_summary(config.wandb_env, result_paths)
