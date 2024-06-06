"""
lm-buddy entrypoint to run summary evaluation using huggingface eval
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from lm_buddy.configs.jobs.hf_evaluate import HuggingFaceEvalJobConfig
from lm_buddy.constants import LM_BUDDY_RESULTS_PATH
from lm_buddy.jobs.asset_loader import (
    HuggingFaceDatasetLoader,
    HuggingFaceModelLoader,
    HuggingFaceTokenizerLoader,
)
from lm_buddy.jobs.common import EvaluationResult


@dataclass
class BadResponseError(Exception):
    def __init__(self, message, error=None):
        self.message = message
        self.error = error


def run_eval(config: HuggingFaceEvalJobConfig) -> Path:
    # Init loaders
    hf_dataset_loader = HuggingFaceDatasetLoader()
    hf_model_loader = HuggingFaceModelLoader()
    hf_tokenizer_loader = HuggingFaceTokenizerLoader()

    # Load dataset given its URI
    dataset = hf_dataset_loader.load_dataset(config.dataset)

    # Enable / disable tqdm
    input_samples = dataset.select(range(10))["examples"]
    dataset_iterable = tqdm(input_samples) if config.evaluation.enable_tqdm else input_samples
    results = []

    # depending on config, use the summarizer pipeline or directly call the model
    # for inference
    if config.evaluation.use_pipeline:
        logger.info("Using summarization pipeline")
        summarizer = pipeline(
            "summarization",
            model=hf_model_loader.resolve_asset_path(config.model.path),
            device=0 if torch.cuda.is_available() else -1,
        )

        t = time.time()
        # for sample_txt in dataset_iterable:
        #     # summarizer output is a list (1 element in this case) of dict with key = "summary_text"
        #     results += summarizer(sample_txt, min_length=30, do_sample=False)

        # alternative: run on the whole dataset
        results = summarizer(dataset.select(range(10))["examples"], min_length=30, do_sample=False)

        logger.info(f"Summarization performed in {time.time()-t} seconds")

        results = [r["summary_text"] for r in results]

    else:
        logger.info("Using direct HF model invocation")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = hf_model_loader.load_pretrained_model(config.model).to(device)
        tokenizer = hf_tokenizer_loader.load_pretrained_tokenizer(config.tokenizer)

        for sample_txt in dataset_iterable:
            inputs = tokenizer(sample_txt, truncation=True, padding=True, return_tensors="pt").to(
                device
            )
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            output_txt = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            results += output_txt

    print(results)

    return "/tmp/dataset"


def run_hf_evaluation(config: HuggingFaceEvalJobConfig) -> EvaluationResult:
    # Run eval and store output in local filename
    result_dataset_path = run_eval(config)
    logger.info(f"Prometheus evaluation dataset stored at {result_dataset_path}")

    return EvaluationResult(
        artifacts=[],
        dataset_path=result_dataset_path,
        tables={},
    )
