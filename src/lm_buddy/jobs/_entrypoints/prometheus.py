# lm-buddy entrypoint to run evaluations using a Prometheus inference server
# see https://github.com/kaistAI/prometheus/blob/main/evaluation/benchmark/run_absolute_scoring.py

import copy
import json
from pathlib import Path

from datasets import Dataset, load_dataset
from fastchat.conversation import get_conv_template
from openai import Completion, OpenAI, OpenAIError
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from lm_buddy.integrations.huggingface import HuggingFaceAssetLoader
from lm_buddy.integrations.huggingface.tokenizer_config import AutoTokenizerConfig
from lm_buddy.integrations.wandb import (
    ArtifactLoader,
    ArtifactType,
    build_directory_artifact,
    wandb_init_from_config,
)
from lm_buddy.jobs.common import LMBuddyJobType
from lm_buddy.jobs.configs import PrometheusJobConfig


class BadResponseError(Exception):
    def __init__(self, message, error=None):
        self.message = message
        self.error = error


def openai_completion(config: PrometheusJobConfig, client: OpenAI, prompt: str) -> Completion:
    """Connects to a remote OpenAI-API-compatible Prometheus endpoint
    and returns a Completion holding the model's response.
    """

    return client.completions.create(
        model=config.prometheus.inference.engine,
        prompt=prompt,
        best_of=config.prometheus.best_of,
        max_tokens=config.prometheus.max_tokens,
        frequency_penalty=config.prometheus.frequency_penalty,
        temperature=config.prometheus.temperature,
        top_p=config.prometheus.top_p,
    )


def parse_response(config: PrometheusJobConfig, response: Completion) -> tuple[str, str]:
    """Given a Prometheus eval response as returned by the OpenAI API
    endpoint (i.e. in Completion format), extract feedback
    and score.
    """

    if response is None:
        raise BadResponseError("Server returned an empty response")

    try:
        response_text = response.choices[0].text
        # note: this can raise a ValueError if the message is malformed
        feedback, score = response_text.split("[RESULT]")
        feedback = feedback.strip()
        score = score.strip()
        if score not in [
            str(s) for s in range(config.evaluation.min_score, config.evaluation.max_score + 1)
        ]:
            raise BadResponseError(f"Score {score} is not in range")
    except (ValueError, BadResponseError) as e:
        raise BadResponseError(f"Server returned a malformed response ({e})", e)

    return feedback, score


def instruction_to_prompt(instruction: str) -> str:
    """Given some text containing Prometheus instructions, transform it
    into an actual llama-2 prompt.
    """
    conv = get_conv_template("llama-2")
    conv.set_system_message("You are a fair evaluator language model.")
    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def run_eval(
    config: PrometheusJobConfig, data: Dataset, tokenizer: PreTrainedTokenizer, client: OpenAI
) -> str:
    # enable / disable tqdm
    dataset_iterable = tqdm(data) if config.evaluation.enable_tqdm else data

    # open the output file for writing and iterate on samples
    tracking_name = config.tracking.name if config.tracking is not None else "output.json"
    output_fname = Path(config.evaluation.tmp_folder) / tracking_name
    with output_fname.open("w") as file:
        for sample in dataset_iterable:
            # convert instructions from the dataset (`text_field` in a dict) to
            # prompts that prometheus accepts
            prompt = instruction_to_prompt(sample[config.dataset.text_field])

            # skip those examples which are too long
            tokenized_prompt = tokenizer(prompt, truncation=False)
            if len(tokenized_prompt["input_ids"]) > 3072:
                continue

            # prepare output
            result = copy.deepcopy(sample)
            result["prometheus_output"] = []
            result["prometheus_score"] = []

            for idx in range(config.evaluation.num_answers):
                i = 0
                while i < config.evaluation.max_retries:
                    try:
                        response = openai_completion(config, client, prompt)
                        feedback, score = parse_response(config, response)
                        break
                    except (OpenAIError, BadResponseError) as e:
                        print(f"[w] {e.message}, retrying ({i+1}/{config.evaluation.max_retries})")
                        i += 1
                        if i == config.evaluation.max_retries:
                            raise e

                result["prometheus_output"].append(feedback)
                result["prometheus_score"].append(score)

            # dump sample results incrementally
            file.write(json.dumps(result) + "\n")

    # convert plain json dataset in HF format
    output_hf_name = str(Path(config.evaluation.tmp_folder) / "hf" / tracking_name)
    ds = load_dataset("json", data_files=str(output_fname), split="train")
    ds.save_to_disk(output_hf_name)

    return str(output_hf_name)


def run_prometheus(config: PrometheusJobConfig, artifact_loader: ArtifactLoader):
    # load dataset from W&B artifact
    hf_loader = HuggingFaceAssetLoader(artifact_loader)
    data = hf_loader.load_dataset(config.dataset)

    # get the tokenizer
    tokenizer_config = AutoTokenizerConfig(load_from=config.prometheus.inference.engine)
    tokenizer = hf_loader.load_pretrained_tokenizer(tokenizer_config)

    # instantiate OpenAI client to speak with the vLLM endpoint
    client = OpenAI(base_url=config.prometheus.inference.base_url)

    # Register a dataset file artifact if tracking is enabled
    if config.tracking:
        with wandb_init_from_config(config.tracking, job_type=LMBuddyJobType.EVALUATION):
            # run eval and store output in local filename
            output_dataset_name = run_eval(config, data, tokenizer, client)

            # store HF dataset as a directory artifact
            artifact = build_directory_artifact(
                dir_path=output_dataset_name,
                artifact_name=config.tracking.name,
                artifact_type=ArtifactType.DATASET,
                reference=False,
            )
            print("[i] Logging artifact for evaluation results...")
            artifact_loader.log_artifact(artifact)
    else:
        output_dataset_name = run_eval(config, data, tokenizer, client)
        print(f"[i] Evaluation results stored in {output_dataset_name}")
