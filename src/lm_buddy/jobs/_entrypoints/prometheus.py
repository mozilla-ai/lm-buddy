from lm_buddy.jobs.configs import PrometheusJobConfig
from lm_buddy.integrations.huggingface import HuggingFaceAssetLoader
from lm_buddy.integrations.wandb import (
    ArtifactType, 
    ArtifactLoader, 
    build_file_artifact, 
    wandb_init_from_config,
)
from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer
from openai import OpenAIError, OpenAI

from tqdm import tqdm
import os
import json
import copy

class BadResponseException(Exception):
    def __init__(self, message, error):
        self.message = message
        self.error = error


def openai_completion(config, client, prompt):
    return client.completions.create(
        model = "kaist-ai/prometheus-13b-v1.0",
        prompt = prompt,
        best_of = config.prometheus.best_of,
        max_tokens = config.prometheus.max_tokens,
        frequency_penalty = config.prometheus.frequency_penalty,
        temperature = config.prometheus.temperature,
        top_p = config.prometheus.top_p
    )


def parse_response(response):
    try:
        assert response is not None
        response_text = response.choices[0].text
        feedback, score = response_text.split('[RESULT]')
        feedback = feedback.strip()
        score = score.strip()
        assert score in ["1","2","3","4","5"]
    except (ValueError, AssertionError) as e:
        raise BadResponseException("Server returned a bad response", e)

    return feedback, score


def instruction_to_prompt(instruction):
    conv = get_conv_template("llama-2")
    conv.set_system_message("You are a fair evaluator language model.")
    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def run_prometheus(config: PrometheusJobConfig, artifact_loader: ArtifactLoader):

    # load dataset from W&B artifact
    hf_loader = HuggingFaceAssetLoader(artifact_loader)
    artifact_path,_ = hf_loader.resolve_asset_path(config.dataset.load_from)
    dataset_fname = os.path.join(artifact_path, config.dataset.load_from.name)
    
    with open(dataset_fname,'r') as f:
        # eval samples are JSON-encoded, each takes one line in the dataset file
        data = [json.loads(line) for line in f.readlines()]

    # get the tokenizer
    tokenizer = hf_loader.load_pretrained_tokenizer(config.prometheus.tokenizer)

    # instantiate OpenAI client to speak with the vLLM endpoint
    client = OpenAI(
        base_url = config.prometheus.inference.base_url
    )

    # open the output file for writing and iterate on samples
    output_fname = os.path.join("/tmp", config.tracking.name)
    with open(output_fname,'w') as file:
        for sample in tqdm(data):
            # convert instructions from the dataset (`text_field` in a dict) to
            # prompts that prometheus accepts
            prompt = instruction_to_prompt(sample[config.dataset.text_field])

            # skip those examples which are too long 
            tokenized_prompt  = tokenizer(prompt, truncation=False)
            if(len(tokenized_prompt['input_ids'])>3072):
                continue

            # prepare output
            result = copy.deepcopy(sample)
            result['prometheus_output'] = []
            result['prometheus_score'] = []

            for idx in range(config.prometheus.num_answers):

                i = 0
                while i < config.prometheus.max_retries: 
                    try:
                        response = openai_completion(config, client, prompt)
                        feedback, score = parse_response(response)
                        print(feedback, score)
                        break
                    except (OpenAIError, BadResponseException) as e:
                        print(f"[w] {e.message}, retrying ({i+1}/{config.prometheus.max_retries})")
                        i += 1
                        if i == config.prometheus.max_retries:
                            raise e
                
                result['prometheus_output'].append(feedback)
                result['prometheus_score'].append(score)

            # dump sample results
            file.write(json.dumps(result)+"\n")


    # Register a dataset file artifact if tracking is enabled
    if config.tracking:
        
        with wandb_init_from_config(config.tracking) as run:
            file_artifact = build_file_artifact(
                artifact_name = config.tracking.name, 
                artifact_type = ArtifactType.DATASET,
                file_path = output_fname,
                reference = False,
            )
            print("[i] Logging artifact for evaluation results...")
            artifact_loader.log_artifact(file_artifact)
