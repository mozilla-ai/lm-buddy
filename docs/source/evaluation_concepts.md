Evaluation
====================================

## lm-evaluation-harness

Evaluation is currently done via [EleutherAI's lm-evaluation-harness package](https://github.com/EleutherAI/lm-evaluation-harness) run as a process. Evaluation can either happen on HuggingFace models hosted on the Hub, or on local models in shared storage on a Linux filesystem that resolve to [Weights and Biases Artifacts](https://docs.wandb.ai/ref/python/artifact) objects

In the `evaluation` directory, there are sample files for running evaluation on a model in HuggingFace (`lm_harness_hf_config.yaml`), or using a local inference server hosted on vLLM, (`lm_harness_inference_server_config.yaml`).

## Prometheus

Evaluation relies on [Prometheus](https://github.com/kaistAI/Prometheus) as LLM judge. We internally serve it via [vLLM](https://github.com/vllm-project/vllm) but any other OpenAI API compatible service should work (e.g. llamafile via their `api_like_OAI.py` script).

Input datasets _must_ be in HuggingFace format. The code below shows how to convert Prometheus benchmark datasets and optionally save them as wandb artifacts:

```
import wandb
from datasets import load_dataset
from lm_buddy.tracking.artifact_utils import (
    ArtifactType,
    build_directory_artifact,
)
from lm_buddy.jobs.common import JobType

artifact_name = "tutorial_vicuna_eval"
dataset_fname = "/path/to/prometheus/evaluation/benchmark/data/vicuna_eval.json"
output_path = "/tmp/tutorial_vicuna_eval"

# load the json dataset and save it in HF format
ds = load_dataset("json", data_files = dataset_fname, split='train')
ds.save_to_disk(output_path)

with wandb.init(job_type=JobType.PREPROCESSING,
                project="wandb-project-name",
                entity="wandb-entity-name",
                name=artifact_name
               ):
    artifact = build_directory_artifact(
        dir_path=output_path,
        artifact_name=artifact_name,
        artifact_type=ArtifactType.DATASET,
        reference=False,
    )
    wandb.log_artifact(artifact)
```

In the `evaluation` directory, you will find a sample `prometheus_config.yaml` file for running Prometheus evaluation. Before using it, you will need to specify the `path` of the input dataset, the `base_url` where the Prometheus model is served, and
the `tracking` options to save the evaluation output on wandb.

You can then run the evaluation as:

```
python -m lm_buddy evaluate prometheus --config /path/to/prometheus_config.yaml
```