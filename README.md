# LM Buddy

LM Buddy is a collection of jobs for finetuning and evaluating open-source (large) language models.
The library makes use of YAML-based configuration files as inputs to CLI commands for each job,
and tracks input/output artifacts on [Weights & Biases](https://docs.wandb.ai/).

The package currently exposes two types of jobs:
1. **finetuning job** using HuggingFace model/training implementations and 
[Ray Train](https://docs.ray.io/en/latest/train/train.html)
for compute scaling, or an
2. **evaluation job** using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 
with inference performed via an in-process HuggingFace model or an externally-hosted 
[vLLM](https://github.com/vllm-project/vllm) server.

> [!WARNING]
>
> LM Buddy is in the early stages of development.
> It is missing important features and documentation.
> You should expect breaking changes in the core interfaces and configuration structures
> as development continues.
> Use only if you are comfortable working in this environment.

## Installation

TODO: Finalize this section once published to PyPi

```
pip install lm-buddy
```

### Minimum Python version

LM Buddy is intended to be used in production on a Ray cluster 
(see section below on [Ray job submission](#ray-job-submission)).
Currently, we are utilizing Ray clusters running Python 3.10.8.
In order to avoid dependency/syntax errors when executing LM Buddy on Ray,
installation of this package requires Python between `[3.10, 3.11)`.

## CLI usage

LM Buddy exposes a CLI with a few commands, one for each type of job.
To see all available job commands, run `lm_buddy run --help`

Once LM Buddy is installed in your local Python environment, usage is as follows:
```
# Simple test
lm_buddy run simple --config simple_config.yaml

# LLM finetuning
lm_buddy run finetuning --config finetuning_config.yaml

# LLM evaluation
lm_buddy run lm-harness --config lm_harness_config.yaml
```

See the `examples/configs` folder for examples of the job configuration structure. 
For a full end-to-end interactive workflow for using the package, see the example notebooks.

## Ray job submission

Although the LM Buddy CLI can be used as a standalone tool,
its commands are intended to be used as the entrypoints for jobs on a
[Ray](https://docs.ray.io/en/latest/index.html) compute cluster.
The suggested method for submitting an LM Buddy job to Ray is by using the 
[Ray Python SDK](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html) 
within a local Python driver script.
This requires you to specify a Ray runtime environment containing:
1) A `working_dir` for the local directory containing your job config YAML file, and
2) A `pip` dependency for your desired version of `lm-buddy`.

An example of the submission process is as follows:

```
from ray.job_submission import JobSubmissionClient

# If using a remote cluster, replace 127.0.0.1 with the head node's IP address.
client = JobSubmissionClient("http://127.0.0.1:8265")

runtime_env = {
    "working_dir": "/path/to/working/directory",
    "pip": ["lm-buddy==X.X.X"]
}

# Assuming 'config.yaml' is present in the working directory
client.submit_job(
    entrypoint="lm_buddy run <job-name> --config config.yaml", 
    runtime_env=runtime_env
)
```

See the `examples/` folder for more examples of submitting Ray jobs.

## Development

See the [contributing](CONTRIBUTING.md) guide for more information on development workflows 
and/or building locally.
