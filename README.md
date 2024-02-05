# flamingo

<p align="center">
    <img src="https://github.com/mozilla-ai/flamingo/blob/main/assets/flamingo.png" width="300">
</p>

Flamingo is a library of tools for managing the finetuning and evaluation lifecycle of open-source large language models, using YAML-based configs and CLI primitives as input to jobs for Ray on Kubernetes.

The package currently allows users to launch either a:
1. **finetuning job** using HuggingFace style model paths or Weights&Biases artifact locations
2. **evaluation job** using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with either a HuggingFace Hub model, or on a model inference server [via vLLM](https://github.com/vllm-project/vllm)


### Installation

```
pip install mzai-flamingo
```

## Workflow

There are several possible workflows:

1) A workflow for users on Ray
2) A workflow for users on GPU-based machines
3) A workflow for library developers and maintainers.

For all workflow examples, (see `/examples`)

## User Ray Workflow

The user starts by pip-installing `mzai-flamingo` into a new project environment and creating a directory for their experiment jobs. In that new directory, create a a YAML config file
(see `/examples`).

This file can then be passed as an argument via Flamingo CLI.

### CLI Usage

`flamingo` exposes a CLI with a few commands, one for each Ray job type.
To see all commands, run `flamingo run --help`

Once flamingo is installed in your local Python environment, usage is as follows:
```
# Simple test
flamingo run simple --config simple_config.yaml

# LLM finetuning
flamingo run finetuning --config finetuning_config.yaml

# LLM evaluation
flamingo run lm-harness --config lm_harness_config.yaml
```

See the `examples/configs` folder for examples of the configuration structure. For a full end-to-end interactive workflow running from within Flamingo, see the sample notebooks.

### Ray Usage
When submitting a job to Ray, the above commands should be used as your job entrypoints. An additional option is to wrap the Flamingo submission in a Python script that sets up the Ray client and additional parameters and is then passed to Ray's Job Submission SDK.

```
client.submit_job(
    entrypoint="python -m flamingo run lm-harness --config lm_harness.yaml", runtime_env=runtime_env
)```

For a full sample job with the correct directory structure that you can run as a stand-alone if you have flamingo installed in your Python environment , see the `sample_job` directory.


### Minimum Python version

This library is developed with the same Python version as the Ray cluster
to avoid dependency/syntax errors when executing code remotely.
Currently, installation requires Python between `[3.10, 3.11)` to match the global
cluster environment (Ray cluster is running 3.10.8).


### Development

See the [contributing](CONTRIBUTING.md) guide for more information on development workflows and/or building locally.
