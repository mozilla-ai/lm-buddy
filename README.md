# LM Buddy

LM Buddy is a library of tools for managing the finetuning and evaluation lifecycle
of open-source large language models, 
using YAML-based configs and CLI primitives as input to jobs for Ray on Kubernetes.

The package currently allows users to launch either a:
1. **finetuning job** using HuggingFace style model paths or Weights&Biases artifact locations
2. **evaluation job** using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with either a HuggingFace Hub model, or on a model inference server [via vLLM](https://github.com/vllm-project/vllm)


### Installation

```
pip install lm-buddy
```

### CLI Usage

LM Buddy exposes a CLI with a few commands, one for each Ray job type.
To see all commands, run `lm_buddy run --help`

Once LM Buddy is installed in your local Python environment, usage is as follows:
```
# Simple test
lm_buddy run simple --config simple_config.yaml

# LLM finetuning
lm_buddy run finetuning --config finetuning_config.yaml

# LLM evaluation
lm_buddy run lm-harness --config lm_harness_config.yaml
```

See the `examples/configs` folder for examples of the configuration structure. 
For a full end-to-end interactive workflow for using the package, see the example notebooks.

## Workflow

There are several possible workflows:

1) A workflow for users on Ray
2) A workflow for users on GPU-based machines
3) A workflow for library developers and maintainers.

For all workflow examples, (see `/examples`)

## User Ray Workflow

The user starts by pip-installing `lm-buddy` into a new project environment and creating a directory for their experiment jobs. In that new directory, create a a YAML config file
(see `/examples`).

This file can then be passed as an argument via LM Buddy CLI.

### Ray Usage
When submitting a job to Ray, the above commands should be used as your job entrypoints. 
An additional option is to wrap the job submission in a Python script 
that sets up the Ray client and additional parameters and is then passed to Ray's Job Submission SDK.

```
client.submit_job(
    entrypoint="python -m lm_buddy run lm-harness --config lm_harness.yaml", 
    runtime_env=runtime_env
)
```

### Minimum Python version

This library is developed with the same Python version as the Ray cluster
to avoid dependency/syntax errors when executing code remotely.
Currently, installation requires Python between `[3.10, 3.11)` to match the global
cluster environment (Ray cluster is running 3.10.8).


### Development

See the [contributing](CONTRIBUTING.md) guide for more information on development workflows and/or building locally.
