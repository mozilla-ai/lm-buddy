# flamingo

<p align="center">
    <img src="https://github.com/mozilla-ai/flamingo/blob/main/assets/flamingo.png" width="300">
</p>

Flamingo is a library of tools for managing the finetuning and evaluation lifecycle of open-source large language models, making use of YAML-based configs as input to jobs for Ray on Kubernetes.

The package currently allows users to launch either a:
1. **fine-tuning job** using HuggingFace style model paths or Weights&Biases artifact locations
2. **evaluation job** using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) against either a HuggingFace Hub model, or on a local model inference server via vLLM
3. **simple job** as an example job

## Workflow
For all workflow examples, (see `/examples`)

The user starts by writing a YAML config file (see `/examples`) and an accompanying Ray job entrypoint. See the `examples/configs` folder for examples of the configuration structure. For a full end-to-end interactive workflow running from within Flamingo, see the sample notebooks. For a full sample job with the correct directory structure that you can run as a stand-alone if you have flaimgo installed in your Python environment , see the `sample_job` directory.

## Usage

### Minimum Python version

This library is developed with the same Python version as the Ray cluster
to avoid dependency/syntax errors when executing code remotely.
Currently, installation requires Python between `[3.10, 3.11)` to match the global
cluster environment (Ray cluster is running 3.10.8).

### Installation

```
pip install mzai-flamingo
```

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

When submitting a job to Ray, the above commands should be used as your job entrypoints.

### Development

See the [contributing](CONTRIBUTING.md) guide for more information on development workflows and/or building locally.
