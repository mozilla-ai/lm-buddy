## Working with Flamingo

Submitting a Flamingo job includes two parts: a YAML file that specifies configuration for your finetuning or evaluation job, and a driver script that either invokes the Flamingo CLI directly or submits a job to Ray that invokes Flamingo as its entrypoint.

## Examples

For a full end-to-end interactive workflow running from within Flamingo, see the sample notebooks under `notebooks`.

For a full sample job with the correct directory structure that you can run as a part of stand-alone repo with a simple Python script that is [run locally to submit to the Job Submission SDK](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html#submitting-a-ray-job), see the `sample_job` directory.

## Fine-tuning Details



## Evaluation Details

All evaluation is currently done via [EleutherAI's lm-evaluation-harness package](https://github.com/EleutherAI/lm-evaluation-harness) run as a process. Evaluation can either happen on HuggingFace models hosted on the Hub, or on local models in shared storage on a Linux filesystem that resolve to [Weights and Biases Artifacts](https://docs.wandb.ai/ref/python/artifact) objects.

In the `evaluation` directory, there are sample files for running evaluation on a model in HuggingFace (`lm_harness_hf_config.yaml`), or using a local inference server hosted on vLLM, (`lm_harness_inference_server_config.yaml`).
