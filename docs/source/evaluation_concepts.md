Evaluation
====================================

All evaluation is currently done via [EleutherAI's lm-evaluation-harness package](https://github.com/EleutherAI/lm-evaluation-harness) run as a process. Evaluation can either happen on HuggingFace models hosted on the Hub, or on local models in shared storage on a Linux filesystem that resolve to [Weights and Biases Artifacts](https://docs.wandb.ai/ref/python/artifact) objects.

In the `evaluation` directory, there are sample files for running evaluation on a model in HuggingFace (`lm_harness_hf_config.yaml`), or using a local inference server hosted on vLLM, (`lm_harness_inference_server_config.yaml`).
