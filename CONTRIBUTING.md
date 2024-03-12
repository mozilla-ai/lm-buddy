# Contributing

## Setup

The LM Buddy package can be installed for development by running:

```
pip install -e ".[dev]"
```

This will install an editable version of the package and all of its development requirements
in your active virtual environment.
Note that installation currently requires a Python version between `[3.10, 3.11)`.

## Code style

This repository uses [Ruff](https://docs.astral.sh/ruff/) for Python formatting and linting.
Ruff should be installed automatically in your environment as part of the package's
development dependencies.

You can execute Ruff by calling `ruff --fix .` or `ruff format .` from the workspace root.
Ruff will pick up the configuration defined in the `pyproject.toml` file automatically.

## Testing a development branch on Ray

LM Buddy is intended to be installed as a pip requirement in the runtime environment of a Ray job.
However, it is often desirable to test local branches on Ray before publishing a new version of the library.

This is possible by submitting a Ray job with a runtime environment that points to your
local development branch of the `lm-buddy` repo.

To do so, follow the steps:

1. Compile a locked version of the package requirements from the `pyproject.toml` file, 
which will create a `requirements.txt` file in the `lm-buddy` repository.
This can be done using multiple open-source tools, such as
[pip-tools](https://github.com/jazzband/pip-tools),
as shown below:

    ```
    # pip-tools
    pip install pip-tools
    pip-compile -o requirements.txt pyproject.toml

    ```

2. When submitting a job to a Ray cluster, specify in the Ray runtime environment the following:

    - `py_modules`: Local path to the LM Buddy module folder (located at `src/lm_buddy` in the repo).
    - `pip`: Local path to the `requirements.txt` file generated above. Make sure the path to this file
    matches the location of the requirements file generated in the first step. 

3. Submit your job with an entrypoint command that invokes `lm_buddy` directly as a module, eg:

    ```
    python -m lm_buddy run finetuning --config config.yaml
    ```

    This is necessary because `py_modules` uploads the `lm_buddy` module to the Ray cluster
    but does not install its entrypoint in the the Ray worker environment.

An example of this workflow can be found in the `examples/notebooks/dev_workflow.ipynb` notebook.

For a full sample job with a directory structure that you can run with a simple Python script that is 
[run locally to submit to the Job Submission SDK](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html#submitting-a-ray-job), 
see the `examples/dev_submission` directory.

## Publishing

> [!NOTE] 
>
> This section is intended for only maintainers at Mozilla.ai.

`.github/workflows/publish.yaml` contains the GitHub Action used to publish wheels to PyPI.

The publishing workflow can be triggered in two ways:
1. Manually triggered on the GitHub Actions UI, which publishes the package to TestPyPI, or
2. Triggered by the creation of a GitHub release, which publishes to real PyPI.

Note that there are some limitations on publishing to PyPI which are relevant to `lm-buddy`. 
For instance, we can't publish a package that has git-hash pinned dependencies because [PyPI requirements dictate](https://github.com/pypi/warehouse/blob/fca2efaee722cceef87e3e61926426de090db03b/warehouse/forklift/legacy.py#L280) that all packages be version-based.  

When creating a GitHub release, make sure that the tag you create in the GitHub release UI
matches the version of the target code branch specified in `pyproject.toml`.
