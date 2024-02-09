# Contributing

## Setup

This project is built using the [Poetry](https://python-poetry.org/docs/) build tool.
First, install Poetry in your local environment via
```
curl -sSL https://install.python-poetry.org | python3 - -y
```
or see the [installation guide](https://python-poetry.org/docs/#installation)
for alternate installation methods.

Once Poetry is installed, you can install `flamingo` for development by running
```
poetry lock
poetry install
```
This will install an editable version of the package along with all of its dependency groups.

Poetry should recognize your active virtual environment during installation
If you have an active Conda environment, Poetry should recognize it during installation
and install the package dependencies there.
This hasn't been explicitly tested with other virtual python environments, but will likely work.

Alternatively, you can use poetry's own environment by running
```
poetry lock
poetry env use python3.10
poetry install
```
where `python3.10` is your python interpreter.

The `pyproject.toml` file defines dependency groups for the logical job types in the package.
Individual dependency groups can be installed by running
`poetry install --with <group1>,<group2>` or `poetry install --only <group>`.

Python should be [3.10, 3.11).

## Code style

This repository uses [Ruff](https://docs.astral.sh/ruff/) for Python formatting and linting.
Ruff should be installed automatically in your environment as part of the package's
development dependencies.

You can execute Ruff by calling `ruff --fix .` or `ruff format .` from the workspace root.
Ruff will pick up the configuration defined in the `pyproject.toml` file automatically.

## Testing a development branch

`flamingo` is intended to be installed as a pip requirement in the runtime environment of a Ray job.
However, it is often desirable to test local branches on Ray before publishing a new version of the library.

This is possible submitting a Ray job with a runtime environment that points to your
development branch of the `flamingo` repo.

To do so, follow the steps:

1. Export a copy of the package dependencies by running:

    ```
    poetry export --without-hashes --with finetuning,evaluation -o requirements.txt
    ```

    The following command will create a `requirements.txt` file in the repository
    that contains the dependencies for the `finetuning` and `evaluation` job groups:

2. When submitting a job to a Ray cluster, specify in the Ray runtime environment the following:

    - `py_modules`: Local path to the `flamingo` module folder (located at `src/flamingo` in the workspace).
    - `pip`: Local path to the `requirements.txt` file generated above.

3. Submit your job with an entrypoint command that invokes `flamingo` directly as a module, eg:

    ```
    python -m flamingo run finetuning --config config.yaml
    ```

    This is necessary because `py_modules` uploads the `flamingo` module
    but does not install its entrypoint in the environment path.

An example of this workflow can be found in the `examples/notebooks/dev_workflow.ipynb` notebook.

For a full sample job with a directory structure that you can run as a part of stand-alone repo with a simple Python script that is [run locally to submit to the Job Submission SDK](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html#submitting-a-ray-job), see the `examples/dev_submission` directory.


# Publishing

This is only relevant for maintainers / Mozilla.ai internal people.
Use the local installable package workflow above for iteration locally.

## Setup

This only needs to be done once.
You should have access to the API key via 1password. Make sure the 1password cli is installed (`brew install 1password-cli`).

Set up poetry to use the key(s):

```
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry config pypi-token.testpypi $(op read "op://<VAULT>/PyPI-test/pypi/api_key")
poetry config pypi-token.pypi $(op read "op://<VAULT>/PyPI/pypi/api_key")
```

## Test 

Then build and publish to PyPI Test:

```

poetry publish --repository testpypi --dry-run --build
poetry publish --repository testpypi --build
```

## Publish

When you're ready, run:

```
poetry publish --build
```
