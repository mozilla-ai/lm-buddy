# Contributing

## Code style

This repository uses [Ruff](https://docs.astral.sh/ruff/) for Python formatting and linting.
Ruff should be installed automatically in your environment as part of the package's
development dependencies.

You can execute Ruff by calling `ruff --fix .` or `ruff format .` from the workspace root.
Ruff will pick up the configuration defined in the `pyproject.toml` file automatically.

## Testing a development branch

`flamingo` is intended to be installed as a pip requirement in the runtime environment of a Ray job.
However, when developing the package locally it is desirable to be able to test your branch
by running jobs from it before publishing a new library version.
This is possible by submitting your Ray job with a runtime environment that points to your local,
in-development copy of the `flamingo` repo.

This can be done by the following steps:
1. Export a copy of the package dependencies by running 
`poetry export --without-hashes --with finetuning,evaluation -o requirements.txt`. 
This will create a `requirements.txt` file in the repository that contains the dependencies
for the `finetuning` and `evaluation` job groups.
2. In your Ray runtime environment, specify the following:
    - `py_modules`: Local path to the `flamingo` module folder (located at `src/flamingo` in the workspace).
    - `pip`: Local path to the `requirements.txt` file generated above.
3. Submit your job with an entrypoint command that invokes `flamingo` directly as a module,
e.g., `python -m flamingo run finetuning --config cofig.yaml`.
This is necessary because `py_modules` simply uploads the `flamingo` module
but does not install its entrypoint in the environment path.

An example of this workflow can be found in the `examples/dev_workflow.ipynb` notebook.
