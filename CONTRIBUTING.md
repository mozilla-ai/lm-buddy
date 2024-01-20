# Contributing

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

2. In your Ray runtime environment, specify the following:

    - `py_modules`: Local path to the `flamingo` module folder (located at `src/flamingo` in the workspace).
    - `pip`: Local path to the `requirements.txt` file generated above.

3. Submit your job with an entrypoint command that invokes `flamingo` directly as a module, eg:

    ```
    python -m flamingo run finetuning --config config.yaml
    ```

    This is necessary because `py_modules` uploads the `flamingo` module
    but does not install its entrypoint in the environment path.

An example of this workflow can be found in the `examples/dev_workflow.ipynb` notebook.

