# flamingo

<p align="center">
    <img src="https://github.com/mozilla-ai/flamingo/blob/main/assets/flamingo.png" width="300">
</p>

## Getting started

### Installation

This project is built using the [Poetry](https://python-poetry.org/docs/) build tool.
First, install Poetry in your local environment via
```
curl -sSL https://install.python-poetry.org | python3 - -y
```
or see the [installation guide](https://python-poetry.org/docs/#installation)
for more instructions.

Once Poetry is installed, you can install `flamingo` for development by running
```
poetry lock
poetry install
```
This will install an editable version of the package along with all of its dependency groups.
Poetry should recognize your active virtual environment during installation
and install the package dependencies there.

The `pyproject.toml` file defines dependency groups for the logical job types in the package.
Individual dependency groups can be installed by running 
`poetry install --with <group1>,<group2>` or `poetry install --only <group>`.

See the [contributing](CONTRIBUTING.md) guide for more information on development workflows.

### Python version

This library is developed with the same Python version as the Ray cluster
to avoid dependency/syntax errors when executing code remotely.
Currently, installation requires at least Python 3.10 to match the global
cluster environment (Ray cluster is running 3.10.8).
