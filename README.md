# flamingo

<p align="center">
    <img src="https://github.com/mozilla-ai/flamingo/blob/main/assets/flamingo.png" width="450">
</p>

## Installation

Install the package for local development in your chosen Python environment by running:

```
pip install -e ".[all]"
```

Dependency groups are defined for the logical job groups accessible from the library.
See `pyproject.toml` for exact information.

### Python version

This library is developed with the same Python version as the Ray cluster
to avoid dependency/syntax errors when executing code remotely.
Currently, installation requires at least Python 3.10 to match the global
cluster environment (Ray cluster is running 3.10.8).
