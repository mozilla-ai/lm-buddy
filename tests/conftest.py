"""
This file is used to provide fixtures for the test session accessible to all Flamingo submodules.
"""
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def examples_dir():
    return Path(__file__).parents[1] / "examples"


@pytest.fixture(scope="session")
def resources_dir():
    return Path(__file__).parent / "resources"
