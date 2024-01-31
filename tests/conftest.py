"""
Tests for the Flamingo.

This file is used to provide fixtures for the test session that are accessible to all submodules.
"""
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def examples_dir():
    return Path(__file__).parents[1] / "examples"


@pytest.fixture(scope="session")
def resources_dir():
    return Path(__file__).parent / "resources"
