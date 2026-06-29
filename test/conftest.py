"""pytest configuration for site-rate-dating tests."""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring an installed CLI (treetime on PATH via uv run)",
    )
