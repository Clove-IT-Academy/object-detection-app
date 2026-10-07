import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-model-tests", action="store_true", help="Use locally provisioned weights"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-model-tests"):
        skip = pytest.mark.skip(reason="Requires provisioned local model; use --run-model-tests")
        for item in items:
            if "model" in item.keywords:
                item.add_marker(skip)
