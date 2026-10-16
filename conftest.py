"""Root conftest: pytest only reads command-line options registered here."""
import os


def pytest_addoption(parser):
    parser.addoption("--network", action="store_true", help="run tests that talk to dl.google.com")
    parser.addoption("--models", metavar="DIR", default=os.environ.get("OFFLINE_TRANSLATE_TEST_MODELS"),
                     help="models folder with Hebrew installed, for end-to-end translation tests")
