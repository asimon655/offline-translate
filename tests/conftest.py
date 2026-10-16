import os
from pathlib import Path

import pytest

from fakes import FakeGoogle, FakeModel


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "network" in item.keywords and not config.getoption("--network"):
            item.add_marker(pytest.mark.skip(reason="needs --network"))
        if "models" in item.keywords and not config.getoption("--models"):
            item.add_marker(pytest.mark.skip(reason="needs --models=DIR"))


@pytest.fixture
def google(monkeypatch):
    """A fake dl.google.com serving Hebrew and French (supported) and Bosnian (old format)."""
    return FakeGoogle().install(monkeypatch)


@pytest.fixture
def fake_engine(monkeypatch):
    from offline_translate import engine
    FakeModel.loaded = []
    monkeypatch.setattr(engine, "NmtModel", FakeModel)
    return FakeModel


@pytest.fixture
def models_dir(request):
    return Path(request.config.getoption("--models")).expanduser()
