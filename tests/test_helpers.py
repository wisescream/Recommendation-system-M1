"""Tests for the Kaggle download helper module."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from helpers.download_kaggle import TARGET_FILE, require_credentials


class TestRequireCredentials:
    def test_raises_when_both_missing(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        with pytest.raises(SystemExit, match="Missing Kaggle credentials"):
            require_credentials()

    def test_raises_when_username_missing(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
        with pytest.raises(SystemExit, match="KAGGLE_USERNAME"):
            require_credentials()

    def test_raises_when_key_missing(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "dummy_user")
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        with pytest.raises(SystemExit, match="KAGGLE_KEY"):
            require_credentials()

    def test_passes_when_both_present(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "dummy_user")
        monkeypatch.setenv("KAGGLE_KEY", "dummy_key")
        # Should not raise
        require_credentials()


class TestTargetFile:
    def test_path_ends_with_csv(self):
        assert str(TARGET_FILE).endswith(".csv")
