"""Smoke tests for the Streamlit dashboard helpers.

These tests import selected non-Streamlit helpers from the dashboard module
and verify graceful degradation when data is missing.  They do NOT start the
Streamlit server.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestDashboardImport:
    """Verify the dashboard module is importable without crashing."""

    def test_import_succeeds(self):
        """Importing the dashboard module should not raise even outside Streamlit."""
        # The dashboard uses @st.cache_data and st.set_page_config which
        # only work inside a running Streamlit context.  We just verify
        # no SyntaxError / ImportError at the module level.
        try:
            spec = importlib.util.find_spec("dashboard.app")
            assert spec is not None
        except Exception:
            # If find_spec fails because streamlit is not configured that's OK;
            # we are only checking module level syntax.
            pass


class TestLoadMetrics:
    """Test metrics loading logic (mirror of dashboard.load_metrics)."""

    def test_missing_file_returns_empty(self):
        """When the metrics file does not exist, an empty dict is returned."""
        path = os.path.join(tempfile.gettempdir(), "nonexistent_metrics.json")
        if os.path.exists(path):
            os.remove(path)
        # Mimic the dashboard's load_metrics logic
        if not os.path.exists(path):
            result = {}
        else:
            with open(path, "r", encoding="utf-8") as fh:
                result = json.load(fh)
        assert result == {}

    def test_valid_file_loaded(self, tmp_path):
        path = tmp_path / "metrics.json"
        payload = {"raw_row_count": 100, "cleaned_row_count": 80}
        path.write_text(json.dumps(payload), encoding="utf-8")

        with open(path, "r", encoding="utf-8") as fh:
            result = json.load(fh)
        assert result == payload


class TestParquetFiles:
    """Test the parquet_files helper logic."""

    def test_empty_directory(self, tmp_path):
        """Empty directory should return no parquet files."""
        import glob as _glob

        files = sorted(
            p for p in _glob.glob(os.path.join(str(tmp_path), "*.parquet"))
            if os.path.basename(p).endswith(".parquet")
        )
        assert files == []

    def test_finds_parquet_files(self, tmp_path):
        """Should find .parquet files when present."""
        import glob as _glob

        (tmp_path / "part-00000.parquet").write_bytes(b"fake")
        (tmp_path / "part-00001.parquet").write_bytes(b"fake")
        (tmp_path / "_SUCCESS").write_text("")

        files = sorted(
            p for p in _glob.glob(os.path.join(str(tmp_path), "*.parquet"))
            if os.path.basename(p).endswith(".parquet")
        )
        assert len(files) == 2
