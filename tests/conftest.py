"""Shared pytest fixtures for the DSA Tutor test suite."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch):
    """Import backend.app without importing the heavyweight ML stack."""

    class PlaceholderRAGPipeline:
        pass

    fake_rag_pipeline = types.ModuleType("RAG.rag_pipeline")
    fake_rag_pipeline.RAGPipeline = PlaceholderRAGPipeline
    monkeypatch.setitem(sys.modules, "RAG.rag_pipeline", fake_rag_pipeline)

    sys.modules.pop("backend.app", None)
    module = importlib.import_module("backend.app")
    yield module

    sys.modules.pop("backend.app", None)
