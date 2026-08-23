"""Regression tests for the FastAPI application helpers and endpoints."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError


def make_document(
    page_content: str,
    *,
    source: str | None = None,
    page: int | None = None,
):
    """Create the small portion of a LangChain Document used by app.py."""

    metadata = {}
    if source is not None:
        metadata["source"] = source
    if page is not None:
        metadata["page"] = page
    return SimpleNamespace(page_content=page_content, metadata=metadata)


def test_get_pipeline_loads_database_once_and_caches_instance(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    vector_db = tmp_path / "vector_db"
    vector_db.mkdir()
    (vector_db / "index.faiss").touch()
    data_dir = tmp_path / "Data"

    pipeline = Mock()
    pipeline_class = Mock(return_value=pipeline)
    monkeypatch.setattr(app_module, "VECTOR_DB_PATH", vector_db)
    monkeypatch.setattr(app_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(app_module, "RAGPipeline", pipeline_class)
    monkeypatch.setattr(app_module, "_pipeline", None)

    first_result = app_module.get_pipeline()
    second_result = app_module.get_pipeline()

    assert first_result is pipeline
    assert second_result is pipeline
    pipeline_class.assert_called_once_with(str(data_dir))
    pipeline.load_existing_vectordb.assert_called_once_with(str(vector_db))


def test_get_pipeline_raises_when_faiss_index_is_missing(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    vector_db = tmp_path / "missing-vector-db"
    pipeline_class = Mock()
    monkeypatch.setattr(app_module, "VECTOR_DB_PATH", vector_db)
    monkeypatch.setattr(app_module, "RAGPipeline", pipeline_class)
    monkeypatch.setattr(app_module, "_pipeline", None)

    with pytest.raises(RuntimeError, match="Vector DB not found"):
        app_module.get_pipeline()

    pipeline_class.assert_not_called()
    assert app_module._pipeline is None


def test_get_pipeline_does_not_cache_a_failed_load(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    vector_db = tmp_path / "vector_db"
    vector_db.mkdir()
    (vector_db / "index.faiss").touch()
    pipeline = Mock()
    pipeline.load_existing_vectordb.side_effect = RuntimeError("invalid index")
    monkeypatch.setattr(app_module, "VECTOR_DB_PATH", vector_db)
    monkeypatch.setattr(app_module, "RAGPipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(app_module, "_pipeline", None)

    with pytest.raises(RuntimeError, match="invalid index"):
        app_module.get_pipeline()

    assert app_module._pipeline is None


def test_get_pipeline_returns_cached_instance_even_if_index_is_removed(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cached_pipeline = object()
    monkeypatch.setattr(app_module, "VECTOR_DB_PATH", tmp_path / "missing")
    monkeypatch.setattr(app_module, "_pipeline", cached_pipeline)

    assert app_module.get_pipeline() is cached_pipeline


def test_format_sources_normalizes_content_and_preserves_metadata(app_module):
    document = make_document(
        "  Binary\nsearch\tremoves   half the search space.  ",
        source="algorithms.pdf",
        page=0,
    )

    sources = app_module.format_sources([(document, 0.875)])

    assert len(sources) == 1
    assert sources[0].source == "algorithms.pdf"
    assert sources[0].page == 0
    assert sources[0].score == pytest.approx(0.875)
    assert sources[0].preview == "Binary search removes half the search space."


def test_format_sources_uses_defaults_and_truncates_preview(app_module):
    document = make_document("x" * 400)

    source = app_module.format_sources([(document, "0.5")])[0]

    assert source.source == "Unknown"
    assert source.page is None
    assert source.score == 0.5
    assert source.preview == "x" * 320


def test_format_sources_accepts_no_results(app_module):
    assert app_module.format_sources([]) == []


def test_health_reports_artifact_and_hosted_generation_state(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    vector_db = tmp_path / "vector_db"
    vector_db.mkdir()
    (vector_db / "index.faiss").touch()
    (tmp_path / "models" / "phi-4-mini-dsa-adapter").mkdir(parents=True)
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "VECTOR_DB_PATH", vector_db)
    monkeypatch.setattr(app_module, "HF_ENDPOINT_URL", "https://example.test")
    monkeypatch.setattr(app_module, "HF_TOKEN", "secret")

    assert app_module.health() == {
        "status": "ok",
        "vector_db_exists": True,
        "adapter_exists": True,
        "hosted_generation_configured": True,
    }


def test_ask_returns_answer_and_formatted_sources(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
):
    document = make_document("A stack is last-in, first-out.", source="stacks.pdf", page=2)
    pipeline = Mock()
    pipeline.query_with_cosine_similarity.return_value = [(document, 0.91)]
    pipeline.generate_answer.return_value = "Use a stack."
    monkeypatch.setattr(app_module, "get_pipeline", lambda: pipeline)
    request = app_module.AskRequest(
        query="What should I use for LIFO?",
        mode="summary",
        top_k=4,
        max_new_tokens=200,
        similarity_threshold=0.6,
    )

    response = app_module.ask(request)

    assert response.answer == "Use a stack."
    assert response.mode == "summary"
    assert response.sources[0].source == "stacks.pdf"
    pipeline.query_with_cosine_similarity.assert_called_once_with(
        request.query,
        top_k=4,
    )
    pipeline.generate_answer.assert_called_once_with(
        request.query,
        mode="summary",
        top_k=4,
        max_new_tokens=200,
        similarity_threshold=0.6,
    )


@pytest.mark.parametrize(
    ("pipeline_error", "expected_status"),
    [(ValueError("bad query"), 400), (RuntimeError("model unavailable"), 503)],
)
def test_ask_translates_pipeline_errors_to_http_errors(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_error: Exception,
    expected_status: int,
):
    def raise_error():
        raise pipeline_error

    monkeypatch.setattr(app_module, "get_pipeline", raise_error)

    with pytest.raises(HTTPException) as exc_info:
        app_module.ask(app_module.AskRequest(query="Explain queues"))

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == str(pipeline_error)


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"query": ""},
        {"query": "valid", "mode": "verbose"},
        {"query": "valid", "top_k": 0},
        {"query": "valid", "top_k": 9},
        {"query": "valid", "max_new_tokens": 63},
        {"query": "valid", "similarity_threshold": 1.1},
    ],
)
def test_ask_request_rejects_invalid_boundaries(app_module, invalid_fields):
    with pytest.raises(ValidationError):
        app_module.AskRequest(**invalid_fields)
