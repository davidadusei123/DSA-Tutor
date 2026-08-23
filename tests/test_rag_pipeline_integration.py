"""Opt-in integration coverage for the real embedding model and FAISS index."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_RAG_INTEGRATION") != "1",
    reason="Set RUN_RAG_INTEGRATION=1 to load the real embedding model and index.",
)
def test_existing_vector_database_can_answer_a_query():
    """Load the checked-in project artifacts without rebuilding or overwriting them."""

    from RAG.rag_pipeline import RAGPipeline

    project_root = Path(__file__).resolve().parents[1]
    pipeline = RAGPipeline(str(project_root / "Data"))
    vector_store = pipeline.load_existing_vectordb(str(project_root / "vector_db"))

    results = pipeline.query("Explain depth first search in simple terms", top_k=3)

    assert vector_store.index.ntotal > 0
    assert len(results) == 3
    assert all(document.page_content for document, _score in results)
