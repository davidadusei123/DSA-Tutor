"""FastAPI backend for the DSA Tutor RAG app."""

from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from RAG.rag_pipeline import RAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["raw_answer", "summary"] = "summary"
    top_k: int = Field(default=3, ge=1, le=8)
    max_new_tokens: int = Field(default=300, ge=64, le=800)


class SourceChunk(BaseModel):
    source: str
    page: int | None = None
    score: float
    preview: str


class AskResponse(BaseModel):
    answer: str
    mode: str
    sources: list[SourceChunk]


app = FastAPI(title="DSA Tutor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RAGPipeline | None = None
_pipeline_lock = Lock()


def get_pipeline() -> RAGPipeline:
    """Create and cache the RAG pipeline for API requests."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            if not (VECTOR_DB_PATH / "index.faiss").exists():
                raise RuntimeError(
                    f"Vector DB not found at {VECTOR_DB_PATH}. "
                    "Build it before starting the API."
                )
            pipeline = RAGPipeline(str(DATA_DIR))
            pipeline.load_existing_vectordb(str(VECTOR_DB_PATH))
            _pipeline = pipeline
        return _pipeline


def format_sources(results: list[tuple]) -> list[SourceChunk]:
    """Convert FAISS results into API-friendly source previews."""
    sources: list[SourceChunk] = []
    for document, score in results:
        preview = " ".join(document.page_content.split())
        sources.append(
            SourceChunk(
                source=document.metadata.get("source", "Unknown"),
                page=document.metadata.get("page"),
                score=float(score),
                preview=preview[:320],
            )
        )
    return sources


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "vector_db_exists": (VECTOR_DB_PATH / "index.faiss").exists(),
        "adapter_exists": (PROJECT_ROOT / "models" / "phi-4-mini-dsa-adapter").exists(),
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        pipeline = get_pipeline()
        results = pipeline.query(request.query, top_k=request.top_k)
        answer = pipeline.generate_answer(
            request.query,
            mode=request.mode,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
        )
        return AskResponse(
            answer=answer,
            mode=request.mode,
            sources=format_sources(results),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
