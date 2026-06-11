"""RAG 検索ツール — v1 retriever の薄いラッパー（DESIGN.md §4.4）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import Settings
from src.rag.retriever import OpenFOAMRetriever

from .base import ToolResult


def _error(message: str) -> ToolResult:
    return ToolResult(ok=False, content=message)


def _make_retriever(settings: Settings) -> OpenFOAMRetriever:
    db_path = Path(__file__).parent.parent.parent / "knowledge_base" / "chroma_db"
    return OpenFOAMRetriever(
        db_path=str(db_path),
        openai_api_key=settings.openai_api_key,
    )


def _format_hit(rank: int, meta: dict, document: str, distance: float) -> str:
    case_id = meta.get("case_id", "unknown")
    title = meta.get("title_ja") or meta.get("summary_ja") or case_id
    solver = meta.get("solver", "?")
    snippet = document.replace("\n", " ").strip()
    if len(snippet) > 240:
        snippet = snippet[:240] + "..."
    return (
        f"{rank}. {title}\n"
        f"   case_id={case_id}, solver={solver}, distance={distance:.3f}\n"
        f"   {snippet}"
    )


def rag_search(
    query: str,
    scope: str = "case",
    top_k: int = 3,
    filters: dict[str, Any] | None = None,
    *,
    settings: Settings | None = None,
) -> ToolResult:
    """
    ChromaDB をベクトル検索する。

    Phase 1 では scope=\"case\" のみ。scope=\"file\" は未実装。
    """
    if scope == "file":
        return _error("scope='file' is not implemented in Phase 1.")
    if scope != "case":
        return _error(f"Unknown scope: {scope}. Use 'case' or 'file'.")

    query = query.strip()
    if not query:
        return _error("query must not be empty.")

    top_k = max(1, min(int(top_k), 5))
    settings = settings or Settings()
    retriever = _make_retriever(settings)

    if not retriever.is_available:
        return ToolResult(
            ok=True,
            content="RAG index not available. Build with: python -m src.main build-index",
            data={"results": [], "scope": scope, "filters": filters or {}},
        )

    selector = retriever.selector
    embedding = selector._embed(query)
    count = selector.collection.count()
    n_results = min(top_k, count)
    if n_results == 0:
        return ToolResult(ok=True, content="No documents in RAG index.", data={"results": []})

    try:
        raw = selector.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        return _error(f"RAG search failed: {exc}")

    docs = raw["documents"][0] if raw.get("documents") else []
    metas = raw["metadatas"][0] if raw.get("metadatas") else []
    dists = raw["distances"][0] if raw.get("distances") else []

    hits: list[dict[str, Any]] = []
    lines: list[str] = [f"RAG search ({scope}, top_k={top_k}):"]
    for index, (meta, document, distance) in enumerate(zip(metas, docs, dists), start=1):
        hit = {
            "rank": index,
            "case_id": meta.get("case_id", ""),
            "case_path": meta.get("case_path", ""),
            "solver": meta.get("solver", ""),
            "title_ja": meta.get("title_ja", ""),
            "summary_ja": meta.get("summary_ja", ""),
            "distance": distance,
            "document": document,
            "metadata": meta,
        }
        hits.append(hit)
        lines.append(_format_hit(index, meta, document, distance))

    if filters:
        lines.append(f"(filters accepted but not applied in Phase 1: {filters})")

    return ToolResult(
        ok=True,
        content="\n".join(lines),
        data={"results": hits, "scope": scope, "filters": filters or {}},
    )
