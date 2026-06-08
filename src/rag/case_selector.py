"""
ケース選定 — SimulationSpec から参照チュートリアルケースを 1 件選ぶ
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb
from openai import OpenAI
from rich.console import Console

from ..models import SimulationSpec
from .case_catalog import load_case_files
from .case_intent import phenomenon_matches
from .indexer import COLLECTION_NAME

console = Console()

# solver 互換マップ（spec.solver → 許容 solver 一覧）
SOLVER_COMPAT: dict[str, set[str]] = {
    "simpleFoam": {"simpleFoam"},
    "pimpleFoam": {"pimpleFoam", "pisoFoam"},
    "icoFoam": {"icoFoam", "pisoFoam"},
    "pisoFoam": {"pisoFoam", "icoFoam", "pimpleFoam"},
    "nonNewtonianIcoFoam": {"nonNewtonianIcoFoam", "icoFoam"},
}

# RAS モデル同士は参照ケース選定で互換とみなす
REFERENCE_RAS_MODELS = frozenset({
    "kOmegaSST", "kEpsilon", "RNGkEpsilon", "SpalartAllmaras",
    "kkLOmega", "RAS", "realizableKE", "LaunderSharmaKE",
})

TURBULENCE_COMPAT: dict[str, set[str]] = {
    "laminar": {"laminar", "unknown"},
    "kOmegaSST": {"kOmegaSST", "RAS", "unknown"},
    "kEpsilon": {"kEpsilon", "RNGkEpsilon", "RAS", "unknown"},
    "SpalartAllmaras": {"SpalartAllmaras", "RAS", "unknown"},
    "LES": {"LES"},
}


def _meta_bool(meta: dict, key: str) -> bool:
    v = meta.get(key)
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def _turbulence_compatible(spec_model: str, case_model: str) -> bool:
    """参照ケース選定用の乱流モデル互換判定。"""
    if spec_model == "laminar":
        return case_model in (
            "laminar", "unknown", "RAS", "LES",
            "kEpsilon", "kOmegaSST", "kkLOmega", "SpalartAllmaras",
        )
    if spec_model in REFERENCE_RAS_MODELS:
        return case_model in REFERENCE_RAS_MODELS or case_model in ("unknown", "laminar")
    if spec_model == "LES":
        return case_model in ("LES", "unknown")
    allowed = TURBULENCE_COMPAT.get(spec_model, {spec_model, "unknown", "RAS"})
    return case_model in allowed or case_model in REFERENCE_RAS_MODELS


def _has_usable_mesh(meta: dict) -> bool:
    """blockMeshDict / snappy / 事前メッシュのいずれかが使えるか。"""
    return (
        _meta_bool(meta, "has_blockmesh")
        or _meta_bool(meta, "has_snappy")
        or _meta_bool(meta, "mesh_prebuilt")
    )


class CaseSelector:
    """ChromaDB から reference case を 1 件選定する。"""

    def __init__(self, db_path: str, openai_api_key: str):
        self.db_path = Path(db_path)
        self.openai = OpenAI(api_key=openai_api_key)
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            client = chromadb.PersistentClient(path=str(self.db_path))
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def is_available(self) -> bool:
        try:
            return self.db_path.exists() and self.collection.count() > 0
        except Exception:
            return False

    def select(
        self,
        spec: SimulationSpec,
        exclude_case_ids: list[str] | None = None,
        n_candidates: int = 20,
    ) -> dict | None:
        """
        reference case を選定する。

        Returns:
            dict with keys: case_id, case_path, reference_files, metadata
            or None if no case found
        """
        if not self.is_available:
            return None

        exclude = set(exclude_case_ids or [])
        query = self._build_query(spec)
        embedding = self._embed(query)

        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(n_candidates, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            console.print(f"[yellow]  ケース検索エラー: {e}[/yellow]")
            return None

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else []

        for meta, dist in zip(metas, dists):
            case_id = meta.get("case_id", "")
            if case_id in exclude:
                continue
            if not self._passes_hard_filter(spec, meta):
                continue
            if dist > 0.85:
                continue

            case_path = meta.get("case_path", "")
            if not case_path or not Path(case_path).exists():
                continue

            reference_files = load_case_files(case_path)
            if len(reference_files) < 3:
                continue

            console.print(
                f"  [green]参照ケース: {case_id}[/green] "
                f"(solver={meta.get('solver')}, phenomenon={meta.get('phenomenon', 'general')}, "
                f"dist={dist:.3f})"
            )
            return {
                "case_id": case_id,
                "case_path": case_path,
                "reference_files": reference_files,
                "metadata": meta,
                "distance": dist,
                "title_ja": meta.get("title_ja", ""),
                "summary_ja": meta.get("summary_ja", ""),
            }

        return None

    def _passes_hard_filter(self, spec: SimulationSpec, meta: dict) -> bool:
        case_solver = meta.get("solver", "")
        allowed = SOLVER_COMPAT.get(spec.solver, {spec.solver})
        if case_solver not in allowed:
            return False

        meta_steady = meta.get("steady_state")
        if isinstance(meta_steady, str):
            meta_steady = meta_steady.lower() in ("true", "1", "yes")
        if bool(meta_steady) != spec.steady_state:
            return False

        meta_dims = int(meta.get("dimensions", 3))
        if meta_dims != spec.dimensions:
            return False

        case_turb = meta.get("turbulence_model", "unknown")
        if not _turbulence_compatible(spec.turbulence_model, case_turb):
            return False

        # incompressible カテゴリを優先（圧縮性除外）
        if meta.get("category", "") == "compressible":
            return False

        if meta.get("requires_preprocessing") in (True, "True", "true", "1"):
            return False

        # meshing_demo は通常の解析選定から除外
        case_phenomenon = meta.get("phenomenon", "general") or "general"
        if case_phenomenon == "meshing_demo" and spec.phenomenon != "meshing_demo":
            return False

        if spec.phenomenon and spec.phenomenon != "general":
            if not phenomenon_matches(spec.phenomenon, case_phenomenon):
                return False

        if not _has_usable_mesh(meta):
            return False

        return True

    def _build_query(self, spec: SimulationSpec) -> str:
        parts = [
            f"OpenFOAM {spec.solver}",
            spec.case_type.replace("_", " "),
            spec.turbulence_model,
            "steady state" if spec.steady_state else "transient unsteady",
            f"{spec.dimensions}D",
            spec.description,
        ]
        if spec.phenomenon:
            parts.append(f"phenomenon {spec.phenomenon}")
        if spec.observables:
            parts.append(" ".join(spec.observables))
        if spec.re_number:
            parts.append(f"Reynolds number {spec.re_number:.0f}")
        # geometry hint from case_type / phenomenon
        if spec.phenomenon == "karman_vortex_shedding" or "cylinder" in spec.case_type or "ogrid" in spec.case_type:
            parts.append("cylinder vortex shedding external flow カルマン渦")
        if spec.phenomenon == "airfoil_steady":
            parts.append("airfoil wing steady flow 翼 揚力")
        if spec.phenomenon == "backward_facing_step":
            parts.append("backward facing step reattachment 後方ステップ")
        if spec.phenomenon == "channel_internal" or "channel" in spec.case_type:
            parts.append("channel internal flow チャンネル")
        return " ".join(filter(None, parts))

    def _embed(self, text: str) -> list[float]:
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def list_cases_matching(self, spec: SimulationSpec, limit: int = 5) -> list[dict]:
        """デバッグ用: フィルタを通過するケース一覧。"""
        if not self.is_available:
            return []
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        matched = []
        for meta in all_meta:
            if self._passes_hard_filter(spec, meta):
                matched.append(meta)
            if len(matched) >= limit:
                break
        return matched
