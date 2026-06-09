"""参照ケースと SimulationSpec の一致度スコア。"""
from __future__ import annotations

from ..models import SimulationSpec

FAST_PATH_THRESHOLD = 0.8


def compute_match_score(spec: SimulationSpec, meta: dict) -> float:
    """0..1 の一致度。fast path 判定に使用。"""
    score = 0.0
    weights = 0.0

    def add(w: float, ok: bool) -> None:
        nonlocal score, weights
        weights += w
        if ok:
            score += w

    case_solver = meta.get("solver", "")
    add(0.25, case_solver == spec.solver or not case_solver)

    case_phen = meta.get("phenomenon", "general") or "general"
    if spec.phenomenon:
        add(0.2, case_phen == spec.phenomenon)
    else:
        add(0.2, True)

    add(0.15, int(meta.get("dimensions", 3)) == spec.dimensions)

    meta_steady = meta.get("steady_state") in (True, "True", "true", "1")
    add(0.1, bool(meta_steady) == spec.steady_state)

    case_turb = meta.get("turbulence_model", "unknown")
    if spec.turbulence_model == "laminar":
        add(0.1, case_turb in ("laminar", "unknown"))
    else:
        add(0.1, case_turb in (spec.turbulence_model, "RAS", "unknown"))

    add(0.1, meta.get("category", "") != "compressible")
    add(0.1, not _meta_bool(meta, "requires_preprocessing"))

    if weights <= 0:
        return 0.0
    return round(score / weights, 3)


def should_use_fast_path(spec: SimulationSpec, meta: dict, score: float) -> bool:
    if score < FAST_PATH_THRESHOLD:
        return False
    if _meta_bool(meta, "mesh_prebuilt"):
        return True
    return _meta_bool(meta, "has_blockmesh") or _meta_bool(meta, "has_snappy")


def _meta_bool(meta: dict, key: str) -> bool:
    v = meta.get(key)
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)
