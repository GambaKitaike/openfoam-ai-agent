"""
チュートリアルケースの「検証目的」メタデータ（XSim 風）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

# LLM が選択する phenomenon タグ（enum 固定）
PHENOMENON_TAGS = frozenset({
    "karman_vortex_shedding",
    "airfoil_steady",
    "channel_internal",
    "backward_facing_step",
    "cavity_flow",
    "external_building",
    "meshing_demo",
    "general",
})

# ユーザー phenomenon → 互換チュートリアル phenomenon
PHENOMENON_COMPAT: dict[str, set[str]] = {
    "karman_vortex_shedding": {"karman_vortex_shedding"},
    "airfoil_steady": {"airfoil_steady", "general"},
    "channel_internal": {"channel_internal", "cavity_flow", "general"},
    "backward_facing_step": {"backward_facing_step", "general"},
    "cavity_flow": {"cavity_flow", "channel_internal", "general"},
    "external_building": {"external_building", "general"},
    "meshing_demo": {"meshing_demo"},
    "general": {"general", "karman_vortex_shedding", "airfoil_steady", "channel_internal",
                "backward_facing_step", "cavity_flow", "external_building"},
}


@dataclass
class CaseIntent:
    """1 チュートリアルケースの意図メタデータ（LLM 生成 + 機械的事実）。"""
    case_id: str
    title_ja: str = ""
    summary_ja: str = ""
    phenomenon: str = "general"
    geometry: str = "general"
    observables: list[str] = field(default_factory=list)
    bc_summary_ja: str = ""
    mesh_notes_ja: str = ""
    mesh_prebuilt: bool = False
    has_blockmesh_in_allrun: bool = False
    run_commands: list[str] = field(default_factory=list)
    suitable_for_ja: list[str] = field(default_factory=list)
    not_suitable_for_ja: list[str] = field(default_factory=list)
    source_hash: str = ""

    def normalize_phenomenon(self) -> None:
        if self.phenomenon not in PHENOMENON_TAGS:
            self.phenomenon = "general"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseIntent:
        intent = cls(
            case_id=data.get("case_id", ""),
            title_ja=data.get("title_ja", ""),
            summary_ja=data.get("summary_ja", ""),
            phenomenon=data.get("phenomenon", "general"),
            geometry=data.get("geometry", "general"),
            observables=list(data.get("observables", [])),
            bc_summary_ja=data.get("bc_summary_ja", ""),
            mesh_notes_ja=data.get("mesh_notes_ja", ""),
            mesh_prebuilt=bool(data.get("mesh_prebuilt", False)),
            has_blockmesh_in_allrun=bool(data.get("has_blockmesh_in_allrun", False)),
            run_commands=list(data.get("run_commands", [])),
            suitable_for_ja=list(data.get("suitable_for_ja", [])),
            not_suitable_for_ja=list(data.get("not_suitable_for_ja", [])),
            source_hash=data.get("source_hash", ""),
        )
        intent.normalize_phenomenon()
        return intent

    def to_metadata(self) -> dict[str, Any]:
        """ChromaDB 用メタデータ（スカラー + JSON 文字列）。"""
        return {
            "phenomenon": self.phenomenon,
            "title_ja": self.title_ja[:500],
            "summary_ja": self.summary_ja[:1000],
            "intent_geometry": self.geometry,
            "mesh_prebuilt": self.mesh_prebuilt,
            "suitable_for_ja": json.dumps(self.suitable_for_ja, ensure_ascii=False)[:2000],
            "not_suitable_for_ja": json.dumps(self.not_suitable_for_ja, ensure_ascii=False)[:1000],
            "observables": json.dumps(self.observables, ensure_ascii=False)[:500],
        }

    def embedding_snippet(self) -> str:
        """ベクトル検索用テキスト断片。"""
        parts = [
            self.title_ja,
            self.summary_ja,
            f"phenomenon {self.phenomenon}",
            f"geometry {self.geometry}",
            " ".join(self.suitable_for_ja),
            " ".join(self.observables),
            self.bc_summary_ja,
            self.mesh_notes_ja,
        ]
        return " ".join(p for p in parts if p)


def phenomenon_matches(spec_phenomenon: str, case_phenomenon: str) -> bool:
    """ユーザー意図とケース phenomenon の互換性判定。"""
    if not spec_phenomenon or spec_phenomenon == "general":
        return True
    allowed = PHENOMENON_COMPAT.get(spec_phenomenon, {spec_phenomenon})
    return case_phenomenon in allowed
