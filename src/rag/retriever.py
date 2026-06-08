"""
RAG リトリーバー — ケース単位

SimulationSpec から参照チュートリアルケースを 1 件選び EnrichedContext を構築する。
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from ..models import SimulationSpec, EnrichedContext
from .case_selector import CaseSelector
from .reference_case_params import extract_reference_params

console = Console()


class OpenFOAMRetriever:
    """ChromaDB から参照ケースを選定するクラス。"""

    def __init__(self, db_path: str, openai_api_key: str):
        self.db_path = Path(db_path)
        self.selector = CaseSelector(str(db_path), openai_api_key)

    @property
    def is_available(self) -> bool:
        return self.selector.is_available

    @property
    def collection(self):
        """後方互換: Agent② の件数表示用。"""
        return self.selector.collection

    def retrieve(
        self,
        spec: SimulationSpec,
        exclude_case_ids: list[str] | None = None,
    ) -> EnrichedContext:
        """SimulationSpec に基づいて参照ケースを選定し EnrichedContext を返す。"""
        if not self.is_available:
            console.print(
                "[yellow]  RAGインデックスが未構築です。"
                "python -m src.main build-index を実行してください。[/yellow]"
            )
            return self._fallback_context(spec)

        query_preview = self.selector._build_query(spec)
        console.print(f"  [dim]ケース検索: {query_preview[:80]}...[/dim]")

        selected = self.selector.select(spec, exclude_case_ids=exclude_case_ids)
        if selected:
            meta = selected["metadata"]
            # 参照ケースの solver を spec に同期（整合性確保）
            case_solver = meta.get("solver", spec.solver)
            if case_solver:
                spec.solver = case_solver
            mesh_prebuilt = meta.get("mesh_prebuilt") in (True, "True", "true", "1")
            if mesh_prebuilt:
                case_turb = meta.get("turbulence_model", "")
                if case_turb and case_turb not in ("unknown", "RAS"):
                    spec.turbulence_model = case_turb
            typical = extract_reference_params(
                case_id=selected["case_id"],
                case_path=selected["case_path"],
                reference_files=selected["reference_files"],
                title_ja=selected.get("title_ja", meta.get("title_ja", "")),
                summary_ja=selected.get("summary_ja", meta.get("summary_ja", "")),
                metadata=meta,
            )
            return EnrichedContext(
                spec=spec,
                rag_available=True,
                rag_sources=[selected["case_path"]],
                reference_case_id=selected["case_id"],
                reference_case_path=selected["case_path"],
                reference_files=selected["reference_files"],
                reference_title_ja=selected.get("title_ja", meta.get("title_ja", "")),
                reference_summary_ja=selected.get("summary_ja", meta.get("summary_ja", "")),
                reference_phenomenon=meta.get("phenomenon", "general"),
                reference_mesh_prebuilt=mesh_prebuilt,
                reference_typical_params=typical,
                mesh_template_name=self._mesh_from_case(meta),
                mesh_params_suggestion=self._suggest_mesh_params(spec),
            )

        console.print("[yellow]  条件に合う参照ケースが見つかりません。テンプレートにフォールバック。[/yellow]")
        return self._fallback_context(spec)

    def _mesh_from_case(self, meta: dict) -> str:
        if meta.get("has_snappy"):
            return "box_snappy"
        if int(meta.get("dimensions", 3)) == 2:
            return "box_channel_2d"
        return "box_channel_3d"

    def _suggest_mesh_params(self, spec: SimulationSpec) -> dict:
        re = spec.re_number or 10000
        if re < 1000:
            nx, ny, nz = 40, 20, 1
        elif re < 50000:
            nx, ny, nz = 80, 40, 1
        else:
            nx, ny, nz = 120, 60, 1
        if spec.dimensions == 2:
            nz = 1
        return {"nx": nx, "ny": ny, "nz": nz, "lx": 20.0, "ly": 10.0, "lz": 0.01}

    def _fallback_context(self, spec: SimulationSpec) -> EnrichedContext:
        mesh = "box_channel_2d" if spec.dimensions == 2 else "box_channel_3d"
        if spec.case_type in ("snappy_2d", "snappy_external", "external_snappy"):
            mesh = "box_snappy_2d" if spec.dimensions == 2 else "box_snappy"
        if spec.case_type == "cylinder_2d_ogrid":
            mesh = "ogrid_cylinder_2d"
        return EnrichedContext(
            spec=spec,
            rag_available=False,
            rag_sources=[],
            mesh_template_name=mesh,
            mesh_params_suggestion=self._suggest_mesh_params(spec),
        )

    # 後方互換
    def _fallback_context_legacy(self, spec: SimulationSpec) -> EnrichedContext:
        return self._fallback_context(spec)
