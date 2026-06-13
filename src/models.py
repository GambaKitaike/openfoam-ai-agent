"""
共有データモデル定義 - 4エージェント間のデータ契約
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.rag.reference_case_params import ReferenceCaseParams


# ─────────────────────────────────────────────────────────────────────────────
# Agent① → Agent② へ渡すデータ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationSpec:
    """前処理エージェントが自然言語から抽出した解析仕様。"""
    solver: str                          # "simpleFoam", "pimpleFoam", etc.
    case_type: str                       # "external_flow" | "internal_flow" | "heat_transfer"
    mesh_template: str                   # "box_external" | "box_internal" | "box_2d"
    turbulence_model: str                # "kOmegaSST", "kEpsilon", "laminar", etc.
    steady_state: bool
    inlet_velocity: float                # 単位除去済みの数値 [m/s]
    dimensions: int = 3                  # 2 または 3
    characteristic_length: float = 1.0  # Re数計算用の代表長さ [m]
    nu: float = 1.5e-5                   # 動粘度 [m^2/s] (デフォルト: 空気20°C)
    re_number: float | None = None       # 計算済みRe数
    description: str = ""
    phenomenon: str = ""                 # karman_vortex_shedding | airfoil_steady | ...
    observables: list[str] = field(default_factory=list)
    boundary_conditions: dict = field(default_factory=dict)
    mesh_params: dict = field(default_factory=dict)
    defaults_applied: list[str] = field(default_factory=list)  # LLMが補完した項目
    defaulted_fields: list[str] = field(default_factory=list)  # data 欠損時の決定的デフォルト
    raw_llm_output: dict = field(default_factory=dict)
    stl_path: str = ""                   # ユーザー提供STLファイルのパス（snappyHexMesh用）

    def __post_init__(self):
        if self.re_number is None and self.inlet_velocity and self.nu:
            self.re_number = (self.inlet_velocity * self.characteristic_length) / self.nu


# ─────────────────────────────────────────────────────────────────────────────
# Agent② → Agent① 要件プロファイル
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RequirementField:
    """Agent① ヒアリングで確認すべき 1 項目。"""
    key: str
    required: bool
    reason: str
    suggested: float | str | bool | None = None
    parser: str = "float"  # float | str | bool


@dataclass
class ReferenceHint:
    """類似チュートリアルから抽出した典型パラメータ（Agent② プロファイル用）。"""
    case_id: str = ""
    title_ja: str = ""
    inlet_velocity: float | None = None
    nu: float | None = None
    turbulence_model: str = ""
    solver: str = ""
    steady_state: bool | None = None
    characteristic_length: float | None = None
    re_number: float | None = None


@dataclass
class RequirementProfile:
    """現象・ケースタイプに応じた必要十分条件。"""
    phenomenon: str
    fields: list[RequirementField] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    similar_case_ids: list[str] = field(default_factory=list)
    reference_hints: list[ReferenceHint] = field(default_factory=list)


@dataclass
class SpecReviewAlternative:
    """同一矛盾に対する別の修正方針。"""
    key: str
    suggested: float | str | bool | None
    label: str


@dataclass
class SpecReviewIssue:
    """Agent② が Agent① の spec に対して指摘する 1 項目。"""
    key: str
    message: str
    suggested: float | str | bool | None = None
    severity: str = "warning"  # warning | error
    user_locked: bool = False  # 説明文でユーザーが明示した値
    alternatives: list[SpecReviewAlternative] = field(default_factory=list)


@dataclass
class ReferenceMatch:
    """参照ケース選定結果 + fast path 判定。"""
    context: "EnrichedContext"
    score: float = 0.0
    use_fast_path: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Agent③ 段階的ケース生成状態
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CaseBuildState:
    """段階的生成パイプラインの中間状態。"""
    case_dir: str
    spec: SimulationSpec
    files_created: list[str] = field(default_factory=list)
    patch_names: list[str] = field(default_factory=list)
    build_path: str = "staged"  # staged | reference_fast
    completed_steps: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent② → Agent③ へ渡すデータ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnrichedContext:
    """RAGエージェントが検索結果を付加した拡張コンテキスト。"""
    spec: SimulationSpec
    rag_available: bool = True
    rag_sources: list[str] = field(default_factory=list)

    # ケース単位 RAG（主経路）
    reference_case_id: str = ""
    reference_case_path: str = ""
    reference_files: dict[str, str] = field(default_factory=dict)  # rel_path → content
    reference_title_ja: str = ""
    reference_summary_ja: str = ""
    reference_phenomenon: str = ""
    reference_mesh_prebuilt: bool = False
    reference_typical_params: "ReferenceCaseParams | None" = None

    # フォールバック用（参照ケースが見つからない場合）
    mesh_template_name: str = ""
    mesh_params_suggestion: dict = field(default_factory=dict)

    # 後方互換（非推奨・未使用）
    relevant_examples: list[str] = field(default_factory=list)
    recommended_schemes: str = ""
    recommended_bcs: str = ""
    reference_fvschemes: str = ""
    reference_fvsolution: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Agent③ → Agent④ へ渡すデータ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """ケース生成の結果。"""
    output_path: str
    case_type: str
    files_created: list[str] = field(default_factory=list)
    mesh_built: bool = False
    build_path: str = "staged"  # staged | reference_fast


@dataclass
class CaseArtifacts:
    """OpenFOAMGPTエージェントが生成・実行した成果物。"""
    case_dir: str
    spec: SimulationSpec
    generation_result: GenerationResult
    block_mesh_success: bool = False
    block_mesh_retries: int = 0
    solver_success: bool = False
    solver_retries: int = 0
    final_residuals: dict[str, float] = field(default_factory=dict)
    converged: bool = False
    log_files: dict[str, str] = field(default_factory=dict)     # コマンド名 → ログパス
    foam_file: str = ""                  # ParaView用 .foam ファイルパス
    vtk_dir: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Agent④ が生成する最終レポート
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicsCheck:
    """物理妥当性チェックの個別項目。"""
    name: str
    passed: bool
    message: str
    value: Any = None


@dataclass
class AnalysisReport:
    """後処理エージェントが生成する解析レポート。"""
    artifacts: CaseArtifacts
    physics_checks: list[PhysicsCheck] = field(default_factory=list)
    overall_valid: bool = False
    summary_text: str = ""
    report_file: str = ""               # report.md のパス
    windows_paraview_path: str = ""     # Windows から開くための UNC パス

    @property
    def passed_checks(self) -> list[PhysicsCheck]:
        return [c for c in self.physics_checks if c.passed]

    @property
    def failed_checks(self) -> list[PhysicsCheck]:
        return [c for c in self.physics_checks if not c.passed]
