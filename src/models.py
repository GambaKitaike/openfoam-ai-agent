"""
共有データモデル定義 - 4エージェント間のデータ契約
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    boundary_conditions: dict = field(default_factory=dict)
    mesh_params: dict = field(default_factory=dict)
    defaults_applied: list[str] = field(default_factory=list)  # LLMが補完した項目
    raw_llm_output: dict = field(default_factory=dict)
    stl_path: str = ""                   # ユーザー提供STLファイルのパス（snappyHexMesh用）

    def __post_init__(self):
        if self.re_number is None and self.inlet_velocity and self.nu:
            self.re_number = (self.inlet_velocity * self.characteristic_length) / self.nu


# ─────────────────────────────────────────────────────────────────────────────
# Agent② → Agent③ へ渡すデータ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnrichedContext:
    """RAGエージェントが検索結果を付加した拡張コンテキスト。"""
    spec: SimulationSpec
    relevant_examples: list[str] = field(default_factory=list)   # 検索取得ファイル断片
    recommended_schemes: str = ""        # 類似ケースから推薦される数値スキーム
    recommended_bcs: str = ""            # 推薦される境界条件の記述例
    mesh_template_name: str = ""         # 選択すべきテンプレート名
    mesh_params_suggestion: dict = field(default_factory=dict)   # LLMへのメッシュパラメータ提案
    rag_sources: list[str] = field(default_factory=list)         # 参照元（デバッグ用）
    rag_available: bool = True           # RAGインデックスが存在するか
    # チュートリアルから直接取得したシステムファイル（空文字ならテンプレートにフォールバック）
    reference_fvschemes: str = ""        # RAGで見つかったチュートリアルの fvSchemes
    reference_fvsolution: str = ""       # RAGで見つかったチュートリアルの fvSolution


# ─────────────────────────────────────────────────────────────────────────────
# Agent③ → Agent④ へ渡すデータ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """ケース生成の結果。"""
    output_path: str
    case_type: str
    files_created: list[str] = field(default_factory=list)


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
