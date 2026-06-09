"""
Agent① Pre-processing Agent
自然言語の入力を構造化された SimulationSpec に変換する
曖昧なパラメータには物理的に妥当なデフォルト値を補完する
"""
from __future__ import annotations

import json
import re

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config import Settings
from ..llm_client import LLMClient
from ..models import SimulationSpec
from .spec_clarification import clarify_spec, hearing_loop_with_agent2

console = Console()

PREPROCESSING_SYSTEM_PROMPT = """あなたはOpenFOAMの前処理専門エキスパートです。
ユーザーの自然言語入力から解析仕様を正確に抽出してください。

抽出ルール:
- 物理量は必ずSI単位系（m, kg, s）の数値のみで返す（単位文字列は含めない）
- 明示されていないパラメータはOpenFOAM標準的なデフォルト値を設定し、defaults_applied に記録する
- solver は解析タイプと定常/非定常に基づいて最適なものを選択する
- mesh_template は case_type と dimensions から決定する
- phenomenon はユーザーが言及した物理現象を正規化タグで返す

【phenomenon タグ — 必ず以下のいずれか1つ】
- "karman_vortex_shedding": カルマン渦、円柱周り非定常、渦の放出
- "airfoil_steady": 翼、翼型、揚力、定常翼解析
- "channel_internal": チャンネル、パイプ、ダクト、内部流れ
- "backward_facing_step": 後方ステップ、再付着、バックステップ
- "cavity_flow": キャビティ、駆動流、リド駆動キャビティ
- "external_building": 建物周り、風環境、都市風
- "meshing_demo": メッシュ生成デモのみ（通常は選ばない）
- "general": 上記に当てはまらない

【observables — ユーザーが観察したい量】
- "velocity_U", "pressure_p", "lift_drag", "vorticity", "streamlines" 等

【ソルバー選択ルール】
- 定常流れ（"定常","steady","static"等）              → simpleFoam, steady_state: true
- 非定常・過渡流れ（"非定常","transient","unsteady",
  "時間変化","カルマン渦","渦の放出","振動","脈動"等） → pimpleFoam, steady_state: false
  ただし "層流","laminar" かつ Re < 2000 の場合      → icoFoam, steady_state: false （最も安定）
  ★ 例外: case_type が cylinder_2d_ogrid または phenomenon が karman_vortex_shedding
    → 必ず pimpleFoam（O-グリッド + icoFoam は fvSolution/phi で失敗しやすい）
- 圧縮性流れ（マッハ数 > 0.3 or "圧縮性"）           → rhoCentralFoam, steady_state: false
- solver が明示されていない場合: Re < 500 かつ定常 → simpleFoam、それ以外で物体周り → pimpleFoam

【デフォルト流入速度ルール — Reが高くなりすぎないよう注意】
- 室内・建物内部流れ           : 0.5〜2.0 m/s（デフォルト 1.0）
- 管路・ダクト内部流れ         : 1.0〜5.0 m/s（デフォルト 2.0）
- 建物・構造物周り外部流れ     : 5.0〜15.0 m/s（デフォルト 5.0）
- 自動車・航空機周り           : 10〜50 m/s（デフォルト 20.0）
- 円柱・基礎形状テスト         : 1.0 m/s（定常解析ならこれで Re を低く保つ）
★ 定常(simpleFoam)で Re > 10,000 になる場合は inlet_velocity を下げて Re ≈ 1,000〜5,000 に調整

【case_type 選択ルール — 収束に直結するため重要】
- "channel_2d"      : 2Dチャンネル・パイプ断面・バックステップ等、壁あり2D解析（pitzDailyスタイル）
                      noSlip壁が存在 → SIMPLEC収束安定。最もよく使う検証ケース
- "channel_3d"      : 3Dダクト・管路・内部流れ、壁あり3D解析
                      noSlip壁が存在 → SIMPLEC収束安定
- "cylinder_2d_ogrid": 2D円柱周り外部流れ（最推奨）STL不要、blockMesh O-グリッド使用
                      スキューなし完全六面体メッシュ → pimpleFoam で安定なカルマン渦観察
                      ★ 「2D円柱」「カルマン渦」「Re=100〜2000」等を言及したらこれを選択
- "snappy_2d"       : 2D外部流れ（円柱以外の翼断面等）でSTLありのケース
                      dimensions=2、z方向1セル・empty境界、snappyHexMesh使用
- "external_snappy" : 3D外部流れ（飛行機・車・3D物体）でSTLありのケース
                      snappyHexMesh向け設定
- "heat_transfer"   : 熱伝導・強制対流・浮力流れを含む解析

【SIMPLE収束の鉄則】
- blockMesh単独の場合、必ずnoSlip壁パッチが1つ以上必要（symmetryPlane/empty のみでは収束しない）
- 壁あり → SIMPLEC (consistent yes) + linearUpwind → 50〜300stepで収束
- 壁なし純外部流れ → STLを使ったsnappyHexMeshで物体面をwall化するべき
"""

EXTRACT_PROMPT_TEMPLATE = """以下の解析要件を解析し、OpenFOAMの設定仕様をJSON形式で返してください。

解析要件: {description}

以下のJSON形式で回答してください（コードブロックなし、純粋なJSONのみ）:
{{
  "solver": "使用するソルバー名 (simpleFoam/pimpleFoam/icoFoam/rhoPimpleFoam等)",
  "case_type": "channel_2d または channel_3d または cylinder_2d_ogrid または snappy_2d または external_snappy または heat_transfer",
  "dimensions": 2 または 3,
  "turbulence_model": "kOmegaSST または kEpsilon または SpalartAllmaras または laminar",
  "steady_state": true または false,
  "description": "解析の簡潔な説明（日本語可）",
  "phenomenon": "karman_vortex_shedding | airfoil_steady | channel_internal | backward_facing_step | cavity_flow | external_building | meshing_demo | general",
  "observables": ["velocity_U", "pressure_p"],
  "inlet_velocity": 数値のみ（単位なし、例: 10.0）,
  "characteristic_length": 代表長さ[m]（数値のみ）,
  "nu": 動粘度[m^2/s]（数値のみ、空気=1.5e-5、水=1e-6）,
  "boundary_conditions": {{
    "inlet": {{"velocity": 数値のみ}},
    "outlet": {{"type": "zeroGradient"}},
    "wall": {{"type": "noSlip"}}
  }},
  "mesh_params": {{
    "domain_description": "計算領域の説明",
    "refinement_level": 1から5の整数
  }},
  "defaults_applied": ["補完したパラメータ名のリスト"]
}}\n\n【重要】円柱・翼断面等の2D外部流れ → snappy_2d。3D物体 → external_snappy。"""


class PreprocessingAgent:
    """Agent①: 自然言語 → SimulationSpec 変換エージェント。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMClient(settings)

    def extract(self, description: str, stl_path: str = "") -> SimulationSpec:
        """自然言語から draft SimulationSpec を抽出（ヒアリング前）。"""
        prompt = EXTRACT_PROMPT_TEMPLATE.format(description=description)
        raw = self.llm.chat(prompt, system=PREPROCESSING_SYSTEM_PROMPT)
        data = self._parse_json(raw)
        spec = self._build_spec(data, description)
        if stl_path:
            self._apply_stl_path(spec, stl_path)
        return spec

    def complete_hearing(
        self,
        spec: SimulationSpec,
        agent2,
        description: str = "",
        interactive: bool = True,
        trace=None,
    ) -> SimulationSpec:
        """Agent② と内部ループし spec を完成・レビューする。"""
        spec = hearing_loop_with_agent2(
            spec, agent2, description, interactive=interactive, trace=trace
        )
        self._print_spec_summary(spec)
        return spec

    def run(
        self,
        description: str,
        stl_path: str = "",
        interactive: bool = True,
    ) -> SimulationSpec:
        """
        自然言語の解析説明を SimulationSpec に変換する。

        Args:
            description: ユーザーの自然言語入力
            stl_path: STLファイルパス（指定時はsnappyHexMeshフローに切り替え）

        Returns:
            SimulationSpec
        """
        spec = self.extract(description, stl_path=stl_path)
        spec = clarify_spec(spec, description, interactive=interactive)
        self._print_spec_summary(spec)
        return spec

    def _apply_stl_path(self, spec: SimulationSpec, stl_path: str) -> None:
        """STL が指定された場合は snappyHexMesh フローに上書き。"""
        from pathlib import Path as _Path
        if not _Path(stl_path).exists():
            console.print(f"[red]警告: STLファイルが見つかりません: {stl_path}[/red]")
            return
        spec.stl_path = stl_path
        if spec.case_type != "snappy_2d":
            is_2d_input = spec.dimensions == 2
            spec.case_type = "snappy_2d" if is_2d_input else "snappy_external"
        spec.mesh_template = "box_snappy_2d" if spec.case_type == "snappy_2d" else "box_snappy"
        console.print(
            f"  [green]STL検出: {_Path(stl_path).name} → "
            f"{'2D ' if spec.case_type == 'snappy_2d' else ''}snappyHexMesh モードで実行[/green]"
        )

    def _parse_json(self, text: str) -> dict:
        """LLM 出力から JSON を抽出する。"""
        text = text.strip()
        # コードブロック除去
        text = re.sub(r'^```[^\n]*\n', '', text)
        text = re.sub(r'\n```$', '', text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    def _build_spec(self, data: dict, original_description: str) -> SimulationSpec:
        """dict から SimulationSpec を構築し、欠損値を補完する。"""
        defaults_applied = data.get("defaults_applied", [])

        # 速度の数値化（単位が混入していても除去）
        raw_velocity = data.get("inlet_velocity", data.get("boundary_conditions", {}).get("inlet", {}).get("velocity", 10.0))
        inlet_velocity = self._to_float(raw_velocity, 10.0)

        # mesh_template の決定（case_type と dimensions から自動選択）
        dimensions = int(data.get("dimensions", 3))
        raw_case_type = data.get("case_type", "channel_2d")

        # 旧 case_type との後方互換
        _LEGACY_MAP = {
            "external_flow": "external_snappy",
            "internal_flow": "channel_3d",
            "2d_flow":       "channel_2d",
        }
        case_type = _LEGACY_MAP.get(raw_case_type, raw_case_type)

        # mesh_template: 壁あり系は channel, 外部流れ系は box_snappy
        if case_type == "cylinder_2d_ogrid":
            # O-グリッドは Python ジェネレータで生成するため template 名は "ogrid_cylinder_2d"
            mesh_template = "ogrid_cylinder_2d"
        elif case_type == "snappy_2d":
            mesh_template = "box_snappy_2d"
        elif case_type == "channel_2d" or dimensions == 2:
            mesh_template = "box_channel_2d"
        elif case_type == "channel_3d":
            mesh_template = "box_channel_3d"
        elif case_type in ("external_snappy", "snappy_external"):
            mesh_template = "box_snappy"
        elif case_type == "heat_transfer":
            mesh_template = "box_channel_3d"
        else:
            # 不明な case_type は安全のため壁あり 2D へフォールバック
            case_type = "channel_2d"
            mesh_template = "box_channel_2d"

        spec = SimulationSpec(
            solver=data.get("solver", "simpleFoam"),
            case_type=case_type,
            mesh_template=mesh_template,
            turbulence_model=data.get("turbulence_model", "kOmegaSST"),
            steady_state=bool(data.get("steady_state", True)),
            inlet_velocity=inlet_velocity,
            dimensions=dimensions,
            characteristic_length=self._to_float(data.get("characteristic_length"), 1.0),
            nu=self._to_float(data.get("nu"), 1.5e-5),
            description=data.get("description", original_description),
            phenomenon=self._normalize_phenomenon(data.get("phenomenon", ""), original_description),
            observables=list(data.get("observables", [])),
            boundary_conditions=data.get("boundary_conditions", {}),
            mesh_params=data.get("mesh_params", {}),
            defaults_applied=defaults_applied,
            raw_llm_output=data,
        )

        # O-グリッド円柱カルマン渦は pimpleFoam 固定（icoFoam + O-grid はテンプレ不整合）
        if case_type == "cylinder_2d_ogrid" or spec.phenomenon == "karman_vortex_shedding":
            if spec.solver == "icoFoam":
                spec.solver = "pimpleFoam"
                spec.steady_state = False

        return spec

    @staticmethod
    def _normalize_phenomenon(raw: str, description: str) -> str:
        """LLM 出力 + キーワードフォールバックで phenomenon を正規化。"""
        valid = {
            "karman_vortex_shedding", "airfoil_steady", "channel_internal",
            "backward_facing_step", "cavity_flow", "external_building",
            "meshing_demo", "general",
        }
        if raw in valid:
            return raw
        hay = f"{raw} {description}".lower()
        if any(k in hay for k in ("カルマン", "karman", "渦", "vortex", "円柱")):
            return "karman_vortex_shedding"
        if any(k in hay for k in ("翼", "airfoil", "foil", "揚力", "naca")):
            return "airfoil_steady"
        if any(k in hay for k in ("後方ステップ", "backward", "バックステップ", "再付着")):
            return "backward_facing_step"
        if any(k in hay for k in ("キャビティ", "cavity", "lid")):
            return "cavity_flow"
        if any(k in hay for k in ("建物", "building", "風環境")):
            return "external_building"
        if any(k in hay for k in ("チャンネル", "channel", "パイプ", "pipe", "ダクト")):
            return "channel_internal"
        return "general"

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        """様々な形式の値を float に変換する。"""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r'[\d.eE+\-]+', str(value))
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return default

    def _print_spec_summary(self, spec: SimulationSpec):
        """解析仕様のサマリーを表示する。"""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("項目", style="dim")
        table.add_column("値", style="bold white")

        table.add_row("ソルバー", spec.solver)
        table.add_row("解析タイプ", spec.case_type)
        table.add_row("メッシュテンプレート", spec.mesh_template)
        table.add_row("乱流モデル", spec.turbulence_model)
        table.add_row("定常/非定常", "定常" if spec.steady_state else "非定常")
        if spec.phenomenon:
            table.add_row("現象タグ", spec.phenomenon)
        table.add_row("流入速度", f"{spec.inlet_velocity} m/s")
        if spec.re_number:
            table.add_row("Re数", f"{spec.re_number:,.0f}")
        if spec.characteristic_length:
            table.add_row("代表長さ", f"{spec.characteristic_length} m")
        if spec.nu:
            table.add_row("動粘度 nu", f"{spec.nu:g} m²/s")
        if spec.defaults_applied:
            table.add_row("自動補完", ", ".join(spec.defaults_applied))

        console.print(Panel(table, title="[bold cyan]解析仕様[/bold cyan]", border_style="cyan"))
