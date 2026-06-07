"""
LLMクライアント - OpenAI / Anthropic への統一インターフェース
"""
from __future__ import annotations

import json
from typing import Any

from .config import Settings


SYSTEM_PROMPT = """あなたはOpenFOAMの専門家AIエージェントです。
ユーザーの解析要件を理解し、適切なOpenFOAMケース設定を生成します。
以下の点に注意してください：
- 物理的に妥当な境界条件を設定する
- 解析の目的に合った数値スキームを選択する
- 乱流モデルは解析条件（Re数など）に応じて適切に選択する
- 単位系はSI単位系（m, kg, s, K, mol, A, cd）を使用する
"""


class AnalysisSpec:
    """解析仕様を格納するデータクラス。"""

    def __init__(self, data: dict[str, Any]):
        self.solver: str = data.get("solver", "simpleFoam")
        self.case_type: str = data.get("case_type", "external_flow")
        self.dimensions: int = data.get("dimensions", 3)
        self.turbulence_model: str = data.get("turbulence_model", "kOmegaSST")
        self.steady_state: bool = data.get("steady_state", True)
        self.description: str = data.get("description", "")
        self.boundary_conditions: dict = data.get("boundary_conditions", {})
        self.mesh_params: dict = data.get("mesh_params", {})
        self.raw: dict = data


class LLMClient:
    """OpenAI / Anthropic への統一LLMクライアント。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = self._init_client()

    def _init_client(self):
        if self.settings.llm_provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=self.settings.openai_api_key)
        elif self.settings.llm_provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=self.settings.anthropic_api_key)
        else:
            raise ValueError(f"未対応のLLMプロバイダー: {self.settings.llm_provider}")

    def parse_analysis_spec(self, description: str) -> AnalysisSpec:
        """
        自然言語の解析説明をOpenFOAM設定仕様に変換する。
        """
        prompt = f"""以下の解析要件を解析し、OpenFOAMの設定仕様をJSON形式で返してください。

解析要件: {description}

以下のJSON形式で回答してください（コードブロックなし、純粋なJSONのみ）:
{{
  "solver": "使用するソルバー名 (例: simpleFoam, pimpleFoam, rhoCentralFoam)",
  "case_type": "解析タイプ (例: external_flow, internal_flow, heat_transfer)",
  "dimensions": 2または3,
  "turbulence_model": "乱流モデル (例: kOmegaSST, kEpsilon, Spalart-Allmaras, laminar)",
  "steady_state": true または false,
  "description": "解析の簡潔な説明",
  "boundary_conditions": {{
    "inlet": {{"type": "入口条件の説明", "velocity": "速度 (m/s)"}},
    "outlet": {{"type": "出口条件の説明"}},
    "wall": {{"type": "壁面条件の説明"}}
  }},
  "mesh_params": {{
    "domain_size": "計算領域のサイズ",
    "refinement_level": 1から5の整数
  }}
}}"""

        response_text = self._chat(prompt)

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # JSONの抽出を試みる
            import re
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = {"solver": "simpleFoam", "description": description}

        return AnalysisSpec(data)

    def generate_block_mesh_dict(self, spec: "AnalysisSpec") -> str:
        """
        解析仕様に基づいてblockMeshDictを生成する。
        blockMeshDictはケース形状に強く依存するためLLMで動的に生成する。
        """
        prompt = f"""以下のOpenFOAM解析仕様に対応するblockMeshDictを生成してください。

解析仕様:
- ソルバー: {spec.solver}
- 解析タイプ: {spec.case_type}
- 次元: {spec.dimensions}D
- 説明: {spec.description}
- 境界条件: {spec.boundary_conditions}
- メッシュパラメータ: {spec.mesh_params}

要件:
- OpenFOAM v2512 形式で記述すること
- FoamFile ヘッダーを含めること
- vertices、blocks、edges、boundary、mergePatchPairs を含めること
- boundary のパッチ名は境界条件の inlet/outlet/wall に対応させること
- 2D解析の場合は front/back を empty に設定すること
- セル数は解析精度と計算コストのバランスを考慮すること（最初は粗めでよい）
- コードブロック記号(```)は絶対に含めないこと
- blockMeshDictの内容のみを出力し、説明文やコメントは不要

blockMeshDict の内容のみ出力してください:"""

        raw = self._chat(prompt)
        return self._strip_code_fences(raw)

    def fix_block_mesh_error(self, block_mesh_dict: str, error_message: str) -> str:
        """
        blockMesh 実行時のエラーを修正した blockMeshDict を返す。
        """
        prompt = f"""以下のblockMeshDictでblockMeshを実行したところエラーが発生しました。
エラーを修正したblockMeshDictを出力してください（説明不要、コードブロック記号(```)は絶対に含めないこと）。

エラーメッセージ:
{error_message}

現在のblockMeshDict:
{block_mesh_dict}"""

        raw = self._chat(prompt)
        return self._strip_code_fences(raw)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """LLM がマークダウンコードブロック(```)で囲んだ場合に除去する。"""
        import re
        # ```openfoam ... ``` や ``` ... ``` を除去
        text = re.sub(r'^```[^\n]*\n', '', text.strip())
        text = re.sub(r'\n```$', '', text.strip())
        text = text.strip('`').strip()
        return text

    def review_case(self, case_files: dict[str, str]) -> str:
        """OpenFOAMケースのファイルをレビューする。"""
        files_text = "\n\n".join(
            f"=== {path} ===\n{content}"
            for path, content in case_files.items()
        )
        prompt = f"""以下のOpenFOAMケースファイルをレビューしてください。
問題点、警告、改善案を日本語で具体的に指摘してください。

{files_text}"""
        return self._chat(prompt)

    def _chat(self, user_message: str) -> str:
        """LLMにメッセージを送信して応答を得る（デフォルトシステムプロンプト使用）。"""
        return self.chat(user_message, system=SYSTEM_PROMPT)

    def chat(self, user_message: str, system: str | None = None) -> str:
        """
        LLMにメッセージを送信して応答を得る汎用メソッド。
        各エージェントが独自のシステムプロンプトで呼び出せる。
        """
        system_prompt = system or SYSTEM_PROMPT
        if self.settings.llm_provider == "openai":
            response = self._client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
            )
            return response.choices[0].message.content

        elif self.settings.llm_provider == "anthropic":
            response = self._client.messages.create(
                model=self.settings.llm_model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                temperature=0,
            )
            return response.content[0].text
        return ""
