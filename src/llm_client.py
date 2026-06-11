"""
LLMクライアント - OpenAI / Anthropic への統一インターフェース
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import Settings


@dataclass
class ChatResponse:
    """ツール呼び出し対応チャットの応答。"""

    text: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


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

    def chat_with_tools(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatResponse:
        """ツール呼び出し対応のチャット。既存の chat() インターフェースは変更しない。"""
        if self.settings.llm_provider == "openai":
            return self._chat_with_tools_openai(system, messages, tools)
        if self.settings.llm_provider == "anthropic":
            return self._chat_with_tools_anthropic(system, messages, tools)
        raise ValueError(f"未対応のLLMプロバイダー: {self.settings.llm_provider}")

    def _chat_with_tools_openai(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatResponse:
        response = self._client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{"role": "system", "content": system}, *messages],
            tools=tools,
            temperature=0,
        )
        message = response.choices[0].message
        tool_calls: list[dict[str, Any]] = []
        if message.tool_calls:
            for call in message.tool_calls:
                tool_calls.append(
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                )
        return ChatResponse(text=message.content, tool_calls=tool_calls)

    @staticmethod
    def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            function = tool.get("function", {})
            converted.append(
                {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return converted

    @staticmethod
    def _openai_messages_to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            nonlocal pending_tool_results
            if pending_tool_results:
                converted.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []

        for message in messages:
            role = message.get("role")
            if role == "tool":
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message.get("content", ""),
                    }
                )
                continue

            flush_tool_results()

            if role == "assistant" and message.get("tool_calls"):
                content_blocks: list[dict[str, Any]] = []
                text = message.get("content")
                if text:
                    content_blocks.append({"type": "text", "text": text})
                for call in message["tool_calls"]:
                    function = call.get("function", {})
                    raw_args = function.get("arguments", "{}")
                    if isinstance(raw_args, dict):
                        tool_input = raw_args
                    else:
                        tool_input = json.loads(raw_args or "{}")
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": function.get("name", ""),
                            "input": tool_input,
                        }
                    )
                converted.append({"role": "assistant", "content": content_blocks})
                continue

            converted.append(
                {
                    "role": role,
                    "content": message.get("content", ""),
                }
            )

        flush_tool_results()
        return converted

    def _chat_with_tools_anthropic(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatResponse:
        anthropic_messages = self._openai_messages_to_anthropic(messages)
        response = self._client.messages.create(
            model=self.settings.llm_model,
            max_tokens=4096,
            system=system,
            messages=anthropic_messages,
            tools=self._openai_tools_to_anthropic(tools),
            temperature=0,
        )

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input, ensure_ascii=False),
                        },
                    }
                )

        text = "\n".join(text_parts) if text_parts else None
        return ChatResponse(text=text, tool_calls=tool_calls)
