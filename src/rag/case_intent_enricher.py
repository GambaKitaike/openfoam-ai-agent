"""
チュートリアルケースの意図メタデータを LLM で自動生成し、ディスクキャッシュする。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.llm_client import LLMClient
from src.rag.case_catalog import CaseRecord, _find_zero_dir, _read_text
from src.rag.case_intent import PHENOMENON_TAGS, CaseIntent

console = Console()

INTENT_SYSTEM_PROMPT = """あなたはOpenFOAMチュートリアルケースの解説を書く専門家です。
XSim のような分かりやすい日本語説明を JSON で返してください。
出力は JSON のみ（コードブロック禁止）。幻覚禁止 — 入力の事実のみ使用。"""

INTENT_PROMPT_TEMPLATE = """以下の OpenFOAM チュートリアルケースについて、検証目的・観察対象・実行上の注意を JSON で返してください。

【ケース情報】
case_id: {case_id}
solver: {solver}
steady: {steady}
dimensions: {dimensions}D
geometry (機械推定): {geometry}
turbulence: {turbulence}
mesh_prebuilt: {mesh_prebuilt}
has_blockmesh_in_allrun: {has_blockmesh_in_allrun}
run_commands: {run_commands}
requires_preprocessing: {requires_preprocessing}
patch_names: {patch_names}

【controlDict 抜粋】
{control_excerpt}

【Allrun / Allrun.pre 抜粋】
{allrun_excerpt}

【0/U boundaryField 抜粋】
{u_excerpt}

【turbulenceProperties 抜粋】
{turb_excerpt}

【README 抜粋（あれば）】
{readme_excerpt}

phenomenon は次のいずれかのみ: {phenomenon_tags}

JSON 形式:
{{
  "title_ja": "1行タイトル（例: 2次元翼周りの定常流れ）",
  "summary_ja": "2〜4文の概要",
  "phenomenon": "上記 enum の1つ",
  "geometry": "cylinder_2d | airfoil_2d | backward_step | channel | cavity | building | general",
  "observables": ["velocity_U", "pressure_p", "lift_drag", ...],
  "bc_summary_ja": "境界条件の日本語要約",
  "mesh_notes_ja": "メッシュに関する注意（例: メッシュ済み blockMesh 不要）",
  "suitable_for_ja": ["ユーザーが言いそうな日本語指示 3〜5件"],
  "not_suitable_for_ja": ["向かない指示例 1〜3件"]
}}"""


class CaseIntentEnricher:
    """CaseRecord に LLM 生成の CaseIntent を付与する。"""

    def __init__(
        self,
        cache_dir: str | Path,
        llm: LLMClient | None = None,
        openai_api_key: str = "",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if llm is not None:
            self.llm = llm
        else:
            from src.config import Settings
            settings = Settings()
            if openai_api_key:
                settings.openai_api_key = openai_api_key
            self.llm = LLMClient(settings)

    def enrich_all(
        self,
        records: list[CaseRecord],
        force: bool = False,
    ) -> dict[str, int]:
        """
        全ケースに intent を付与する（キャッシュ hit 時は LLM スキップ）。

        Returns:
            stats: enriched, cached, failed
        """
        stats = {"enriched": 0, "cached": 0, "failed": 0, "skipped": 0}

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("意図メタデータ生成...", total=len(records))
            for record in records:
                try:
                    intent, from_cache = self.enrich_record(record, force=force)
                    record.intent = intent
                    record.embedding_text = record.build_embedding_text()
                    if from_cache:
                        stats["cached"] += 1
                    else:
                        stats["enriched"] += 1
                except Exception as e:
                    stats["failed"] += 1
                    console.print(f"  [yellow]intent 失敗: {record.case_id} - {e}[/yellow]")
                    record.intent = self._fallback_intent(record)
                    record.embedding_text = record.build_embedding_text()
                progress.advance(task)

        return stats

    def enrich_record(
        self,
        record: CaseRecord,
        force: bool = False,
    ) -> tuple[CaseIntent, bool]:
        """1 ケースを enrich。戻り値: (intent, from_cache)"""
        case_dir = Path(record.case_path)
        source_hash = compute_source_hash(case_dir)
        cache_path = self._cache_path(record.case_id)

        if not force and cache_path.exists():
            cached = CaseIntent.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))
            if cached.source_hash == source_hash:
                self._merge_mechanical(cached, record)
                return cached, True

        context = self._build_llm_context(record, case_dir)
        intent = self._call_llm(record, context)
        intent.source_hash = source_hash
        self._merge_mechanical(intent, record)
        intent.normalize_phenomenon()
        cache_path.write_text(
            json.dumps(intent.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return intent, False

    def _cache_path(self, case_id: str) -> Path:
        safe = case_id.replace("/", "__")
        return self.cache_dir / f"{safe}.json"

    def _merge_mechanical(self, intent: CaseIntent, record: CaseRecord) -> None:
        """機械抽出メタデータで LLM 出力を上書き補正。"""
        intent.case_id = record.case_id
        intent.mesh_prebuilt = record.mesh_prebuilt
        intent.has_blockmesh_in_allrun = record.has_blockmesh_in_allrun
        if record.run_commands:
            intent.run_commands = list(record.run_commands)

    def _fallback_intent(self, record: CaseRecord) -> CaseIntent:
        """LLM 失敗時の最小 intent。"""
        intent = CaseIntent(
            case_id=record.case_id,
            title_ja=record.case_id.split("/")[-1],
            summary_ja=f"{record.solver} による {record.geometry} 解析",
            phenomenon="general",
            geometry=record.geometry,
            mesh_prebuilt=record.mesh_prebuilt,
            has_blockmesh_in_allrun=record.has_blockmesh_in_allrun,
            run_commands=list(record.run_commands),
        )
        return intent

    def _build_llm_context(self, record: CaseRecord, case_dir: Path) -> dict:
        control = _read_text(case_dir / "system" / "controlDict")[:1500]
        allrun_parts = []
        for name in ("Allrun", "Allrun.pre"):
            p = case_dir / name
            if p.exists():
                allrun_parts.append(f"--- {name} ---\n{p.read_text(errors='ignore')[:2000]}")
        zero_dir = _find_zero_dir(case_dir)
        u_excerpt = ""
        if zero_dir:
            u = _read_text(zero_dir / "U")
            if "boundaryField" in u:
                idx = u.index("boundaryField")
                u_excerpt = u[idx:idx + 1200]
        turb = _read_text(case_dir / "constant" / "turbulenceProperties")[:800]
        return {
            "case_id": record.case_id,
            "solver": record.solver,
            "steady": record.steady_state,
            "dimensions": record.dimensions,
            "geometry": record.geometry,
            "turbulence": record.turbulence_model,
            "mesh_prebuilt": record.mesh_prebuilt,
            "has_blockmesh_in_allrun": record.has_blockmesh_in_allrun,
            "run_commands": record.run_commands,
            "requires_preprocessing": record.requires_preprocessing,
            "patch_names": record.patch_names[:12],
            "control_excerpt": control,
            "allrun_excerpt": "\n".join(allrun_parts) or "(なし)",
            "u_excerpt": u_excerpt or "(なし)",
            "turb_excerpt": turb or "(なし)",
            "readme_excerpt": record.readme_excerpt or "(なし)",
            "phenomenon_tags": ", ".join(sorted(PHENOMENON_TAGS)),
        }

    def _call_llm(self, record: CaseRecord, context: dict) -> CaseIntent:
        prompt = INTENT_PROMPT_TEMPLATE.format(**context)
        raw = self.llm.chat(prompt, system=INTENT_SYSTEM_PROMPT)
        data = _parse_json(raw)
        return CaseIntent(
            case_id=record.case_id,
            title_ja=data.get("title_ja", ""),
            summary_ja=data.get("summary_ja", ""),
            phenomenon=data.get("phenomenon", "general"),
            geometry=data.get("geometry", record.geometry),
            observables=list(data.get("observables", [])),
            bc_summary_ja=data.get("bc_summary_ja", ""),
            mesh_notes_ja=data.get("mesh_notes_ja", ""),
            suitable_for_ja=list(data.get("suitable_for_ja", [])),
            not_suitable_for_ja=list(data.get("not_suitable_for_ja", [])),
        )

    def enrich_only(
        self,
        records: list[CaseRecord],
        force: bool = False,
    ) -> dict[str, int]:
        """キャッシュ更新のみ（インデックス化なし）。"""
        return self.enrich_all(records, force=force)


def compute_source_hash(case_dir: Path) -> str:
    """主要ファイルの MD5 で変更検出。"""
    h = hashlib.md5()
    paths: list[Path] = [case_dir / "system" / "controlDict"]
    for name in ("Allrun", "Allrun.pre"):
        p = case_dir / name
        if p.exists():
            paths.append(p)
    zero_dir = _find_zero_dir(case_dir)
    if zero_dir:
        u = zero_dir / "U"
        if u.exists():
            paths.append(u)
    for p in sorted(paths, key=lambda x: str(x)):
        try:
            h.update(str(p).encode())
            h.update(p.read_bytes())
        except OSError:
            pass
    return h.hexdigest()


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'^```[^\n]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    return {}


def enrich_all_intents(
    records: list[CaseRecord],
    cache_dir: str | Path,
    openai_api_key: str,
    skip: bool = False,
    enrich_only: bool = False,
    force: bool = False,
) -> dict[str, int]:
    """build-index から呼ばれるエントリポイント。"""
    if skip:
        return {"enriched": 0, "cached": 0, "failed": 0, "skipped": len(records)}
    enricher = CaseIntentEnricher(cache_dir=cache_dir, openai_api_key=openai_api_key)
    return enricher.enrich_all(records, force=force)
