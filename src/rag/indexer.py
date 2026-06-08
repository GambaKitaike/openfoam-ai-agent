"""
RAG インデクサー — ケース単位

OpenFOAM チュートリアルを 1 ケース = 1 ドキュメントとして ChromaDB に保存する。
"""
from __future__ import annotations

import hashlib

import chromadb
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .case_catalog import TUTORIALS_ROOT, CaseRecord, discover_cases
from .case_intent_enricher import CaseIntentEnricher, enrich_all_intents

console = Console()

COLLECTION_NAME = "openfoam_cases"


class OpenFOAMIndexer:
    """OpenFOAM チュートリアルケースをケース単位でインデックス化する。"""

    def __init__(self, db_path: str, openai_api_key: str):
        from pathlib import Path
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.openai_api_key = openai_api_key
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.openai = OpenAI(api_key=openai_api_key)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def build(
        self,
        include_web: bool = False,
        skip_enrich: bool = False,
        enrich_only: bool = False,
        force_enrich: bool = False,
        intent_cache_dir: str | None = None,
    ) -> dict:
        """
        ケースカタログを構築する。

        Args:
            include_web: 互換のため残すが、ケース単位インデックスでは未使用
            skip_enrich: LLM 意図メタデータ生成をスキップ
            enrich_only: インデックス化せず enrich のみ実行
            force_enrich: キャッシュを無視して LLM 再生成
            intent_cache_dir: case_intents キャッシュディレクトリ
        """
        stats = {"cases": 0, "skipped": 0, "total": 0, "enriched": 0, "cached": 0, "intent_failed": 0}

        console.print("[bold cyan]ケース単位 RAG インデックスを構築中...[/bold cyan]")
        if not TUTORIALS_ROOT.exists():
            console.print(f"[yellow]  チュートリアルが見つかりません: {TUTORIALS_ROOT}[/yellow]")
            return stats

        cases = discover_cases()
        stats["total"] = len(cases)
        console.print(f"  発見ケース数: {len(cases)}")

        if intent_cache_dir is None:
            intent_cache_dir = str(self.db_path.parent / "case_intents")

        if not skip_enrich:
            console.print("[bold cyan]意図メタデータを生成中 (LLM)...[/bold cyan]")
            enrich_stats = enrich_all_intents(
                cases,
                cache_dir=intent_cache_dir,
                openai_api_key=self.openai_api_key,
                skip=False,
                force=force_enrich,
            )
            stats["enriched"] = enrich_stats.get("enriched", 0)
            stats["cached"] = enrich_stats.get("cached", 0)
            stats["intent_failed"] = enrich_stats.get("failed", 0)
            console.print(
                f"  intent: 新規 {stats['enriched']}, キャッシュ {stats['cached']}, "
                f"失敗 {stats['intent_failed']}"
            )
        else:
            enricher = CaseIntentEnricher(
                cache_dir=intent_cache_dir,
                openai_api_key=self.openai_api_key,
            )
            loaded = 0
            for record in cases:
                cache_path = enricher._cache_path(record.case_id)
                if cache_path.exists():
                    from .case_intent import CaseIntent
                    import json
                    intent = CaseIntent.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))
                    enricher._merge_mechanical(intent, record)
                    record.intent = intent
                    record.embedding_text = record.build_embedding_text()
                    loaded += 1
            stats["cached"] = loaded
            if loaded:
                console.print(f"  [dim]キャッシュから intent 読込: {loaded} 件[/dim]")

        if enrich_only:
            console.print("[bold green]✓ enrich-only 完了[/bold green]")
            return stats

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("ケースをインデックス化...", total=len(cases))
            for record in cases:
                if self._upsert_case(record):
                    stats["cases"] += 1
                else:
                    stats["skipped"] += 1
                progress.advance(task)

        console.print(
            f"\n[bold green]✓ 完了: {stats['cases']} ケース "
            f"(スキップ {stats['skipped']})[/bold green]"
        )
        return stats

    def get_collection_count(self) -> int:
        return self.collection.count()

    def _upsert_case(self, record: CaseRecord) -> bool:
        case_hash = hashlib.md5(record.case_id.encode()).hexdigest()[:12]
        doc_id = f"case_{case_hash}"

        text = record.embedding_text or record.build_embedding_text()
        try:
            embedding = self._embed(text)
        except Exception as e:
            console.print(f"  [yellow]エンベディング失敗: {record.case_id} - {e}[/yellow]")
            return False

        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[record.to_metadata()],
        )
        return True

    def _embed(self, text: str) -> list[float]:
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],
        )
        return response.data[0].embedding
