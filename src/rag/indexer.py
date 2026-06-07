"""
RAG インデクサー
OpenFOAMチュートリアル（ローカル）と公式Webドキュメントを
ChromaDB ベクトルストアにインデックス化する
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Generator

import chromadb
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# インデックス対象のローカルチュートリアルパス
TUTORIALS_ROOT = Path("/usr/lib/openfoam/openfoam2512/tutorials")

# Webスクレイピング対象URL
WEB_SOURCES = [
    "https://www.openfoam.com/documentation/guides/latest/doc/guide-applications-solvers-incompressible.html",
    "https://www.openfoam.com/documentation/guides/latest/doc/guide-turbulence.html",
    "https://www.openfoam.com/documentation/guides/latest/doc/guide-bcs.html",
    "https://www.openfoam.com/documentation/guides/latest/doc/guide-fvschemes.html",
    "https://www.openfoam.com/documentation/guides/latest/doc/guide-fvsolution.html",
    "https://www.openfoam.com/documentation/guides/latest/doc/guide-meshing-blockmesh.html",
]

# インデックス対象とするファイル名パターン
TARGET_FILENAMES = {
    "blockMeshDict", "controlDict", "fvSchemes", "fvSolution",
    "turbulenceProperties", "transportProperties",
    "U", "p", "k", "omega", "epsilon", "nut", "nuTilda",
    "decomposeParDict", "snappyHexMeshDict",
}

# チャンクサイズ（文字数）
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# ChromaDB コレクション名
COLLECTION_NAME = "openfoam_knowledge"


class OpenFOAMIndexer:
    """OpenFOAMのチュートリアルとWebドキュメントをインデックス化するクラス。"""

    def __init__(self, db_path: str, openai_api_key: str):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.openai = OpenAI(api_key=openai_api_key)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def build(self, include_web: bool = True) -> dict:
        """
        インデックスを構築する。

        Args:
            include_web: True の場合 Web ドキュメントもスクレイピングしてインデックス化

        Returns:
            dict: インデックス化の統計情報
        """
        stats = {"local_docs": 0, "web_docs": 0, "total_chunks": 0, "skipped": 0}

        console.print("[bold cyan]RAG インデックスを構築中...[/bold cyan]")

        # ── ローカルチュートリアル ──────────────────────────────────────
        console.print("\n[bold]1/2 ローカルチュートリアルをインデックス化[/bold]")
        local_docs = list(self._iter_tutorial_documents())
        stats["local_docs"] = len(local_docs)

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("ローカル文書を処理中...", total=len(local_docs))
            for doc in local_docs:
                added = self._upsert_document(doc)
                if added:
                    stats["total_chunks"] += 1
                else:
                    stats["skipped"] += 1
                progress.advance(task)

        console.print(f"  → {stats['local_docs']} ファイル, {stats['total_chunks']} チャンク")

        # ── Web ドキュメント ───────────────────────────────────────────
        if include_web:
            console.print("\n[bold]2/2 Web ドキュメントをスクレイピング[/bold]")
            web_docs = list(self._iter_web_documents())
            stats["web_docs"] = len(web_docs)
            web_chunks = 0
            for doc in web_docs:
                added = self._upsert_document(doc)
                if added:
                    web_chunks += 1
            stats["total_chunks"] += web_chunks
            console.print(f"  → {stats['web_docs']} ページ, {web_chunks} チャンク")

        console.print(f"\n[bold green]✓ インデックス構築完了: 合計 {stats['total_chunks']} チャンク[/bold green]")
        return stats

    def get_collection_count(self) -> int:
        """現在のコレクション内のドキュメント数を返す。"""
        return self.collection.count()

    # ──────────────────────────────────────────────────────────────────
    # ローカルチュートリアル読み込み
    # ──────────────────────────────────────────────────────────────────

    def _iter_tutorial_documents(self) -> Generator[dict, None, None]:
        """チュートリアルディレクトリから対象ファイルを読み込む。"""
        if not TUTORIALS_ROOT.exists():
            console.print(f"[yellow]  チュートリアルディレクトリが見つかりません: {TUTORIALS_ROOT}[/yellow]")
            return

        for file_path in TUTORIALS_ROOT.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.name not in TARGET_FILENAMES:
                continue
            try:
                content = file_path.read_text(errors="ignore")
                if len(content.strip()) < 50:
                    continue
                # チュートリアルのカテゴリ（incompressible/simpleFoam/pitzDailyなど）
                rel = file_path.relative_to(TUTORIALS_ROOT)
                parts = rel.parts
                category = parts[0] if len(parts) > 1 else "unknown"
                solver_hint = parts[1] if len(parts) > 2 else ""

                yield {
                    "content": content,
                    "source": str(file_path),
                    "filename": file_path.name,
                    "category": category,
                    "solver": solver_hint,
                    "doc_type": "tutorial",
                }
            except Exception:
                continue

    # ──────────────────────────────────────────────────────────────────
    # Web スクレイピング
    # ──────────────────────────────────────────────────────────────────

    def _iter_web_documents(self) -> Generator[dict, None, None]:
        """Web ドキュメントをスクレイピングして読み込む。"""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            console.print("[yellow]  requests / beautifulsoup4 が未インストール。Webスクレイピングをスキップ。[/yellow]")
            return

        for url in WEB_SOURCES:
            try:
                console.print(f"  → {url}")
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                # メインコンテンツを抽出（ナビゲーションなどを除く）
                main = soup.find("main") or soup.find("article") or soup.find("body")
                if main is None:
                    continue
                text = main.get_text(separator="\n", strip=True)
                # 短すぎる・長すぎるものをスキップ
                if len(text) < 200:
                    continue
                yield {
                    "content": text[:50000],   # 50KB上限
                    "source": url,
                    "filename": url.split("/")[-1],
                    "category": "documentation",
                    "solver": "",
                    "doc_type": "web",
                }
            except Exception as e:
                console.print(f"  [yellow]  スキップ: {url} ({e})[/yellow]")
                continue

    # ──────────────────────────────────────────────────────────────────
    # ChromaDB へのアップサート
    # ──────────────────────────────────────────────────────────────────

    def _upsert_document(self, doc: dict) -> bool:
        """
        ドキュメントをチャンク分割してベクトル化し ChromaDB に保存する。
        既存のドキュメント（同一ソース）はスキップする。
        Returns True if new chunks were added.
        """
        content = doc["content"]
        source = doc["source"]

        # 既存チェック（ソースURLのハッシュで識別）
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        existing = self.collection.get(where={"source_hash": source_hash})
        if existing["ids"]:
            return False

        chunks = self._split_text(content)
        if not chunks:
            return False

        # バッチでエンベディング
        try:
            embeddings = self._embed_batch(chunks)
        except Exception as e:
            console.print(f"  [yellow]エンベディング失敗: {source} - {e}[/yellow]")
            return False

        ids = [f"{source_hash}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": source,
                "source_hash": source_hash,
                "filename": doc["filename"],
                "category": doc["category"],
                "solver": doc["solver"],
                "doc_type": doc["doc_type"],
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        return True

    def _split_text(self, text: str) -> list[str]:
        """テキストをオーバーラップ付きでチャンク分割する。"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """OpenAI API でテキストをベクトル化する。"""
        # OpenAI API は一度に最大 2048 テキスト
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )
            all_embeddings.extend([r.embedding for r in response.data])
        return all_embeddings
