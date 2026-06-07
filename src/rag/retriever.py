"""
RAG リトリーバー
ChromaDB から SimulationSpec に関連する OpenFOAM ドキュメントを検索し
EnrichedContext を構築する
"""
from __future__ import annotations

from pathlib import Path

import chromadb
from openai import OpenAI
from rich.console import Console

from ..models import SimulationSpec, EnrichedContext

console = Console()

COLLECTION_NAME = "openfoam_knowledge"
DEFAULT_N_RESULTS = 8   # 検索で取得する上位件数


class OpenFOAMRetriever:
    """ChromaDB から関連ドキュメントを検索するクラス。"""

    def __init__(self, db_path: str, openai_api_key: str):
        self.db_path = Path(db_path)
        self.openai = OpenAI(api_key=openai_api_key)
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            client = chromadb.PersistentClient(path=str(self.db_path))
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def is_available(self) -> bool:
        """インデックスが存在して検索可能かどうか。"""
        try:
            return self.db_path.exists() and self.collection.count() > 0
        except Exception:
            return False

    def retrieve(self, spec: SimulationSpec, n_results: int = DEFAULT_N_RESULTS) -> EnrichedContext:
        """
        SimulationSpec に基づいて関連ドキュメントを検索し EnrichedContext を返す。

        Args:
            spec: 解析仕様
            n_results: 取得する上位件数

        Returns:
            EnrichedContext
        """
        if not self.is_available:
            console.print("[yellow]  RAGインデックスが未構築です。python -m src.main build-index を実行してください。[/yellow]")
            return self._fallback_context(spec)

        query = self._build_query(spec)
        console.print(f"  [dim]RAG検索クエリ: {query[:80]}...[/dim]")

        try:
            embedding = self._embed(query)
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            console.print(f"[yellow]  RAG検索エラー: {e}[/yellow]")
            return self._fallback_context(spec)

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        # 距離でフィルタリング（cosine距離 > 0.8 は無関係として除外）
        filtered = [
            (doc, meta)
            for doc, meta, dist in zip(documents, metadatas, distances)
            if dist < 0.8
        ]

        relevant_examples = [doc for doc, _ in filtered]
        rag_sources = [meta.get("source", "") for _, meta in filtered]

        # 検索結果からスキームと境界条件の推薦を抽出
        recommended_schemes = self._extract_schemes(relevant_examples)
        recommended_bcs = self._extract_bcs(relevant_examples, spec)

        # メッシュテンプレートの選択
        mesh_template_name = self._select_mesh_template(spec)

        # チュートリアルから fvSchemes / fvSolution を直接取得
        ref_fvschemes, ref_fvsolution = self._find_best_system_files(spec)

        console.print(f"  [dim]関連ドキュメント {len(relevant_examples)} 件ヒット[/dim]")

        return EnrichedContext(
            spec=spec,
            relevant_examples=relevant_examples,
            recommended_schemes=recommended_schemes,
            recommended_bcs=recommended_bcs,
            mesh_template_name=mesh_template_name,
            mesh_params_suggestion=self._suggest_mesh_params(spec),
            rag_sources=rag_sources,
            rag_available=True,
            reference_fvschemes=ref_fvschemes,
            reference_fvsolution=ref_fvsolution,
        )

    # ──────────────────────────────────────────────────────────────────
    # 内部ヘルパー
    # ──────────────────────────────────────────────────────────────────

    def _build_query(self, spec: SimulationSpec) -> str:
        """検索クエリ文字列を構築する。"""
        parts = [
            f"OpenFOAM {spec.solver}",
            spec.case_type.replace("_", " "),
            spec.turbulence_model,
            "steady state" if spec.steady_state else "transient unsteady",
            f"inlet velocity {spec.inlet_velocity} m/s",
            spec.description,
        ]
        if spec.re_number:
            parts.append(f"Reynolds number {spec.re_number:.0f}")
        return " ".join(filter(None, parts))

    def _embed(self, text: str) -> list[float]:
        """テキストをベクトル化する。"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def _extract_schemes(self, documents: list[str]) -> str:
        """検索結果から fvSchemes の記述例を抽出する。"""
        for doc in documents:
            if "divSchemes" in doc and "gradSchemes" in doc:
                # fvSchemes らしき内容を含む最初のドキュメントを返す
                lines = doc.split("\n")
                relevant = [l for l in lines if any(
                    kw in l for kw in ["divSchemes", "gradSchemes", "laplacianSchemes", "Gauss", "upwind", "linear"]
                )]
                if relevant:
                    return "\n".join(relevant[:20])
        return ""

    def _extract_bcs(self, documents: list[str], spec: SimulationSpec) -> str:
        """検索結果から境界条件の記述例を抽出する。"""
        for doc in documents:
            if "boundaryField" in doc and "internalField" in doc:
                lines = doc.split("\n")
                bc_lines = []
                in_bc = False
                for line in lines:
                    if "boundaryField" in line:
                        in_bc = True
                    if in_bc:
                        bc_lines.append(line)
                    if in_bc and len(bc_lines) > 30:
                        break
                if bc_lines:
                    return "\n".join(bc_lines[:30])
        return ""

    def _find_best_system_files(self, spec: SimulationSpec) -> tuple[str, str]:
        """
        解析仕様に最も近いチュートリアルの fvSchemes と fvSolution を
        ChromaDB から検索し、ヒットしたファイルをディスクから丸ごと読んで返す。
        見つからない・読めない場合は空文字を返す（テンプレートにフォールバック）。
        """
        query = self._build_query(spec)
        ref_fvschemes = ""
        ref_fvsolution = ""

        # 非圧縮性 incompressible 専用チュートリアルに絞るフィルタキーワード
        INCOMPRESSIBLE_MARKERS = {"simpleFoam", "pisoFoam", "pimpleFoam", "icoFoam"}
        COMPRESSIBLE_MARKERS = {"rhoSimpleFoam", "rhoPimpleFoam", "sonicFoam",
                                "rhoEnergyFoam", "muEff", "rho*nu", "phid,p",
                                "div(phi,h)", "div(phi,K)"}

        for filename in ["fvSchemes", "fvSolution"]:
            try:
                embedding = self._embed(query)
                results = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=10,
                    where={"filename": filename},
                    include=["documents", "metadatas", "distances"],
                )
                docs = results["documents"][0] if results["documents"] else []
                metas = results["metadatas"][0] if results["metadatas"] else []
                dists = results["distances"][0] if results["distances"] else []

                for doc, meta, dist in zip(docs, metas, dists):
                    if dist > 0.6:
                        continue

                    source_path = meta.get("source", "")

                    # 圧縮性チュートリアルを除外
                    if any(m in source_path for m in COMPRESSIBLE_MARKERS):
                        continue
                    if any(m in doc for m in COMPRESSIBLE_MARKERS):
                        continue

                    # fvSolution: SIMPLE ブロック必須
                    if filename == "fvSolution" and "SIMPLE" not in doc:
                        continue
                    # fvSchemes: divSchemes 必須
                    if filename == "fvSchemes" and "divSchemes" not in doc:
                        continue

                    # ソースパスからファイル全体をディスクで読む
                    full_content = ""
                    if source_path and Path(source_path).exists():
                        try:
                            full_content = Path(source_path).read_text(errors="ignore")
                        except Exception:
                            full_content = ""

                    # 読めなかった場合はチャンクを使わずスキップ
                    if not full_content:
                        continue

                    # 不要・有害なエントリを含むファイルを除外
                    INVALID_MARKERS = {
                        "p_rgh",          # buoyantFoam用（simpleFoam不要）
                        "muEff",          # 圧縮性用
                        "rho*nu",         # 圧縮性用
                        "div(phi,h)",     # エネルギー方程式用
                        "dev(T(grad(U)))" # 古い書式（v2012以前）← dev2に統一済み
                    }
                    if any(m in full_content for m in INVALID_MARKERS):
                        continue

                    solver_hint = meta.get("solver", "")
                    short_path = "/".join(Path(source_path).parts[-3:]) if source_path else solver_hint
                    console.print(f"  [dim]RAG参照 {filename}: {short_path}[/dim]")

                    if filename == "fvSchemes":
                        ref_fvschemes = full_content
                    else:
                        ref_fvsolution = full_content
                    break

            except Exception as e:
                console.print(f"  [yellow]system file 検索エラー ({filename}): {e}[/yellow]")

        return ref_fvschemes, ref_fvsolution

    def _select_mesh_template(self, spec: SimulationSpec) -> str:
        """解析タイプに基づいてメッシュテンプレート名を選択する。"""
        if spec.dimensions == 2 or spec.case_type == "2d_flow":
            return "box_2d"
        if spec.case_type == "internal_flow":
            return "box_internal"
        return "box_external"  # デフォルト: 外部流れ

    def _suggest_mesh_params(self, spec: SimulationSpec) -> dict:
        """解析条件に応じたメッシュパラメータを提案する。"""
        v = spec.inlet_velocity
        # Re数に応じてセル数を調整（高Re → 細かいメッシュ）
        re = spec.re_number or 10000
        if re < 1000:
            nx, ny, nz = 20, 10, 10
        elif re < 50000:
            nx, ny, nz = 40, 20, 20
        else:
            nx, ny, nz = 60, 30, 30

        if spec.case_type == "internal_flow":
            # パイプ・ダクト: 流れ方向に細かく
            return {"nx": nx * 2, "ny": ny, "nz": nz,
                    "lx": 10.0, "ly": 1.0, "lz": 1.0}
        else:
            # 外部流れ: 上下方向も広く
            return {"nx": nx, "ny": ny, "nz": nz,
                    "lx": 20.0, "ly": 10.0, "lz": 10.0,
                    "x_min": -5.0}

    def _fallback_context(self, spec: SimulationSpec) -> EnrichedContext:
        """RAG が使えない場合のフォールバックコンテキスト。"""
        return EnrichedContext(
            spec=spec,
            relevant_examples=[],
            recommended_schemes="",
            recommended_bcs="",
            mesh_template_name=self._select_mesh_template(spec),
            mesh_params_suggestion=self._suggest_mesh_params(spec),
            rag_sources=[],
            rag_available=False,
            reference_fvschemes="",
            reference_fvsolution="",
        )
