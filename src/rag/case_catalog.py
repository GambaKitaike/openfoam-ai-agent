"""
OpenFOAM チュートリアルケースの発見・メタデータ抽出・ファイル収集

インデックス単位は「1 OpenFOAM case directory」。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.case_intent import CaseIntent

TUTORIALS_ROOT = Path("/usr/lib/openfoam/openfoam2512/tutorials")

# ケースに含めるファイル名
CASE_FILE_NAMES = {
    "blockMeshDict", "controlDict", "fvSchemes", "fvSolution",
    "turbulenceProperties", "transportProperties",
    "decomposeParDict", "snappyHexMeshDict", "surfaceFeatureExtractDict",
    "U", "p", "k", "omega", "epsilon", "nut", "nuTilda", "T", "alphat",
}

# 深すぎる・中間セットアップディレクトリは除外
SKIP_PATH_PARTS = {
    "setups.orig", "reEval", "processor", "processors",
    "dynamicCode", "polyMesh",
}

GEOMETRY_KEYWORDS = {
    "cylinder": ("cylinder", "vortex", "vortexshed"),
    "ellipse_external": ("ellipse", "elipse"),
    "channel": ("channel", "pipe", "duct"),
    "cavity": ("cavity",),
    "backward_step": ("backward", "step", "bfs", "bump"),
    "airfoil": ("airfoil", "foil", "naca", "wing"),
    "building": ("building", "wind"),
    "pitz": ("pitz",),
    "motorBike": ("motorbike", "motorBike", "bike"),
}


@dataclass
class CaseRecord:
    """1 チュートリアルケースのメタデータ。"""
    case_id: str
    case_path: str
    category: str
    solver: str
    steady_state: bool
    turbulence_model: str
    dimensions: int
    geometry: str
    patch_names: list[str] = field(default_factory=list)
    indexed_files: list[str] = field(default_factory=list)
    embedding_text: str = ""
    has_blockmesh: bool = False
    has_snappy: bool = False
    requires_preprocessing: bool = False
    mesh_prebuilt: bool = False
    has_blockmesh_in_allrun: bool = False
    run_commands: list[str] = field(default_factory=list)
    readme_excerpt: str = ""
    intent: "CaseIntent | None" = None

    def build_embedding_text(self) -> str:
        """ChromaDB ベクトル検索用テキストを構築する。"""
        parts = [
            f"OpenFOAM case {self.case_id}",
            f"solver {self.solver}",
            f"{'steady' if self.steady_state else 'transient unsteady'}",
            f"turbulence {self.turbulence_model}",
            f"{self.dimensions}D",
            f"geometry {self.geometry}",
            f"category {self.category}",
        ]
        if self.patch_names:
            parts.append(f"patches {' '.join(self.patch_names[:8])}")
        if self.readme_excerpt:
            parts.append(self.readme_excerpt[:500])
        if self.intent:
            parts.insert(0, self.intent.embedding_snippet())
        return " ".join(p for p in parts if p)

    def to_metadata(self) -> dict:
        meta = {
            "case_id": self.case_id,
            "case_path": self.case_path,
            "category": self.category,
            "solver": self.solver,
            "steady_state": self.steady_state,
            "turbulence_model": self.turbulence_model,
            "dimensions": self.dimensions,
            "geometry": self.geometry,
            "patch_names": json.dumps(self.patch_names),
            "indexed_files": json.dumps(self.indexed_files),
            "has_blockmesh": self.has_blockmesh,
            "has_snappy": self.has_snappy,
            "requires_preprocessing": self.requires_preprocessing,
            "mesh_prebuilt": self.mesh_prebuilt,
            "has_blockmesh_in_allrun": self.has_blockmesh_in_allrun,
            "run_commands": json.dumps(self.run_commands[:10]),
            "doc_type": "case",
        }
        if self.intent:
            meta.update(self.intent.to_metadata())
        return meta


def discover_cases(root: Path | None = None) -> list[CaseRecord]:
    """チュートリアルツリーから OpenFOAM ケースを列挙する。"""
    root = root or TUTORIALS_ROOT
    if not root.exists():
        return []

    records: list[CaseRecord] = []
    seen_paths: set[str] = set()

    for control_dict in root.rglob("system/controlDict"):
        case_dir = control_dict.parent.parent
        case_str = str(case_dir.resolve())
        if case_str in seen_paths:
            continue
        if _should_skip_case(case_dir):
            continue
        seen_paths.add(case_str)
        record = _build_case_record(case_dir, root)
        if record and record.solver and record.indexed_files:
            records.append(record)
    return records


def load_case_files(case_path: str | Path) -> dict[str, str]:
    """ケースディレクトリから対象ファイルを relative path → content で読み込む。"""
    case_dir = Path(case_path)
    files: dict[str, str] = {}

    for sub in ("system", "constant"):
        d = case_dir / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.is_file() and f.name in CASE_FILE_NAMES:
                try:
                    files[f"{sub}/{f.name}"] = f.read_text(errors="ignore")
                except OSError:
                    pass

    zero_dir = _find_zero_dir(case_dir)
    if zero_dir:
        for f in zero_dir.iterdir():
            if f.is_file() and f.name in CASE_FILE_NAMES:
                try:
                    files[f"0/{f.name}"] = f.read_text(errors="ignore")
                except OSError:
                    pass

    return files


def _should_skip_case(case_dir: Path) -> bool:
    parts = case_dir.parts
    if any(p in SKIP_PATH_PARTS for p in parts):
        return True
    # 最適化チュートリアル等の深いネストを除外
    try:
        rel = case_dir.relative_to(TUTORIALS_ROOT)
        if len(rel.parts) > 7:
            return True
    except ValueError:
        pass
    return False


def _build_case_record(case_dir: Path, tutorials_root: Path) -> CaseRecord | None:
    try:
        rel = case_dir.relative_to(tutorials_root)
        case_id = str(rel).replace("\\", "/")
    except ValueError:
        case_id = case_dir.name

    category = rel.parts[0] if rel.parts else "unknown"

    control = _read_text(case_dir / "system" / "controlDict")
    if not control:
        return None

    solver = _extract_word(control, "application") or ""
    end_time = _extract_number(control, "endTime")
    delta_t = _extract_number(control, "deltaT")
    adjust = "adjustTimeStep" in control and "yes" in _extract_block(control, "adjustTimeStep")

    turb = _read_text(case_dir / "constant" / "turbulenceProperties")
    turb_model = _parse_turbulence_model(turb)

    blockmesh = _read_text(case_dir / "system" / "blockMeshDict")
    fvschemes = _read_text(case_dir / "system" / "fvSchemes")
    zero_dir = _find_zero_dir(case_dir)
    u_content = ""
    if zero_dir:
        u_content = _read_text(zero_dir / "U") or ""

    dimensions = _infer_dimensions(blockmesh, u_content, case_dir)
    geometry = _infer_geometry(case_id, case_dir.name)
    patch_names = _extract_patch_names(blockmesh, u_content)

    indexed = list(load_case_files(case_dir).keys())

    steady = _infer_steady_state(solver, control, fvschemes)
    requires_pre = _requires_preprocessing(case_dir)
    mesh_prebuilt = _detect_mesh_prebuilt(case_dir)
    has_blockmesh_in_allrun = _has_blockmesh_in_allrun(case_dir)
    run_commands = _extract_run_commands(case_dir)

    readme_excerpt = ""
    readme = case_dir / "README.md"
    if readme.exists():
        readme_excerpt = readme.read_text(errors="ignore")[:500]

    embedding_parts = [
        f"OpenFOAM case {case_id}",
        f"solver {solver}",
        f"{'steady' if steady else 'transient unsteady'}",
        f"turbulence {turb_model}",
        f"{dimensions}D",
        f"geometry {geometry}",
        f"category {category}",
    ]
    if patch_names:
        embedding_parts.append(f"patches {' '.join(patch_names[:8])}")
    if readme_excerpt:
        embedding_parts.append(readme_excerpt)

    record = CaseRecord(
        case_id=case_id,
        case_path=str(case_dir.resolve()),
        category=category,
        solver=solver,
        steady_state=steady,
        turbulence_model=turb_model,
        dimensions=dimensions,
        geometry=geometry,
        patch_names=patch_names,
        indexed_files=indexed,
        has_blockmesh=bool(blockmesh),
        has_snappy=(case_dir / "system" / "snappyHexMeshDict").exists(),
        requires_preprocessing=requires_pre,
        mesh_prebuilt=mesh_prebuilt,
        has_blockmesh_in_allrun=has_blockmesh_in_allrun,
        run_commands=run_commands,
        readme_excerpt=readme_excerpt,
    )
    record.embedding_text = record.build_embedding_text()
    return record


def _requires_preprocessing(case_dir: Path) -> bool:
    """blockMesh 単体では完結しないケース（mirrorMesh, extrudeMesh 等）。"""
    for name in ("Allrun.pre", "Allrun"):
        script = case_dir / name
        if not script.exists():
            continue
        text = script.read_text(errors="ignore")
        extra = ("mirrorMesh", "extrudeMesh", "createPatch", "topoSet", "transformPoints")
        if any(cmd in text for cmd in extra):
            return True
    return False


def _detect_mesh_prebuilt(case_dir: Path) -> bool:
    """constant/polyMesh または polyMesh.orig が存在するか。"""
    if (case_dir / "constant" / "polyMesh").is_dir():
        return True
    if (case_dir / "constant" / "polyMesh.orig").is_dir():
        return True
    return False


def _has_blockmesh_in_allrun(case_dir: Path) -> bool:
    """Allrun / Allrun.pre に blockMesh が含まれるか。"""
    for name in ("Allrun", "Allrun.pre"):
        script = case_dir / name
        if script.exists() and "blockMesh" in script.read_text(errors="ignore"):
            return True
    return False


def _extract_run_commands(case_dir: Path) -> list[str]:
    """Allrun から runApplication / runParallel コマンドを抽出する。"""
    commands: list[str] = []
    for name in ("Allrun", "Allrun.pre"):
        script = case_dir / name
        if not script.exists():
            continue
        text = script.read_text(errors="ignore")
        for m in re.finditer(r"run(?:Application|Parallel)\s+(\w+)", text):
            cmd = m.group(1)
            if cmd not in commands:
                commands.append(cmd)
    return commands


def _find_zero_dir(case_dir: Path) -> Path | None:
    for name in ("0", "0.orig"):
        d = case_dir / name
        if d.is_dir():
            return d
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore") if path.exists() else ""
    except OSError:
        return ""


def _extract_word(content: str, key: str) -> str | None:
    m = re.search(rf"\b{re.escape(key)}\s+(\w+)\s*;", content)
    return m.group(1) if m else None


def _extract_number(content: str, key: str) -> float | None:
    m = re.search(rf"\b{re.escape(key)}\s+([\d.eE+-]+)\s*;", content)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_block(content: str, key: str) -> str:
    m = re.search(rf"\b{re.escape(key)}\s+([^;]+);", content)
    return m.group(1).strip() if m else ""


def _parse_turbulence_model(turb: str) -> str:
    if not turb:
        return "unknown"
    sim = _extract_word(turb, "simulationType") or ""
    if sim.upper() == "LAMINAR" or "simulationType      laminar" in turb.lower():
        return "laminar"
    if "LES" in turb:
        return "LES"
    for model in ("kOmegaSST", "kEpsilon", "SpalartAllmaras", "kkLOmega", "RNGkEpsilon"):
        if model in turb:
            return model
    if sim == "RAS":
        return "RAS"
    return sim or "unknown"


def _infer_steady_state(solver: str, control: str, fvschemes: str) -> bool:
    if solver in ("simpleFoam", "porousSimpleFoam", "rhoSimpleFoam"):
        return True
    if solver in ("icoFoam", "pimpleFoam", "pisoFoam", "nonNewtonianIcoFoam"):
        if "steadyState" in fvschemes and "ddtSchemes" in fvschemes:
            # pimpleFoam can run pseudo-steady; prefer transient if Euler/localEuler
            if re.search(r"ddtSchemes\s*\{[^}]*Euler", fvschemes):
                return False
            if re.search(r"ddtSchemes\s*\{[^}]*localEuler", fvschemes):
                return False
        return False
    if "steadyState" in fvschemes:
        return True
    return False


def _infer_dimensions(blockmesh: str, u_content: str, case_dir: Path | None = None) -> int:
    combined = blockmesh + u_content
    if re.search(r"type\s+empty", combined, re.IGNORECASE):
        return 2
    if re.search(r"frontAndBack|front\s*\{|back\s*\{", combined, re.IGNORECASE):
        if re.search(r"type\s+empty", combined, re.IGNORECASE):
            return 2
    if case_dir is not None:
        for pm_name in ("polyMesh", "polyMesh.orig"):
            boundary = case_dir / "constant" / pm_name / "boundary"
            if boundary.is_file():
                btext = _read_text(boundary)
                if re.search(r"type\s+empty", btext, re.IGNORECASE):
                    return 2
    return 3


def _infer_geometry(case_id: str, case_name: str) -> str:
    hay = f"{case_id} {case_name}".lower()
    for tag, keywords in GEOMETRY_KEYWORDS.items():
        if any(kw.lower() in hay for kw in keywords):
            return tag
    return "general"


def _extract_patch_names(blockmesh: str, u_content: str) -> list[str]:
    names: list[str] = []
    for content in (blockmesh, u_content):
        in_boundary = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("boundary"):
                in_boundary = True
                continue
            if in_boundary:
                if stripped.startswith(")"):
                    break
                m = re.match(r"(\w+)\s*$", stripped)
                if m and m.group(1) not in ("type", "faces", "inGroups", "nFaces"):
                    name = m.group(1)
                    if name not in names and not name.startswith("//"):
                        names.append(name)
    return names
