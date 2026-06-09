"""CaseIntent enricher のユニットテスト"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag.case_catalog import CaseRecord
from src.rag.case_intent import CaseIntent, PHENOMENON_TAGS, phenomenon_matches
from src.rag.case_intent_enricher import (
    CaseIntentEnricher,
    compute_source_hash,
    _parse_json,
)


@pytest.fixture
def airfoil_record(tmp_path: Path) -> CaseRecord:
    case_dir = tmp_path / "incompressible" / "simpleFoam" / "airFoil2D"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "constant" / "polyMesh").mkdir(parents=True)
    (case_dir / "0").mkdir()
    (case_dir / "system" / "controlDict").write_text(
        "application     simpleFoam;\nendTime         1000;\n"
    )
    (case_dir / "Allrun").write_text(
        "#!/bin/bash\nrunApplication simpleFoam\n"
    )
    (case_dir / "0" / "U").write_text(
        "boundaryField\n{\n    freestream { type freestreamVelocity; }\n    foil { type noSlip; }\n}\n"
    )
    return CaseRecord(
        case_id="incompressible/simpleFoam/airFoil2D",
        case_path=str(case_dir),
        category="incompressible",
        solver="simpleFoam",
        steady_state=True,
        turbulence_model="laminar",
        dimensions=2,
        geometry="airfoil",
        patch_names=["freestream", "foil"],
        indexed_files=["system/controlDict", "0/U"],
        mesh_prebuilt=True,
        has_blockmesh_in_allrun=False,
        run_commands=["simpleFoam"],
    )


def test_parse_json_strips_code_fence():
    raw = '```json\n{"title_ja": "テスト", "phenomenon": "general"}\n```'
    data = _parse_json(raw)
    assert data["title_ja"] == "テスト"
    assert data["phenomenon"] == "general"


def test_case_intent_normalize_invalid_phenomenon():
    intent = CaseIntent(case_id="test", phenomenon="invalid_tag")
    intent.normalize_phenomenon()
    assert intent.phenomenon == "general"


def test_phenomenon_matches_compat():
    assert phenomenon_matches("karman_vortex_shedding", "karman_vortex_shedding")
    assert not phenomenon_matches("karman_vortex_shedding", "general")
    assert not phenomenon_matches("karman_vortex_shedding", "airfoil_steady")
    assert phenomenon_matches("", "airfoil_steady")


def test_compute_source_hash_changes_with_file(tmp_path: Path):
    case_dir = tmp_path / "case"
    (case_dir / "system").mkdir(parents=True)
    cd = case_dir / "system" / "controlDict"
    cd.write_text("application simpleFoam;")
    h1 = compute_source_hash(case_dir)
    cd.write_text("application pimpleFoam;")
    h2 = compute_source_hash(case_dir)
    assert h1 != h2


def test_enricher_cache_hit(tmp_path: Path, airfoil_record: CaseRecord):
    cache_dir = tmp_path / "case_intents"
    source_hash = compute_source_hash(Path(airfoil_record.case_path))
    cached = CaseIntent(
        case_id=airfoil_record.case_id,
        title_ja="2次元翼周りの定常流れ",
        summary_ja="simpleFoam による翼周り定常流れの検証ケース。",
        phenomenon="airfoil_steady",
        geometry="airfoil_2d",
        mesh_prebuilt=True,
        run_commands=["simpleFoam"],
        source_hash=source_hash,
        suitable_for_ja=["2D翼の定常流れ", "揚力の確認"],
    )
    safe = airfoil_record.case_id.replace("/", "__")
    cache_dir.mkdir()
    (cache_dir / f"{safe}.json").write_text(
        json.dumps(cached.to_dict(), ensure_ascii=False), encoding="utf-8"
    )

    mock_llm = MagicMock()
    enricher = CaseIntentEnricher(cache_dir=cache_dir, llm=mock_llm)
    intent, from_cache = enricher.enrich_record(airfoil_record)
    assert from_cache is True
    assert intent.phenomenon == "airfoil_steady"
    assert intent.mesh_prebuilt is True
    mock_llm.chat.assert_not_called()


def test_enricher_llm_generates_and_caches(tmp_path: Path, airfoil_record: CaseRecord):
    cache_dir = tmp_path / "case_intents"
    llm_response = json.dumps({
        "title_ja": "2次元翼周りの定常流れ",
        "summary_ja": "既存メッシュを用いた翼周り定常流れ。",
        "phenomenon": "airfoil_steady",
        "geometry": "airfoil_2d",
        "observables": ["velocity_U", "pressure_p"],
        "bc_summary_ja": "freestream と noSlip 壁",
        "mesh_notes_ja": "メッシュ済み。blockMesh 不要",
        "suitable_for_ja": ["2D翼 定常 simpleFoam"],
        "not_suitable_for_ja": ["カルマン渦"],
    }, ensure_ascii=False)

    mock_llm = MagicMock()
    mock_llm.chat.return_value = llm_response
    enricher = CaseIntentEnricher(cache_dir=cache_dir, llm=mock_llm)

    intent, from_cache = enricher.enrich_record(airfoil_record)
    assert from_cache is False
    assert intent.phenomenon == "airfoil_steady"
    assert intent.mesh_prebuilt is True
    assert "simpleFoam" in intent.run_commands

    safe = airfoil_record.case_id.replace("/", "__")
    assert (cache_dir / f"{safe}.json").exists()


def test_enricher_fallback_on_llm_error(tmp_path: Path, airfoil_record: CaseRecord):
    cache_dir = tmp_path / "case_intents"
    mock_llm = MagicMock()
    mock_llm.chat.side_effect = RuntimeError("API error")
    enricher = CaseIntentEnricher(cache_dir=cache_dir, llm=mock_llm)

    intent = enricher._fallback_intent(airfoil_record)
    assert intent.case_id == airfoil_record.case_id
    assert intent.mesh_prebuilt is True


def test_all_phenomenon_tags_in_enum():
    for tag in PHENOMENON_TAGS:
        intent = CaseIntent(case_id="x", phenomenon=tag)
        intent.normalize_phenomenon()
        assert intent.phenomenon == tag
