"""参照ケース fast path。"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ..case_applier import CaseApplier, copy_zero_orig_if_needed
from ..models import CaseBuildState, EnrichedContext, GenerationResult, ReferenceMatch

console = Console()


def apply_reference_fast_path(
    match: ReferenceMatch,
    output_dir: str,
) -> tuple[CaseBuildState, GenerationResult]:
    """高一致参照ケースをコピーしてパラメータ差し替え。"""
    context = match.context
    spec = context.spec
    import re
    case_name = re.sub(r"[^\w-]", "_", f"{spec.solver}_{spec.case_type}")
    case_path = Path(output_dir) / case_name
    applier = CaseApplier()
    files_created = applier.apply(context, case_path)
    copy_zero_orig_if_needed(case_path)

    if spec.case_type == "cylinder_2d_ogrid" and not context.reference_mesh_prebuilt:
        from .mesh_generators import render_block_mesh_dict
        bmd = render_block_mesh_dict(context)
        (case_path / "system" / "blockMeshDict").write_text(bmd)
        if "system/blockMeshDict" not in files_created:
            files_created.append("system/blockMeshDict")

    console.print(f"  [green]参照ケース fast path[/green] (score={match.score:.2f})")
    state = CaseBuildState(
        case_dir=str(case_path),
        spec=spec,
        files_created=files_created,
        build_path="reference_fast",
        completed_steps=["reference_copy"],
    )
    gen = GenerationResult(
        output_path=str(case_path),
        case_type=spec.case_type,
        files_created=files_created,
        mesh_built=False,
        build_path="reference_fast",
    )
    return state, gen
