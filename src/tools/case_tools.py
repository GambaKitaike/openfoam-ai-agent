"""ケース生成ツール — v1 パイプラインの生成部のみをラップ（DESIGN.md §4.3）。"""
from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.agents.openfoam_gpt import OpenFOAMGPTAgent
from src.agents.preprocessing import PreprocessingAgent
from src.agents.prompt_generation import PromptGenerationAgent
from src.agents.spec_clarification import clarify_from_reference
from src.case_builder.pipeline import CaseBuildPipeline
from src.config import Settings
from src.models import EnrichedContext, GenerationResult, SimulationSpec

from .base import ToolResult


def _error(message: str) -> ToolResult:
    return ToolResult(ok=False, content=message)


def _scaffold_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"llm_model": settings.llm_model_mini})


def _spec_to_dict(spec: SimulationSpec) -> dict[str, Any]:
    data = asdict(spec)
    for key, value in list(data.items()):
        if isinstance(value, float) and value is not None:
            data[key] = value
    return data


def _format_bc_summary(boundary_conditions: dict) -> str:
    if not boundary_conditions:
        return "(none)"
    parts: list[str] = []
    for patch, cfg in boundary_conditions.items():
        if isinstance(cfg, dict):
            detail = ", ".join(f"{key}={value}" for key, value in cfg.items())
            parts.append(f"{patch}: {detail}")
        else:
            parts.append(f"{patch}: {cfg}")
    return "; ".join(parts)


_DEFAULTED_FIELD_MESSAGES: dict[str, str] = {
    "inlet_velocity": "流入速度が指定されていないため {value} m/s を仮定しました。結果が大きく変わる可能性があります",
    "characteristic_length": "代表長さが指定されていないため {value} m を仮定しました。結果が大きく変わる可能性があります",
    "nu": "動粘度 nu が指定されていないため {value:g} m²/s を仮定しました。結果が大きく変わる可能性があります",
    "turbulence_model": "乱流モデルが指定されていないため {value} を仮定しました。結果が大きく変わる可能性があります",
}


def _warnings_from_defaulted_fields(spec: SimulationSpec) -> list[str]:
    warnings: list[str] = []
    for key in spec.defaulted_fields:
        template = _DEFAULTED_FIELD_MESSAGES.get(key)
        if template is None:
            continue
        value = getattr(spec, key, None)
        if value is None:
            continue
        warnings.append(template.format(value=value))
    return warnings


def _format_warnings_section(warnings: list[str]) -> str:
    if not warnings:
        return ""
    lines = ["Warnings:"]
    lines.extend(f"- {msg}" for msg in warnings)
    return "\n".join(lines) + "\n"


def _format_spec_summary(spec: SimulationSpec) -> str:
    steady = "steady" if spec.steady_state else "transient"
    lines = [
        f"solver: {spec.solver}",
        f"case_type: {spec.case_type}",
        f"flow: {steady}, {spec.dimensions}D, turbulence={spec.turbulence_model}",
        f"boundary_conditions: {_format_bc_summary(spec.boundary_conditions)}",
    ]
    if spec.inlet_velocity:
        lines.append(f"inlet_velocity: {spec.inlet_velocity} m/s")
    if spec.phenomenon:
        lines.append(f"phenomenon: {spec.phenomenon}")
    if spec.stl_path:
        lines.append(f"stl: {Path(spec.stl_path).name}")
    return "\n".join(lines)


def _promote_case_to_workspace(workspace: Path, gen: GenerationResult) -> GenerationResult:
    """pipeline が作ったサブディレクトリを workspace 直下へ昇格する。"""
    generated = Path(gen.output_path).resolve()
    target = workspace.resolve()
    if generated == target:
        return GenerationResult(
            output_path=str(target),
            case_type=gen.case_type,
            files_created=gen.files_created,
            mesh_built=False,
            build_path=gen.build_path,
        )
    if not generated.is_dir():
        return gen

    for child in generated.iterdir():
        dest = target / child.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(child), str(dest))
    generated.rmdir()

    return GenerationResult(
        output_path=str(target),
        case_type=gen.case_type,
        files_created=gen.files_created,
        mesh_built=False,
        build_path=gen.build_path,
    )


def _generate_case_files_only(
    agent3: OpenFOAMGPTAgent,
    context: EnrichedContext,
    output_dir: str,
    reference_match,
) -> GenerationResult:
    """Agent③ のケース生成のみ。mesh / solver 実行は行わない。"""
    original_run = agent3.pipeline.run

    def run_files_only(
        ctx: EnrichedContext,
        out_dir: str,
        *,
        run_mesh: bool = True,
    ) -> tuple:
        return original_run(ctx, out_dir, run_mesh=False)

    agent3.pipeline.run = run_files_only  # type: ignore[method-assign]
    try:
        return agent3._generate_case(context, output_dir, reference_match)
    finally:
        agent3.pipeline.run = original_run


def case_scaffold(
    workspace: Path,
    description: str,
    stl_path: str | None = None,
    *,
    settings: Settings | None = None,
) -> ToolResult:
    """
    自然言語から OpenFOAM ケース一式を workspace に生成する（実行はしない）。

    v1 の Agent①（spec 変換）+ Agent②（参照選定）+ Agent③（ファイル生成）のみ呼ぶ。
    """
    settings = settings or Settings()
    scaffold_settings = _scaffold_settings(settings)
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    agent1 = PreprocessingAgent(scaffold_settings)
    agent2 = PromptGenerationAgent(settings)
    agent3 = OpenFOAMGPTAgent(settings)

    stl = stl_path or ""
    spec = agent1.extract(description, stl_path=stl)
    spec = agent1.complete_hearing(
        spec,
        agent2,
        description,
        interactive=False,
    )

    match = agent2.retrieve_match(spec)
    scaffold_warnings: list[str] = _warnings_from_defaulted_fields(spec)
    if match.context.reference_case_id:
        spec, ref_warnings = clarify_from_reference(spec, match.context, interactive=False)
        scaffold_warnings.extend(ref_warnings)
        match.context.spec = spec

    context = match.context

    def guidance_fn(rel: str, s: SimulationSpec, patch_names: list[str] | None = None) -> str:
        return agent2.get_file_guidance(rel, s, context, patch_names=patch_names)

    agent3.pipeline = CaseBuildPipeline(
        settings,
        guidance_fn=lambda rel, s, patches: guidance_fn(rel, s, patches),
    )

    try:
        gen = _generate_case_files_only(
            agent3,
            context,
            str(workspace),
            match,
        )
    except Exception as exc:
        return _error(f"Case generation failed: {exc}")

    gen = _promote_case_to_workspace(workspace, gen)
    spec = context.spec

    content = (
        "Case scaffold complete (files only; run blockMesh/solver via run_openfoam).\n"
        f"{_format_warnings_section(scaffold_warnings)}"
        f"{_format_spec_summary(spec)}\n"
        f"files_created: {len(gen.files_created)}\n"
        f"output: {gen.output_path}"
    )
    return ToolResult(
        ok=True,
        content=content,
        data={
            "spec": _spec_to_dict(spec),
            "files_created": gen.files_created,
            "output_path": gen.output_path,
            "reference_case_id": context.reference_case_id or None,
            "build_path": gen.build_path,
        },
    )
