"""既知形状の blockMeshDict 生成（Python）。"""
from __future__ import annotations

from ..mesh.cylinder_2d_ogrid import generate as generate_cylinder_2d_ogrid
from ..models import SimulationSpec, EnrichedContext


def render_block_mesh_dict(context: EnrichedContext) -> str:
    spec = context.spec
    if spec.case_type == "cylinder_2d_ogrid" or spec.mesh_template == "ogrid_cylinder_2d":
        r = (spec.characteristic_length or 0.1) / 2.0
        lchar = r * 2
        return generate_cylinder_2d_ogrid(
            cx=0.0, cy=0.0, r=r,
            ring_factor=2.0,
            x_in=round(-lchar * 8, 4),
            x_out=round(lchar * 20, 4),
            y_min=round(-lchar * 10, 4),
            y_max=round(lchar * 10, 4),
            z_min=0.0,
            z_max=0.01,
        )
    params = context.mesh_params_suggestion or {}
    nx = params.get("nx", 40)
    ny = params.get("ny", 20)
    lx = params.get("lx", 20.0)
    ly = params.get("ly", 10.0)
    depth = params.get("depth", 0.1)
    return _box_channel_2d(lx, ly, depth, nx, ny)


def _box_channel_2d(lx: float, ly: float, depth: float, nx: int, ny: int) -> str:
    x_min, x_max = -lx * 0.25, lx * 0.75
    y_half = ly / 2
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
scale 1;
vertices
(
    ({x_min} {-y_half} 0) ({x_max} {-y_half} 0) ({x_max} {y_half} 0) ({x_min} {y_half} 0)
    ({x_min} {-y_half} {depth}) ({x_max} {-y_half} {depth}) ({x_max} {y_half} {depth}) ({x_min} {y_half} {depth})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    top {{ type wall; faces ((3 7 6 2)); }}
    bottom {{ type wall; faces ((0 1 5 4)); }}
    front {{ type empty; faces ((0 3 2 1)); }}
    back {{ type empty; faces ((4 5 6 7)); }}
);
mergePatchPairs ();
"""
