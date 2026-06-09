"""OpenFOAM ケースファイルの決定的ビルダー（Jinja 代替）。"""
from __future__ import annotations

from ..models import SimulationSpec
from .policy import compute_time_settings

FOAM_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Version:  v2512
    \\\\  /    A nd           | Website:  www.openfoam.com
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
"""


def build_transport_properties(spec: SimulationSpec) -> str:
    return f"""{FOAM_HEADER}
    class       dictionary;
    object      transportProperties;
}}
transportModel  Newtonian;
nu              {spec.nu:g};
"""


def build_turbulence_properties(spec: SimulationSpec) -> str:
    if spec.turbulence_model == "laminar":
        body = "simulationType  laminar;"
    else:
        body = f"""simulationType  RAS;
RAS
{{
    RASModel        {spec.turbulence_model};
    turbulence      on;
    printCoeffs     on;
}}"""
    return f"""{FOAM_HEADER}
    class       dictionary;
    object      turbulenceProperties;
}}
{body}
"""


def build_fv_schemes(spec: SimulationSpec) -> str:
    ddt = "steadyState" if spec.steady_state else "Euler"
    if spec.steady_state:
        div_u = "bounded Gauss linearUpwind grad(U)"
        div_t = "bounded Gauss upwind"
    else:
        div_u = "Gauss linearUpwind grad(U)"
        div_t = "Gauss upwind"
    return f"""{FOAM_HEADER}
    class       dictionary;
    object      fvSchemes;
}}
ddtSchemes {{ default {ddt}; }}
gradSchemes {{ default Gauss linear; grad(U) cellLimited Gauss linear 1; }}
divSchemes
{{
    default none;
    div(phi,U)      {div_u};
    div(phi,k)      {div_t};
    div(phi,omega)  {div_t};
    div(phi,epsilon) {div_t};
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default corrected; }}
wallDist {{ method meshWave; }}
"""


def build_fv_solution(spec: SimulationSpec) -> str:
    if spec.steady_state:
        return _fv_solution_steady(spec)
    if spec.case_type == "cylinder_2d_ogrid":
        return _fv_solution_ogrid_pimple()
    return _fv_solution_pimple_generic()


def _fv_solution_steady(spec: SimulationSpec) -> str:
    return f"""{FOAM_HEADER}
    class       dictionary;
    object      fvSolution;
}}
solvers
{{
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-6; relTol 0.1; }}
    "(U|k|omega|epsilon)" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-5; relTol 0.1; }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    consistent yes;
    residualControl {{ p 1e-2; U 1e-3; "(k|omega|epsilon)" 1e-3; }}
}}
relaxationFactors {{ equations {{ U 0.9; ".*" 0.9; }} }}
"""


def _fv_solution_ogrid_pimple() -> str:
    return f"""{FOAM_HEADER}
    class       dictionary;
    object      fvSolution;
}}
solvers
{{
    Phi {{ solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0.01; }}
    p {{ solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0.05; }}
    pFinal {{ solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0; }}
    "(U|k|omega|epsilon)" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-5; relTol 0.1; }}
    "(U|k|omega|epsilon)Final" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-5; relTol 0; }}
}}
potentialFlow {{ nNonOrthogonalCorrectors 10; }}
PIMPLE {{ nOuterCorrectors 1; nCorrectors 4; nNonOrthogonalCorrectors 2; }}
relaxationFactors {{ equations {{ U 0.9; k 0.9; omega 0.9; }} }}
"""


def _fv_solution_pimple_generic() -> str:
    return f"""{FOAM_HEADER}
    class       dictionary;
    object      fvSolution;
}}
solvers
{{
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-6; relTol 0.1; }}
    pFinal {{ solver GAMG; smoother GaussSeidel; tolerance 1e-6; relTol 0; }}
    "(U|k|omega|epsilon)" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-5; relTol 0.1; }}
    "(U|k|omega|epsilon)Final" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-5; relTol 0; }}
}}
PIMPLE {{ nOuterCorrectors 3; nCorrectors 2; nNonOrthogonalCorrectors 1; maxCo 0.9; }}
relaxationFactors {{ equations {{ U 0.9; k 0.9; omega 0.9; }} }}
"""


def build_control_dict(spec: SimulationSpec) -> str:
    ts = compute_time_settings(spec)
    wc = ts.get("write_control", "timeStep")
    if wc == "runTime":
        write_block = f"writeControl    runTime;\nwriteInterval   {ts['write_interval']};"
    else:
        wi = max(1, int(ts["write_interval"] / ts["delta_t"]))
        write_block = f"writeControl    timeStep;\nwriteInterval   {wi};"
    adjust = ""
    if not spec.steady_state and spec.case_type != "cylinder_2d_ogrid":
        adjust = f"""
adjustTimeStep  yes;
maxCo           0.5;
maxDeltaT       {ts['delta_t'] * 100:g};
"""
    return f"""{FOAM_HEADER}
    class       dictionary;
    object      controlDict;
}}
application     {spec.solver};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {ts['end_time']};
deltaT          {ts['delta_t']};
{write_block}
purgeWrite      {ts['purge_write']};
{adjust}
writeFormat     ascii;
writePrecision  6;
runTimeModifiable true;
"""


def build_u_field(spec: SimulationSpec, patch_names: list[str]) -> str:
    u = spec.inlet_velocity
    lines = [
        FOAM_HEADER.rstrip(),
        "    class       volVectorField;",
        "    object      U;",
        "}",
        "dimensions      [0 1 -1 0 0 0 0];",
        f"internalField   uniform ({u:g} 0 0);",
        "boundaryField",
        "{",
    ]
    patches = set(patch_names)
    if "inlet" in patches:
        lines += [
            "    inlet { type fixedValue; value uniform (%g 0 0); }" % u,
            "    outlet { type zeroGradient; }",
        ]
    if spec.case_type == "cylinder_2d_ogrid":
        for name, bc in [
            ("top", "symmetryPlane"),
            ("bottom", "symmetryPlane"),
            ("frontAndBack", "empty"),
            ("cylinder", "noSlip"),
        ]:
            if name in patches:
                lines.append(f"    {name} {{ type {bc}; }}")
    elif spec.dimensions == 2:
        for name, bc in [("top", "noSlip"), ("bottom", "noSlip"), ("front", "empty"), ("back", "empty")]:
            if name in patches:
                lines.append(f"    {name} {{ type {bc}; }}")
    else:
        for name in patches:
            if name in ("inlet", "outlet"):
                continue
            bc = "empty" if name in ("front", "back") else "noSlip"
            lines.append(f"    {name} {{ type {bc}; }}")
    lines += ["}", ""]
    return "\n".join(lines)


def build_p_field(spec: SimulationSpec, patch_names: list[str]) -> str:
    lines = [
        FOAM_HEADER.rstrip(),
        "    class       volScalarField;",
        "    object      p;",
        "}",
        "dimensions      [0 2 -2 0 0 0 0];",
        "internalField   uniform 0;",
        "boundaryField",
        "{",
    ]
    patches = set(patch_names)
    if "inlet" in patches:
        lines += ["    inlet { type zeroGradient; }", "    outlet { type fixedValue; value uniform 0; }"]
    if spec.case_type == "cylinder_2d_ogrid":
        for name, bc in [
            ("top", "symmetryPlane"), ("bottom", "symmetryPlane"),
            ("frontAndBack", "empty"), ("cylinder", "zeroGradient"),
        ]:
            if name in patches:
                lines.append(f"    {name} {{ type {bc}; }}")
    else:
        for name in patches:
            if name in ("inlet", "outlet"):
                continue
            bc = "empty" if name in ("front", "back") else "zeroGradient"
            lines.append(f"    {name} {{ type {bc}; }}")
    lines += ["}", ""]
    return "\n".join(lines)


def build_set_fields_dict(spec: SimulationSpec) -> str:
    r = (spec.characteristic_length or 1.0) / 2.0
    wake_x0 = round(r * 1.05, 4)
    wake_x1 = round(r * 6.0, 4)
    wake_y = round(r * 2.0, 4)
    pv = round(spec.inlet_velocity * 0.05, 6)
    u = spec.inlet_velocity
    depth = 0.01
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object setFieldsDict; }}
defaultFieldValues ( volVectorFieldValue U ({u:g} 0 0) );
regions
(
    boxToCell {{ box ({wake_x0} -{wake_y} 0) ({wake_x1} 0 {depth}); fieldValues ( volVectorFieldValue U ({u:g} {pv} 0) ); }}
    boxToCell {{ box ({wake_x0} 0 0) ({wake_x1} {wake_y} {depth}); fieldValues ( volVectorFieldValue U ({u:g} {-pv} 0) ); }}
);
"""
