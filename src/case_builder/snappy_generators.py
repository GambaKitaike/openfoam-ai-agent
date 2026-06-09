"""snappyHexMesh 関連ファイルの決定的ビルダー。"""
from __future__ import annotations

from ..models import SimulationSpec

FOAM_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     |
    \\\\  /    A nd           |
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
"""


def render_snappy_block_mesh_dict(
    *,
    stl_name: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int = 120,
    ny: int = 70,
    is_2d: bool = True,
) -> str:
    """STL bbox に合わせた snappy 用背景 blockMeshDict。"""
    if is_2d:
        return f"""{FOAM_HEADER}
    object      blockMeshDict;
}}
convertToMeters 1.0;
vertices
(
    ( {x_min} {y_min} 0    )
    ( {x_max} {y_min} 0    )
    ( {x_max} {y_max} 0    )
    ( {x_min} {y_max} 0    )
    ( {x_min} {y_min} 0.01 )
    ( {x_max} {y_min} 0.01 )
    ( {x_max} {y_max} 0.01 )
    ( {x_min} {y_max} 0.01 )
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    top {{ type symmetryPlane; faces ((3 7 6 2)); }}
    bottom {{ type symmetryPlane; faces ((0 1 5 4)); }}
    front {{ type empty; faces ((4 5 6 7)); }}
    back {{ type empty; faces ((0 3 2 1)); }}
);
mergePatchPairs ();
"""
    z_min, z_max = 0.0, max(y_max - y_min, 0.1)
    nz = 20
    return f"""{FOAM_HEADER}
    object      blockMeshDict;
}}
convertToMeters 1.0;
vertices
(
    ( {x_min} {y_min} {z_min} ) ( {x_max} {y_min} {z_min} )
    ( {x_max} {y_max} {z_min} ) ( {x_min} {y_max} {z_min} )
    ( {x_min} {y_min} {z_max} ) ( {x_max} {y_min} {z_max} )
    ( {x_max} {y_max} {z_max} ) ( {x_min} {y_max} {z_max} )
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    top {{ type wall; faces ((3 7 6 2)); }}
    bottom {{ type wall; faces ((0 1 5 4)); }}
    front {{ type patch; faces ((4 5 6 7)); }}
    back {{ type patch; faces ((0 3 2 1)); }}
);
mergePatchPairs ();
"""


def build_snappy_hex_mesh_dict(
    *,
    stl_name: str,
    stl_solid_name: str,
    location_x: float,
    location_y: float,
    location_z: float,
    spec: SimulationSpec,
) -> str:
    is_2d = spec.case_type == "snappy_2d"
    feature_level = 1 if is_2d else 2
    surface_min = 2 if is_2d else 3
    surface_max = 3 if is_2d else 5
    return f"""{FOAM_HEADER}
    object      snappyHexMeshDict;
}}
castellatedMesh true;
snap            true;
addLayers       false;
geometry
{{
    {stl_name}
    {{
        type triSurfaceMesh;
        file "{stl_name}";
        name {stl_solid_name};
    }}
}}
castellatedMeshControls
{{
    maxLocalCells       1000000;
    maxGlobalCells      4000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;
    features ( {{ file "{stl_solid_name}.eMesh"; level {feature_level}; }} );
    refinementSurfaces
    {{
        {stl_solid_name}
        {{
            level ({surface_min} {surface_max});
            patchInfo {{ type wall; inGroups (wall); }}
        }}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    locationInMesh ({location_x} {location_y} {location_z});
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}
addLayersControls
{{
    relativeSizes true;
    layers {{ {stl_solid_name} {{ nSurfaceLayers 3; }} }}
    expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1;
    nGrow 0; featureAngle 60; slipFeatureAngle 30; nRelaxIter 3;
    nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;
}}
meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minVol 1e-13; minTetQuality 1e-30; minArea -1;
    minTwist 0.05; minDeterminant 0.001; minFaceWeight 0.05;
    minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4;
    errorReduction 0.75; relaxed {{ maxNonOrtho 75; }}
}}
debug 0;
mergeTolerance 1e-6;
"""


def build_surface_feature_extract_dict(stl_name: str) -> str:
    return f"""{FOAM_HEADER}
    object      surfaceFeatureExtractDict;
}}
{stl_name}
{{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs {{ includedAngle 150; }}
    writeObj            yes;
}}
"""
