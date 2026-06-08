"""
2D 円柱周り O-グリッド blockMeshDict ジェネレータ

ellipsekkLOmega / vortexShed チュートリアルと同様の 12 ブロック構造:
  - 4 ブロック: 円柱面〜外 O-リング (内輪)
  - 8 ブロック: 外 O-リング〜計算領域境界 (外枠)

snappyHexMesh を使わず完全六面体メッシュを生成するため、
スキュー問題が発生せず icoFoam/pimpleFoam ともに安定に収束する。
"""
from __future__ import annotations
import math

FOAM_HEADER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  auto                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def generate(
    *,
    cx: float = 0.0,
    cy: float = 0.0,
    r: float = 0.05,
    ring_factor: float = 2.0,   # 外O-リング半径 = r * ring_factor
    x_in: float = -0.8,
    x_out: float = 2.0,
    y_min: float = -1.0,
    y_max: float = 1.0,
    z_min: float = 0.0,
    z_max: float = 0.01,
    # セル数 (各方向)
    n_r: int = 15,        # O-リング径方向 (円柱面→外O-リング)
    n_t: int = 20,        # O-リング接線方向 (1 ブロックあたり)
    n_up: int = 40,       # 上流方向 (inlet → O-リング左端)
    n_down: int = 80,     # 下流方向 (O-リング右端 → outlet)
    n_lat: int = 30,      # 横方向  (O-リング上下端 → top/bottom)
    # グレーディング
    gr_r: float = 0.05,   # 径方向: 外側セル / 内側セル (≪1 → 円柱面が細かい)
    gr_up: float = 0.1,   # 上流: inlet 側が粗く、O-リング側が細かい
    gr_down: float = 10.0,# 下流: O-リング側が細かく、outlet 側が粗い
    gr_lat: float = 0.1,  # 横: 境界が粗く、O-リング側が細かい
) -> str:
    """
    2D 円柱周りの blockMeshDict 文字列を返す。

    座標系:
      - 円柱中心: (cx, cy, -)
      - 流入方向: +x
      - 2D方向:   z (z_min〜z_max の 1 セル, empty patch)

    境界パッチ:
      inlet    : x = x_in   (type patch)
      outlet   : x = x_out  (type patch)
      top      : y = y_max  (type symmetryPlane)
      bottom   : y = y_min  (type symmetryPlane)
      cylinder : 円柱面      (type wall)
      frontAndBack: ±z 面   (type empty, defaultPatch 扱い)
    """
    sin45 = math.sqrt(0.5)
    R = r * ring_factor        # 外O-リング半径 (円柱面のRing_factor倍)

    # ── 2D 頂点座標 (20 頂点) ─────────────────────────────────────────
    # 頂点番号は vortexShed チュートリアルの命名に準拠
    p = [None] * 20
    # 外枠 (計算領域境界との交点)
    p[0]  = (x_in,           y_min)
    p[1]  = (cx - R*sin45,   y_min)
    p[2]  = (cx + R*sin45,   y_min)
    p[3]  = (x_out,          y_min)
    p[4]  = (x_out,          cy - R*sin45)
    p[5]  = (x_out,          cy + R*sin45)
    p[6]  = (x_out,          y_max)
    p[7]  = (cx + R*sin45,   y_max)
    p[8]  = (cx - R*sin45,   y_max)
    p[9]  = (x_in,           y_max)
    p[10] = (x_in,           cy + R*sin45)
    p[11] = (x_in,           cy - R*sin45)
    # 外O-リング (arc 頂点)
    p[12] = (cx - R*sin45,   cy - R*sin45)   # SW outer
    p[13] = (cx + R*sin45,   cy - R*sin45)   # SE outer
    p[14] = (cx + R*sin45,   cy + R*sin45)   # NE outer
    p[15] = (cx - R*sin45,   cy + R*sin45)   # NW outer
    # 円柱面 (arc 頂点)
    p[16] = (cx - r*sin45,   cy - r*sin45)   # SW cylinder
    p[17] = (cx + r*sin45,   cy - r*sin45)   # SE cylinder
    p[18] = (cx + r*sin45,   cy + r*sin45)   # NE cylinder
    p[19] = (cx - r*sin45,   cy + r*sin45)   # NW cylinder

    # ── 3D 頂点リスト (z_min 面 = 0〜19, z_max 面 = 20〜39) ──────────
    def vt(i: int, z: float) -> str:
        x, y = p[i]
        return f"    ( {x:>10.6f} {y:>10.6f} {z:>8.5f} )"

    verts = ["vertices", "("]
    for i in range(20):
        verts.append(f"{vt(i, z_min)}  // {i}")
    for i in range(20):
        verts.append(f"{vt(i, z_max)}  // {i+20}")
    verts.append(");")

    # ── ブロック定義ヘルパー ──────────────────────────────────────────
    # hex2D(a,b,c,d): z_min 面の四角形 → z_max 方向に押し出し
    def hex2D(a: int, b: int, c: int, d: int,
              nx: int, ny: int, gx: float, gy: float) -> str:
        return (
            f"    hex ({a} {b} {c} {d} {a+20} {b+20} {c+20} {d+20})"
            f"  ({nx} {ny} 1)"
            f"  simpleGrading ({gx:.6g} {gy:.6g} 1)"
        )

    # ── ブロック ──────────────────────────────────────────────────────
    # 外枠 8 ブロック + 内輪 4 ブロック (合計 12)
    blocks = ["", "blocks", "("]

    # ─ 外枠ブロック (O-リング外側〜計算領域境界) ─
    # 上流左下 / 上流左上 は n_up セル (x 方向: inlet 粗く O-リング 細かく)
    # 下流は n_down セル (x 方向: O-リング細かく outlet 粗く)
    # 横は n_lat セル (y 方向: top/bottom 粗く O-リング 細かく)

    # Block 0: SW 外枠  (p0, p1, p12, p11) ─ inlet 下半分
    blocks.append(f"    // Block 0: SW outer (inlet bottom half)")
    blocks.append(hex2D(0, 1, 12, 11,   n_up,  n_lat,  gr_up,  gr_lat))

    # Block 1: S  外枠  (p1, p2, p13, p12) ─ O-リング下側
    blocks.append(f"    // Block 1: S outer (below O-ring)")
    blocks.append(hex2D(1, 2, 13, 12,   n_t,   n_lat,  1.0,    gr_lat))

    # Block 2: SE 外枠  (p2, p3, p4, p13) ─ outlet 下半分
    blocks.append(f"    // Block 2: SE outer (outlet bottom half)")
    blocks.append(hex2D(2, 3, 4, 13,    n_down, n_lat, gr_down, gr_lat))

    # Block 3: E  外枠  (p13, p4, p5, p14) ─ O-リング右側
    blocks.append(f"    // Block 3: E outer (right of O-ring)")
    blocks.append(hex2D(13, 4, 5, 14,   n_down, n_t,   gr_down, 1.0))

    # Block 4: NE 外枠  (p14, p5, p6, p7) ─ outlet 上半分
    blocks.append(f"    // Block 4: NE outer (outlet top half)")
    blocks.append(hex2D(14, 5, 6, 7,    n_down, n_lat, gr_down, 1/gr_lat))

    # Block 5: N  外枠  (p15, p14, p7, p8) ─ O-リング上側
    blocks.append(f"    // Block 5: N outer (above O-ring)")
    blocks.append(hex2D(15, 14, 7, 8,   n_t,   n_lat,  1.0,    1/gr_lat))

    # Block 6: NW 外枠  (p10, p15, p8, p9) ─ inlet 上半分
    blocks.append(f"    // Block 6: NW outer (inlet top half)")
    blocks.append(hex2D(10, 15, 8, 9,   n_up,  n_lat,  gr_up,  1/gr_lat))

    # Block 7: W  外枠  (p11, p12, p15, p10) ─ O-リング左側
    blocks.append(f"    // Block 7: W outer (left of O-ring)")
    blocks.append(hex2D(11, 12, 15, 10, n_up,   n_t,   gr_up,  1.0))

    # ─ 内輪 4 ブロック (円柱面〜外O-リング) ─
    # Block 8: W O-ring  (p12, p16, p19, p15)
    blocks.append(f"    // Block 8: W O-ring")
    blocks.append(hex2D(12, 16, 19, 15, n_r,    n_t,   gr_r,   1.0))

    # Block 9: S O-ring  (p12, p13, p17, p16)
    blocks.append(f"    // Block 9: S O-ring")
    blocks.append(hex2D(12, 13, 17, 16, n_t,    n_r,   1.0,    gr_r))

    # Block 10: E O-ring (p17, p13, p14, p18)
    blocks.append(f"    // Block 10: E O-ring")
    blocks.append(hex2D(17, 13, 14, 18, n_r,    n_t,   1/gr_r, 1.0))

    # Block 11: N O-ring (p19, p18, p14, p15)
    blocks.append(f"    // Block 11: N O-ring")
    blocks.append(hex2D(19, 18, 14, 15, n_t,    n_r,   1.0,    1/gr_r))

    blocks.append(");")

    # ── エッジ (arc) ──────────────────────────────────────────────────
    def arc_mid(angle_deg: float, radius: float, z: float) -> str:
        a = math.radians(angle_deg)
        return f"( {cx + radius*math.cos(a):.6f} {cy + radius*math.sin(a):.6f} {z:.5f} )"

    edges = ["", "edges", "("]
    # 円柱面 arc (z_min)
    edges.append(f"    arc 16 17  {arc_mid(-90, r, z_min)}  // S: SW→SE")
    edges.append(f"    arc 17 18  {arc_mid(  0, r, z_min)}  // E: SE→NE")
    edges.append(f"    arc 18 19  {arc_mid( 90, r, z_min)}  // N: NE→NW")
    edges.append(f"    arc 19 16  {arc_mid(180, r, z_min)}  // W: NW→SW")
    # 円柱面 arc (z_max)
    edges.append(f"    arc 36 37  {arc_mid(-90, r, z_max)}")
    edges.append(f"    arc 37 38  {arc_mid(  0, r, z_max)}")
    edges.append(f"    arc 38 39  {arc_mid( 90, r, z_max)}")
    edges.append(f"    arc 39 36  {arc_mid(180, r, z_max)}")
    # 外O-リング arc (z_min)
    edges.append(f"    arc 12 13  {arc_mid(-90, R, z_min)}  // S: SW→SE outer")
    edges.append(f"    arc 13 14  {arc_mid(  0, R, z_min)}  // E: SE→NE outer")
    edges.append(f"    arc 14 15  {arc_mid( 90, R, z_min)}  // N: NE→NW outer")
    edges.append(f"    arc 15 12  {arc_mid(180, R, z_min)}  // W: NW→SW outer")
    # 外O-リング arc (z_max)
    edges.append(f"    arc 32 33  {arc_mid(-90, R, z_max)}")
    edges.append(f"    arc 33 34  {arc_mid(  0, R, z_max)}")
    edges.append(f"    arc 34 35  {arc_mid( 90, R, z_max)}")
    edges.append(f"    arc 35 32  {arc_mid(180, R, z_max)}")
    edges.append(");")

    # ── 境界パッチ ────────────────────────────────────────────────────
    # quad2D(a, b) = z 方向に伸びた面: (a, a+20, b+20, b) の向きで法線を計算
    # vortexShed チュートリアルと同じ頂点順序を採用
    def qf(a: int, b: int) -> str:
        """quad face through z: (a, a+20, b+20, b)"""
        return f"            ({a} {a+20} {b+20} {b})"

    bnd = ["", "defaultPatch", "{",
           "    name    frontAndBack;",
           "    type    empty;",
           "}", "",
           "boundary", "("]

    # inlet  (x = x_in, normal = -x)
    # 頂点順: 上から下へ (p9→p10→p11→p0) で quad: p9-p10, p10-p11, p11-p0
    bnd += [
        "    inlet",
        "    {",
        "        type    patch;",
        "        faces",
        "        (",
        qf(9, 10),
        qf(10, 11),
        qf(11, 0),
        "        );",
        "    }",
    ]
    # outlet (x = x_out, normal = +x)
    bnd += [
        "    outlet",
        "    {",
        "        type    patch;",
        "        faces",
        "        (",
        qf(3, 4),
        qf(4, 5),
        qf(5, 6),
        "        );",
        "    }",
    ]
    # top (y = y_max, normal = +y)
    bnd += [
        "    top",
        "    {",
        "        type    symmetryPlane;",
        "        faces",
        "        (",
        qf(6, 7),
        qf(7, 8),
        qf(8, 9),
        "        );",
        "    }",
    ]
    # bottom (y = y_min, normal = -y)
    bnd += [
        "    bottom",
        "    {",
        "        type    symmetryPlane;",
        "        faces",
        "        (",
        qf(0, 1),
        qf(1, 2),
        qf(2, 3),
        "        );",
        "    }",
    ]
    # cylinder (wall, normal = -r = 向心方向)
    # 円柱面は内輪ブロックの内側面 (p16-p19 の arc 面)
    # vortexShed と同じ頂点順: 16→17→18→19→16
    bnd += [
        "    cylinder",
        "    {",
        "        type    wall;",
        "        faces",
        "        (",
        qf(16, 17),
        qf(17, 18),
        qf(18, 19),
        qf(19, 16),
        "        );",
        "    }",
    ]
    bnd.append(");")
    bnd.append("")
    bnd.append("mergePatchPairs")
    bnd.append("(")
    bnd.append(");")
    bnd.append("")
    bnd.append("// ************************************************************************* //")

    return FOAM_HEADER + "scale   1;\n\n" + "\n".join(verts + blocks + edges + bnd) + "\n"
