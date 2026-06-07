"""直径0.1m、高さ0.3mの円柱STLを生成する"""
import math
import struct

def write_binary_stl(filename, triangles):
    with open(filename, 'wb') as f:
        f.write(b'\x00' * 80)  # header
        f.write(struct.pack('<I', len(triangles)))
        for normal, v1, v2, v3 in triangles:
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<3f', *v3))
            f.write(struct.pack('<H', 0))

def make_cylinder(radius=0.05, height=0.3, n_segs=36):
    """円柱の三角形リストを生成"""
    triangles = []
    pi2 = 2 * math.pi

    for i in range(n_segs):
        a0 = pi2 * i / n_segs
        a1 = pi2 * (i + 1) / n_segs
        x0, y0 = radius * math.cos(a0), radius * math.sin(a0)
        x1, y1 = radius * math.cos(a1), radius * math.sin(a1)

        # 側面（2三角形で1矩形）
        nx = math.cos((a0 + a1) / 2)
        ny = math.sin((a0 + a1) / 2)
        triangles.append(((nx, ny, 0), (x0, y0, 0),      (x1, y1, 0),      (x1, y1, height)))
        triangles.append(((nx, ny, 0), (x0, y0, 0),      (x1, y1, height), (x0, y0, height)))

        # 下面（法線 -z）
        triangles.append(((0, 0, -1), (0, 0, 0), (x1, y1, 0), (x0, y0, 0)))
        # 上面（法線 +z）
        triangles.append(((0, 0, 1),  (0, 0, height), (x0, y0, height), (x1, y1, height)))

    return triangles

tris = make_cylinder(radius=0.05, height=0.3, n_segs=36)
out = "/home/akari/openfoam-ai-agent/cylinder.stl"
write_binary_stl(out, tris)
print(f"生成完了: {out}")
print(f"三角形数: {len(tris)}")
print(f"直径: 0.1m, 高さ: 0.3m, 円周分割数: 36")
