"""2D解析用の円柱STLを生成する。
ドメインz=[0, 0.01]に対してz=[-0.005, 0.015]の薄い円柱（ドメインを完全に貫く）"""
import math, struct

def write_binary_stl(filename, triangles):
    with open(filename, 'wb') as f:
        f.write(b'\x00' * 80)
        f.write(struct.pack('<I', len(triangles)))
        for normal, v1, v2, v3 in triangles:
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<3f', *v3))
            f.write(struct.pack('<H', 0))

def make_cylinder_2d(radius=0.05, z_min=-0.005, z_max=0.015, n_segs=36):
    triangles = []
    pi2 = 2 * math.pi
    h = z_max - z_min
    for i in range(n_segs):
        a0 = pi2 * i / n_segs
        a1 = pi2 * (i + 1) / n_segs
        x0, y0 = radius * math.cos(a0), radius * math.sin(a0)
        x1, y1 = radius * math.cos(a1), radius * math.sin(a1)
        nx = math.cos((a0 + a1) / 2)
        ny = math.sin((a0 + a1) / 2)
        # 側面
        triangles.append(((nx, ny, 0), (x0, y0, z_min), (x1, y1, z_min), (x1, y1, z_max)))
        triangles.append(((nx, ny, 0), (x0, y0, z_min), (x1, y1, z_max), (x0, y0, z_max)))
        # 下面
        triangles.append(((0, 0, -1), (0, 0, z_min), (x1, y1, z_min), (x0, y0, z_min)))
        # 上面
        triangles.append(((0, 0, 1),  (0, 0, z_max), (x0, y0, z_max), (x1, y1, z_max)))
    return triangles

tris = make_cylinder_2d(radius=0.05, z_min=-0.005, z_max=0.015, n_segs=36)
out = "/home/akari/openfoam-ai-agent/cylinder_2d.stl"
write_binary_stl(out, tris)
print(f"生成完了: {out}")
print(f"直径: 0.1m, z: -0.005〜0.015m（ドメイン z=[0,0.01] を貫通）")
