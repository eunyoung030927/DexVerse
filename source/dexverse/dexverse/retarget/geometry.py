# Ported from github.com/HyeonseokE/bench2dex_retarget (retarget/geometry.py).
"""Object surface geometry for fingertip contacts.

Meshes are read from the USD stage the simulator built, so they are exactly the shapes (and
scales) that collide -- the scene yamls point at .usd files of every flavour (textured.usd,
base0.usd, mobility.usd, ...) and only some ship a loose .obj.

Contact is geometric, as in the task-06 prototype: a fingertip within CONTACT_DIST of a body's
surface touches it. PhysX contact reports on distal links came back empty there.
"""

from __future__ import annotations

import numpy as np
import torch

CONTACT_DIST = 0.02
# A fingertip cannot be more than a few cm inside a body. Deeper "inside" readings come from the
# inside test failing on open meshes (articulated appliances: every tip read as touching the
# microwave, task 44), so they are treated as far away, not as contact.
MAX_DEPTH = 0.05


def effective_distance(d):
    """Signed distance -> distance used for contact and proximity: |d|, with deep 'inside' = far."""
    return torch.where(d < -MAX_DEPTH, torch.full_like(d, float("inf")), d.abs())


def body_meshes(stage, body_prim_paths: list[str]) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Triangle soup of every body, expressed in the body's rigid frame (scale baked into points).

    A body owns the meshes below its prim up to the next rigid body. Instance proxies are
    traversed: Bench2Dex assets reference their geometry as instanceable prims.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    out = []
    for path in body_prim_paths:
        body = stage.GetPrimAtPath(path)
        if not body.IsValid():
            out.append(None)
            continue
        T = Gf.Transform(cache.GetLocalToWorldTransform(body))
        rigid = Gf.Matrix4d(T.GetRotation(), T.GetTranslation())        # body frame: scale removed
        inv = rigid.GetInverse()
        verts, faces, off = [], [], 0
        it = iter(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
        for p in it:
            if p != body and p.HasAPI(UsdPhysics.RigidBodyAPI):
                it.PruneChildren()
                continue
            if not p.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(p)
            pts = m.GetPointsAttr().Get()
            counts = m.GetFaceVertexCountsAttr().Get()
            idx = m.GetFaceVertexIndicesAttr().Get()
            if not pts or not counts or not idx:
                continue
            M = np.array(cache.GetLocalToWorldTransform(p) * inv)           # row-vector convention
            P = np.asarray(pts, dtype=np.float64) @ M[:3, :3] + M[3, :3]
            idx = np.asarray(idx)
            tris, k = [], 0
            for c in counts:                                         # fan-triangulate polygons
                for j in range(1, c - 1):
                    tris.append((idx[k], idx[k + j], idx[k + j + 1]))
                k += c
            verts.append(P)
            faces.append(np.asarray(tris, dtype=np.int64) + off)
            off += len(P)
        out.append((np.concatenate(verts), np.concatenate(faces)) if verts else None)
    return out


def closest_on_triangles(p: torch.Tensor, tri: torch.Tensor, chunk: int = 2_000_000):
    """Closest surface point of a triangle soup to each query point.

    p (M,3), tri (F,3,3) -> (dist (M,), closest (M,3), face index (M,)). Ericson's region test,
    vectorised over (points x faces) in chunks so a 50k-face fridge does not exhaust memory.
    """
    M, F = p.shape[0], tri.shape[0]
    rows = max(1, chunk // max(F, 1))
    best_d = torch.full((M,), float("inf"), device=p.device)
    best_c = torch.zeros(M, 3, device=p.device)
    best_f = torch.zeros(M, dtype=torch.long, device=p.device)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    for s in range(0, M, rows):
        q = p[s:s + rows, None]                                     # (m,1,3)
        ap = q - a
        d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
        bp = q - b
        d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
        cp = q - c
        d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        den = (va + vb + vc).clamp(min=1e-18)
        v, w = vb / den, vc / den
        res = a + ab * v[..., None] + ac * w[..., None]              # interior
        # edges
        t_ab = (d1 / (d1 - d3).clamp(min=1e-18)).clamp(0, 1)
        e_ab = a + ab * t_ab[..., None]
        t_ac = (d2 / (d2 - d6).clamp(min=1e-18)).clamp(0, 1)
        e_ac = a + ac * t_ac[..., None]
        t_bc = ((d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp(min=1e-18)).clamp(0, 1)
        e_bc = b + (c - b) * t_bc[..., None]
        res = torch.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], e_ab, res)
        res = torch.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], e_ac, res)
        res = torch.where(((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[..., None], e_bc, res)
        # vertices
        res = torch.where(((d1 <= 0) & (d2 <= 0))[..., None], a.expand_as(res), res)
        res = torch.where(((d3 >= 0) & (d4 <= d3))[..., None], b.expand_as(res), res)
        res = torch.where(((d6 >= 0) & (d5 <= d6))[..., None], c.expand_as(res), res)
        d = (q - res).norm(dim=-1)                                   # (m,F)
        dm, fi = d.min(1)
        best_d[s:s + rows], best_f[s:s + rows] = dm, fi
        best_c[s:s + rows] = res[torch.arange(len(fi), device=p.device), fi]
    return best_d, best_c, best_f


def signed_distance(p, verts, faces):
    """Signed distance (negative inside) of p (M,3) to a mesh given in the same frame."""
    tri = verts[faces]                                              # (F,3,3)
    d, cpt, f = closest_on_triangles(p, tri)
    n = torch.cross(tri[f, 1] - tri[f, 0], tri[f, 2] - tri[f, 0], dim=-1)
    inside = ((p - cpt) * n).sum(-1) < 0
    return torch.where(inside, -d, d), cpt
