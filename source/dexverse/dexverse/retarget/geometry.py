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


def simplify(verts: np.ndarray, faces: np.ndarray, max_faces: int = 20000):
    """Vertex-clustering decimation to at most ``max_faces`` triangles (the grid cell grows until it fits).

    ``signed_distance`` is brute force over every triangle; a 660k-face prop (BimanualLiftTray's food) made one
    stage-2 run take over an hour. Cell size = the approximation error (a few mm for the meshes that need it).
    """
    if len(faces) <= max_faces:
        return verts, faces
    ext = float(np.ptp(verts, axis=0).max())
    cell = ext / 256.0
    while True:
        key = np.floor((verts - verts.min(0)) / cell).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        f = inv[faces]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        f = np.unique(f, axis=0)
        if len(f) <= max_faces:
            n = inv.max() + 1
            v = np.zeros((n, 3))
            np.add.at(v, inv, verts)
            v /= np.bincount(inv, minlength=n)[:, None]
            return v.astype(verts.dtype), f.astype(faces.dtype)
        cell *= 1.5


def signed_distance(p, verts, faces):
    """Signed distance (negative inside) of p (M,3) to a mesh given in the same frame."""
    tri = verts[faces]                                              # (F,3,3)
    d, cpt, f = closest_on_triangles(p, tri)
    n = torch.cross(tri[f, 1] - tri[f, 0], tri[f, 2] - tri[f, 0], dim=-1)
    inside = ((p - cpt) * n).sum(-1) < 0
    return torch.where(inside, -d, d), cpt

def collision_samples(stage, body_prim_paths: list[str], per_prim: int = 64, seed: int = 0) -> list[np.ndarray | None]:
    """Surface sample points (body frame, scale baked) of every body's COLLISION geometry.

    What PhysX collides with is what matters for a penetration term: a hand link's collision prim is often a
    primitive (Allegro: boxes, Shadow: cylinders / spheres, Inspire: boxes + a cylinder) that is fatter than its
    visual mesh, and ``body_meshes`` (UsdGeom.Mesh only) misses it. Prims with ``UsdPhysics.CollisionAPI`` are
    used; a body without any falls back to its meshes. Mesh prims: ``per_prim`` random vertices; Cube: corners,
    edge midpoints, face centres and a grid on each face; Sphere / Cylinder / Capsule / Cone: rings.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    rng = np.random.default_rng(seed)
    cache = UsdGeom.XformCache()

    def ring(n, r, z):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([r * np.cos(a), r * np.sin(a), np.full(n, z)], 1)

    def primitive_points(p):
        t = p.GetTypeName()
        g = UsdGeom.Gprim(p)
        if t == "Cube":
            h = 0.5 * float(UsdGeom.Cube(p).GetSizeAttr().Get() or 2.0)
            u = np.linspace(-1, 1, 5)
            pts = [np.array([x, y, s * 1.0]) for s in (-1, 1) for x in u for y in u]          # faces z
            pts += [np.array([x, s * 1.0, z]) for s in (-1, 1) for x in u for z in u]         # faces y
            pts += [np.array([s * 1.0, y, z]) for s in (-1, 1) for y in u for z in u]         # faces x
            return np.unique(np.array(pts), axis=0) * h
        axis_attr = p.GetAttribute("axis")
        axis = (axis_attr.Get() if axis_attr else "Z") or "Z"
        if t == "Sphere":
            r = float(UsdGeom.Sphere(p).GetRadiusAttr().Get() or 1.0)
            pts = np.concatenate([ring(12, r * np.cos(e), r * np.sin(e)) for e in np.linspace(-1.2, 1.2, 5)]
                                 + [np.array([[0, 0, r], [0, 0, -r]])])
        elif t in ("Cylinder", "Capsule", "Cone"):
            geom = {"Cylinder": UsdGeom.Cylinder, "Capsule": UsdGeom.Capsule, "Cone": UsdGeom.Cone}[t](p)
            r = float(geom.GetRadiusAttr().Get() or 1.0)
            hgt = float(geom.GetHeightAttr().Get() or 2.0)
            zs = np.linspace(-hgt / 2, hgt / 2, 5)
            pts = [ring(12, r if t != "Cone" else r * (0.5 - z / hgt), z) for z in zs]
            if t == "Capsule":
                pts += [ring(12, r * np.cos(e), hgt / 2 + r * np.sin(e)) for e in (0.5, 1.0)]
                pts += [ring(12, r * np.cos(e), -hgt / 2 - r * np.sin(e)) for e in (0.5, 1.0)]
                pts += [np.array([[0, 0, hgt / 2 + r], [0, 0, -hgt / 2 - r]])]
            else:
                pts += [np.array([[0, 0, hgt / 2], [0, 0, -hgt / 2]])]
            pts = np.concatenate(pts)
        else:
            return None
        if axis == "X":
            pts = pts[:, [2, 1, 0]] * np.array([1, 1, -1])
        elif axis == "Y":
            pts = pts[:, [0, 2, 1]] * np.array([1, 1, -1])
        return pts

    out = []
    for path in body_prim_paths:
        body = stage.GetPrimAtPath(path)
        if not body.IsValid():
            out.append(None)
            continue
        T = Gf.Transform(cache.GetLocalToWorldTransform(body))
        inv = Gf.Matrix4d(T.GetRotation(), T.GetTranslation()).GetInverse()
        coll, meshes = [], []
        it = iter(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
        for p in it:
            if p != body and p.HasAPI(UsdPhysics.RigidBodyAPI):
                it.PruneChildren()
                continue
            is_coll = p.HasAPI(UsdPhysics.CollisionAPI)
            if p.IsA(UsdGeom.Mesh):
                pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
                if not pts:
                    continue
                P = np.asarray(pts, dtype=np.float64)
                P = P[rng.choice(len(P), min(len(P), per_prim), replace=False)]
            elif p.GetTypeName() in ("Cube", "Sphere", "Cylinder", "Capsule", "Cone"):
                P = primitive_points(p)
                if P is None:
                    continue
            else:
                continue
            M = np.array(cache.GetLocalToWorldTransform(p) * inv)
            P = P @ M[:3, :3] + M[3, :3]
            (coll if is_coll else meshes).append(P)
        src = coll or meshes
        out.append(np.concatenate(src).astype(np.float32) if src else None)
    return out
