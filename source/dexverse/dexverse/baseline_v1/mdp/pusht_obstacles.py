"""Seeded goal-local Push-T rods with a collision-checked XY/yaw path witness.

No Isaac imports: tests use the actual two-bar footprint, not a bounding disk.
"""

from functools import lru_cache
import heapq
import math

import numpy as np
import torch

# tee.usda: x=[-.1,.1], y=[-.0625,.1375], relative to the recorded root.
TEE_XY_BOUNDS = ((-0.10, 0.10), (-0.0625, 0.1375))
TEE_RECT_CENTERS = np.array([[0., -.0375], [0., .0625]])
TEE_RECT_HALF_SIZES = np.array([[.10, .025], [.025, .075]])
TEE_CORNERS = np.concatenate([
    center + half * np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    for center, half in zip(TEE_RECT_CENTERS, TEE_RECT_HALF_SIZES)
])
TEE_BOUNDING_BOX_DIAGONAL = math.hypot(0.20, 0.20)
TEE_ROOT_RADIUS = math.hypot(0.10, 0.1375)
PATH_MARGIN = 0.015
TEE_CLEARANCE_RADIUS = TEE_ROOT_RADIUS + PATH_MARGIN
ROD_RADIUS = 0.018
ROD_HEIGHT = 0.04
ROD_COUNT = 10
ROD_NAMES = tuple(f"push_t_rod_{i}" for i in range(ROD_COUNT))
ROD_COLOR = (1.0, 0.32, 0.015)
SEPARATION_SLACK = 0.01
ROD_MIN_SURFACE_GAP = 0.15
ROD_MIN_SPACING = ROD_MIN_SURFACE_GAP + 2 * ROD_RADIUS
GOAL_ROD_RADII = (0.20, 0.40)
# Require rods on all sides, without prescribing a ring/grid or a fixed entry.
MAX_ROD_ANGULAR_GAP = 2 * math.pi / 3
ROD_EDGE_INSET = ROD_RADIUS + 0.02
START_GOAL_MIN_DISTANCE = 2 * TEE_ROOT_RADIUS + SEPARATION_SLACK
PATH_GRID_STEP = 0.03
PATH_YAW_BINS = 24
PATH_SWEEP_STEP = 0.006
DIRECT_BLOCK_DISTANCE = ROD_RADIUS - 0.003


def segment_distances(start, end, points):
    """Distances from points (..., K, 2) to segments (..., 2), with projection."""
    delta = end - start
    t = ((points - start[..., None, :]) * delta[..., None, :]).sum(-1) / np.maximum((delta * delta).sum(-1)[..., None], 1e-12)
    nearest = start[..., None, :] + np.clip(t, 0, 1)[..., None] * delta[..., None, :]
    return np.linalg.norm(points - nearest, axis=-1), t


def direct_path_blocked(start, goal, rods):
    """A rod intersects the root trajectory itself, not just an inflated bound.

    The root lies inside the T at every yaw. Requiring its straight XY segment
    to penetrate a rod blocks straight translation even with in-place rotation.
    """
    distance, t = segment_distances(np.asarray(start), np.asarray(goal), np.asarray(rods))
    return np.any((distance < DIRECT_BLOCK_DISTANCE) & (t > .2) & (t < .8), axis=-1)


def angle_delta(start, end):
    """Shortest signed angle, including connections across the 0/2pi seam."""
    return (end - start + math.pi) % (2 * math.pi) - math.pi


def pose_clearance(poses, rods, *, table_size=(1.5, 1.5), table_center=(0., 0.)):
    """Minimum T-to-rod surface / T-to-table-edge clearance for (..., 3) poses.

    Transform rod centres into the T frame, then use exact point-to-rectangle
    distance for each of the two USD collision boxes. All box corners determine
    table containment. Negative clearance means collision or overhang.
    """
    poses, rods = np.asarray(poses), np.asarray(rods).reshape(-1, 2)
    c, s = np.cos(poses[..., 2]), np.sin(poses[..., 2])
    rotation = np.stack((c, -s, s, c), axis=-1).reshape(*c.shape, 2, 2)
    corners = np.einsum("...ij,kj->...ki", rotation, TEE_CORNERS) + poses[..., None, :2]
    edge = (np.asarray(table_size) / 2 - np.abs(corners - table_center)).min(axis=(-1, -2))
    if not len(rods):
        return edge
    local = np.einsum("...ji,...kj->...ki", rotation, rods - poses[..., None, :2])
    outside = np.maximum(np.abs(local[..., :, None, :] - TEE_RECT_CENTERS) - TEE_RECT_HALF_SIZES, 0.)
    obstacle = np.linalg.norm(outside, axis=-1).min(axis=(-1, -2)) - ROD_RADIUS
    return np.minimum(edge, obstacle)


def motion_samples(start, end):
    """Sample XY / shortest-yaw interpolation and bound inter-sample motion."""
    delta = np.asarray(end) - start
    delta[2] = angle_delta(start[2], end[2])
    travel = np.linalg.norm(delta[:2]) + TEE_ROOT_RADIUS * abs(delta[2])
    steps = max(1, int(np.ceil(travel / PATH_SWEEP_STEP)))
    return np.asarray(start) + np.linspace(0., 1., steps + 1)[:, None] * delta, travel / (2 * steps)


def motion_is_clear(start, end, rods, *, table_size=(1.5, 1.5), table_center=(0., 0.)):
    """Certify the full swept footprint, not only endpoint/sample collisions.

    Distance changes by at most translation + root radius * yaw change. An
    extra half-sample travel margin bounds every point between sample poses.
    """
    poses, sweep_margin = motion_samples(start, end)
    return bool(np.all(pose_clearance(poses, rods, table_size=table_size, table_center=table_center)
                       >= PATH_MARGIN + sweep_margin))


@lru_cache(maxsize=8)
def _pose_grid(table_size, table_center):
    half = np.asarray(table_size) / 2 - PATH_MARGIN
    axes = [np.linspace(table_center[i] - half[i], table_center[i] + half[i],
                        int(np.ceil(2 * half[i] / PATH_GRID_STEP)) + 1) for i in range(2)]
    angles = np.arange(PATH_YAW_BINS) * (2 * math.pi / PATH_YAW_BINS)
    nodes = np.stack(np.meshgrid(*axes, angles, indexing="ij"), axis=-1)
    return nodes.reshape(-1, 3), nodes.shape[:3]


def find_clear_path(start, goal, rods, *, table_size=(1.5, 1.5), table_center=(0., 0.)):
    """A* over XY/yaw using the actual T, returning (N, 3) poses or None.

    Translation and in-place turning edges are swept-collision checked, as are
    exact endpoint connectors. A failed finite-grid search rejects a layout;
    it does not prove that the continuous problem is impossible.
    """
    start, goal, rods = map(np.asarray, (start, goal, rods))
    geometry = dict(table_size=table_size, table_center=table_center)
    if np.any(np.asarray(table_size) <= 2 * PATH_MARGIN):
        return None
    if np.any(pose_clearance(np.stack((start, goal)), rods, **geometry) < PATH_MARGIN):
        return None
    flat, (nx, ny, na) = _pose_grid(tuple(table_size), tuple(table_center))
    clearances = np.concatenate([pose_clearance(chunk, rods, **geometry) for chunk in np.array_split(flat, 16)])
    safe = clearances >= PATH_MARGIN

    def cost(poses, point):
        return np.linalg.norm(poses[..., :2] - point[:2], axis=-1) + TEE_ROOT_RADIUS * np.abs(angle_delta(poses[..., 2], point[2]))

    def connectors(point):
        near = np.flatnonzero(safe & (np.linalg.norm(flat[:, :2] - point[:2], axis=-1) <= 1.5 * PATH_GRID_STEP)
                              & (np.abs(angle_delta(flat[:, 2], point[2])) <= 2 * math.pi / na))
        return [node for node in near if motion_is_clear(point, flat[node], rods, **geometry)]

    source, target = connectors(start), set(connectors(goal))
    if not source or not target:
        return None
    heuristic = cost(flat, goal)
    parents = np.full(len(flat), -1, dtype=np.int32)
    best = np.full(len(flat), np.inf)
    closed = np.zeros(len(flat), dtype=bool)
    queue = []
    for node in source:
        best[node] = cost(flat[node], start)
        heapq.heappush(queue, (best[node] + heuristic[node], int(node)))
    while queue:
        _, node = heapq.heappop(queue)
        if closed[node]:
            continue
        closed[node] = True
        if node in target:
            chain = []
            while node != -1:
                chain.append(flat[node])
                node = parents[node]
            return np.vstack((start, chain[::-1], goal))
        xy, a = divmod(node, na)
        x, y = divmod(xy, ny)
        neighbors = [(x, y, (a + da) % na) for da in (-1, 1)]
        neighbors.extend((x + dx, y + dy, a) for dx, dy in
                         ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)))
        for ix, iy, ia in neighbors:
            if not (0 <= ix < nx and 0 <= iy < ny):
                continue
            other = (ix * ny + iy) * na + ia
            if closed[other] or not safe[other]:
                continue
            travel = cost(flat[other], flat[node])
            candidate = best[node] + travel
            if candidate >= best[other]:
                continue
            # Endpoint clearance can certify an entire edge without sampling.
            if min(clearances[node], clearances[other]) < PATH_MARGIN + travel / 2:
                if not motion_is_clear(flat[node], flat[other], rods, **geometry):
                    continue
            best[other], parents[other] = candidate, node
            heapq.heappush(queue, (candidate + heuristic[other], other))
    return None


def sample_layouts(count, *, device="cpu", table_size=(1.5, 1.5), table_center=(0., 0.),
                   spawn_x=(-0.406, 0.094), spawn_y=(-0.35, 0.15), max_rounds=256):
    """Return [start x,y,yaw; goal x,y,yaw; ten rod x,y] for each env.

    Uniform-area annulus proposals surround the goal, with 15 cm surface gaps.
    Uses Torch's seeded RNG; geometry is deterministic NumPy. Failed bounded
    rejection raises explicitly, never relaxing spacing or reachability.
    """
    if count < 0 or max_rounds < 1 or any(hi <= lo for lo, hi in (spawn_x, spawn_y)):
        raise ValueError("Invalid Push-T layout sampling arguments")
    half = np.asarray(table_size) / 2
    if np.any(half <= 2 * TEE_CLEARANCE_RADIUS):
        raise ValueError("Table too small for T-sized obstacle/edge clearance")
    center = np.asarray(table_center)
    low = np.array([spawn_x[0], spawn_y[0]]) + center
    high = np.array([spawn_x[1], spawn_y[1]]) + center
    safe_half = half - TEE_CLEARANCE_RADIUS
    rod_half = half - ROD_EDGE_INSET
    geometry = dict(table_size=table_size, table_center=table_center)
    result = np.empty((count, 6 + 2 * ROD_COUNT), dtype=np.float32)
    for row in range(count):
        for _ in range(max_rounds):
            rand_goal = torch.rand(3, device=device).cpu().numpy()
            goal = np.r_[low + rand_goal[:2] * (high - low), rand_goal[2] * 2 * math.pi]
            if np.any(np.abs(goal[:2] - center) > safe_half):
                continue
            rand_rods = torch.rand(ROD_COUNT, 256, 2, device=device).cpu().numpy()
            radius = np.sqrt(GOAL_ROD_RADII[0]**2 + rand_rods[..., 0] * (GOAL_ROD_RADII[1]**2 - GOAL_ROD_RADII[0]**2))
            angle = rand_rods[..., 1] * 2 * math.pi
            proposals = goal[:2] + radius[..., None] * np.stack((np.cos(angle), np.sin(angle)), axis=-1)
            rods = []
            for candidates in proposals:
                valid = (np.abs(candidates - center) <= rod_half).all(-1)
                if rods:
                    valid &= (np.linalg.norm(candidates[:, None, :] - np.asarray(rods)[None, :, :], axis=-1) >= ROD_MIN_SPACING).all(-1)
                available = np.flatnonzero(valid)
                if not len(available):
                    break
                rods.append(candidates[available[0]])
            if len(rods) != ROD_COUNT:
                continue
            rods = np.asarray(rods)
            bearings = np.sort(np.arctan2(rods[:, 1] - goal[1], rods[:, 0] - goal[0]))
            if np.diff(np.r_[bearings, bearings[0] + 2 * math.pi]).max() > MAX_ROD_ANGULAR_GAP:
                continue
            if pose_clearance(goal, rods, **geometry) < PATH_MARGIN:
                continue
            rand = torch.rand(256, 3, device=device).cpu().numpy()
            starts = np.column_stack((low + rand[:, :2] * (high - low), rand[:, 2] * 2 * math.pi))
            valid = (np.abs(starts[:, :2] - center) <= safe_half).all(-1)
            valid &= np.linalg.norm(starts[:, :2] - goal[:2], axis=-1) >= START_GOAL_MIN_DISTANCE
            valid &= pose_clearance(starts, rods, **geometry) >= PATH_MARGIN + SEPARATION_SLACK
            valid &= direct_path_blocked(starts[:, :2], goal[:2], rods)
            chosen = None
            for index in np.flatnonzero(valid):
                if find_clear_path(starts[index], goal, rods, **geometry) is not None:
                    chosen = index
                    break
            if chosen is not None:
                result[row] = np.concatenate((starts[chosen], goal, rods.ravel()))
                break
        else:
            raise RuntimeError("Could not sample collision-free Push-T layout within budget; enlarge table/spawn region, never weaken clearance")
    return torch.as_tensor(result, device=device)


def reset_push_t_with_rods(env, env_ids, *, table_size=(1.5, 1.5), table_center=(0., 0.),
                         spawn_x=(-0.406, 0.094), spawn_y=(-0.35, 0.15)):
    """Jointly reset tee, goal, rods; their poses are ordinary recorded scene state."""
    if env_ids is None:
        ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if not len(ids):
        return
    samples = sample_layouts(len(ids), device=env.device, table_size=table_size,
                             table_center=table_center, spawn_x=spawn_x, spawn_y=spawn_y)
    origins = env.scene.env_origins[ids]
    for name, xy, yaw, dynamic in [
        ("object", samples[:, :2], samples[:, 2], True),
        ("goal_tee", samples[:, 3:5], samples[:, 5], False),
        *[(name, samples[:, 6 + i*2:8 + i*2], torch.zeros(len(ids), device=env.device), False)
          for i, name in enumerate(ROD_NAMES)],
    ]:
        asset = env.scene[name]
        pose = asset.data.default_root_state[ids, :7].clone()
        pose[:, :2] = xy
        pose[:, :3] += origins
        pose[:, 3:] = 0
        pose[:, 3] = torch.cos(yaw / 2)
        pose[:, 6] = torch.sin(yaw / 2)
        asset.write_root_pose_to_sim(pose, env_ids=ids)
        if dynamic:
            asset.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


def rod_positions_in_table(env):
    """Three coordinates per rod for state policies; also visible in RGB."""
    table = env.scene["table"].data.root_pos_w
    return torch.cat([env.scene[name].data.root_pos_w - table for name in ROD_NAMES], dim=-1)
