# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode deterministic reset seeding for demo recording.

The reset events (object / goal poses, materials) draw from the global torch / numpy / python RNGs. Seeding once at
start-up (``cfg.seed``) makes the *stream* reproducible, but how much of it is consumed before the k-th saved
episode depends on failed attempts, episode lengths and per-step randomisation, so the k-th episode of two
recording sessions (e.g. two hands) lands on different scenes (checked 2026-10-29 on GraspPan: five hands, same
seed 42, identical only at reset 0).

:func:`seed_for_episode` re-seeds every RNG right before a reset with a value that depends only on the base seed
and the index of the episode about to be recorded. Consequences:

* episode k has the same scene in every session / for every hand (same task, same base seed);
* a failed or manually RESET attempt is retried on the same scene (the index has not advanced);
* after a saved episode the next reset moves to the next scene.
"""

from __future__ import annotations

EPISODE_SEED_STRIDE = 1_000_003  # prime, so seeds of different base seeds never overlap for < 1e6 episodes
CURRENT_SEED: int | None = None  # the episode seed set by seed_for_episode (read by the per-term wrappers)


def episode_seed(base_seed: int | None, episode_index: int) -> int:
    base = 42 if base_seed is None or int(base_seed) < 0 else int(base_seed)
    return (base * EPISODE_SEED_STRIDE + int(episode_index)) % (2**31 - 1)


def seed_for_episode(env, base_seed: int | None, episode_index: int, log=print) -> int:
    """Seed torch / numpy / python (and warp / replicator through Isaac Lab's ``env.seed``) for ``episode_index``.

    Call immediately before ``env.reset()``. Returns the seed used."""
    global CURRENT_SEED
    seed = episode_seed(base_seed, episode_index)
    CURRENT_SEED = seed
    target = getattr(env, "unwrapped", env)
    if hasattr(target, "seed"):
        target.seed(seed)
    else:  # plain fallback
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if log is not None:
        log(f"[episode seed] episode {episode_index}: seed {seed} (base {base_seed})")
    return seed


def _term_seed(name: str) -> int:
    import zlib

    return (CURRENT_SEED + zlib.crc32(name.encode())) % (2**31 - 1)


def install_per_term_seeding(env, log=print) -> list[str]:
    """Give every *reset* event term its own deterministic RNG state: seed = episode seed + hash(term name).

    Without this the terms share one stream, and a term whose draw count depends on the robot (joint resets,
    material randomisation over the hand's bodies) shifts every later term: checked 2026-10-29 on GraspPan, the
    object pose matched across hands but the stove / goal poses did not. With per-term seeds each pose term is
    independent of the others and of the hand. Returns the wrapped term names."""
    import random

    import numpy as np
    import torch

    target = getattr(env, "unwrapped", env)
    em = getattr(target, "event_manager", None)
    if em is None or "reset" not in getattr(em, "_mode_term_cfgs", {}):
        return []
    names = list(em._mode_term_names["reset"])
    for name, cfg in zip(names, em._mode_term_cfgs["reset"]):
        if getattr(cfg.func, "_episode_seed_wrapped", False):
            continue
        inner = cfg.func

        def wrapped(*a, _inner=inner, _name=name, **kw):
            if CURRENT_SEED is not None:
                s = _term_seed(_name)
                random.seed(s)
                np.random.seed(s)
                torch.manual_seed(s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(s)
            return _inner(*a, **kw)

        wrapped._episode_seed_wrapped = True
        if hasattr(inner, "reset"):  # class-based terms: the manager calls term_cfg.func.reset(env_ids=...)
            wrapped.reset = inner.reset
        wrapped.__name__ = getattr(inner, "__name__", type(inner).__name__)
        cfg.func = wrapped
    if log is not None:
        log(f"[episode seed] reset event terms seeded individually: {names}")
    return names
