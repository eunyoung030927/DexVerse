#!/usr/bin/env python
"""handmask (261115): online hand masking for the cv2dp eval (new file; the target eval script is NOT modified).

  python.sh hm_eval_masked.py --mask_hand eye_in_hand,third_person [--mask_dilate 2] [--mask_target <eval .py>]
            [--mask_examples 3] <all usual arguments of the target eval>

Loads the target eval script (default xhand_latent/cv2dp_eval_pooled.py) as a module (its argparse / AppLauncher run
with the remaining argv, exactly as when it is launched directly), then patches two things before calling its main():
  1. gymnasium.make: the cameras of the masked views get the instance_id_segmentation_fast annotator (hm_seg.add_segmentation;
     rgb settings unchanged), so the mask comes from the same render as the rgb image.
  2. observe(): after the target's own observe() builds the message, the hand pixels (instance prim under the env Robot
     prim, any hand incl. sharpa; dilated --mask_dilate px) of each masked view are set to 0 in the uint8 image bytes.
Same segmentation rule and dilation as the training masks (hm_render_masks.py + hm_build_masked.py).
Without --mask_hand (or with 'none') nothing is patched and the run is identical to the target script.
--mask_examples N: save the first N masked observations of the first episode as PNG to <output_dir>/handmask_examples/.
A summary (views, dilation, per-view mean masked fraction over all steps) goes to <output_dir>/handmask_eval.json.
"""
import json
import os
import sys
import types

LC = "/workspace/xhand/latent_codec_v1"
VIEW_KEY = {"eye_in_hand": "observation.images.eye_in_hand", "third_person": "observation.images.third_person"}


def _pop(argv, name, default=None):
    if name in argv:
        i = argv.index(name)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    for i, a in enumerate(argv):
        if a.startswith(name + "="):
            del argv[i]
            return a.split("=", 1)[1]
    return default


argv = sys.argv[1:]
MASK = _pop(argv, "--mask_hand", "none")
DILATE = int(_pop(argv, "--mask_dilate", "2"))
TARGET = _pop(argv, "--mask_target", f"{LC}/xhand_latent/cv2dp_eval_pooled.py")
N_EX = int(_pop(argv, "--mask_examples", "3"))
views = [] if MASK in ("", "none") else MASK.split(",")
for v in views:
    if v not in VIEW_KEY:
        sys.exit(f"--mask_hand: unknown view {v!r} (choose from {list(VIEW_KEY)})")
sys.argv = [TARGET] + argv
sys.path.insert(0, LC)
sys.path.insert(0, os.path.dirname(TARGET))

# --- load the target eval as a module (no __main__ block): runs its argparse + AppLauncher with the remaining argv ---
mod = types.ModuleType("hm_target_eval")
mod.__file__ = TARGET
with open(TARGET) as f:
    code = compile(f.read(), TARGET, "exec")
exec(code, mod.__dict__)  # noqa: S102
for need in ("observe", "main", "make_env", "simulation_app", "args"):
    if need not in mod.__dict__:
        sys.exit(f"{TARGET} has no {need}; the handmask hook expects the p3h/ho/cv2dp eval layout")

STATS = {"views": views, "dilate_px": DILATE, "target": TARGET, "steps": 0, "masked_frac_sum": {v: 0.0 for v in views},
         "labels": {}}

if views:
    import gymnasium
    import numpy as np

    from xhand_latent.hm_seg import add_segmentation, dilate, hand_mask_from_camera

    _make = gymnasium.make
    _keymap = {}  # dataset key -> camera name, filled by the patched make

    def make_with_seg(task, cfg=None, **kw):
        if cfg is not None:
            cams = mod.camera_keys(cfg)  # same key->camera mapping the target eval uses
            for v in views:
                name = cams[VIEW_KEY[v]]
                add_segmentation(getattr(cfg.scene, name))
                _keymap[VIEW_KEY[v]] = name
        return _make(task, cfg=cfg, **kw)

    gymnasium.make = make_with_seg
    _observe = mod.observe
    ex_dir = os.path.join(mod.args.output_dir, "handmask_examples")

    def observe_masked(env, keys, order, refresh):
        msg = _observe(env, keys, order, refresh)  # renders / updates the cameras (rgb + segmentation together)
        for v in views:
            key = VIEW_KEY[v]
            cam = env.scene[keys[key]]
            m, labels = hand_mask_from_camera(cam)
            m = dilate(m, DILATE).cpu().numpy()
            shape, dtype, data = msg[key]
            img = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            assert img.shape[:2] == m.shape, (img.shape, m.shape)
            img[m] = 0
            msg[key] = (img.shape, "uint8", img.tobytes())
            STATS["masked_frac_sum"][v] += float(m.mean())
            if not STATS["labels"].get(v):
                STATS["labels"][v] = sorted(set(labels))[:40]
            if STATS["steps"] < N_EX:
                from PIL import Image
                os.makedirs(ex_dir, exist_ok=True)
                Image.fromarray(img).save(os.path.join(ex_dir, f"step{STATS['steps']:02d}_{v}.png"))
        STATS["steps"] += 1
        return msg

    mod.observe = observe_masked


def _write_summary():
    out = dict(STATS)
    out["masked_frac_mean"] = {v: STATS["masked_frac_sum"][v] / max(1, STATS["steps"]) for v in views}
    os.makedirs(mod.args.output_dir, exist_ok=True)
    with open(os.path.join(mod.args.output_dir, "handmask_eval.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    print(f"[handmask] views={views or 'none'} dilate={DILATE}px target={TARGET}", flush=True)
    try:
        mod.main()
        _write_summary()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        mod.simulation_app.close()
