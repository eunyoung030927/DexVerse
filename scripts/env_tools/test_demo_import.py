"""CPU regressions for non-destructive NumPy-compatible demo import."""

import importlib.util
import io
import pickle
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "demo_tools/import_versioned_demos.py"
spec = importlib.util.spec_from_file_location("demo_import", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def session():
    return {
        "format": "dexbench_trajectory",
        "task": "Dexbench-GraspCup-v0",
        "env_name": "Dexbench-GraspCup-v0",
        "robot_type": "floating_shadow_right",
        "num_episodes": 1,
        "episodes": [
            {
                "actions": np.arange(56, dtype=np.float32).reshape(2, 28),
                "success": True,
                "states": [{}, {}, {}],
                "initial_state": {},
            }
        ],
    }


def test_numpy2_module_name_remaps_without_changing_array_values():
    payload = session()
    # Protocol 0 has newline-delimited module names and no framed byte lengths.
    raw = pickle.dumps(payload, protocol=0).replace(b"cnumpy.core.multiarray\n", b"cnumpy._core.multiarray\n")
    loader = module.NumpyCompatUnpickler(io.BytesIO(raw))
    restored = loader.load()
    assert loader.remapped
    assert module.array_digest(restored) == module.array_digest(payload)
    normalized = pickle.loads(pickle.dumps(restored, protocol=4))
    assert module.array_digest(normalized) == module.array_digest(payload)


def test_unsupported_pickle_constructor_is_rejected():
    with pytest.raises(pickle.UnpicklingError, match="Unsupported"):
        module.NumpyCompatUnpickler(io.BytesIO(b"cos\nsystem\n.")).load()


def test_legacy_byte_constructor_only_accepts_latin1():
    assert module._latin1_encode("\u00ff") == b"\xff"
    with pytest.raises(pickle.UnpicklingError, match="latin1"):
        module._latin1_encode("data", "utf-8")


def test_import_preserves_source_and_arrays_and_does_not_overwrite(tmp_path):
    source = tmp_path / "Dexbench-GraspCup-v0.pkl"
    source.write_bytes(pickle.dumps(session(), protocol=4))
    original = source.read_bytes()
    item = {
        "source": str(source),
        "target_task": "Dexverse-GraspCup-v0",
        "version": 0,
        "collection": "test",
        "sha256": module.sha256(source),
    }
    dest = tmp_path / "demonstrations"
    result = module.convert(item, source, dest)
    assert result["status"] == "imported" and not result["review_reasons"]
    assert source.read_bytes() == original
    output = dest / result["output"]
    data = pickle.loads(output.read_bytes())
    assert data["task"] == "Dexverse-GraspCup-v0" and data["task_version"] == 0
    assert data["task_source_revision"] is None
    assert data["demo_import"]["original_identity"]["task"] == "Dexbench-GraspCup-v0"
    np.testing.assert_array_equal(data["episodes"][0]["actions"], session()["episodes"][0]["actions"])
    before = output.read_bytes()
    assert module.convert(item, source, dest)["status"] == "already_imported"
    assert output.read_bytes() == before
    modified = session()
    modified["episodes"][0]["actions"] *= 2
    source.write_bytes(pickle.dumps(modified, protocol=4))
    item["sha256"] = module.sha256(source)
    with pytest.raises(FileExistsError):
        module.convert(item, source, dest)
    assert output.read_bytes() == before


def test_review_recordings_are_not_in_usable_version_tree(tmp_path):
    source = tmp_path / "prototype.pkl"
    payload = session()
    payload["episodes"][0]["states"] = []
    source.write_bytes(pickle.dumps(payload, protocol=4))
    result = module.convert(
        {"source": str(source), "target_task": "Dexverse-GraspCup-v0", "version": 0, "collection": "test"},
        source,
        tmp_path / "demos",
    )
    assert result["output"].startswith("_review/v0/")
    assert "missing_or_misaligned_recorded_states" in result["review_reasons"]


def test_interrupted_serialization_never_publishes_partial_output(tmp_path, monkeypatch):
    source = tmp_path / "session.pkl"
    source.write_bytes(pickle.dumps(session(), protocol=4))
    dest = tmp_path / "demos"

    def fail_dump(payload, stream, protocol):
        stream.write(b"partial")
        raise OSError("simulated interruption")

    monkeypatch.setattr(module.pickle, "dump", fail_dump)
    with pytest.raises(OSError, match="simulated interruption"):
        module.convert(
            {"source": str(source), "target_task": "Dexverse-GraspCup-v0", "version": 0, "collection": "test"},
            source,
            dest,
        )
    assert not list(dest.rglob("*.pkl"))
    assert not list(dest.rglob("*.tmp"))


def test_review_recording_requires_explicit_replay_decision():
    from dexverse.benchmark import validate_replay_identity

    payload = {
        "task": "Dexverse-GraspCup-v1",
        "demo_import": {"compatibility": "needs_review", "review_reasons": ["prototype_scene"]},
    }
    with pytest.raises(ValueError, match="needs review"):
        validate_replay_identity(payload, payload["task"])
    validate_replay_identity(payload, payload["task"], explicit_override=True)


def test_bulk_replay_discovery_skips_review_tree(tmp_path):
    import sys
    sys.path.insert(0, str(SCRIPT.parent))
    from demo_selection import discover_groups
    usable = tmp_path / "v1/functional/task/session.pkl"
    review = tmp_path / "_review/v1/functional/task/session.pkl"
    for path in (usable, review):
        path.parent.mkdir(parents=True)
        path.touch()
    groups, _ = discover_groups(tmp_path, all_tasks=True, legacy_collections=True)
    assert [path for group in groups for path in group.pickles] == [usable]
