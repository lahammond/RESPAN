"""Unit tests for associate_spines_with_necks_per_component.

Regression guard: the previous implementation computed connected components
over (spines | necks) and silently dropped any component that didn't contain
a spine voxel. This dropped every nnU-Net neck fragment that sat in a
segmentation gap from its parent spine — the root cause of the 24% neck
preservation on T7.

The rewritten implementation computes CC(necks) alone and assigns each
fragment to its nearest spine via mode-vote within a dilated bbox, gated by
an anisotropic µm cap.

Run with:
    python -m pytest tests/test_neck_association.py -v

Or as a standalone script (prints PASS/FAIL):
    python tests/test_neck_association.py
"""

import logging
import numpy as np
import sys
from pathlib import Path

try:
    import pytest
except ImportError:               # standalone-script mode — see __main__ block
    class _MarkStub:
        def __getattr__(self, _name):
            return lambda fn: fn  # markers degrade to no-op decorators

    class _PytestStub:
        mark = _MarkStub()

    pytest = _PytestStub()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RESPAN.ImageAnalysis.ChunkedProcessing import (
    associate_spines_with_necks_per_component,
    bridge_spine_neck_gaps,
)


def _make_logger():
    lg = logging.getLogger("neck-test")
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    return lg


@pytest.mark.gpu
def test_disconnected_neck_preserved():
    """nnU-Net neck fragment in a segmentation gap MUST be assigned to the nearest spine.

    Synthetic 3D volume with:
      - A 3x3x3 spine block at z=10
      - A disconnected 2x2x2 neck fragment at z=14 (4 voxels of gap between them)
    Expected: after association, the neck fragment has the spine's label (1).
    Regression: old CC(spines|necks) code would put them in separate components
    and silently drop the neck fragment.
    """
    shape = (32, 32, 32)
    spines = np.zeros(shape, dtype=np.uint32)
    necks = np.zeros(shape, dtype=np.uint32)

    # Spine label = 1 at z=10..12
    spines[10:13, 15:18, 15:18] = 1

    # Neck fragment at z=14..15 — disconnected from the spine by z=13 (empty).
    necks[14:16, 16:18, 16:18] = 1

    result = associate_spines_with_necks_per_component(
        spines, necks, _make_logger(),
        spine_dist_um=2.0, input_resXY=0.065, input_resZ=0.15,
    )

    assigned = result[14:16, 16:18, 16:18]
    assert np.all(assigned == 1), (
        f"Disconnected neck fragment dropped or mislabeled. "
        f"Expected all 1s, got unique values: {np.unique(assigned)}"
    )


@pytest.mark.gpu
def test_far_neck_dropped_as_noise():
    """Neck fragment > spine_dist × 3 µm from every spine is dropped (noise)."""
    shape = (64, 64, 64)
    spines = np.zeros(shape, dtype=np.uint32)
    necks = np.zeros(shape, dtype=np.uint32)

    # Spine at one corner.
    spines[5:8, 5:8, 5:8] = 1
    # Neck fragment at the opposite corner — far beyond 6 µm at 0.065 µm XY spacing.
    # Corner-to-corner distance ~55 * 0.065 = 3.6 µm in XY, 55 * 0.15 = 8.25 µm in Z
    # → total 3D > 9 µm, well beyond the 6 µm cap.
    necks[58:61, 58:61, 58:61] = 1

    result = associate_spines_with_necks_per_component(
        spines, necks, _make_logger(),
        spine_dist_um=2.0, input_resXY=0.065, input_resZ=0.15,
    )

    # Neck voxels should stay unassigned (value 0) — the dilated bbox
    # of the neck CC never sees the spine.
    dropped = result[58:61, 58:61, 58:61]
    assert np.all(dropped == 0), (
        f"Far neck fragment should be dropped as noise. "
        f"Got unique values: {np.unique(dropped)}"
    )
    # Spine voxels preserved.
    assert np.all(result[5:8, 5:8, 5:8] == 1)


@pytest.mark.gpu
def test_chain_fragments_assigned_via_progressive_passes():
    """Mid-chain fragments are picked up after the corridor is extended.

    We simulate the two-pass flow by hand:
      Pass 1: reference=spines → assigns only near-spine fragments.
      (extend_objects_GPU would bridge through the mid-chain here in the full pipeline.)
      Pass 2: reference = pass1_result | bridge → should assign mid-chain.

    This test asserts Pass 1 preserves the near-spine fragment and leaves the
    far fragment unassigned, which is the prerequisite for Pass 2 rescue.
    """
    shape = (48, 48, 48)
    spines = np.zeros(shape, dtype=np.uint32)
    necks = np.zeros(shape, dtype=np.uint32)

    # Spine at z=10
    spines[9:12, 20:23, 20:23] = 1
    # Near-spine fragment at z=13 (within 3 µm at Z=0.15 µm)
    necks[13:15, 20:22, 20:22] = 1
    # Mid-chain fragment at z=22 (further; depends on config whether in cap)
    necks[22:24, 20:22, 20:22] = 1

    pass1 = associate_spines_with_necks_per_component(
        spines, necks, _make_logger(),
        spine_dist_um=1.0, input_resXY=0.1, input_resZ=0.3,
        # cap = 1 µm × 3 = 3 µm; Z distance from z=11 (spine top) to z=13 (near
        # frag top) = 2 * 0.3 = 0.6 µm → assigned. Spine top z=11 to z=22 (mid
        # frag) = 11 * 0.3 = 3.3 µm → dropped (slightly > cap).
    )

    # Near fragment assigned
    near = pass1[13:15, 20:22, 20:22]
    assert np.all(near == 1), f"Near-spine fragment lost: uniques={np.unique(near)}"

    # Mid-chain fragment NOT assigned in Pass 1 (would be rescued in Pass 2 after
    # extend_objects_GPU bridges through it).
    mid = pass1[22:24, 20:22, 20:22]
    assert np.all(mid == 0), (
        f"Mid-chain fragment should be left for Pass 2 rescue, but was assigned: "
        f"uniques={np.unique(mid)}"
    )


@pytest.mark.gpu
def test_mode_vote_handles_two_spines():
    """When a neck fragment sits between two spines, it is assigned whole-fragment
    to the spine that the majority of its voxels is nearest to (mode vote)."""
    shape = (32, 32, 64)
    spines = np.zeros(shape, dtype=np.uint32)
    necks = np.zeros(shape, dtype=np.uint32)

    # Two spines on opposite sides along X.
    spines[15:17, 15:17, 5:7] = 1     # spine 1 at x=5-7
    spines[15:17, 15:17, 57:59] = 2   # spine 2 at x=57-59

    # Neck fragment skewed toward spine 1 (x=10-20, so all voxels nearer spine 1).
    necks[15:17, 15:17, 10:21] = 1

    result = associate_spines_with_necks_per_component(
        spines, necks, _make_logger(),
        spine_dist_um=2.0, input_resXY=0.065, input_resZ=0.15,
    )

    assigned = result[15:17, 15:17, 10:21]
    uniq = np.unique(assigned)
    assert uniq.size == 1 and uniq[0] == 1, (
        f"Mode vote should have assigned whole fragment to spine 1. "
        f"Got uniques: {uniq}"
    )


def test_bridge_fills_1voxel_gap():
    """Spine head and assigned neck 1 voxel apart → bridge voxel fills the gap."""
    shape = (16, 16, 16)
    spines = np.zeros(shape, dtype=np.int32)
    necks = np.zeros(shape, dtype=np.int32)
    dendrites = np.zeros(shape, dtype=np.uint8)

    # Spine block at z=5..7, neck block at z=9..11 (gap at z=8).
    spines[5:8, 7:10, 7:10] = 7       # spine label = 7
    necks[9:12, 7:10, 7:10] = 7        # neck assigned to same spine (gap at z=8)

    out, _still_split = bridge_spine_neck_gaps(necks, spines, dendrites, _make_logger())

    # After closing, the gap at z=8 should be filled with label 7 in `necks`
    # (since spines retains its original values; bridge writes to necks).
    gap = out[8, 7:10, 7:10]
    assert np.all(gap == 7), f"Expected bridge label 7 at gap, got uniques {np.unique(gap)}"

    # Spine voxels unchanged.
    assert np.all(spines[5:8, 7:10, 7:10] == 7)


def test_bridge_skips_connected_spine():
    """Spine and neck already connected → no bridge needed, function is a no-op."""
    shape = (16, 16, 16)
    spines = np.zeros(shape, dtype=np.int32)
    necks = np.zeros(shape, dtype=np.int32)
    dendrites = np.zeros(shape, dtype=np.uint8)

    spines[5:8, 7:10, 7:10] = 3
    necks[8:11, 7:10, 7:10] = 3  # adjacent to spine — already connected

    before = necks.copy()
    out, still_split = bridge_spine_neck_gaps(necks, spines, dendrites, _make_logger())
    assert np.array_equal(out, before), "Connected spine should not be modified"
    assert still_split == [], f"No still-split labels expected, got {still_split}"


def test_bridge_respects_dendrite():
    """Bridge must NOT overwrite dendrite voxels even if they fill the gap."""
    shape = (16, 16, 16)
    spines = np.zeros(shape, dtype=np.int32)
    necks = np.zeros(shape, dtype=np.int32)
    dendrites = np.zeros(shape, dtype=np.uint8)

    spines[5:8, 7:10, 7:10] = 11
    necks[9:12, 7:10, 7:10] = 11
    # Dendrite is in the gap.
    dendrites[8, 7:10, 7:10] = 1

    out, _still_split = bridge_spine_neck_gaps(necks, spines, dendrites, _make_logger())

    # Gap voxels should stay 0 in necks (dendrite in the way, so not writable).
    gap = out[8, 7:10, 7:10]
    assert np.all(gap == 0), f"Dendrite voxels must not be overwritten, got {np.unique(gap)}"


def test_bridge_respects_other_spine():
    """Bridge must not overwrite another spine's label."""
    shape = (16, 24, 16)
    spines = np.zeros(shape, dtype=np.int32)
    necks = np.zeros(shape, dtype=np.int32)
    dendrites = np.zeros(shape, dtype=np.uint8)

    # Spine 1 with a gapped neck
    spines[5:8, 7:10, 7:10] = 1
    necks[9:12, 7:10, 7:10] = 1
    # A different spine (label 2) sits in the gap — must not be overwritten.
    spines[8, 7:10, 7:10] = 2

    out, _still_split = bridge_spine_neck_gaps(necks, spines, dendrites, _make_logger())

    gap = out[8, 7:10, 7:10]
    # Gap voxel is spine-2; our bridge shouldn't have written 1 on top of it.
    assert np.all(gap == 0), f"Other spine must not be overwritten, got necks={np.unique(gap)}"
    # Spine volume unchanged.
    assert np.all(spines[8, 7:10, 7:10] == 2)


if __name__ == "__main__":
    tests = [
        test_disconnected_neck_preserved,
        test_far_neck_dropped_as_noise,
        test_chain_fragments_assigned_via_progressive_passes,
        test_mode_vote_handles_two_spines,
        test_bridge_fills_1voxel_gap,
        test_bridge_skips_connected_spine,
        test_bridge_respects_dendrite,
        test_bridge_respects_other_spine,
    ]
    failed = []
    for t in tests:
        print(f"\n=== {t.__name__} ===")
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            print(f"FAIL: {t.__name__}\n  {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"ERROR: {t.__name__}\n  {type(e).__name__}: {e}")
            failed.append(t.__name__)

    print("\n" + "=" * 50)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)}: {failed}")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} TESTS PASSED")
        sys.exit(0)
