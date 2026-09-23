"""Unit tests for soma-label resolution on the chunked path.

Regression guard: the chunked slab scan resolved the soma label as
`channel_map.get(3, channel_map.get(5, -1))`. That map is keyed by LABEL VALUE and
holds CLASS NAMES, so the expression returned a name ('spine_membranes' for model
type 4, 'soma' for types 1 and 3, -1 for type 2) and the label array was compared
against a string. NumPy answers all-False without raising, so `soma_present` was
False on EVERY chunked run for EVERY model type.

Measured on T4 (500 MB, Dataset226, 2026-09-20): 1,289,594 soma voxels segmented,
soma reported absent. The cost was not only `euclidean_dist_to_soma` (0.0 before
v1.5.1, NaN after) - `soma_present=False` also nulls the soma volume passed to
`calculate_dend_ID_and_geo_distance`, so no soma KDTree is built and geodesic
distances are measured from the corner-most dendrite voxel while still being
reported as `geodesic_dist_to_soma`.

Introduced 2026-04-12 in eb5cd93 ("Label extraction to zarr slab-by-slab"), which
split one correct line, `soma_present = bool(np.max(soma) > 0)`, into a chunked
and a non-chunked branch. The non-chunked branch stayed correct.

Run with:
    python -m pytest tests/test_soma_detection.py -v
Or as a standalone script (prints PASS/FAIL):
    python tests/test_soma_detection.py
"""

import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:               # standalone-script mode — see __main__ block
    class _MarkStub:
        def __getattr__(self, _name):
            return lambda fn: fn

    class _PytestStub:
        mark = _MarkStub()

    pytest = _PytestStub()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RESPAN.ImageAnalysis.label_semantics import (   # noqa: E402
    LABEL_CHANNEL_MAP,
    soma_label_from_channel_map,
)

# The soma label RESPAN's model types actually use, independent of the map.
EXPECTED_SOMA_LABEL = {1: 3, 2: 2, 3: 3, 4: 5}


def _slab_with_soma(soma_label):
    """A label slab that genuinely contains soma voxels plus a few other classes."""
    slab = np.zeros((4, 8, 8), dtype=np.uint8)
    slab[0, :2, :2] = 1
    slab[1, :2, :2] = 3
    slab[2, :4, :4] = soma_label
    return slab


def test_soma_label_resolved_for_every_model_type():
    """Every model type's soma label comes back as the integer label value."""
    for model_type, channel_map in LABEL_CHANNEL_MAP.items():
        got = soma_label_from_channel_map(channel_map)
        assert got == EXPECTED_SOMA_LABEL[model_type], (
            f"model_type {model_type}: expected soma label "
            f"{EXPECTED_SOMA_LABEL[model_type]}, got {got!r}")
        assert isinstance(got, int), f"model_type {model_type}: {got!r} is not an int"


def test_soma_is_detected_in_a_slab_for_every_model_type():
    """The resolved label finds soma voxels in a slab that has them."""
    for model_type, channel_map in LABEL_CHANNEL_MAP.items():
        label = soma_label_from_channel_map(channel_map)
        slab = _slab_with_soma(label)
        assert bool(np.any(slab == label)), f"model_type {model_type}: soma not detected"


def test_absent_soma_is_not_detected():
    """A slab with no soma voxels must not report one (false-positive direction)."""
    for model_type, channel_map in LABEL_CHANNEL_MAP.items():
        label = soma_label_from_channel_map(channel_map)
        slab = np.zeros((4, 8, 8), dtype=np.uint8)
        slab[0, :2, :2] = 1          # dendrite only
        assert not bool(np.any(slab == label)), (
            f"model_type {model_type}: soma reported in a slab that has none")


def test_missing_soma_class_returns_none():
    """A map with no soma class yields None rather than a wrong label."""
    assert soma_label_from_channel_map({1: 'dendrites', 2: 'spines'}) is None


def test_the_old_expression_would_still_fail():
    """Pin the defect itself, so a revert cannot pass quietly.

    This is the exact expression that shipped, evaluated against a slab that does
    contain soma voxels. It must fail to detect them for every model type.
    """
    for model_type, channel_map in LABEL_CHANNEL_MAP.items():
        label = EXPECTED_SOMA_LABEL[model_type]
        slab = _slab_with_soma(label)
        old = channel_map.get(3, channel_map.get(5, -1))
        assert not bool(np.any(slab == old)), (
            f"model_type {model_type}: the old expression detected soma - "
            f"this test no longer pins the defect")
        assert old != label, (
            f"model_type {model_type}: the old expression returned the right "
            f"label by coincidence; the guard needs rethinking")


def test_map_is_keyed_by_label_value_not_name():
    """The direction of the map is what the bug got wrong; assert it explicitly."""
    for model_type, channel_map in LABEL_CHANNEL_MAP.items():
        for key, value in channel_map.items():
            assert isinstance(key, int), (
                f"model_type {model_type}: key {key!r} is not a label value")
            assert isinstance(value, str), (
                f"model_type {model_type}: value {value!r} is not a class name")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        name = fn.__name__
        print(f"\n=== {name} ===")
        try:
            fn()
            print(f"PASS: {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL: {name}\n  {exc}")
    print("\n" + "=" * 50)
    print(f"ALL {len(tests)} TESTS PASSED" if not failed
          else f"{failed} of {len(tests)} TESTS FAILED")
    sys.exit(1 if failed else 0)
