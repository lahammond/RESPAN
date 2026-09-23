# RESPAN tests

RESPAN's analysis pipeline requires an NVIDIA GPU, so the test suite is split by
what a machine without one can honestly run.

## The split

| Selection | Where it runs | What it covers |
|---|---|---|
| `pytest -m "not gpu"` | GitHub Actions, any CPU host | Pure-CPU logic on small synthetic volumes |
| `pytest -m gpu` | Local GPU workstation | CuPy kernels, GPU EDT, per-component neck association |
| `pytest` | Local GPU workstation | Everything |

Tests that need a GPU are marked with `@pytest.mark.gpu` and are deselected in
CI. They are not deleted or hidden — they run before a release on the hardware
RESPAN actually targets.

## Why CI does not run the GPU tests

There is no CPU fallback for the GPU code paths, by design. A CPU emulation of
the GPU distance transform would be slow enough to be useless and would test
different code than users run. `ChunkedProcessing` imports CuPy defensively so
the module still imports without it, and any attempt to reach a GPU operation
raises immediately with an actionable message rather than degrading silently.

The consequence is that the CI job is fast and cheap. The CPU-safe tests operate
on 16x16x16 synthetic label volumes and complete in a few milliseconds; the job
is dominated entirely by dependency installation.

CI also asserts that CuPy is *absent* from the runner, so the `not gpu`
deselection cannot quietly stop being meaningful if the image ever changes.

## Running them

```bash
# CPU-safe subset (what CI runs)
pytest -m "not gpu" -v

# Everything, on a GPU host
pytest -v

# No pytest installed? Each test file also runs standalone:
python tests/test_neck_association.py
```

Test dependencies for the CPU subset are in `requirements-test.txt` — numpy,
scipy, scikit-image, zarr, numcodecs, psutil and pytest. No CuPy, no PyTorch,
no TensorFlow, no nnU-Net, and no image data.

## What is covered

`test_neck_association.py` — regression guard for spine/neck association and
gap bridging. The earlier implementation computed connected components over
`(spines | necks)` and silently dropped any component containing no spine voxel,
which discarded every nnU-Net neck fragment sitting in a segmentation gap from
its parent spine. That was the root cause of 24% neck preservation on the 10 GB
benchmark. These tests pin the corrected behaviour.

| Test | Marker | Asserts |
|---|---|---|
| `test_disconnected_neck_preserved` | `gpu` | Fragment in a gap is assigned to the nearest spine |
| `test_far_neck_dropped_as_noise` | `gpu` | Fragment beyond the distance cap is dropped |
| `test_chain_fragments_assigned_via_progressive_passes` | `gpu` | Chained fragments resolve across passes |
| `test_mode_vote_handles_two_spines` | `gpu` | Mode vote assigns a whole fragment to one spine |
| `test_bridge_fills_1voxel_gap` | — | A one-voxel gap is bridged |
| `test_bridge_skips_connected_spine` | — | Already-connected spine is left untouched |
| `test_bridge_respects_dendrite` | — | Bridging never overwrites dendrite voxels |
| `test_bridge_respects_other_spine` | — | Bridging never overwrites another spine |

## Larger-scale testing

Unit tests are the floor, not the whole story. Each release is also validated end
to end against real datasets from 10 MB to 55 GB, with SHA-256 comparison against
stored reference outputs.
