# RESPAN v1.5: What's new

## v1.5.1 (20 September 2026)

A patch release. The pipeline, models and settings are unchanged. Several measurements
were being reported incorrectly and are now fixed, so **some column values change**.
Results from v1.5.1 are not comparable with earlier releases for the columns below.
Everything else is unchanged: on our regression dataset only these columns differ, and
all other values, the segmentation labels and the validation images are bit-identical.

### Columns whose values change

| column | affected runs | what was wrong |
|---|---|---|
| `spine_extent`, `spine_solidity` | all | ratio computed from two different geometries, could exceed 1 |
| `head_extent`, `head_solidity` | all | same |
| `neck_extent`, `neck_solidity` | all | same, plus 0.0 written for spines with no neck |
| `euclidean_dist_to_soma` | all | 0.0 when no soma was measured |
| `geodesic_dist_to_soma` | large images only | measured from the wrong starting point |

Volumes, surface areas, lengths, widths, intensities, spine counts and classifications
are unaffected in every case.

### `extent` and `solidity` could exceed 1

Both are ratios that cannot exceed 1. RESPAN counted voxels for the numerator but
derived the denominator from the triangulated surface, which sits about half a voxel
inside the voxel block. The denominator therefore described a slightly smaller object
than the numerator, and the ratio read high, increasingly so for smaller objects. On
our regression image 51 of 84 spine heads reported a solidity above 1, the largest
2.12, and one neck reported 6.00. Both measures now come from the surface mesh
throughout and agree with the `_vol`, `_bbox_vol` and `_convex_vol` columns beside them.

You can recompute either from columns you already have, which is exactly what v1.5.1
writes:

```
solidity = <prefix>_vol / <prefix>_convex_vol
extent   = <prefix>_vol / <prefix>_bbox_vol
```

Two cases now report `NaN` instead of a number, because no shape can be measured:
objects one voxel in size, whose triangulated surface is a fixed octahedron whatever
the underlying structure, and spines whose neck does not reach the dendrite. The
second case matters for summary tables: a partial spine used to contribute 0.0 to the
average. On our regression image 24 of 84 spines are partial and `avg_neck_solidity`
read 34.5% low as a result, 0.277 against 0.423. The size of that error depended on
how many spines were partial, so it was not a constant offset between datasets.

### Soma distances on large images

On images processed through the low-memory (chunked) path, RESPAN did not detect the
soma even when the model had segmented one correctly. Two columns were affected.

`euclidean_dist_to_soma` was reported as 0.0. `geodesic_dist_to_soma` measures the
path along the dendrite from a starting point; with no soma found, that starting point
fell back to the end of each dendrite nearest the image origin. Distances were still
real path lengths, but measured from the wrong end.

How much that matters depends on where the soma sits. Where the two ends coincide,
values came out close to correct: on our 500 MB test image, five of six dendrites
(527 of 546 spines) differed by less than 1.5 um on average and kept their ordering
exactly. For one dendrite that ran back past the soma, the ordering was reversed, so
spines near the soma were reported as far from it. That was 19 spines on this image,
but the proportion is a property of the geometry, not a bound: an image with the soma
at the opposite corner would invert more dendrites than it preserved.

Both columns are now measured from the soma whenever one is present, on images of any
size. Small images were never affected, and v1.0 predates this code entirely.

When an image contains no soma, nothing changes. `euclidean_dist_to_soma` is `NaN`
rather than 0.0, which no longer looks like a spine sitting on the soma, and
`geodesic_dist_to_soma` continues to measure from one end of each dendrite. That
remains useful as position along the branch for datasets of isolated dendrite
segments, which is why it is kept.

### An image could be dropped from a batch

If the path to a validation image exceeded the Windows 260-character limit, writing it
failed, that image was excluded from the run, and results from a previous run could be
left in the output folder looking current. Deep folder paths, such as cloud-synced
drives, reach that limit easily. RESPAN now writes long paths correctly, and a run that
loses an image reports an error instead of finishing silently.

### Running without the GUI

RESPAN can now run headlessly from a command line, which suits batch jobs,
scheduled runs and remote machines with no display. It performs exactly what the
GUI Run button performs, and every default matches the GUI's.

```
RESPAN_v1_5_01.exe --cli --data-dir "D:/MyExperiment" ^
                   --model-dir "D:/Models/Dataset226..." --model-type 4
```

`--data-dir` is the parent folder, as in the GUI: every subfolder containing an
`Analysis_Settings.yaml` is processed in turn. `--cli --help` lists all options,
including voxel sizes (`--input-xy`, `--input-z`, or `--use-yaml-res` to take them
from the settings file), the spine filters (`--spine-vol`, `--spine-dist`,
`--min-dendrite-vol`), the optional stages (`--recover-filopodia`,
`--detect-multi-head-spines`, `--dendrite-repair`, `--image-restore`,
`--axial-restore`, `--track`), and `--log-file` to record the whole run.

This is new in 1.5.1 and the option names may still change.

### Other fixes

- The Run button in the GUI could close the application immediately. This affected the
  v1.5.0 source release; the packaged application was not affected.
- GUI settings are saved to your user profile rather than the application folder, so
  they persist when that folder is read-only.
- An unhandled error writes a dated crash log next to the application instead of
  closing the window without explanation.
- The model type selector now starts on the current model scheme rather than the
  oldest one. A saved settings file still takes precedence.
- Diagnostic text that served no purpose for users has been removed from the console.

---


This release focuses on scalability, accuracy at high spine density, and fixes for several user-reported issues. The pipeline now runs end to end on datasets up to 55 GB+ on a 128 GB RAM / 24 GB VRAM workstation, and adds three new spine classes (filopodia, partial-spine, multi-head) so dense or occluded regions are better quantified.

A new fine-tuning guide is included for users who want to adapt the pretrained nnU-Net models to their own data without retraining from scratch.

## Highlights

- Validated end-to-end on 55 GB images on a 128 GB RAM / 24 GB VRAM machine. The chunked zarr path is designed to scale down to smaller-RAM hosts via adaptive chunk sizing, but lower-memory configurations have not been benchmarked.
- Spine neck preservation improvements
- New: filopodia recovery, partial-spine classification, multi-head spine grouping.
- New: step-by-step fine-tuning guide for the segmentation models.
- New: GUI controls for spine thresholds
- Fixed: CARE training silent failure when training folders are misconfigured (#10).
- Fixed: nnU-Net training silent failures from the GUI. Pre-flight check + actionable error messages naming the actual cause.
- Fixed: ZeroDivisionError in GPU memory check on Windows WDDM systems (#16).
- Fixed: out-of-memory crashes during spine measurement and neck association on large datasets.
- Improved: GUI startup logging and path-with-spaces / OneDrive path warnings, addressing several silent-failure reports (#11, #12).

## Bug fixes

### nnU-Net training failures from the GUI (multiple devices, multiple users)
Several users reported that "Plan and preprocess" consistently fails when launched from the GUI's nnU-Net Training tab, returning a generic "return code 1" with the traceback hidden inside the bundled executable. The most common root causes were silent and difficult to diagnose: training folder in OneDrive / cloud-synced paths (placeholder files), paths with spaces, missing or malformed dataset.json, or nonstandard image filenames not matching nnU-Net v2's `CASE_NNNN.tif` convention.

v1.5 adds a pre-flight check that validates the training folder layout (`Dataset###_NAME/imagesTr`, `labelsTr`, `dataset.json`) before invoking nnU-Net. It catches the most common silent failures ahead of time and surfaces a specific, actionable error message naming the path, the cause, and the fix. When the underlying nnU-Net call still returns non-zero, the error log now lists the six most common causes with one-line fixes instead of just "return code 1".

### CARE model training (#10)
The CARE training pipeline used to fail with a generic `FileNotFoundError: Didn't find any images.` from csbdeep when the source/target folders were missing, empty, named differently, or contained mismatched filenames. The error gave no indication of which folder was wrong. RESPAN now validates both directories before starting training and fails fast with a specific message naming the folder, the number of files found, the accepted extensions, and the next step. If filenames in `lowSNR/` and `highSNR/` do not match exactly, the error lists the offending files.

### Spine batch processing on Windows WDDM (#16)
On Windows display GPUs, `torch.cuda.mem_get_info()` can report near-zero free VRAM because the WDDM driver pre-allocates memory pageably. The percentage calculation now guards against zero-free-memory and reports infinity instead of crashing.

### GUI startup and runtime diagnostics (#11, #12)
v1.0 surfaced almost no information when the GUI failed at startup or when "Run" produced no output. v1.5 logs a startup banner and runs early-stage checks before the pipeline starts:

- RESPAN install location is logged at startup, with explicit warnings if the path contains spaces or sits inside a OneDrive-synced folder. Both have caused silent failures in the past (Issue #12).
- GPU availability is checked and logged (device name, VRAM, driver) before the user can hit Run.
- Bundled environment setup status is reported in the GUI log with a critical-error dialog if it fails (was previously a silent no-op).
- DLL load failures during startup now print the resolved DLL path, the current `PATH`, `CUDA_PATH`, and `LD_LIBRARY_PATH` to the GUI log.
- When Run is clicked, both the data directory and model directory are checked for spaces and OneDrive paths. The user gets a confirmation dialog rather than a silent crash mid-pipeline.
- Empty / mis-organized data directories produce a clear error message showing the expected layout (`data_dir/dataset_name/Analysis_Settings.yaml + image files`), not an unhandled exception.

These changes do not fix the underlying "AttributeError" referenced in Issue #11, but they should make the next failure trace easy to reproduce and report.

### Out-of-memory on large dendrite datasets (#14, related)
Multiple memory crashes during spine analysis on large or dense datasets are addressed by the chunked-processing rewrite. `safe_unique`, `safe_max`, `find_objects_streaming`, and `connected_components_streaming` slab-iterate on both numpy and zarr inputs and avoid the full-volume transient allocations that previously triggered OOM at 1+ GB scale. For images above 1 GB, geodesic distances now use a coarser XY resolution (0.3 micrometers, Z unchanged) to avoid OOM on giant single dendrites that span most of the volume.

### Other fixes
- Phantom-label overcounting in the tiny-head filter no longer reports indices that correspond to label-space holes.
- Boolean-dtype handling in per-component spine detection no longer mis-interprets uint8 watershed masks as integer indexing.
- Volume filter bounds are now inclusive: spines of exactly the configured minimum size are no longer silently dropped.
- nnU-Net first-run install on the bundled distribution no longer requires manually unsetting `PIP_REQUIRE_VIRTUALENV`.
- Distance-map MIPs now write to separate single-channel TIFFs so the main composite stays within ImageJ's 7-channel display limit.
- Geodesic distance values for spines on dendrites that exceed the geodesic memory limit are preserved as NaN (rather than zero), so they are not confused with spines at the soma. Scaled geodesic distance should ensure measurements regardless of dendrite size and machine resources.

## New features

### Filopodia recovery
Spines whose head is detected without a traceable neck connection to the dendrite, plus dendrite-attached neck fragments without a paired spine head, are now retained and labeled in the output CSV. A new `spine_type` column reports `'spine'`, `'partial-spine'`, or `'filopodia'`. Summary CSVs gain `n_filopodia`, `n_spines_excl_filopodia`, and `filopodia_per_um`. Enabled by default; opt out via the `recover_filopodia` setting in YAML or the GUI checkbox.

### Partial-spine classification
When a spine head is detected but the neck does not reach the dendrite (typically because an adjacent head occludes the neck path), the spine is now labeled `partial-spine` rather than silently dropped or merged. Distance and head metrics are retained; neck metrics are zeroed. This aligns with the SpineS 2022 precedent (PMID 35859092). Controlled by `keep_partial_spines` (default on).

### Multi-head spine detection
Optional feature that surfaces spines with multiple heads sharing the same neck. When enabled, a `parent_spine_id` column links sibling heads, a grouped channel is added to the validation MIP, and a `Multi_Head_Spine_Groups.csv` is written. Off by default; enable via the GUI or `--detect-multi-head`.

### Geodesic distance at lower resolution for large images
For input images larger than 1 GB, geodesic distance from each spine to the soma is computed on an XY-downsampled dendrite mask (target 0.3 micrometers in XY, Z unchanged). The resulting per-spine distance is within sub-micrometer precision of the full-resolution computation, while avoiding the multi-hundred-GB float64 buffer that the full-resolution Dijkstra requires on giant dendrites. Smaller images are unchanged.

### Spine threshold controls in the GUI
A new "Spine Thresholding" panel exposes:
- Spurious-neck length minimum (default 1.0 micrometer)
- nnU-Net support minimum (default 0.2)
- Toggles for: drop spurious necks, drop disconnected neck fragments, drop wrong-direction neck fragments, recover filopodia, keep partial spines, keep head if flagged.

Every field has a hover tooltip explaining its purpose and the default value. YAML still works; the GUI takes precedence when both are set.

### Pre-merge audit CSVs
When the tiny-head filter drops labels, a `{filename}_tiny_head_dropped_labels.csv` is written next to the main spine CSV with each dropped label and its voxel count. Drop samples are also logged for the disconnected and wrong-direction neck filters.

## Performance and scalability

### Chunked / streaming processing for large images
The full pipeline now operates on zarr-backed intermediate arrays for any single volume above 1 GB. The image, neck and spine accumulators, watershed labels, distance maps, and skeleton are all spilled to disk and accessed slab by slab. On a 128 GB RAM / 24 GB VRAM host the canonical T7 (10 GB / 5.5 billion voxels) dataset runs end to end in ~3 hours 49 minutes with ~108-113 GB RAM plateau and 1.34 GB peak VRAM. The larger T_LARGE benchmark (55 GB / 29.8 billion voxels) runs end to end on the same hardware. Smaller-memory hosts are supported by the adaptive chunk-sizing logic but have not been benchmarked.

### Per-spine bounding-box processing
Per-spine measurements (volume, surface area, distances, intensity, head metrics) now read only the spine's own bounding box rather than scanning the full volume per spine. Scales linearly with spine count rather than spine count times volume, which was untenable above ~10,000 spines.

### Two-tier spine-to-neck gap bridging
A morphological closing pass fills 1-voxel gaps; a signal-aware pathfinder (which favors fluorescence intensity along the dendrite-bound corridor) handles multi-voxel gaps. Pathfinder excludes neighbour spine heads as obstacles so necks no longer route through adjacent heads. T4 dataset: 173 of 198 gapped spines rescued.

### Voronoi-based neck assignment
Neck connected components that bridge multiple spine heads are now split per-voxel via the Voronoi region of each head's distance transform, instead of mode-vote over the whole CC. Eliminates ~21% of T2 neck contamination cases observed in v1.0.

### Other performance improvements
- Streaming connected components and `find_objects` for huge label volumes (no full-volume int32 materialization).
- GPU-chunked Euclidean distance transform with halo (replaces full-volume scipy EDT, which allocated tens of GB).
- Dendrite skeletonization per object via slab-streamed bounding-box reads.
- nnU-Net VRAM management: opt-in patching for images above 500 MB to prevent VRAM spillover into shared system memory.
- Multi-hour T7 stalls in `per_spine_regionprops` on dual-zarr inputs are guarded by a log-silence heartbeat monitor (Phase H temporarily disabled at the call site pending a batched-zarr rewrite).

## GUI changes

### Added
- Spine Thresholding panel (see "New features" above).
- Hover tooltips on threshold fields.
- "Drop spurious necks" toggle (replaces the silent default).
- "Recover filopodia" toggle.
- "Keep partial spines" toggle.
- "Keep head if flagged" toggle.

### Removed
- "Minimum neck/head intensity ratio" field. The control was inert (the underlying setting `spurious_require_intensity` defaulted to False in v1.0) and removing it avoids confusion. Intensity thresholds remain available via YAML for expert users.

### Changed
- "Spine Thresholding" group box layout updated for the new controls.
- Default spurious-neck mode changed from "drop both head and neck" to "drop neck only". Spine heads detected by nnU-Net are now preserved when the neck is flagged as spurious; the head reverts to a partial-spine classification rather than being deleted entirely.
- Default `spurious_path_length_min_um` lowered from 2.0 to 1.0 micrometers. Pathfinder-synthesized necks shorter than 2.0 micrometers with no nnU-Net backing previously slipped through the filter; the new threshold catches them while keeping the AND-gate with `nnunet_support < 0.2` so genuine short necks with model support are preserved.

## Migration notes

These defaults changed from v1.0. If you have a `Analysis_Settings.yaml` from a previous release, the new defaults apply unless you set the value explicitly.

| Setting | v1.0 | v1.5 | Effect |
|---|---|---|---|
| `spurious_path_length_min_um` | 2.0 | 1.0 | Drops shorter spurious necks; spine count in dense regions may decrease slightly. |
| `spurious_exclude_mode` | both | neck_only | Heads of flagged spines are preserved as partial-spines. |
| Volume filter bounds | strict (`>`, `<`) | inclusive (`>=`, `<=`) | Spines of exactly the minimum size are no longer dropped. |
| `recover_filopodia` | (n/a) | true | New spine type in CSV. |
| `keep_partial_spines` | (n/a) | true | New spine type in CSV. |
| `min_spine_head_voxels` | (n/a) | 2 | Drops 1-2 voxel stray heads; reduces phantom labels in nearest-spine assignment. |

New CSV columns added (backward compatible: existing scripts work, but new analyses can use them):
- `spine_type` (`'spine'`, `'partial-spine'`, `'filopodia'`).
- `parent_spine_id` (multi-head feature, when enabled).
- `head_euclidean_dist_to_dend` (straight-line distance, used as a fallback when the traced neck is incomplete).

Summary CSV columns added:
- `n_filopodia`, `n_spines_excl_filopodia`, `filopodia_per_um`.


## Documentation

- `docs/Fine_Tuning_Guide.md`: step-by-step guide for fine-tuning the pretrained nnU-Net models to your data, including correction strategy, folder layout, dataset.json, plans transfer, and the actual `nnUNetv2_train` invocation.
- `docs/testing_strategy.md`: per-profile runtime budgets, per-stage timing tables, and the six kill triggers used to detect stalls or runaway runs at scale.

## Acknowledgements

Issue reporters and contributors:
- @ritapgv (#10) for surfacing the CARE training silent-failure mode.
- @nicoperedo (#16) for the Windows WDDM ZeroDivision diagnosis and the suggested fix direction.
- @dcp2153 (#14) for the large-dendrite memory error report that motivated several of the chunked-pipeline fixes.
- @ardennorthchaim (#12) for the GUI startup diagnosis that prompted the new startup-banner logging and path warnings.
- Sergio Bernal-Garcia and the Polleux Lab for ongoing discussions and feedback.

## Citation

If RESPAN supports your research, please cite:

Sergio Bernal-Garcia, Alexa P. Schlotter, Daniela Pereira, Franck Polleux, Luke A. Hammond. (2025). A deep learning pipeline for accurate and automated restoration, segmentation, and quantification of dendritic spines. *Cell Reports Methods* 5(10):101179. doi:10.1016/j.crmeth.2025.101179
