# -*- coding: utf-8 -*-
"""
GPU-Accelerated Chunked Processing for Large Images
=====================================================

Replaces the broken Dask path with a proper stage-and-sweep architecture:
- At each pipeline stage, only that stage's inputs are in memory
- Outputs write to zarr immediately; previous arrays are released
- GPU processes chunks with proper halo overlap
- Decision rule: if estimated peak RAM > 70% system RAM, use chunked path

Classes:
    MemoryBudget    — RAM/GPU capacity detection + fits_in_ram()/fits_on_gpu()
    ZarrWorkspace   — Manages zarr cache dir under {data_dir}/Validation_Data/.zarr_cache/
    ChunkIterator   — Yields chunk coords with configurable halo, handles edges

Core operations:
    distance_transform_gpu_chunked()    — GPU EDT with anisotropic halo
    connected_components_streaming()    — 2-pass slab-based CC with union-find merge
    skeletonize_per_object()            — Per-dendrite bbox extraction + skeletonize
    regionprops_streaming()             — Streaming regionprops from zarr
    geodesic_per_dendrite()             — Per-dendrite geodesic distance
"""

__author__ = 'Luke Hammond <luke.hammond@osumc.edu>'
__license__ = 'GPL-3.0 License (see LICENSE)'

import os
import shutil
import math
import logging
import numpy as np
import psutil
import zarr
from numcodecs import Blosc

try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as gpu_edt
    _GPU_AVAILABLE = True
except ModuleNotFoundError as _cupy_import_error:
    # Tolerate only a genuinely ABSENT CuPy (CPU-only hosts, e.g. CI runners).
    #
    # An installed-but-BROKEN CuPy — mismatched CUDA DLLs, wrong nvrtc version,
    # the exact failure this project pins CUDA_PATH to avoid — raises plain
    # ImportError, not ModuleNotFoundError. That must keep failing loudly right
    # here: swallowing it would resurface later as a misleading "install cupy"
    # message on a machine where CuPy is in fact installed.
    _GPU_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "CuPy not found (%s). GPU code paths in ChunkedProcessing are disabled; "
        "only CPU-safe operations can run in this process.", _cupy_import_error
    )

    class _CuPyUnavailable:
        """Stand-in for cupy on hosts where it is not installed.

        This is a backstop, not the primary guard. GPU entry points call
        _require_cupy() and fail with a direct message. Two call sites
        (MemoryBudget.__init__ and the VRAM probe in
        distance_transform_gpu_chunked) deliberately wrap their cupy call in
        `except Exception` to degrade gracefully, and depend on this object
        raising rather than being None.

        Note this does NOT cover associate_spines_with_necks_per_component,
        which re-imports cupy locally and so shadows `cp` for its whole body.
        """

        def __getattr__(self, name):
            if name.startswith("__"):
                # Let hasattr()/introspection behave normally.
                raise AttributeError(name)
            raise RuntimeError(
                "This operation requires CuPy and an NVIDIA GPU (tried to use "
                "cupy.{}). Install cupy-cuda11x to run it, or restrict a test "
                "run to CPU-safe cases with: pytest -m \"not gpu\"".format(name)
            )

    cp = _CuPyUnavailable()

    def gpu_edt(*_args, **_kwargs):
        raise RuntimeError(
            "GPU distance transform requires CuPy and an NVIDIA GPU. "
            "Install cupy-cuda11x to run it."
        )


def _require_cupy(operation):
    """Fail fast, and clearly, when a GPU-only entry point is called on a CPU host."""
    if not _GPU_AVAILABLE:
        raise RuntimeError(
            "{} requires CuPy and an NVIDIA GPU, which are not available in this "
            "process. Install cupy-cuda11x, or restrict a test run to CPU-safe "
            "cases with: pytest -m \"not gpu\"".format(operation)
        )

from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage import morphology, measure

GB = 1024 ** 3
MB = 1024 ** 2

# Chunk size constants — two access patterns, two chunk strategies.
# Streaming arrays (dendrite_mask, skeleton) are read Z-slab by Z-slab.
# Per-spine arrays (distances, labels) are read by small 3D bounding boxes.
STREAMING_CHUNKS = (128, 512, 512)


def spine_access_chunks(settings):
    """Resolution-adaptive chunk size for per-spine random-access arrays.

    Targets ~5 µm per XY chunk dimension so each chunk contains 2-4 spines,
    keeping NTFS file count manageable while minimizing read amplification.
    """
    res_xy = getattr(settings, 'input_resXY', 0.065)
    c_xy = max(64, int(round(5.0 / res_xy / 64)) * 64)
    return (64, c_xy, c_xy)


# ============================================================================
# Safe reducers — avoid np.max/np.any/np.unique silently materializing zarr
# ============================================================================
# numpy reductions like np.max(zarr_array) silently call np.asarray(arr), which
# materializes the full volume (catastrophic at 22 GB+). These helpers slab-
# iterate on zarr input and fall through to the native numpy call otherwise.

def _safe_slab_depth(arr):
    """Per-slab Z depth capped at ~2 GB per slab for a 3D array."""
    if arr.ndim < 3:
        return max(1, arr.shape[0])
    row_bytes = int(arr.shape[1]) * int(arr.shape[2]) * arr.dtype.itemsize
    if row_bytes <= 0:
        return 64
    return max(1, min(64, int(2 * GB / row_bytes)))


def safe_max(arr):
    """Max reduction that works for numpy and zarr. Never materializes zarr."""
    if hasattr(arr, 'chunks'):
        m = None
        d = _safe_slab_depth(arr)
        for z in range(0, arr.shape[0], d):
            slab = np.asarray(arr[z:min(z + d, arr.shape[0])])
            if slab.size == 0:
                continue
            sm = slab.max()
            m = sm if m is None else max(m, sm)
        return m if m is not None else 0
    return arr.max()


def safe_any(arr):
    """Short-circuit any() over slabs. Works for numpy and zarr."""
    if hasattr(arr, 'chunks'):
        d = _safe_slab_depth(arr)
        for z in range(0, arr.shape[0], d):
            if np.any(np.asarray(arr[z:min(z + d, arr.shape[0])])):
                return True
        return False
    return bool(np.any(arr))


def safe_unique(arr):
    """Unique reduction that works for numpy and zarr.

    np.unique on a full 5.47B-voxel numpy volume (T7 spine_labels_vol) allocates
    a 5 GB bool mask plus sort workspace ≈ 22 GB transient — sufficient to OOM
    even a 128 GB system when other large arrays are live. Slab-iterate large
    numpy arrays the same way we do for zarr — the set-union approach avoids
    the full-volume sort.

    Threshold: slab-iterate anything >500M voxels (below that, direct
    np.unique is fine and marginally faster).
    """
    if hasattr(arr, 'chunks'):
        # Zarr input
        s = set()
        d = _safe_slab_depth(arr)
        for z in range(0, arr.shape[0], d):
            s.update(np.unique(np.asarray(arr[z:min(z + d, arr.shape[0])])).tolist())
        if not s:
            return np.array([], dtype=arr.dtype)
        return np.array(sorted(s))
    # Numpy input — slab-iterate if large to avoid full-volume sort workspace.
    shape = arr.shape
    n_vox = int(np.prod(shape)) if len(shape) > 0 else arr.size
    if n_vox > 500_000_000 and len(shape) >= 3:
        s = set()
        d = _safe_slab_depth(arr)
        for z in range(0, shape[0], d):
            s.update(np.unique(arr[z:min(z + d, shape[0])]).tolist())
        if not s:
            return np.array([], dtype=arr.dtype)
        return np.array(sorted(s))
    return np.unique(arr)


# ============================================================================
# MemoryBudget — RAM/GPU capacity detection
# ============================================================================
class MemoryBudget:
    """Detects system RAM and GPU memory, provides capacity checks."""

    def __init__(self, ram_threshold=0.70, gpu_threshold=0.70):
        """
        Parameters
        ----------
        ram_threshold : float
            Fraction of total system RAM considered "safe" for in-memory processing.
        gpu_threshold : float
            Fraction of total GPU memory considered "safe" for GPU operations.
        """
        self.ram_threshold = ram_threshold
        self.gpu_threshold = gpu_threshold

        # System RAM
        mem = psutil.virtual_memory()
        self.total_ram = mem.total
        self.available_ram = mem.available

        # GPU memory
        try:
            self.gpu_free, self.gpu_total = cp.cuda.runtime.memGetInfo()
        except Exception:
            self.gpu_free = 0
            self.gpu_total = 0
        self.gpu_available = self.gpu_total > 0

    def fits_in_ram(self, estimated_bytes):
        """Check if estimated peak memory fits within safe RAM threshold."""
        return estimated_bytes < self.total_ram * self.ram_threshold

    def fits_on_gpu(self, estimated_bytes):
        """Check if estimated GPU memory fits within safe GPU threshold."""
        return estimated_bytes < self.gpu_total * self.gpu_threshold

    def safe_ram_bytes(self):
        """Return number of bytes available under the threshold."""
        return int(self.total_ram * self.ram_threshold)

    def safe_gpu_bytes(self):
        """Return number of GPU bytes available under the threshold."""
        return int(self.gpu_total * self.gpu_threshold)

    def estimate_pipeline_peak(self, volume_shape, dtype=np.float32):
        """Estimate peak RAM for full in-memory pipeline.

        The pipeline needs roughly 8x the volume size at peak
        (neuron, dendrite, spine_labels, head_labels, distance maps, skeleton, etc.)
        """
        itemsize = np.dtype(dtype).itemsize
        single_vol = int(np.prod(np.array(volume_shape, dtype=np.int64))) * itemsize
        return single_vol * 8

    def needs_chunked_processing(self, volume_shape, dtype=np.float32, force=False):
        """Decision rule: True if estimated peak > threshold or force=True."""
        if force:
            return True
        peak = self.estimate_pipeline_peak(volume_shape, dtype)
        return not self.fits_in_ram(peak)

    def log_summary(self, logger):
        """Log memory budget summary."""
        logger.info(f"     Memory budget:")
        logger.info(f"       System RAM: {self.total_ram / GB:.1f} GB "
                     f"(available: {self.available_ram / GB:.1f} GB, "
                     f"threshold: {self.ram_threshold:.0%})")
        if self.gpu_total > 0:
            logger.info(f"       GPU memory: {self.gpu_total / GB:.1f} GB "
                         f"(free: {self.gpu_free / GB:.1f} GB, "
                         f"threshold: {self.gpu_threshold:.0%})")


# ============================================================================
# ZarrWorkspace — Manages zarr cache directory
# ============================================================================
class ZarrWorkspace:
    """Manages a zarr cache directory for intermediate chunked processing results.

    Cache lives under a local temp directory to avoid Windows MAX_PATH issues
    on long network/cloud paths (Google Drive, NAS).
    Cleaned up after processing completes.
    """

    COMPRESSOR = Blosc(cname='lz4', clevel=1)
    DEFAULT_CHUNKS = (64, 512, 512)

    PREFERRED_SSD_ROOTS = ['E:/respan_workspace', 'D:/respan_workspace']

    def __init__(self, locations, filename, logger=None):
        import tempfile
        import hashlib

        data_path = locations.tables.replace("Tables", "Validation_Data")
        path_hash = hashlib.md5(data_path.encode()).hexdigest()[:12]

        # Priority: fast local SSD → system temp → data directory (last resort)
        ssd_root = None
        for candidate in self.PREFERRED_SSD_ROOTS:
            drive = os.path.splitdrive(candidate)[0]
            if drive and os.path.isdir(drive + '/'):
                ssd_root = candidate
                break

        cloud_indicators = ['OneDrive', 'My Drive', 'Google Drive', 'Dropbox']
        is_cloud = any(ind in data_path for ind in cloud_indicators)

        if ssd_root is not None:
            self.base_dir = os.path.join(ssd_root, path_hash, filename)
            if logger:
                logger.info(f"       Zarr cache: using SSD ({ssd_root})")
        elif is_cloud:
            local_root = os.path.join(tempfile.gettempdir(), "respan_zarr_cache")
            self.base_dir = os.path.join(local_root, path_hash, filename)
            if logger:
                logger.info(f"       Zarr cache: using local temp dir (cloud filesystem detected)")
        else:
            self.base_dir = os.path.join(data_path, ".zarr_cache", filename)

        os.makedirs(self.base_dir, exist_ok=True)
        self.logger = logger
        self._stores = {}
        if self.logger:
            self.logger.info(f"       Zarr cache: {self.base_dir}")

    def create_array(self, name, shape, dtype=np.int32, chunks=None, fill_value=0):
        """Create a new zarr array in the workspace.

        Parameters
        ----------
        name : str
            Array name (e.g., 'dendrite_labels', 'distance_map').
        shape : tuple
            Volume shape (Z, Y, X).
        dtype : numpy dtype
            Data type for the array.
        chunks : tuple, optional
            Chunk shape. Defaults to DEFAULT_CHUNKS clamped to volume.
        fill_value : scalar
            Fill value for uninitialized chunks.

        Returns
        -------
        zarr.Array
        """
        if chunks is None:
            chunks = tuple(min(c, s) for c, s in zip(self.DEFAULT_CHUNKS, shape))

        # Remove any existing array directory before creating a fresh zarr —
        # `zarr.open(mode='w')` only rewrites metadata. Stale chunk files from
        # a prior run survive metadata recreation and can contaminate sparse
        # outputs (e.g. Phase H labeled_spines where un-updated slabs would
        # retain data from a previous pipeline run).
        path = os.path.join(self.base_dir, name + ".zarr")
        if os.path.isdir(path):
            import shutil as _shutil
            _shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
        store = zarr.DirectoryStore(path)
        arr = zarr.open(
            store, mode='w',
            shape=shape, dtype=dtype,
            chunks=chunks,
            compressor=self.COMPRESSOR,
            fill_value=fill_value,
        )
        self._stores[name] = store
        if self.logger:
            size_mb = int(np.prod(np.array(shape, dtype=np.int64))) * np.dtype(dtype).itemsize / MB
            self.logger.info(f"       Zarr array '{name}': shape={shape}, dtype={dtype}, "
                             f"chunks={chunks}, uncompressed={size_mb:.0f} MB")
        return arr

    def open_array(self, name, mode='r'):
        """Open an existing zarr array from the workspace."""
        path = os.path.join(self.base_dir, name + ".zarr")
        os.makedirs(path, exist_ok=True)
        store = zarr.DirectoryStore(path)
        return zarr.open(store, mode=mode)

    def array_to_numpy(self, name):
        """Read zarr array into numpy using slab-by-slab streaming.

        np.array(zarr_array) loads all compressed chunks into a dict
        simultaneously, causing a transient ~1.5x spike over the final array.
        For 20 GB arrays, this can OOM. Reading slab-by-slab keeps the
        transient overhead to one slab (~0.5-2 GB).

        Slab depth is capped at ~2 GB to prevent zarr's internal temp
        allocation from spiking when chunks[0] >= shape[0].
        """
        arr = self.open_array(name)
        out = np.empty(arr.shape, dtype=arr.dtype)
        row_bytes = int(arr.shape[1]) * int(arr.shape[2]) * arr.dtype.itemsize
        max_slab = max(1, int(2 * 1024**3 / row_bytes)) if row_bytes > 0 else 64
        slab_depth = min(arr.chunks[0] if len(arr.chunks) > 0 else 64, max_slab)
        for z in range(0, arr.shape[0], slab_depth):
            ze = min(z + slab_depth, arr.shape[0])
            out[z:ze] = arr[z:ze]
        return out

    def numpy_to_zarr(self, name, data, chunks=None):
        """Write a numpy array to zarr. Returns the zarr array."""
        arr = self.create_array(name, data.shape, data.dtype, chunks=chunks)
        arr[:] = data
        return arr

    def cleanup(self):
        """Remove the entire cache directory."""
        if os.path.isdir(self.base_dir):
            shutil.rmtree(self.base_dir, ignore_errors=True)
            if self.logger:
                self.logger.info(f"     Cleaned up zarr cache: {self.base_dir}")


# ============================================================================
# TIFF -> zarr ingest — page-by-page streaming, never materializes full volume
# ============================================================================
# tifffile.imread() and tifffile.TiffFile.asarray() both allocate a single
# contiguous numpy array of the full series shape. At T_LARGE (55 GB uint16
# image, 27.7 GB uint8 labels) this exceeds available contiguous host RAM and
# fails with MemoryError before any pipeline work. tifffile.aszarr() exposes
# a TIFF-backed zarr Store but is read-only and not all builds support it
# uniformly; the page-by-page approach below is the most portable scale-safe
# pattern and matches the slab-stream contract used elsewhere in this module
# (skeletonize_per_object, labels_to_zarr_channels).
def tiff_metadata(tiff_path):
    """Read shape, dtype, axes from a TIFF without loading data.

    Parameters
    ----------
    tiff_path : str
        Path to source TIFF.

    Returns
    -------
    dict
        ``shape`` (tuple), ``dtype`` (numpy dtype), ``axes`` (str like
        'ZYX'/'ZCYX'/'CZYX'), ``n_pages`` (int), ``page_shape`` (tuple of
        first page YX shape), ``page_dtype`` (numpy dtype),
        ``n_bytes_estimate`` (int) — total uncompressed byte count.
    """
    import tifffile as _tf
    with _tf.TiffFile(tiff_path) as tif:
        series = tif.series[0]
        first_page = tif.pages[0] if len(tif.pages) > 0 else None
        n_bytes = int(np.prod(np.array(series.shape, dtype=np.int64))) * int(np.dtype(series.dtype).itemsize)
        # Normalize tifffile's "Q" (unknown axis) placeholder for TIFFs without
        # explicit axis metadata. Standard ImageJ-saved 3D label TIFFs (RESPAN
        # nnU-Net output) often report axes='QYX' for a 3D ZYX stack — tifffile
        # can't disambiguate Z vs T vs S without metadata. We treat single-Q +
        # YX as Z (RESPAN convention: all label volumes are Z-stacks).
        _raw_axes = str(series.axes)
        if _raw_axes == 'QYX':
            _axes = 'ZYX'
        elif _raw_axes == 'QQYX' and len(series.shape) == 4:
            # Two unknowns + YX: ambiguous, but standard RESPAN multi-channel
            # convention is Z first, C second. Defer to caller via 'ZCYX'.
            _axes = 'ZCYX'
        else:
            _axes = _raw_axes
        return {
            'shape': tuple(int(s) for s in series.shape),
            'dtype': np.dtype(series.dtype),
            'axes': _axes,
            'axes_raw': _raw_axes,  # original tifffile axes string (debug)
            'n_pages': len(tif.pages),
            'page_shape': tuple(int(s) for s in first_page.shape) if first_page is not None else (),
            'page_dtype': np.dtype(first_page.dtype) if first_page is not None else None,
            'n_bytes_estimate': n_bytes,
        }


def tiff_to_zarr_3d(tiff_path, workspace, name, *, chunks=None, slab_pages=8,
                    select_channel=None, logger=None):
    """Stream a TIFF into a 3D (Z, Y, X) zarr without full materialization.

    Two ingest strategies, picked automatically:

    1. **memmap path** — ``tifffile.memmap()`` returns a numpy memmap whose
       shape matches the TIFF series. OS-level demand paging means slice
       reads (``mm[z_start:z_end, ...]``) load only the requested bytes.
       This is the ONLY path that handles ImageJ "big TIFF" hyperstacks
       above 4 GiB, which store all Z planes in a single IFD with
       ``len(tif.pages) == 1``. T_LARGE's 55 GB
       raw is exactly this case.

    2. **per-page fallback** — for compressed or multi-IFD TIFFs where
       memmap is unavailable. Reads ``slab_pages`` pages at a time into a
       small preallocated buffer (≈ ``slab_pages × page_bytes`` peak,
       e.g. 8 × 66 MB = 0.5 GB).

    Either way, never allocates the full source volume.

    Parameters
    ----------
    tiff_path : str
        Path to source TIFF.
    workspace : ZarrWorkspace
        Target zarr workspace.
    name : str
        Output zarr array name.
    chunks : tuple of int, optional
        Output zarr chunk shape. Defaults to ``STREAMING_CHUNKS`` clamped.
    slab_pages : int
        Number of Z planes per read iteration.
    select_channel : int, optional
        For multi-channel TIFFs (axes 'ZCYX' or 'CZYX'), select this channel
        index and stream only that channel. Required for multi-channel
        input. Must be ``None`` for single-channel ('ZYX').
    logger : logging.Logger, optional

    Returns
    -------
    zarr.Array
        Output zarr handle of shape (Z, Y, X) and dtype matching source.

    Raises
    ------
    ValueError
        If TIFF axes are unsupported (anything other than ZYX, ZCYX, CZYX),
        or if select_channel is missing/invalid for multi-channel input,
        or if both memmap and per-page paths are unavailable for the file.
    """
    import tifffile as _tf

    # --- Phase 1: read metadata and validate axes/channel parameters ---
    with _tf.TiffFile(tiff_path) as tif:
        series = tif.series[0]
        axes_raw = str(series.axes)
        shape = tuple(int(s) for s in series.shape)
        dtype = np.dtype(series.dtype)
        n_pages = len(tif.pages)

    # Normalize tifffile 'Q' (unknown) axis placeholder — standard RESPAN
    # convention treats single-Q + YX as Z, dual-Q + YX as ZC.
    if axes_raw == 'QYX':
        axes = 'ZYX'
    elif axes_raw == 'QQYX' and len(shape) == 4:
        axes = 'ZCYX'
    else:
        axes = axes_raw

    if axes == 'ZYX':
        if select_channel is not None:
            raise ValueError(
                f"select_channel={select_channel} requested but TIFF is "
                f"single-channel ZYX")
        out_shape = shape
        n_channels = 1
    elif axes == 'ZCYX':
        n_z, n_c, n_y, n_x = shape
        if select_channel is None:
            raise ValueError(
                f"TIFF is multi-channel ({axes}, C={n_c}); must pass select_channel")
        if not (0 <= select_channel < n_c):
            raise ValueError(
                f"select_channel={select_channel} out of range [0, {n_c})")
        out_shape = (n_z, n_y, n_x)
        n_channels = n_c
    elif axes == 'CZYX':
        n_c, n_z, n_y, n_x = shape
        if select_channel is None:
            raise ValueError(
                f"TIFF is multi-channel ({axes}, C={n_c}); must pass select_channel")
        if not (0 <= select_channel < n_c):
            raise ValueError(
                f"select_channel={select_channel} out of range [0, {n_c})")
        out_shape = (n_z, n_y, n_x)
        n_channels = n_c
    else:
        raise ValueError(
            f"Unsupported TIFF axes {axes!r}; expected ZYX, ZCYX, or CZYX")

    if chunks is None:
        chunks = tuple(min(c, s) for c, s in zip(STREAMING_CHUNKS, out_shape))

    # --- Phase 2: try memmap (handles ImageJ big TIFF and contiguous-storage TIFFs) ---
    mm = None
    try:
        mm_candidate = _tf.memmap(tiff_path, mode='r')
        # dtype comparison ignores byte-order: tifffile.memmap returns the
        # native byte order of the TIFF on disk (e.g. '>u2' big-endian for
        # ImageJ TIFs), while series.dtype reports the canonical numpy form
        # (e.g. 'uint16'). Strict `==` fails on byte-order mismatch even
        # though the data is functionally identical (numpy auto-byteswaps on
        # read). Compare kind+itemsize instead.
        _mm_kind_size = (mm_candidate.dtype.kind, mm_candidate.dtype.itemsize)
        _src_kind_size = (dtype.kind, dtype.itemsize)
        if tuple(mm_candidate.shape) == shape and _mm_kind_size == _src_kind_size:
            mm = mm_candidate
            if logger and mm_candidate.dtype != dtype:
                logger.info(
                    f"     memmap byte-order: src={dtype} mm={mm_candidate.dtype} — "
                    f"numpy will auto-byteswap on read.")
        else:
            # Shape/dtype divergence beyond byte-order — bail to per-page
            if logger:
                logger.warning(
                    f"     memmap shape/dtype mismatch: "
                    f"mm.shape={mm_candidate.shape} src.shape={shape}, "
                    f"mm.dtype={mm_candidate.dtype} ({_mm_kind_size}) vs "
                    f"src.dtype={dtype} ({_src_kind_size})")
            del mm_candidate
    except (ValueError, OSError, TypeError) as _mm_err:
        # tifffile raises ValueError for compressed / non-contiguous TIFFs
        if logger:
            logger.info(f"     memmap unavailable ({type(_mm_err).__name__}); "
                        f"falling back to per-page read")
        mm = None

    zarr_out = workspace.create_array(name, out_shape, dtype=dtype, chunks=chunks)

    n_z = out_shape[0]
    n_slabs = (n_z + slab_pages - 1) // slab_pages
    log_every = max(1, n_slabs // 20)

    if logger:
        ch_str = (f"channel {select_channel}/{n_channels}"
                  if select_channel is not None else "single-channel")
        path_str = "memmap" if mm is not None else "per-page"
        logger.info(
            f"     Streaming TIFF -> zarr '{name}' [{path_str}]: "
            f"src axes={axes} shape={shape} pages={n_pages}, "
            f"out shape={out_shape} dtype={dtype} chunks={chunks}, {ch_str}")

    # --- Phase 3: stream slabs ---
    if mm is not None:
        try:
            for slab_idx, z_start in enumerate(range(0, n_z, slab_pages)):
                z_end = min(z_start + slab_pages, n_z)
                if axes == 'ZYX':
                    slab = np.asarray(mm[z_start:z_end], dtype=dtype)
                elif axes == 'ZCYX':
                    slab = np.asarray(mm[z_start:z_end, select_channel, :, :], dtype=dtype)
                else:  # CZYX
                    slab = np.asarray(mm[select_channel, z_start:z_end, :, :], dtype=dtype)
                zarr_out[z_start:z_end] = slab
                del slab

                if logger and (slab_idx + 1) % log_every == 0:
                    logger.info(f"       TIFF->zarr (memmap): {z_end}/{n_z} Z planes")
        finally:
            # Drop reference; OS reclaims mapped pages after process closes
            # the underlying file handle (numpy memmap).
            del mm
    else:
        # Per-page fallback (multi-IFD non-contiguous TIFFs).
        if axes == 'ZYX':
            expected_pages = n_z
            def page_for_z(z): return z
        elif axes == 'ZCYX':
            expected_pages = n_z * shape[1]  # Z * C
            _ch = select_channel
            _C = shape[1]
            def page_for_z(z, _c=_ch, _CC=_C): return z * _CC + _c
        else:  # CZYX
            expected_pages = shape[0] * n_z  # C * Z
            _ch = select_channel
            _Z = n_z
            def page_for_z(z, _c=_ch, _ZZ=_Z): return _c * _ZZ + z

        if n_pages < expected_pages:
            raise ValueError(
                f"TIFF has {n_pages} pages but axes={axes} shape={shape} "
                f"requires {expected_pages}. This is the ImageJ-big-TIFF case "
                f"(single-IFD contiguous) but memmap was unavailable. Cannot stream.")

        with _tf.TiffFile(tiff_path) as tif:
            pages = tif.pages
            for slab_idx, z_start in enumerate(range(0, n_z, slab_pages)):
                z_end = min(z_start + slab_pages, n_z)
                slab_buf = np.empty((z_end - z_start, out_shape[1], out_shape[2]),
                                     dtype=dtype)
                for local_z, z in enumerate(range(z_start, z_end)):
                    page_idx = page_for_z(z)
                    slab_buf[local_z] = pages[page_idx].asarray()
                zarr_out[z_start:z_end] = slab_buf
                del slab_buf

                if logger and (slab_idx + 1) % log_every == 0:
                    logger.info(f"       TIFF->zarr (pages): {z_end}/{n_z} Z planes")

    if logger:
        logger.info(f"     TIFF->zarr '{name}' complete: {n_z} Z planes streamed")

    return workspace.open_array(name)


def tiff_labels_to_per_class_zarrs(tiff_path, channel_map, workspace, *,
                                    slab_pages=8, logger=None):
    """Stream a label TIFF directly into per-class uint8 zarrs in a single pass.

    Replaces the legacy two-step ``imread → labels_to_zarr_channels`` for the
    chunked entry path. Reads the source TIFF slab-by-slab (memmap if
    available, per-page fallback otherwise), and for each Z-slab dispatches
    boolean masks ``(slab == label_val).astype(uint8)`` into one zarr per
    class. For model_type 4 also writes a combined ``spines = cores |
    membranes`` channel. Never materializes the full labels volume.

    Parameters
    ----------
    tiff_path : str
        Path to source label TIFF (3D ZYX uint8/uint16 expected).
    channel_map : dict
        ``{label_value: class_name}`` (e.g., ``{1: 'dendrites', 2: 'spine_cores',
        3: 'spine_membranes', 4: 'necks', 5: 'soma'}`` for model_type 4).
    workspace : ZarrWorkspace
        Target workspace.
    slab_pages : int
        Z planes per slab read iteration.
    logger : logging.Logger, optional

    Returns
    -------
    dict
        ``{class_name: zarr.Array}`` with one entry per channel_map value plus
        an optional ``'spines'`` combined channel for model_type 4.
    """
    import tifffile as _tf

    with _tf.TiffFile(tiff_path) as tif:
        series = tif.series[0]
        axes_raw = str(series.axes)
        shape = tuple(int(s) for s in series.shape)
        dtype = np.dtype(series.dtype)
        n_pages = len(tif.pages)

    # Normalize tifffile 'Q' axis placeholder for label TIFFs (RESPAN convention).
    if axes_raw == 'QYX':
        axes = 'ZYX'
    else:
        axes = axes_raw

    if axes != 'ZYX':
        raise ValueError(
            f"Label TIFF must be 3D ZYX; got axes={axes!r} (raw={axes_raw!r}) shape={shape}")

    # Try memmap (handles ImageJ big-TIFF and contiguous-storage labels);
    # fallback to per-page reads (handles compressed multi-IFD labels —
    # the common case for nnU-Net-saved labels with zlib compression).
    mm = None
    try:
        mm_candidate = _tf.memmap(tiff_path, mode='r')
        # Same byte-order-tolerant dtype compare as tiff_to_zarr_3d above.
        _mm_kind_size = (mm_candidate.dtype.kind, mm_candidate.dtype.itemsize)
        _src_kind_size = (dtype.kind, dtype.itemsize)
        if tuple(mm_candidate.shape) == shape and _mm_kind_size == _src_kind_size:
            mm = mm_candidate
        else:
            del mm_candidate
    except (ValueError, OSError, TypeError) as _mm_err:
        if logger:
            logger.info(
                f"     Label memmap unavailable ({type(_mm_err).__name__}); "
                f"falling back to per-page read")
        mm = None

    # Allocate per-class zarrs upfront so the dispatch loop is straightforward.
    result = {}
    for label_val, class_name in channel_map.items():
        result[class_name] = workspace.create_array(
            class_name, shape, dtype=np.uint8, chunks=STREAMING_CHUNKS)

    has_cores_and_membranes = (
        'spine_cores' in result and 'spine_membranes' in result)
    if has_cores_and_membranes:
        result['spines'] = workspace.create_array(
            'spines', shape, dtype=np.uint8, chunks=STREAMING_CHUNKS)

    n_z = shape[0]
    n_slabs = (n_z + slab_pages - 1) // slab_pages
    log_every = max(1, n_slabs // 20)

    if logger:
        path_str = "memmap" if mm is not None else "per-page"
        logger.info(
            f"     Streaming label TIFF → per-class zarrs [{path_str}]: "
            f"shape={shape} dtype={dtype} pages={n_pages}, "
            f"classes={list(result.keys())}")

    if mm is not None:
        try:
            for slab_idx, z_start in enumerate(range(0, n_z, slab_pages)):
                z_end = min(z_start + slab_pages, n_z)
                # Single slab read; dispatch to multiple per-class zarrs
                slab = np.asarray(mm[z_start:z_end])
                for label_val, class_name in channel_map.items():
                    result[class_name][z_start:z_end] = (slab == label_val).astype(np.uint8)
                if has_cores_and_membranes:
                    result['spines'][z_start:z_end] = ((slab == 2) | (slab == 3)).astype(np.uint8)
                del slab

                if logger and (slab_idx + 1) % log_every == 0:
                    logger.info(f"       Labels TIFF->zarrs (memmap): {z_end}/{n_z} Z planes")
        finally:
            del mm
    else:
        # Per-page fallback. n_pages == n_z is the standard case for
        # zlib-compressed labels written by RESPAN.
        if n_pages < n_z:
            raise ValueError(
                f"Label TIFF has {n_pages} pages but axes=ZYX shape={shape} "
                f"requires {n_z}. Cannot stream without memmap.")

        with _tf.TiffFile(tiff_path) as tif:
            pages = tif.pages
            for slab_idx, z_start in enumerate(range(0, n_z, slab_pages)):
                z_end = min(z_start + slab_pages, n_z)
                slab_buf = np.empty((z_end - z_start, shape[1], shape[2]), dtype=dtype)
                for local_z, z in enumerate(range(z_start, z_end)):
                    slab_buf[local_z] = pages[z].asarray()
                # Dispatch slab to per-class zarrs (single read amortizes over all classes)
                for label_val, class_name in channel_map.items():
                    result[class_name][z_start:z_end] = (slab_buf == label_val).astype(np.uint8)
                if has_cores_and_membranes:
                    result['spines'][z_start:z_end] = ((slab_buf == 2) | (slab_buf == 3)).astype(np.uint8)
                del slab_buf

                if logger and (slab_idx + 1) % log_every == 0:
                    logger.info(f"       Labels TIFF->zarrs (pages): {z_end}/{n_z} Z planes")

    if logger:
        logger.info(f"     Labels TIFF->zarrs complete: {n_z} Z planes, {len(result)} classes")

    return result


# ============================================================================
# ChunkIterator — Yields chunk coordinates with halo
# ============================================================================
class ChunkIterator:
    """Yields (core_slices, halo_slices, pad_widths) for processing a volume in chunks.

    Parameters
    ----------
    shape : tuple
        Volume shape (Z, Y, X).
    core_size : tuple
        Core chunk size (Z, Y, X) — the region that gets written to output.
    halo : tuple
        Halo size per side (Z, Y, X) — extra context read but not written.
    """

    def __init__(self, shape, core_size=(64, 512, 512), halo=(0, 0, 0)):
        self.shape = shape
        self.core_size = tuple(min(c, s) for c, s in zip(core_size, shape))
        self.halo = halo

    def __iter__(self):
        """Yield (core_slices, read_slices, pad_before, pad_after) tuples.

        core_slices : tuple of slice — where to write in the output
        read_slices : tuple of slice — where to read from the input (with halo)
        pad_before  : tuple of int — padding needed at start (if halo exceeds volume)
        pad_after   : tuple of int — padding needed at end
        """
        for z in range(0, self.shape[0], self.core_size[0]):
            for y in range(0, self.shape[1], self.core_size[1]):
                for x in range(0, self.shape[2], self.core_size[2]):
                    core_slices = []
                    read_slices = []
                    pad_before = []
                    pad_after = []

                    starts = (z, y, x)
                    for dim in range(3):
                        core_start = starts[dim]
                        core_end = min(core_start + self.core_size[dim], self.shape[dim])

                        read_start = core_start - self.halo[dim]
                        read_end = core_end + self.halo[dim]

                        # Clamp to volume bounds
                        actual_start = max(0, read_start)
                        actual_end = min(self.shape[dim], read_end)

                        # Padding needed where halo extends beyond volume
                        pb = actual_start - read_start  # > 0 if halo exceeded start
                        pa = read_end - actual_end      # > 0 if halo exceeded end

                        core_slices.append(slice(core_start, core_end))
                        read_slices.append(slice(actual_start, actual_end))
                        pad_before.append(pb)
                        pad_after.append(pa)

                    yield (
                        tuple(core_slices),
                        tuple(read_slices),
                        tuple(pad_before),
                        tuple(pad_after),
                    )

    def __len__(self):
        n = 1
        for dim in range(3):
            n *= math.ceil(self.shape[dim] / self.core_size[dim])
        return n


# ============================================================================
# GPU-Chunked Distance Transform
# ============================================================================
def distance_transform_gpu_chunked(mask_zarr, output_zarr, spacing, max_dist=None,
                                    core_size=(64, 512, 512), logger=None):
    """Compute distance transform on a binary mask using GPU chunks with halo overlap.

    Parameters
    ----------
    mask_zarr : zarr.Array
        Input binary mask (nonzero = foreground). Distance is computed from foreground.
    output_zarr : zarr.Array
        Output float32 zarr array (same shape). Written in-place.
    spacing : tuple of float
        Voxel spacing (Z, Y, X) in physical units (microns).
    max_dist : float, optional
        Maximum relevant distance in physical units (microns). Determines halo size.
        If None, uses half the smallest volume dimension * min spacing.
    core_size : tuple
        Core chunk size for processing.
    logger : logging.Logger, optional
    """
    _require_cupy("distance_transform_gpu_chunked")

    shape = mask_zarr.shape

    if max_dist is None:
        max_dist = min(shape) * min(spacing) / 2

    # Calculate anisotropic halo: enough voxels to cover max_dist in each axis.
    # Halo is NEVER reduced — it guarantees measurement correctness up to max_dist.
    halo = tuple(int(math.ceil(max_dist / s)) + 2 for s in spacing)

    # Fit chunks in GPU VRAM by shrinking core_size only (never halo).
    # CuPy PBA 3D EDT needs ~5x the chunk volume for temporaries.
    try:
        gpu_free, _ = cp.cuda.runtime.memGetInfo()
    except Exception:
        gpu_free = 8 * GB  # conservative fallback

    gpu_budget = int(gpu_free * 0.60)
    bytes_per_voxel_edt = 5 * 4  # ~5 float32 arrays for PBA internals

    core_size = list(core_size)
    for _ in range(10):
        full_shape = tuple(c + 2 * h for c, h in zip(core_size, halo))
        chunk_bytes = int(np.prod(full_shape)) * bytes_per_voxel_edt
        if chunk_bytes <= gpu_budget:
            break
        # Shrink the largest core dimension — halo stays intact
        max_core_dim = max(range(3), key=lambda d: core_size[d])
        core_size[max_core_dim] = max(16, core_size[max_core_dim] // 2)
        if logger:
            logger.info(f"       VRAM: shrinking core to {tuple(core_size)} (halo preserved at {halo})")
    else:
        # Even minimum core doesn't fit — halo alone exceeds VRAM
        min_halo_shape = tuple(2 * h for h in halo)
        min_bytes = int(np.prod(min_halo_shape)) * bytes_per_voxel_edt
        raise MemoryError(
            f"GPU-chunked EDT: halo alone ({halo}) requires {min_bytes / GB:.1f} GB "
            f"but only {gpu_budget / GB:.1f} GB available. "
            f"Reduce max_dist (currently {max_dist:.1f} \u00b5m) or use a GPU with more VRAM."
        )

    core_size = tuple(core_size)

    # Halo guarantees per-chunk EDT correctness only up to ~min(halo)-2 voxels
    # Euclidean. Beyond that, per-chunk values diverge (seams at chunk boundaries,
    # verified in T_CROP at Y=511/X=511/X=1023/Z=191/255/319). The output is left
    # raw because downstream measurement code (spine filter, neck cap) relies on
    # far-field values being LARGE (>threshold) so spurious far objects are
    # filtered OUT. Visualization callers zero the far-field at MIP-save time
    # using the returned max_reliable_vox so the channel doesn't drown in junk.
    max_reliable_vox = float(max(0, min(halo) - 2))

    full_shape = tuple(c + 2 * h for c, h in zip(core_size, halo))
    chunk_mb = int(np.prod(full_shape)) * bytes_per_voxel_edt / MB
    # CuPy PBA EDT peak VRAM is ~13-16x volume size. Pre-compute
    # the predicted GPU peak so we can warn or refuse if it exceeds available
    # VRAM — common failure mode is users setting spine_dist too high for
    # their GPU (e.g., spine_dist=5 + 45nm XY = 333-vox XY halo → 9.8 GB
    # chunk × 16 = 157 GB VRAM peak on a 24 GB Titan).
    _PBA_VRAM_MULTIPLIER = 16
    _predicted_vram_mb = chunk_mb * _PBA_VRAM_MULTIPLIER
    try:
        import cupy as _cp_for_check
        _gpu_total_mb = _cp_for_check.cuda.runtime.memGetInfo()[1] / MB
    except Exception:
        _gpu_total_mb = None

    if logger:
        logger.info(f"     GPU-chunked EDT: shape={shape}, spacing={spacing}")
        logger.info(f"       max_dist={max_dist:.1f} µm, halo={halo}, core={core_size}, "
                     f"chunk_with_halo={full_shape} ({chunk_mb:.0f} MB est., "
                     f"predicted GPU peak ~{_predicted_vram_mb:.0f} MB at {_PBA_VRAM_MULTIPLIER}x CuPy PBA factor)")
        # WARN (not abort) when prediction is between GPU size and 2x GPU size —
        # often the actual peak comes in lower due to CuPy PBA padding alignment.
        if _gpu_total_mb is not None and _predicted_vram_mb > 0.90 * _gpu_total_mb:
            logger.warning(
                f"     ⚠ Predicted VRAM peak {_predicted_vram_mb:.0f} MB is close to / "
                f"exceeds GPU total {_gpu_total_mb:.0f} MB. Actual PBA peak may come "
                f"in lower (alignment-dependent). If you see CUDA OOM, reduce "
                f"settings.neuron_spine_dist (currently produces halo={halo}).")
        logger.info(f"       max_reliable_vox={max_reliable_vox:.0f} (raw output kept; visualization clamps post-hoc)")

    # Hard guard: only abort when predicted VRAM is catastrophically high
    # (>2x GPU memory). At 1-2x, the 16x PBA multiplier is often a worst-case
    # overestimate — CuPy's PBA implementation handles dim-padding-alignment
    # such that small/medium chunks see lower-than-predicted peaks. Let the
    # try/except below handle the actual cuda.runtime.runtimeError if it
    # really overflows. T7 canonical (9685299, 2962 spines, byte-identical)
    # has chunk_with_halo=2.2 GB → predicted 35 GB but actual peak ~7.5 GB.
    # Pre-existing bench validates that the prediction is conservative here.
    if _gpu_total_mb is not None and _predicted_vram_mb > 2.0 * _gpu_total_mb:
        # Calculate the spine_dist (µm) that would fit. Halo at spine_dist_um
        # is `cap_um / spacing_xy` → halo scales linearly with spine_dist.
        # Volume scales with halo³ (chunk_with_halo dim), so VRAM scales
        # roughly with spine_dist³. Solve: spine_dist_safe = current_dist *
        # (target_vram / predicted_vram)^(1/3).
        _safe_ratio = (0.85 * _gpu_total_mb / _predicted_vram_mb) ** (1.0 / 3.0)
        # max_dist passed in is already 3x spine_dist_um (per neck cap convention)
        _current_spine_dist_um = max_dist / 3.0
        _safe_spine_dist_um = _current_spine_dist_um * _safe_ratio
        msg = (
            f"GPU-chunked EDT predicted VRAM peak {_predicted_vram_mb:.0f} MB exceeds "
            f"90% of GPU total {_gpu_total_mb:.0f} MB. CuPy PBA needs ~{_PBA_VRAM_MULTIPLIER}x "
            f"chunk_with_halo size ({chunk_mb:.0f} MB × {_PBA_VRAM_MULTIPLIER} = "
            f"{_predicted_vram_mb:.0f} MB).\n"
            f"\n"
            f"  Cause: settings.neuron_spine_dist={_current_spine_dist_um:.1f} µm produces "
            f"halo={halo} voxels, which is too large for this GPU.\n"
            f"\n"
            f"  Fix options:\n"
            f"  1. Reduce settings.neuron_spine_dist to ≤{_safe_spine_dist_um:.1f} µm "
            f"(current {_current_spine_dist_um:.1f} µm). Halo size scales linearly with "
            f"spine_dist; VRAM peak scales cubically.\n"
            f"  2. Use a GPU with more VRAM (need ≥{_predicted_vram_mb/1024:.1f} GB).\n"
            f"  3. Disable GPU EDT for this stage (CPU fallback) — slower but works.\n"
            f"\n"
            f"  Halo size is determined by physics (max measurement reach) and CANNOT "
            f"be reduced without losing measurement correctness for distant features."
        )
        if logger:
            logger.error(msg)
        raise MemoryError(msg)

    chunks = ChunkIterator(shape, core_size=core_size, halo=halo)
    n_chunks = len(chunks)

    for idx, (core_sl, read_sl, pad_b, pad_a) in enumerate(chunks):
        # Read chunk with halo from zarr
        chunk_np = np.array(mask_zarr[read_sl])

        # Pad if halo extends beyond volume edges
        if any(p > 0 for p in pad_b) or any(p > 0 for p in pad_a):
            pad_widths = [(pb, pa) for pb, pa in zip(pad_b, pad_a)]
            # Pad with 0 (background) so EDT extends naturally at volume edges.
            # Padding with 1 (foreground) creates fake dendrites at boundaries,
            # capping all distances at half the smallest dimension.
            chunk_np = np.pad(chunk_np, pad_widths, mode='constant', constant_values=0)

        # Transfer to GPU and compute EDT
        chunk_gpu = cp.asarray(chunk_np)
        # EDT of inverted mask: distance from background to nearest foreground.
        # No sampling — output in voxel units to match adaptive_distance_transform
        # (non-chunked path). Downstream code converts voxels→microns where needed.
        edt_gpu = gpu_edt(chunk_gpu == 0)

        # Extract core region (strip halo)
        core_local = []
        for dim in range(3):
            local_start = halo[dim]
            core_extent = core_sl[dim].stop - core_sl[dim].start
            core_local.append(slice(local_start, local_start + core_extent))

        core_result = cp.asnumpy(edt_gpu[tuple(core_local)])

        # Write core to output zarr (raw EDT values; visualization callers
        # apply halo-reach zero-out at MIP save time, see comment at top)
        output_zarr[core_sl] = core_result.astype(np.float32)

        # Free GPU memory
        del chunk_gpu, edt_gpu, core_result
        cp.get_default_memory_pool().free_all_blocks()

        if logger and (idx + 1) % max(1, n_chunks // 5) == 0:
            logger.info(f"       EDT chunk {idx + 1}/{n_chunks}")

    if logger:
        logger.info(f"     GPU-chunked EDT complete.")

    return max_reliable_vox


# ============================================================================
# Connected Components Streaming (2-pass with union-find)
# ============================================================================
class UnionFind:
    """Weighted union-find with path compression for label merging."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def get_mapping(self):
        """Return dict mapping every label to its canonical root."""
        mapping = {}
        for label in self.parent:
            mapping[label] = self.find(label)
        return mapping


def connected_components_streaming(mask_zarr, output_zarr, slab_depth=64,
                                    min_volume=0, connectivity=1, logger=None):
    """Label connected components on a binary mask too large for RAM.

    Two-pass algorithm:
    1. For each Z-slab (with 1-slice overlap), run ndimage.label().
       Offset labels to be globally unique. At overlap boundaries, record
       which labels from adjacent slabs are connected using union-find.
    2. Remap all labels to their canonical (merged) IDs, optionally filter by volume.

    Parameters
    ----------
    mask_zarr : zarr.Array
        Input binary mask (nonzero = object).
    output_zarr : zarr.Array
        Output int32 zarr array. Written in-place.
    slab_depth : int
        Number of Z slices per slab.
    min_volume : int
        Minimum voxel count to keep a connected component (0 = keep all).
    connectivity : int
        Passed to `ndimage.generate_binary_structure(3, connectivity)`. 1 gives 6-conn
        (face neighbors), 2 gives 18-conn (face+edge), 3 gives 26-conn (face+edge+corner).
        Default 1 for backward compat; use 3 for thin diagonal structures (filopodia).
    logger : logging.Logger, optional

    Returns
    -------
    tuple (int, dict)
        - Number of connected components (after filtering).
        - Dict mapping label -> [z0, z1, y0, y1, x0, x1] bbox.
    """
    shape = mask_zarr.shape
    struct = ndimage.generate_binary_structure(3, connectivity)

    # Adapt slab depth: ndimage.label allocates ~3x slab size (input bool + int64 output + temp).
    # Keep total slab memory under 15% of available RAM.
    import psutil
    bytes_per_z = int(shape[1]) * int(shape[2]) * 8 * 3  # ~3x for label overhead
    avail = psutil.virtual_memory().available
    if bytes_per_z * slab_depth > avail * 0.15:
        max_depth = max(1, int(avail * 0.15 / bytes_per_z))
        slab_depth = min(slab_depth, max_depth)

    if logger:
        logger.info(f"     Streaming connected components: shape={shape}, slab_depth={slab_depth}")

    uf = UnionFind()
    label_offset = 0
    prev_bottom_plane = None  # Last Z-plane of previous slab (already offset)

    # Pass 1: label each slab, record overlaps
    for z_start in range(0, shape[0], slab_depth):
        z_end = min(z_start + slab_depth, shape[0])

        slab = np.array(mask_zarr[z_start:z_end, :, :])
        slab_labels, n_labels = ndimage.label(slab > 0, structure=struct)

        if n_labels == 0:
            output_zarr[z_start:z_end, :, :] = 0
            prev_bottom_plane = None
            continue

        # Offset labels to be globally unique
        slab_labels[slab_labels > 0] += label_offset

        # Find connections at boundary with previous slab.
        # For connectivity=1 (6-conn) only same-(y,x) voxels on adjacent Z planes
        # are connected. For 18/26-conn, (y,x) and any 2D neighbor on the
        # adjacent plane also connect. Build the appropriate neighbor footprint
        # and union all label pairs that coincide via any offset.
        if prev_bottom_plane is not None:
            top_plane = slab_labels[0, :, :]  # First plane of current slab
            if connectivity <= 1:
                both_nonzero = (prev_bottom_plane > 0) & (top_plane > 0)
                if np.any(both_nonzero):
                    pairs = np.unique(
                        np.column_stack([prev_bottom_plane[both_nonzero],
                                         top_plane[both_nonzero]]),
                        axis=0
                    )
                    for p, c in pairs:
                        uf.union(int(p), int(c))
            else:
                # 3D connectivity >=2 → 2D boundary footprint is full 3x3
                # (edges for conn=2, corners for conn=3 — both treat the
                # boundary plane the same: any in-plane 3x3 neighbor voxel
                # on the top plane connects to a voxel on the bottom plane).
                # We union across all 9 (dy,dx) offsets.
                Hp, Wp = prev_bottom_plane.shape
                prev_nz = prev_bottom_plane > 0
                if np.any(prev_nz) and np.any(top_plane > 0):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                # fast path — exact-overlap case
                                both_nonzero = prev_nz & (top_plane > 0)
                                if not np.any(both_nonzero):
                                    continue
                                pairs = np.unique(
                                    np.column_stack([prev_bottom_plane[both_nonzero],
                                                     top_plane[both_nonzero]]),
                                    axis=0)
                            else:
                                # Shift top_plane by (dy, dx); keep the
                                # overlapping region only.
                                y0 = max(0, dy); y1 = Hp - max(0, -dy)
                                x0 = max(0, dx); x1 = Wp - max(0, -dx)
                                if y0 >= y1 or x0 >= x1:
                                    continue
                                prev_sub = prev_bottom_plane[y0:y1, x0:x1]
                                top_sub = top_plane[y0 - dy:y1 - dy,
                                                     x0 - dx:x1 - dx]
                                both_nonzero = (prev_sub > 0) & (top_sub > 0)
                                if not np.any(both_nonzero):
                                    continue
                                pairs = np.unique(
                                    np.column_stack([prev_sub[both_nonzero],
                                                     top_sub[both_nonzero]]),
                                    axis=0)
                            for p, c in pairs:
                                uf.union(int(p), int(c))

        # Save bottom plane for next slab's overlap check
        prev_bottom_plane = slab_labels[-1, :, :].copy()

        # Write to output
        output_zarr[z_start:z_end, :, :] = slab_labels

        label_offset += n_labels

        if logger:
            logger.info(f"       CC slab z={z_start}:{z_end} — {n_labels} local labels")

    if label_offset == 0:
        if logger:
            logger.info(f"     No objects found in mask.")
        return 0

    # Build global remapping: every label -> its canonical root -> compact ID
    # First, collect all labels that appear
    all_labels = set()
    for z_start in range(0, shape[0], slab_depth):
        z_end = min(z_start + slab_depth, shape[0])
        slab = np.array(output_zarr[z_start:z_end, :, :])
        all_labels.update(np.unique(slab[slab > 0]).tolist())

    # Build compact mapping
    label_to_root = {}
    for label in all_labels:
        label_to_root[int(label)] = uf.find(int(label))

    unique_roots = sorted(set(label_to_root.values()))
    root_to_compact = {root: (i + 1) for i, root in enumerate(unique_roots)}

    full_map = {}
    for label, root in label_to_root.items():
        full_map[label] = root_to_compact[root]

    # Count voxels + compute bboxes per component during relabeling (no extra pass)
    counts = {}
    bboxes = {}

    # Pass 2: relabel + collect bbox + counts
    for z_start in range(0, shape[0], slab_depth):
        z_end = min(z_start + slab_depth, shape[0])
        slab = np.array(output_zarr[z_start:z_end, :, :])

        nonzero_mask = slab > 0
        if not np.any(nonzero_mask):
            continue

        # Vectorized remapping using a lookup array
        max_label = slab.max()
        lut = np.zeros(max_label + 1, dtype=np.int32)
        for old, new in full_map.items():
            if old <= max_label:
                lut[old] = new
        slab_remapped = lut[slab]
        output_zarr[z_start:z_end, :, :] = slab_remapped

        # Collect per-label bboxes and counts using find_objects (single pass per slab)
        slab_max = int(slab_remapped.max())
        if slab_max > 0:
            slab_slices = ndimage.find_objects(slab_remapped, slab_max)
            slab_bc = np.bincount(slab_remapped.ravel())
            for label_val in range(1, slab_max + 1):
                if slab_bc[label_val] == 0:
                    continue
                counts[label_val] = counts.get(label_val, 0) + int(slab_bc[label_val])
                sl_obj = slab_slices[label_val - 1]
                if sl_obj is None:
                    continue
                z0 = sl_obj[0].start + z_start
                z1 = sl_obj[0].stop + z_start
                y0, y1 = sl_obj[1].start, sl_obj[1].stop
                x0, x1 = sl_obj[2].start, sl_obj[2].stop
                if label_val not in bboxes:
                    bboxes[label_val] = [z0, z1, y0, y1, x0, x1]
                else:
                    bb = bboxes[label_val]
                    bb[0] = min(bb[0], z0)
                    bb[1] = max(bb[1], z1)
                    bb[2] = min(bb[2], y0)
                    bb[3] = max(bb[3], y1)
                    bb[4] = min(bb[4], x0)
                    bb[5] = max(bb[5], x1)

    n_components = len(unique_roots)

    # Volume filtering pass
    if min_volume > 0:
        labels_to_remove = {label for label, count in counts.items() if count < min_volume}
        if labels_to_remove and logger:
            logger.info(f"       Removing {len(labels_to_remove)} components with volume < {min_volume} voxels")

        if labels_to_remove:
            for z_start in range(0, shape[0], slab_depth):
                z_end = min(z_start + slab_depth, shape[0])
                slab = np.array(output_zarr[z_start:z_end, :, :])
                mask = np.isin(slab, list(labels_to_remove))
                if np.any(mask):
                    slab[mask] = 0
                    output_zarr[z_start:z_end, :, :] = slab
            n_components -= len(labels_to_remove)

    # Remove filtered labels from bboxes
    if min_volume > 0:
        for label in list(bboxes.keys()):
            if counts.get(label, 0) < min_volume:
                del bboxes[label]

    if logger:
        logger.info(f"     Streaming CC complete: {n_components} components")

    return n_components, bboxes


# ============================================================================
# Per-Object Skeletonization
# ============================================================================
def skeletonize_per_object(labeled_zarr, output_zarr, logger=None):
    """Skeletonize each labeled object independently by extracting its bounding box.

    Since connected components are independent, per-object skeletonization is
    mathematically equivalent to global skeletonization.

    Parameters
    ----------
    labeled_zarr : zarr.Array
        Labeled volume (each connected component has a unique integer label).
    output_zarr : zarr.Array
        Output binary skeleton volume. Written in-place.
    logger : logging.Logger, optional

    Returns
    -------
    dict
        Mapping of label -> skeleton coordinates (N, 3) numpy arrays.
    """
    shape = labeled_zarr.shape

    # Get all unique labels (read in slabs to avoid loading full volume)
    all_labels = set()
    slab_depth = 64
    for z_start in range(0, shape[0], slab_depth):
        z_end = min(z_start + slab_depth, shape[0])
        slab = np.array(labeled_zarr[z_start:z_end, :, :])
        all_labels.update(np.unique(slab).tolist())
    all_labels.discard(0)

    if logger:
        logger.info(f"     Per-object skeletonization: {len(all_labels)} objects")

    skeleton_coords = {}

    # Get bounding box of each object by scanning slabs
    bboxes = _get_object_bboxes(labeled_zarr, all_labels, slab_depth)

    for label_val in sorted(all_labels):
        if label_val not in bboxes:
            continue

        bbox = bboxes[label_val]
        z0, z1, y0, y1, x0, x1 = bbox

        # Build the per-object binary mask slab-by-slab into uint8 directly.
        # Previous: subvol = np.array(zarr[bbox]) materialized 4-byte int32 +
        # the .astype(uint8) cast added a transient — at T_LARGE one dendrite
        # bbox is (653, 4022, 4698) = 49 GB int32, instant OOM. Direct uint8
        # alloc + per-slab-equality drops peak to ~12 GB for that bbox.
        _bbox_shape = (z1 - z0, y1 - y0, x1 - x0)
        _row_bytes = int(_bbox_shape[1]) * int(_bbox_shape[2]) * 4  # int32 slab read
        _slab_z = max(1, min(64, int(2 * 1024**3 / max(_row_bytes, 1))))
        binary = np.zeros(_bbox_shape, dtype=np.uint8)
        for _z in range(0, _bbox_shape[0], _slab_z):
            _ze = min(_z + _slab_z, _bbox_shape[0])
            _sub_slab = np.asarray(labeled_zarr[z0 + _z:z0 + _ze, y0:y1, x0:x1])
            binary[_z:_ze] = (_sub_slab == label_val).astype(np.uint8, copy=False)
            del _sub_slab

        if int(binary.sum()) < 3:
            continue

        # Skeletonize. skeletonize() on a 12 GB uint8 binary uses up to
        # ~3x its size internally (per-iteration scratch). For a 49 GB
        # bbox at T_LARGE this is the dominant transient stage.
        try:
            skel = morphology.skeletonize(binary > 0)
        except Exception:
            del binary
            continue
        del binary

        # Write skeleton back to output slab-by-slab. Avoids full-bbox
        # out_sub materialization (49 GB at T_LARGE).
        _out_slab_z = max(1, min(64, int(2 * 1024**3 / max(int(_bbox_shape[1]) * int(_bbox_shape[2]), 1))))
        for _z in range(0, _bbox_shape[0], _out_slab_z):
            _ze = min(_z + _out_slab_z, _bbox_shape[0])
            _out_sub_slab = np.asarray(output_zarr[z0 + _z:z0 + _ze, y0:y1, x0:x1])
            _out_sub_slab[skel[_z:_ze] > 0] = 1
            output_zarr[z0 + _z:z0 + _ze, y0:y1, x0:x1] = _out_sub_slab
            del _out_sub_slab

        # Store skeleton coordinates in global space
        coords = np.argwhere(skel > 0)
        coords[:, 0] += z0
        coords[:, 1] += y0
        coords[:, 2] += x0
        skeleton_coords[label_val] = coords

    if logger:
        logger.info(f"     Skeletonization complete: {len(skeleton_coords)} skeletons")

    return skeleton_coords


def labels_to_zarr_channels(labels_vol, channel_map, workspace, slab_depth=128, logger=None):
    """Extract label channels from a segmentation volume to individual zarr arrays.

    Reads labels_vol in Z-slabs and writes boolean/uint8 zarr arrays per channel.
    For model_type 4, also writes a combined 'spines' = cores | membranes channel.
    Never materializes more than one slab of the input at a time.

    Parameters
    ----------
    labels_vol : numpy.ndarray
        Full segmentation volume (Z, Y, X) with integer labels.
    channel_map : dict
        Mapping of {label_value: channel_name}. E.g. {1: 'dendrites', 2: 'spine_cores', ...}
    workspace : ZarrWorkspace
        Zarr workspace for output arrays.
    slab_depth : int
        Z-slices per slab.
    logger : logging.Logger, optional

    Returns
    -------
    dict
        Mapping of {channel_name: zarr.Array} for each extracted channel.
    """
    shape = labels_vol.shape
    result = {}

    for label_val, name in channel_map.items():
        result[name] = workspace.create_array(name, shape, dtype=np.uint8,
                                                chunks=STREAMING_CHUNKS)

    has_cores_and_membranes = 'spine_cores' in result and 'spine_membranes' in result
    if has_cores_and_membranes:
        result['spines'] = workspace.create_array('spines', shape, dtype=np.uint8,
                                                    chunks=STREAMING_CHUNKS)

    for z in range(0, shape[0], slab_depth):
        ze = min(z + slab_depth, shape[0])
        slab = labels_vol[z:ze]

        for label_val, name in channel_map.items():
            result[name][z:ze] = (slab == label_val).astype(np.uint8)

        if has_cores_and_membranes:
            result['spines'][z:ze] = ((slab == 2) | (slab == 3)).astype(np.uint8)

    if logger:
        total_mb = sum(int(np.prod(np.array(shape, dtype=np.int64))) for _ in result) / MB
        logger.info(f"     Extracted {len(result)} label channels to zarr "
                     f"({len(result)} arrays, ~{total_mb:.0f} MB uncompressed each)")

    return result


def chunked_clear_border(labels, faces='yx', slab_depth=None, logger=None):
    """Zero out labels that touch specified boundary faces.

    Equivalent to ``skimage.segmentation.clear_border`` restricted to the given
    faces, but slab-iterated to avoid materializing or padding the full volume.

    Defaults match the legacy RESPAN trick:
        padded = np.pad(labels, ((1,1),(0,0),(0,0)), mode='constant', values=0)
        labels = segmentation.clear_border(padded)[1:-1]
    i.e. only Y and X boundary-touching labels are removed; Z-boundary
    labels are preserved. For a 40+ GB int32 volume the pad trick allocates
    an equal-sized transient — this helper uses O(slab) memory instead.

    Parameters
    ----------
    labels : numpy.ndarray or zarr.Array (3D)
        Labeled volume. Modified in place for both backings (numpy direct
        assignment; zarr slab write-back).
    faces : str
        'yx' (default), 'z', or 'all'. Which boundary faces are treated as
        borders. Multiple letters may be combined (e.g. 'yxz' == 'all').
    slab_depth : int, optional
        Override the adaptive slab depth (default is capped by RAM).
    logger : logger, optional
        If provided, logs the number of border labels discovered.

    Returns
    -------
    labels : same object as input
    """
    shape = labels.shape
    assert len(shape) == 3, "chunked_clear_border expects a 3D labeled volume"

    # Adaptive slab depth — mirrors _get_object_bboxes rule.
    if slab_depth is None:
        per_z = int(shape[1]) * int(shape[2]) * np.dtype(labels.dtype).itemsize
        avail = psutil.virtual_memory().available
        max_slab_bytes = min(avail * 0.25, 6 * GB)
        slab_depth = max(1, min(64, int(max_slab_bytes / max(per_z, 1))))

    _faces = set(faces.lower())
    _do_z = 'z' in _faces or 'all' in _faces or faces == 'all'
    _do_yx = ('y' in _faces or 'x' in _faces
              or 'all' in _faces or faces == 'all' or faces == 'yx')

    # ---- Pass 1: collect labels present on requested boundary faces ----
    border_labels = set()

    if _do_z and shape[0] > 0:
        # Only need the first and last Z slice (1 slab each).
        z_first = np.asarray(labels[0:1])
        border_labels.update(np.unique(z_first).tolist())
        del z_first
        if shape[0] > 1:
            z_last = np.asarray(labels[shape[0] - 1:shape[0]])
            border_labels.update(np.unique(z_last).tolist())
            del z_last

    if _do_yx:
        for zs in range(0, shape[0], slab_depth):
            ze = min(zs + slab_depth, shape[0])
            slab = np.asarray(labels[zs:ze])
            # Y=0 row and Y=last row (full X width, this Z range)
            border_labels.update(np.unique(slab[:, 0, :]).tolist())
            border_labels.update(np.unique(slab[:, -1, :]).tolist())
            # X=0 col and X=last col
            border_labels.update(np.unique(slab[:, :, 0]).tolist())
            border_labels.update(np.unique(slab[:, :, -1]).tolist())
            del slab

    border_labels.discard(0)  # background is not a "border label"

    if logger is not None:
        logger.info(f"    chunked_clear_border: {len(border_labels)} border labels identified (faces={faces})")

    if not border_labels:
        return labels

    # ---- Pass 2: scrub (slab read, np.isin mask, write back) ----
    border_arr = np.array(sorted(border_labels), dtype=labels.dtype)
    is_zarr = hasattr(labels, 'chunks')
    for zs in range(0, shape[0], slab_depth):
        ze = min(zs + slab_depth, shape[0])
        slab = np.asarray(labels[zs:ze])
        mask = np.isin(slab, border_arr)
        if mask.any():
            slab[mask] = 0
            if is_zarr:
                labels[zs:ze] = slab
            else:
                # numpy: in-place update via slice assignment
                labels[zs:ze] = slab
        del slab, mask

    return labels


def _get_object_bboxes(labeled_zarr, labels, slab_depth=64):
    """Compute bounding boxes for each label by scanning the volume in Z-slabs.

    Returns dict of {label: (z0, z1, y0, y1, x0, x1)}.
    """
    shape = labeled_zarr.shape
    # Adapt slab depth to avoid OOM: cap each slab at min(25% of free RAM, 6 GB).
    # The hard cap prevents edge-case failures when available RAM is large but
    # fragmented, while 25% keeps slabs efficient when RAM is plentiful.
    slab_bytes = int(shape[1]) * int(shape[2]) * np.dtype(labeled_zarr.dtype).itemsize * slab_depth
    avail = psutil.virtual_memory().available
    max_slab_bytes = min(avail * 0.25, 6 * GB)
    if slab_bytes > max_slab_bytes:
        per_z = int(shape[1]) * int(shape[2]) * np.dtype(labeled_zarr.dtype).itemsize
        slab_depth = max(1, int(max_slab_bytes / per_z))
    # Initialize with inverted bounds
    bboxes = {}
    for label_val in labels:
        bboxes[label_val] = [shape[0], 0, shape[1], 0, shape[2], 0]  # z_min, z_max, y_min, y_max, x_min, x_max

    labels_set = set(labels)
    for z_start in range(0, shape[0], slab_depth):
        z_end = min(z_start + slab_depth, shape[0])
        slab = np.array(labeled_zarr[z_start:z_end, :, :])

        # For large slabs, avoid np.unique (OOM on 3B+ elements) and
        # np.bincount (OOM when max_label is huge due to CC offset labeling).
        # Instead, directly check which known labels are present.
        if slab.size > 500_000_000:
            present = set()
            for lbl in labels_set:
                if np.any(slab == lbl):
                    present.add(lbl)
        else:
            present = set(np.unique(slab)) & labels_set
        if not present:
            continue

        for label_val in present:
            coords = np.argwhere(slab == label_val)
            z_coords = coords[:, 0] + z_start
            y_coords = coords[:, 1]
            x_coords = coords[:, 2]

            bb = bboxes[label_val]
            bb[0] = min(bb[0], int(z_coords.min()))
            bb[1] = max(bb[1], int(z_coords.max()) + 1)
            bb[2] = min(bb[2], int(y_coords.min()))
            bb[3] = max(bb[3], int(y_coords.max()) + 1)
            bb[4] = min(bb[4], int(x_coords.min()))
            bb[5] = max(bb[5], int(x_coords.max()) + 1)

    # Filter out labels with no voxels found (bounds never updated)
    return {label: tuple(bb) for label, bb in bboxes.items() if bb[1] > bb[0]}


def _streaming_bboxes_and_sizes(labeled_zarr, n_labels, slab_depth=64, logger=None):
    """Compute bounding boxes and voxel counts for all labels in one pass.

    Uses ndimage.find_objects per slab (single O(N) scan) plus np.bincount
    for sizes.  Much faster than _get_object_bboxes for compact sequential
    labels (1..n_labels) from connected_components_streaming.

    Parameters
    ----------
    labeled_zarr : zarr.Array
        Labeled volume with compact labels 1..n_labels.
    n_labels : int
        Number of labels (max label value).
    slab_depth : int
        Z-slices per read.
    logger : logging.Logger, optional

    Returns
    -------
    bboxes : dict
        {label: (z0, z1, y0, y1, x0, x1)} for labels with size > 0.
    sizes : np.ndarray
        int64 array of length n_labels+1 with voxel counts per label.
    """
    shape = labeled_zarr.shape

    # Adapt slab depth: cap at min(25% of free RAM, 6 GB) — same policy as
    # _get_object_bboxes for consistency across streaming operations.
    slab_bytes = int(shape[1]) * int(shape[2]) * np.dtype(labeled_zarr.dtype).itemsize * slab_depth
    avail = psutil.virtual_memory().available
    max_slab_bytes = min(avail * 0.25, 6 * GB)
    if slab_bytes > max_slab_bytes:
        per_z = int(shape[1]) * int(shape[2]) * np.dtype(labeled_zarr.dtype).itemsize
        slab_depth = max(1, int(max_slab_bytes / per_z))

    z_min = np.full(n_labels + 1, shape[0], dtype=np.int32)
    z_max = np.zeros(n_labels + 1, dtype=np.int32)
    y_min = np.full(n_labels + 1, shape[1], dtype=np.int32)
    y_max = np.zeros(n_labels + 1, dtype=np.int32)
    x_min = np.full(n_labels + 1, shape[2], dtype=np.int32)
    x_max = np.zeros(n_labels + 1, dtype=np.int32)
    sizes = np.zeros(n_labels + 1, dtype=np.int64)

    for z_start in range(0, shape[0], slab_depth):
        z_end = min(z_start + slab_depth, shape[0])
        slab = np.array(labeled_zarr[z_start:z_end, :, :])

        # Sizes via bincount
        bc = np.bincount(slab.ravel(), minlength=n_labels + 1)
        sizes[:min(len(bc), n_labels + 1)] += bc[:min(len(bc), n_labels + 1)]

        # Bboxes via find_objects (single O(N) scan of the slab)
        max_label_in_slab = slab.max()
        if max_label_in_slab == 0:
            continue
        slab_slices = ndimage.find_objects(slab, max_label_in_slab)
        for i, sl in enumerate(slab_slices):
            label = i + 1
            if sl is None or label > n_labels:
                continue
            z_min[label] = min(z_min[label], z_start + sl[0].start)
            z_max[label] = max(z_max[label], z_start + sl[0].stop)
            y_min[label] = min(y_min[label], sl[1].start)
            y_max[label] = max(y_max[label], sl[1].stop)
            x_min[label] = min(x_min[label], sl[2].start)
            x_max[label] = max(x_max[label], sl[2].stop)

        if logger:
            n_present = np.count_nonzero(bc[1:min(len(bc), n_labels + 1)])
            logger.info(f"       Bbox slab z={z_start}:{z_end} — {n_present} labels present")

    bboxes = {}
    for label in range(1, n_labels + 1):
        if sizes[label] > 0:
            bboxes[label] = (int(z_min[label]), int(z_max[label]),
                             int(y_min[label]), int(y_max[label]),
                             int(x_min[label]), int(x_max[label]))

    if logger:
        logger.info(f"     Bboxes computed for {len(bboxes)} clusters, "
                     f"total voxels={sizes[1:].sum():,}")

    return bboxes, sizes


# ============================================================================
# Streaming Regionprops
# ============================================================================
def regionprops_streaming(labels_zarr, intensity_zarr=None, properties=None,
                          slab_depth=64, logger=None):
    """Compute regionprops by processing labeled volume in Z-slabs.

    For objects spanning multiple slabs, results are aggregated.

    Parameters
    ----------
    labels_zarr : zarr.Array
        Labeled volume.
    intensity_zarr : zarr.Array, optional
        Intensity volume for intensity-weighted properties.
    properties : list of str, optional
        Properties to compute. Default: ['label', 'area', 'bbox', 'centroid'].
    slab_depth : int
        Z-slab depth.
    logger : logging.Logger, optional

    Returns
    -------
    dict
        Mapping of label -> dict of aggregated properties.
    """
    if properties is None:
        properties = ['label', 'area', 'bbox', 'centroid']

    shape = labels_zarr.shape
    # Accumulate per-label stats across slabs
    label_stats = {}

    for z_start in range(0, shape[0], slab_depth):
        z_end = min(z_start + slab_depth, shape[0])
        slab_labels = np.array(labels_zarr[z_start:z_end, :, :])

        if intensity_zarr is not None:
            slab_intensity = np.array(intensity_zarr[z_start:z_end, :, :])
        else:
            slab_intensity = None

        props = measure.regionprops(slab_labels, intensity_image=slab_intensity)

        for prop in props:
            label_val = prop.label
            if label_val not in label_stats:
                label_stats[label_val] = {
                    'area': 0,
                    'z_min': shape[0], 'z_max': 0,
                    'y_min': shape[1], 'y_max': 0,
                    'x_min': shape[2], 'x_max': 0,
                    'centroid_sum': np.zeros(3),
                }
                if slab_intensity is not None:
                    label_stats[label_val]['intensity_sum'] = 0.0

            stats = label_stats[label_val]
            stats['area'] += prop.area

            # Bounding box (adjusted to global Z)
            bb = prop.bbox  # (z0, y0, x0, z1, y1, x1) for 3D
            stats['z_min'] = min(stats['z_min'], bb[0] + z_start)
            stats['z_max'] = max(stats['z_max'], bb[3] + z_start)
            stats['y_min'] = min(stats['y_min'], bb[1])
            stats['y_max'] = max(stats['y_max'], bb[4])
            stats['x_min'] = min(stats['x_min'], bb[2])
            stats['x_max'] = max(stats['x_max'], bb[5])

            # Weighted centroid accumulation
            centroid = np.array(prop.centroid)
            centroid[0] += z_start  # Adjust Z to global
            stats['centroid_sum'] += centroid * prop.area

            if slab_intensity is not None and hasattr(prop, 'mean_intensity'):
                stats['intensity_sum'] += prop.mean_intensity * prop.area

    # Finalize
    for label_val, stats in label_stats.items():
        if stats['area'] > 0:
            stats['centroid'] = stats['centroid_sum'] / stats['area']
            stats['bbox'] = (stats['z_min'], stats['y_min'], stats['x_min'],
                             stats['z_max'], stats['y_max'], stats['x_max'])
            if 'intensity_sum' in stats:
                stats['mean_intensity'] = stats['intensity_sum'] / stats['area']
        # Clean up intermediate keys
        for k in ['centroid_sum', 'z_min', 'z_max', 'y_min', 'y_max', 'x_min', 'x_max',
                   'intensity_sum']:
            stats.pop(k, None)

    if logger:
        logger.info(f"     Streaming regionprops: {len(label_stats)} objects")

    return label_stats


# ============================================================================
# Per-Spine Regionprops (Zarr-Safe)
# ============================================================================
def per_spine_regionprops(labels, intensity_sources, properties,
                           spine_bboxes=None, logger=None):
    """Compute regionprops per spine via bbox reads — works with zarr or numpy.

    Instead of passing a full-volume intensity_image to regionprops_table
    (which requires the entire array in RAM), this reads per-spine bounding
    box slices from the intensity sources and computes properties locally.

    Parameters
    ----------
    labels : numpy.ndarray or zarr.Array
        Labeled spine volume.
    intensity_sources : dict
        Mapping of {name: numpy.ndarray or zarr.Array}. Each array must have
        the same shape as labels. Properties like min_intensity/max_intensity
        are computed from the first source in the dict.
    properties : list of str
        regionprops properties to compute. Must include 'label'.
    spine_bboxes : list, dict, or None
        Precomputed bboxes. Accepts either an ``ndimage.find_objects`` list
        (indexed by label-1) or a ``find_objects_streaming`` dict
        ``{label: [z0, z1, y0, y1, x0, x1]}``. If None, computed from labels.
    logger : logging.Logger, optional

    Returns
    -------
    pandas.DataFrame
        Per-spine measurements with 'label' column.
    """
    import pandas as pd

    # Get bboxes
    if spine_bboxes is None:
        if isinstance(labels, np.ndarray):
            max_label = int(labels.max())
            spine_bboxes = ndimage.find_objects(labels, max_label) if max_label > 0 else []
        else:
            # Single linear pass via find_objects per slab + bbox merge.
            # Prior _get_object_bboxes was O(N_labels × N_voxels) per slab —
            # at T7 (3554 labels × 3.24B-voxel slabs) it took 4+ hours.
            spine_bboxes = find_objects_streaming(labels, logger=logger)

    # Empty-bboxes early return — handles both the autocompute-empty case and
    # caller-passed empty (e.g., a shared bbox dict from an upstream
    # find_objects_streaming() on an all-zero-label volume). Construct column
    # names in the SAME order _measure_spine_from_crops emits them, since
    # pandas DataFrame column order follows dict insertion order and
    # downstream .rename()/merge() ops can be position-sensitive.
    # Helper emission order (see _measure_spine_from_crops):
    #   label → area → centroid-{0,1,2} → morph props (in properties order)
    #   → for each src in intensity_sources, for each of (min, max, mean): f'{src}_{prop}'
    if not spine_bboxes:
        _morph_set = {'area_bbox', 'extent', 'solidity', 'area_convex',
                      'axis_major_length', 'axis_minor_length',
                      'feret_diameter_max'}
        _empty_cols = ['label']
        if 'area' in properties:
            _empty_cols.append('area')
        if 'centroid' in properties:
            _empty_cols.extend(['centroid-0', 'centroid-1', 'centroid-2'])
        for _prop in properties:
            if _prop in _morph_set:
                _empty_cols.append(_prop)
        for _src in intensity_sources.keys():
            for _prop in ('min_intensity', 'max_intensity', 'mean_intensity'):
                if _prop in properties:
                    _empty_cols.append(f'{_src}_{_prop}')
        return pd.DataFrame(columns=_empty_cols)

    # Determine if bboxes are find_objects format (list) or find_objects_streaming format (dict)
    is_dict_bboxes = isinstance(spine_bboxes, dict)

    # Collect results per spine
    rows = []

    if is_dict_bboxes:
        # find_objects_streaming returns {label: [z0, z1, y0, y1, x0, x1]}
        # (legacy _get_object_bboxes returned the same shape with tuple values;
        # both unpack identically below).
        bbox_items = spine_bboxes.items()
    else:
        # find_objects returns a list indexed by label-1
        bbox_items = []
        for i, sl in enumerate(spine_bboxes):
            if sl is not None:
                label_val = i + 1
                z0, z1 = sl[0].start, sl[0].stop
                y0, y1 = sl[1].start, sl[1].stop
                x0, x1 = sl[2].start, sl[2].stop
                bbox_items.append((label_val, (z0, z1, y0, y1, x0, x1)))

    # Sort by Z to exploit zarr chunk locality
    bbox_items = sorted(bbox_items, key=lambda item: item[1][0])

    # Determine whether to use slab-coalesced path (zarr sources) or direct path (numpy)
    _has_zarr_source = (not isinstance(labels, np.ndarray) or
                        any(not isinstance(a, np.ndarray) for a in intensity_sources.values()))

    if _has_zarr_source and len(bbox_items) > 0:
        rows = _per_spine_regionprops_slab_coalesced(
            labels, intensity_sources, properties, bbox_items, logger=logger)
    else:
        rows = _per_spine_regionprops_direct(
            labels, intensity_sources, properties, bbox_items, logger=logger)

    df = pd.DataFrame(rows)
    # Sort by label to match regionprops_table ordering (bbox reads are Z-sorted
    # for zarr locality, but output must be label-sorted for positional joins)
    if len(df) > 0 and 'label' in df.columns:
        df = df.sort_values('label').reset_index(drop=True)
    if logger and len(df) > 0:
        logger.info(f"       Per-spine regionprops: {len(df)} spines measured")
    return df


def _measure_spine_from_crops(label_val, labels_crop, intensity_crops, properties, z0, y0, x0):
    """Compute properties for a single spine from pre-extracted array crops."""
    mask = labels_crop == label_val
    voxel_count = int(mask.sum())
    if voxel_count == 0:
        return None

    row = {'label': label_val}

    if 'area' in properties:
        row['area'] = voxel_count

    if 'centroid' in properties:
        coords = np.argwhere(mask)
        centroid = coords.mean(axis=0)
        row['centroid-0'] = centroid[0] + z0
        row['centroid-1'] = centroid[1] + y0
        row['centroid-2'] = centroid[2] + x0

    _morph_props = {'area_bbox', 'extent', 'solidity', 'area_convex',
                    'axis_major_length', 'axis_minor_length', 'feret_diameter_max'}
    requested_morph = [p for p in properties if p in _morph_props]
    if requested_morph:
        try:
            rp = measure.regionprops(mask.astype(np.uint8))
            if rp:
                for prop in requested_morph:
                    row[prop] = getattr(rp[0], prop, np.nan)
            else:
                for prop in requested_morph:
                    row[prop] = np.nan
        except (ValueError, IndexError):
            for prop in requested_morph:
                row[prop] = np.nan

    for src_name, src_crop in intensity_crops.items():
        values = src_crop[mask]
        if 'min_intensity' in properties:
            row[f'{src_name}_min_intensity'] = float(values.min()) if len(values) > 0 else np.nan
        if 'max_intensity' in properties:
            row[f'{src_name}_max_intensity'] = float(values.max()) if len(values) > 0 else np.nan
        if 'mean_intensity' in properties:
            row[f'{src_name}_mean_intensity'] = float(values.mean()) if len(values) > 0 else np.nan

    return row


def _per_spine_regionprops_direct(labels, intensity_sources, properties, bbox_items, logger=None):
    """Original per-spine loop — fast for numpy arrays (small volumes)."""
    rows = []
    _total = len(bbox_items)
    _milestone_every = max(100, _total // 20) if _total else 0
    for idx, (label_val, bbox) in enumerate(bbox_items):
        z0, z1, y0, y1, x0, x1 = bbox
        sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        labels_crop = labels[sl]
        intensity_crops = {name: arr[sl] for name, arr in intensity_sources.items()}
        row = _measure_spine_from_crops(label_val, labels_crop, intensity_crops, properties, z0, y0, x0)
        if row is not None:
            rows.append(row)
        if logger is not None and _milestone_every and (idx + 1) % _milestone_every == 0:
            logger.info(f"       Per-spine regionprops: {idx + 1}/{_total} spines measured...")
    return rows


def _per_spine_regionprops_slab_coalesced(labels, intensity_sources, properties, bbox_items, logger=None):
    """Slab-coalesced loop — decompress each Z-slab once, extract all spine crops from RAM.

    Spines fully contained within a slab are processed from cached RAM (fast path).
    Spines spanning slab boundaries fall back to direct zarr reads (correct, slower).
    """
    vol_z = labels.shape[0]
    slab_z = labels.chunks[0] if hasattr(labels, 'chunks') else 128

    # Cap slab size to avoid OOM on large volumes: each slab reads
    # labels (int32) + N intensity sources (typically float32) at full XY.
    n_sources = max(1, len(intensity_sources))
    bytes_per_z = int(labels.shape[1]) * int(labels.shape[2]) * (4 + 4 * n_sources)
    avail = psutil.virtual_memory().available
    max_slab_bytes = int(avail * 0.25)
    if bytes_per_z * slab_z > max_slab_bytes:
        slab_z = max(1, max_slab_bytes // bytes_per_z)

    rows = []
    boundary_spines = []

    # Group spines by slab
    slab_starts = list(range(0, vol_z, slab_z))

    _total = len(bbox_items)
    _milestone_every = max(100, _total // 20) if _total else 0
    _n_done = 0

    spine_idx = 0
    for slab_start in slab_starts:
        slab_end = min(slab_start + slab_z, vol_z)

        # Collect spines fully contained in this slab
        slab_spines = []
        while spine_idx < len(bbox_items):
            label_val, bbox = bbox_items[spine_idx]
            z0, z1 = bbox[0], bbox[1]
            if z0 >= slab_end:
                break
            if z0 >= slab_start and z1 <= slab_end:
                slab_spines.append((label_val, bbox))
            elif z0 < slab_end:
                boundary_spines.append((label_val, bbox))
            spine_idx += 1

        if not slab_spines:
            continue

        # Decompress slab once for labels + all intensity sources
        slab_sl = (slice(slab_start, slab_end), slice(None), slice(None))
        labels_slab = np.array(labels[slab_sl]) if not isinstance(labels, np.ndarray) else labels[slab_sl]
        int_slabs = {}
        for name, arr in intensity_sources.items():
            int_slabs[name] = np.array(arr[slab_sl]) if not isinstance(arr, np.ndarray) else arr[slab_sl]

        # Extract per-spine crops from cached slab
        for label_val, bbox in slab_spines:
            z0, z1, y0, y1, x0, x1 = bbox
            local_z0 = z0 - slab_start
            local_z1 = z1 - slab_start
            local_sl = (slice(local_z0, local_z1), slice(y0, y1), slice(x0, x1))

            labels_crop = labels_slab[local_sl]
            intensity_crops = {name: slab[local_sl] for name, slab in int_slabs.items()}
            row = _measure_spine_from_crops(label_val, labels_crop, intensity_crops, properties, z0, y0, x0)
            if row is not None:
                rows.append(row)
            _n_done += 1
            if logger is not None and _milestone_every and _n_done % _milestone_every == 0:
                logger.info(f"       Per-spine regionprops: {_n_done}/{_total} spines measured...")

        del labels_slab, int_slabs

    # Process boundary-spanning spines via direct zarr read (correct, slower)
    for label_val, bbox in boundary_spines:
        z0, z1, y0, y1, x0, x1 = bbox
        sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        labels_crop = np.array(labels[sl]) if not isinstance(labels, np.ndarray) else labels[sl]
        intensity_crops = {}
        for name, arr in intensity_sources.items():
            intensity_crops[name] = np.array(arr[sl]) if not isinstance(arr, np.ndarray) else arr[sl]
        row = _measure_spine_from_crops(label_val, labels_crop, intensity_crops, properties, z0, y0, x0)
        if row is not None:
            rows.append(row)
        _n_done += 1
        if logger is not None and _milestone_every and _n_done % _milestone_every == 0:
            logger.info(f"       Per-spine regionprops: {_n_done}/{_total} spines measured (boundary pass)...")

    return rows


# ============================================================================
# Per-Dendrite Geodesic Distance
# ============================================================================
def geodesic_per_dendrite(labeled_dend_zarr, skeleton_coords_dict, spacing, logger=None):
    """Compute geodesic distances along dendrite skeletons.

    For each dendrite, extracts the bounding box, computes EDT on the dendrite mask,
    and traces geodesic distance along the skeleton.

    Parameters
    ----------
    labeled_dend_zarr : zarr.Array
        Labeled dendrite volume.
    skeleton_coords_dict : dict
        Mapping of dendrite_label -> skeleton coordinates (N, 3).
    spacing : tuple of float
        Voxel spacing (Z, Y, X).
    logger : logging.Logger, optional

    Returns
    -------
    dict
        Mapping of dendrite_label -> geodesic distances array.
    """
    from scipy.spatial import cKDTree

    shape = labeled_dend_zarr.shape
    geodesic_results = {}

    bboxes = _get_object_bboxes(labeled_dend_zarr,
                                 set(skeleton_coords_dict.keys()))

    for label_val, coords in skeleton_coords_dict.items():
        if label_val not in bboxes or len(coords) < 2:
            continue

        bbox = bboxes[label_val]
        z0, z1, y0, y1, x0, x1 = bbox

        # Extract dendrite subvolume
        subvol = np.array(labeled_dend_zarr[z0:z1, y0:y1, x0:x1])
        dend_mask = (subvol == label_val)

        # Adjust skeleton coords to local space
        local_coords = coords.copy()
        local_coords[:, 0] -= z0
        local_coords[:, 1] -= y0
        local_coords[:, 2] -= x0

        # Filter coords that are within bounds
        valid = (
            (local_coords[:, 0] >= 0) & (local_coords[:, 0] < (z1 - z0)) &
            (local_coords[:, 1] >= 0) & (local_coords[:, 1] < (y1 - y0)) &
            (local_coords[:, 2] >= 0) & (local_coords[:, 2] < (x1 - x0))
        )
        local_coords = local_coords[valid]

        if len(local_coords) < 2:
            continue

        # Compute geodesic distance along skeleton using KDTree nearest-neighbor chain
        physical_coords = local_coords * np.array(spacing)
        tree = cKDTree(physical_coords)
        distances = np.zeros(len(physical_coords))

        # Simple chain: nearest-neighbor traversal from endpoint
        visited = np.zeros(len(physical_coords), dtype=bool)
        current = 0  # Start from first point
        visited[current] = True

        for i in range(1, len(physical_coords)):
            dists, indices = tree.query(physical_coords[current], k=min(20, len(physical_coords)))
            # Find nearest unvisited
            for d, idx in zip(dists, indices):
                if not visited[idx]:
                    distances[idx] = distances[current] + d
                    visited[idx] = True
                    current = idx
                    break

        geodesic_results[label_val] = {
            'coords': coords[valid] if isinstance(valid, np.ndarray) else coords,
            'distances': distances,
        }

    if logger:
        logger.info(f"     Geodesic distances computed for {len(geodesic_results)} dendrites")

    return geodesic_results


# ============================================================================
# Per-Component Neck Association
# ============================================================================
def associate_spines_with_necks_per_component(reference, targets, logger,
                                               spine_dist_um=2.0,
                                               input_resXY=0.065,
                                               input_resZ=0.15,
                                               gpu_threshold=100_000,
                                               workspace=None,
                                               streaming_threshold_voxels=500_000_000,
                                               result_name=None):
    """Assign connected target fragments to their nearest reference label.

    Used in two passes of the RESPAN pipeline:
      Pass 1: reference = labeled spines, targets = nnU-Net neck labels.
      Pass 2: reference = spines + extended-neck corridor (from extend_objects_GPU),
              targets = nnU-Net neck voxels not assigned in Pass 1.

    Algorithm (key change from prior implementation):
      1. CC(targets) alone — NOT CC(reference | targets). Fragments disconnected
         from every reference voxel are no longer auto-dropped; they are
         evaluated by physical distance in µm.
      2. For each target CC, dilate its bbox by `spine_dist_um * 3` in voxels
         (anisotropic Z vs XY) and extract a reference subvolume.
         - If the dilated bbox contains any reference voxel: run local EDT
           (GPU if bbox > gpu_threshold, CPU otherwise) and mode-vote the
           nearest-reference label over the CC. Assign the whole CC to that
           label IF the actual minimum distance is within the cap.
         - Otherwise: drop the CC (no reference within cap = noise in Pass 1 /
           genuine noise in Pass 2). In Pass 1, dropped fragments remain in
           `remaining_necks` for Pass 2 to rescue after corridor extension.
      3. Ties in the mode vote break to the smallest label (np.bincount argmax).

    Parameters
    ----------
    reference : numpy.ndarray or zarr.Array
        Labeled reference volume. Each object has a unique integer ID.
    targets : numpy.ndarray or zarr.Array
        Binary or labeled target volume. Voxels > 0 are targets to assign.
    logger : logging.Logger
    spine_dist_um : float
        Maximum spine→neck distance in µm (from Settings.neuron_spine_dist).
        The distance cap is 3× this value.
    input_resXY, input_resZ : float
        Voxel sampling in µm (XY, Z).
    gpu_threshold : int
        Bbox voxel count above which GPU EDT is used.
    workspace : ZarrWorkspace, optional
        Needed for streaming CC on volumes > 500M voxels.

    Returns
    -------
    numpy.ndarray
        Label array (same shape as reference) where target voxels assigned
        to a reference get that reference's label. Reference voxels retain
        their labels. Unassigned (noise / dropped) target voxels stay at 0.
    """
    _require_cupy("associate_spines_with_necks_per_component")

    # KEEP THIS LOCAL IMPORT. It is not redundant with the module-level one.
    # Assigning `cp`/`gpu_edt` anywhere in this function makes them local for
    # the entire body, which shadows the module-level names — including the
    # `except cp.cuda.memory.OutOfMemoryError:` clause below. If this import is
    # removed, that except clause would resolve `cp` to the _CuPyUnavailable
    # stand-in on a CPU-only host and raise from __getattr__ while Python is
    # matching the exception, masking the original error.
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as gpu_edt

    shape = reference.shape
    n_voxels = int(shape[0]) * int(shape[1]) * int(shape[2])

    # Distance cap in µm and halo radius in voxels (anisotropic).
    cap_um = float(spine_dist_um) * 3.0
    halo_z = max(1, int(math.ceil(cap_um / max(input_resZ, 1e-6))))
    halo_yx = max(1, int(math.ceil(cap_um / max(input_resXY, 1e-6))))

    # --- Step A: CC(targets) — streaming for large volumes, scipy.label otherwise ---
    _use_streaming = n_voxels > streaming_threshold_voxels
    _streaming_tmpdir = None
    target_cc = None    # zarr (streaming) or numpy (small)
    bboxes = None       # dict {label: [z0,z1,y0,y1,x0,x1]} or list from find_objects

    # When workspace + result_name are provided,
    # allocate `result` as a zarr-backed array. Avoids the ~22 GB int32 numpy
    # materialization in _copy_reference_to_result at T7 / 111 GB at T_LARGE.
    # _write_result_subvolume below dispatches numpy-view vs zarr-RMW based
    # on the destination type.
    _result_dest_zarr = None
    if workspace is not None and result_name is not None:
        _result_dest_zarr = workspace.create_array(
            result_name, shape, dtype=reference.dtype, chunks=STREAMING_CHUNKS)

    if _use_streaming:
        import tempfile
        _streaming_tmpdir = tempfile.mkdtemp(prefix='respan_assoc_cc_')

        _mask_zarr = zarr.open(os.path.join(_streaming_tmpdir, 'mask'), mode='w',
                                shape=shape, dtype=np.uint8, chunks=STREAMING_CHUNKS)
        _slab = 64
        for z in range(0, shape[0], _slab):
            ze = min(z + _slab, shape[0])
            _mask_zarr[z:ze] = (np.asarray(targets[z:ze]) > 0).astype(np.uint8)

        _cc_zarr = zarr.open(os.path.join(_streaming_tmpdir, 'cc'), mode='w',
                              shape=shape, dtype=np.int32, chunks=STREAMING_CHUNKS)
        n_cc, bboxes = connected_components_streaming(
            _mask_zarr, _cc_zarr, min_volume=0, logger=logger)
        target_cc = _cc_zarr

        if n_cc == 0:
            logger.info("       No target components found.")
            import shutil
            shutil.rmtree(_streaming_tmpdir, ignore_errors=True)
            return _copy_reference_to_result(reference, dest_zarr=_result_dest_zarr)
    else:
        # Materialize targets if zarr for the small path (bounded RAM).
        tgt_np = np.asarray(targets) if hasattr(targets, 'chunks') else targets
        target_cc, n_cc = ndimage.label(tgt_np > 0)
        if n_cc == 0:
            logger.info("       No target components found.")
            return _copy_reference_to_result(reference, dest_zarr=_result_dest_zarr)
        bboxes = ndimage.find_objects(target_cc, n_cc)

    # --- Step B: Allocate result as a copy of reference ---
    result = _copy_reference_to_result(reference, dest_zarr=_result_dest_zarr)

    # Pre-flatten `reference` access pattern — we always read via slicing, which
    # works for numpy and zarr identically.

    # --- Step C: Iterate target CCs ---
    n_assigned = 0
    n_dropped_far = 0         # no reference within dilated bbox
    n_dropped_cap = 0         # reference in bbox but actual min dist > cap
    n_voxels_assigned = 0
    n_voxels_dropped = 0
    n_gpu = 0
    n_cpu = 0
    gpu_flush_counter = 0

    def _bbox_for(cc_id):
        if isinstance(bboxes, dict):
            bb = bboxes.get(cc_id)
            if bb is None:
                return None
            return int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]), int(bb[4]), int(bb[5])
        # list from ndimage.find_objects
        sl = bboxes[cc_id - 1]
        if sl is None:
            return None
        return (int(sl[0].start), int(sl[0].stop),
                int(sl[1].start), int(sl[1].stop),
                int(sl[2].start), int(sl[2].stop))

    for cc_id in range(1, n_cc + 1):
        bb = _bbox_for(cc_id)
        if bb is None:
            continue
        z0, z1, y0, y1, x0, x1 = bb

        # Dilated bbox — clip to volume edges.
        dz0 = max(0, z0 - halo_z)
        dz1 = min(shape[0], z1 + halo_z)
        dy0 = max(0, y0 - halo_yx)
        dy1 = min(shape[1], y1 + halo_yx)
        dx0 = max(0, x0 - halo_yx)
        dx1 = min(shape[2], x1 + halo_yx)

        dsl = (slice(dz0, dz1), slice(dy0, dy1), slice(dx0, dx1))

        # Extract dilated subvolumes (read through for numpy, slab-bounded for zarr).
        ref_crop = np.asarray(reference[dsl])
        cc_crop = np.asarray(target_cc[dsl])
        tgt_cc_crop = (cc_crop == cc_id)

        n_cc_vox = int(tgt_cc_crop.sum())
        if n_cc_vox == 0:
            continue

        if not np.any(ref_crop > 0):
            # No reference within dilated bbox → drop as noise / far-from-reference.
            n_dropped_far += 1
            n_voxels_dropped += n_cc_vox
            continue

        # Build EDT input: foreground = everywhere except reference voxels.
        edt_input = (ref_crop == 0)
        bbox_total_voxels = edt_input.size

        # Anisotropic physical sampling for distance + index computation.
        # Both the mode-vote nearest-label assignment AND the cap check need
        # physical distance (not voxel distance) — otherwise at Z=0.15 / XY=0.045
        # the nearest-label EDT would underweight Z, picking references that are
        # close in voxels but far in µm.
        sampling = (float(input_resZ), float(input_resXY), float(input_resXY))

        # --- Local EDT (GPU if bbox large; CPU otherwise) ---
        dist_field = None
        nearest_label_np = None
        if bbox_total_voxels > gpu_threshold:
            # Initialize GPU temps to None so the OOM handler can safely
            # `del` any that were partially created before the exception.
            edt_input_gpu = None
            distances_gpu = None
            indices = None
            ref_crop_gpu = None
            nearest_label_gpu = None
            try:
                edt_input_gpu = cp.asarray(edt_input)
                distances_gpu, indices = gpu_edt(edt_input_gpu,
                                                  sampling=sampling,
                                                  return_distances=True,
                                                  return_indices=True)
                ref_crop_gpu = cp.asarray(ref_crop)
                nearest_label_gpu = ref_crop_gpu[tuple(indices)]
                nearest_label_np = nearest_label_gpu.get()
                dist_field = distances_gpu.get()

                del edt_input_gpu, distances_gpu, indices, ref_crop_gpu, nearest_label_gpu
                edt_input_gpu = distances_gpu = indices = None
                ref_crop_gpu = nearest_label_gpu = None
                n_gpu += 1
                gpu_flush_counter += 1
                if gpu_flush_counter >= 50:
                    cp.cuda.Device().synchronize()
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    gpu_flush_counter = 0
            except cp.cuda.memory.OutOfMemoryError:
                # Drop any partially-allocated GPU locals before flushing pools.
                # free_all_blocks() cannot reclaim memory still referenced by
                # live cupy arrays, so rebind them to None first. These were
                # initialized to None before the try, so del-or-rebind is safe.
                edt_input_gpu = None
                distances_gpu = None
                indices = None
                ref_crop_gpu = None
                nearest_label_gpu = None
                cp.cuda.Device().synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                gpu_flush_counter = 0
                logger.warning(f"       GPU OOM on target CC {cc_id} "
                               f"(bbox {edt_input.shape}), falling back to CPU")
                dist_field, indices = distance_transform_edt(
                    edt_input, sampling=sampling,
                    return_distances=True, return_indices=True)
                nearest_label_np = ref_crop[tuple(indices)]
                n_cpu += 1
        else:
            dist_field, indices = distance_transform_edt(
                edt_input, sampling=sampling,
                return_distances=True, return_indices=True)
            nearest_label_np = ref_crop[tuple(indices)]
            n_cpu += 1

        # --- Per-voxel assignment along the Voronoi boundary ---
        # Each target voxel gets ITS OWN nearest reference (from the EDT).
        # When an nnU-Net neck CC bridges between two spines, the CC splits
        # along the Voronoi boundary instead of being stamped uniformly with
        # one label — fixing the "outer spine's neck grows into inner spine's
        # head" pattern (T2 Spine 52/49 case: 13 voxels of neck 52 were
        # physically inside spine 49's territory). Audit
        # (session a79e2a036e4cfc098) confirmed this is the correct
        # semantic — Voronoi tessellation by nearest reference is the
        # standard literature pattern (Imaris, Spinifel, NeuTu,
        # CellProfiler SpineDetector).
        in_cc_assignable = (
            tgt_cc_crop
            & (nearest_label_np > 0)
            & (dist_field <= cap_um)
        )
        n_assigned_in_cc = int(in_cc_assignable.sum())
        n_dropped_in_cc = n_cc_vox - n_assigned_in_cc

        if n_assigned_in_cc == 0:
            # All CC voxels either had no valid nearest reference or were
            # beyond the cap — drop the whole fragment (matches prior
            # behaviour for far-from-reference / beyond-cap cases).
            if not np.any(nearest_label_np[tgt_cc_crop] > 0):
                n_dropped_far += 1
            else:
                n_dropped_cap += 1
            n_voxels_dropped += n_cc_vox
            continue

        # Build the per-voxel label crop for this CC's slice. Voxels not in
        # `in_cc_assignable` stay 0 (no write).
        labels_to_write = np.where(in_cc_assignable, nearest_label_np, 0)
        # Cast to result dtype so the write doesn't promote silently.
        labels_to_write = labels_to_write.astype(reference.dtype, copy=False)

        # --- Write per-voxel labels back to result (dilated bbox slice) ---
        _write_result_subvolume_labels(result, dsl, labels_to_write)

        n_assigned += 1
        n_voxels_assigned += n_assigned_in_cc
        if n_dropped_in_cc > 0:
            n_voxels_dropped += n_dropped_in_cc

    # Final GPU cleanup
    if n_gpu > 0:
        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    logger.info(
        f"       Per-target-CC association: {n_cc} target CCs — "
        f"{n_assigned} assigned, {n_dropped_far} no-reference-within-{cap_um:.1f}µm, "
        f"{n_dropped_cap} beyond-cap ({n_gpu} GPU + {n_cpu} CPU EDT calls)")
    logger.info(
        f"       Voxels: {n_voxels_assigned} assigned / {n_voxels_dropped} dropped")

    if _streaming_tmpdir is not None:
        import shutil
        shutil.rmtree(_streaming_tmpdir, ignore_errors=True)

    return result


def _copy_reference_to_result(reference, dest_zarr=None):
    """Copy `reference` into a result destination.

    When `dest_zarr` is provided, copy reference
    slab-by-slab into the zarr (no full-volume numpy alloc — avoids the
    ~22 GB int32 transient at T7, ~111 GB at T_LARGE). Otherwise the legacy
    behavior: allocate a fresh numpy array.

    Slab-iteration is safe for both numpy and zarr inputs.

    Parameters
    ----------
    reference : numpy.ndarray or zarr.Array
    dest_zarr : zarr.Array, optional
        When provided, this is the destination. Returned as-is so the caller
        gets back the same handle. When None, a fresh numpy is allocated and
        returned.

    Returns
    -------
    numpy.ndarray or zarr.Array
        Same type as caller-provided dest_zarr (numpy if None).
    """
    if dest_zarr is not None:
        # Zarr destination — slab-stream copy. Both reference and dest support
        # slab slicing identically.
        shape = reference.shape
        if len(shape) >= 3:
            row_bytes = int(shape[1]) * int(shape[2]) * np.dtype(reference.dtype).itemsize
            max_slab = max(1, int(2 * GB / row_bytes)) if row_bytes > 0 else 64
        else:
            max_slab = 64
        # Pick a slab size that aligns with both source and dest chunk[0]
        if hasattr(reference, 'chunks'):
            src_chunk = max(int(reference.chunks[0]), 1)
        else:
            src_chunk = max_slab
        if hasattr(dest_zarr, 'chunks'):
            dst_chunk = max(int(dest_zarr.chunks[0]), 1)
        else:
            dst_chunk = max_slab
        slab = max(1, min(src_chunk, dst_chunk, max_slab))
        for z in range(0, shape[0], slab):
            ze = min(z + slab, shape[0])
            dest_zarr[z:ze] = np.asarray(reference[z:ze])
        return dest_zarr

    # Legacy numpy destination.
    if hasattr(reference, 'chunks'):
        shape = reference.shape
        out = np.empty(shape, dtype=reference.dtype)
        if len(shape) >= 3:
            row_bytes = int(shape[1]) * int(shape[2]) * out.dtype.itemsize
            max_slab = max(1, int(2 * GB / row_bytes)) if row_bytes > 0 else 64
            slab = min(max(int(reference.chunks[0]), 1), max_slab)
        else:
            slab = 64
        for z in range(0, shape[0], slab):
            ze = min(z + slab, shape[0])
            out[z:ze] = reference[z:ze]
        return out
    return reference.copy()


def _write_result_subvolume(result, dsl, mask, value):
    """Write `value` into `result[dsl]` wherever `mask` is True.

    Handles both numpy and zarr `result`.

    For numpy: ``result[dsl]`` returns a view; direct mask-assignment is in-place.
    For zarr: ``result[dsl]`` returns a numpy COPY (not a view), so the legacy
    pattern silently no-ops. Read-modify-write the bounded crop instead.

    The dilated bbox `dsl` is always small (cap_um × 3 / voxel_spacing typical
    20-50 voxels per dim → ~10-100k voxel crop), so the RMW is bounded.
    """
    if hasattr(result, 'chunks'):
        # zarr: read crop, modify in numpy, write back.
        crop = np.asarray(result[dsl])
        crop[mask] = value
        result[dsl] = crop
    else:
        # numpy view — direct in-place assignment.
        result[dsl][mask] = value


def _write_result_subvolume_labels(result, dsl, labels_array):
    """Write per-voxel labels into `result[dsl]` where `labels_array > 0`.

    Companion to `_write_result_subvolume` for per-voxel (not scalar) writes.
    Used by the per-voxel neck assignment path in
    `associate_spines_with_necks_per_component` to split bridging CCs along
    Voronoi boundaries instead of stamping the whole CC with a single label.

    Handles both numpy and zarr `result`. labels_array is a same-shape int
    array where 0 means "do not write this voxel".
    """
    nz_mask = labels_array > 0
    if hasattr(result, 'chunks'):
        crop = np.asarray(result[dsl])
        crop[nz_mask] = labels_array[nz_mask]
        result[dsl] = crop
    else:
        result[dsl][nz_mask] = labels_array[nz_mask]


def find_objects_streaming(labeled_volume, logger=None, slab_depth=64):
    """Slab-based find_objects equivalent — returns a dict {label: (z0, z1, y0, y1, x0, x1)}.

    Accepts numpy or zarr input. Never materializes the full volume. At T7
    (22 GB int32 zarr), scipy.ndimage.find_objects would require a numpy copy
    plus its own per-label slice list (~44 GB transient). This helper walks
    the volume slab-by-slab, tracks min/max coordinates per label using a
    dict, and returns a compact bbox dict suitable for per-label iteration.
    """
    shape = labeled_volume.shape
    itemsize = labeled_volume.dtype.itemsize
    row_bytes = int(shape[1]) * int(shape[2]) * itemsize if len(shape) == 3 else itemsize
    # Cap slab at ~2 GB of transient per-slab working memory.
    max_slab = max(1, int(2 * GB / row_bytes)) if row_bytes > 0 else 64
    slab = min(max_slab, slab_depth)

    bboxes = {}  # label -> [z0, z1, y0, y1, x0, x1]
    for z in range(0, shape[0], slab):
        ze = min(z + slab, shape[0])
        sub = np.asarray(labeled_volume[z:ze])
        if sub.size == 0:
            continue
        # Per-label bbox in this slab via scipy on the slab (cheap — slab is
        # small). Merge with running bboxes across slabs.
        slab_max = int(sub.max())
        if slab_max == 0:
            continue
        sub_slices = ndimage.find_objects(sub, slab_max)
        for lbl in range(1, slab_max + 1):
            sl = sub_slices[lbl - 1]
            if sl is None:
                continue
            z0_g = sl[0].start + z; z1_g = sl[0].stop + z
            y0, y1 = sl[1].start, sl[1].stop
            x0, x1 = sl[2].start, sl[2].stop
            bb = bboxes.get(lbl)
            if bb is None:
                bboxes[lbl] = [z0_g, z1_g, y0, y1, x0, x1]
            else:
                bb[0] = min(bb[0], z0_g); bb[1] = max(bb[1], z1_g)
                bb[2] = min(bb[2], y0);   bb[3] = max(bb[3], y1)
                bb[4] = min(bb[4], x0);   bb[5] = max(bb[5], x1)

    if logger:
        logger.info(f"       find_objects_streaming: {len(bboxes)} labels in {shape} volume")
    return bboxes


# ============================================================================
# Per-Label Spine↔Neck Gap Bridging
# ============================================================================
def bridge_spine_neck_gaps(connected_necks, spines_filtered, dendrites_mask,
                            logger, close_iterations=1):
    """Bridge 1-voxel gaps between a spine head and its assigned neck voxels.

    nnU-Net's Model Type 4 classifies each voxel as exactly one of {Spine Core,
    Spine Membrane, Neck, Dendrite, ...}; transition voxels between Spine
    Membrane (L3) and Neck (L4) are sometimes assigned to background, leaving
    a visible 1-voxel gap in ~10% of spines. Pass 1 of the neck association
    pipeline assigns the neck fragment to the spine by physical distance but
    does not fill the geometric gap — this helper closes it.

    For each spine label L, builds the combined mask
    `(spines_filtered == L) | (connected_necks == L)` within a per-label bbox,
    checks 26-connectivity, and — if there are multiple components — applies
    morphological closing to identify candidate bridge voxels. Bridge voxels
    are written only where no other spine/neck/dendrite voxel lives, so no
    other label is ever overwritten.

    Parameters
    ----------
    connected_necks : numpy.ndarray
        Neck label volume (modified in place). Labels match spine labels.
    spines_filtered : numpy.ndarray or zarr.Array
        Spine label volume (read-only). Disjoint from connected_necks.
    dendrites_mask : numpy.ndarray or zarr.Array (binary/uint8)
        Dendrite binary mask. Bridge voxels never written into dendrites.
    logger : logging.Logger
    close_iterations : int
        Morphological closing iterations — 1 bridges 1-voxel gaps, 2 bridges
        2-voxel gaps, etc. Default 1 (conservative — won't bridge real splits).

    Returns
    -------
    numpy.ndarray
        The same `connected_necks` array, updated in place and returned for
        chaining convenience.
    """
    shape = connected_necks.shape
    if connected_necks.ndim != 3:
        logger.warning(f"       bridge_spine_neck_gaps: expected 3D volume, "
                        f"got shape {shape}. Skipping.")
        return connected_necks

    # --- Dtype safety ---
    # If spines_filtered uses a wider integer type than connected_necks (e.g.,
    # spines=int64, necks=int32), silent wrap-around could corrupt labels when
    # we cast spine labels into the neck dtype. Assert safety or abort.
    sp_dtype = getattr(spines_filtered, 'dtype', connected_necks.dtype)
    nk_dtype = connected_necks.dtype
    if (np.issubdtype(sp_dtype, np.integer) and np.issubdtype(nk_dtype, np.integer)
            and np.iinfo(sp_dtype).max > np.iinfo(nk_dtype).max):
        logger.warning(
            f"       bridge_spine_neck_gaps: spines_filtered dtype {sp_dtype} can "
            f"represent labels beyond connected_necks dtype {nk_dtype}. Skipping "
            f"to avoid silent truncation — promote connected_necks dtype to fix.")
        return connected_necks

    # --- Collect per-label bboxes without allocating a union volume ---
    # Previous impl allocated a full-volume `union_labels` (~22 GB at T7) while
    # `neck_labels_updated` (also ~22 GB) was still live, risking OOM. Instead,
    # use find_objects_streaming — slab-iterates the zarr/numpy volume and
    # returns a per-label bbox dict without ever materializing full-volume.
    # At T7 this swaps a transient ~20 GB np.asarray for ~2 GB per-slab work.
    sp_bboxes = find_objects_streaming(spines_filtered)
    nk_bboxes = find_objects_streaming(connected_necks)

    all_labels = set(sp_bboxes) | set(nk_bboxes)
    if not all_labels:
        return connected_necks

    # NOTE (anisotropic closing): we intentionally use a voxel-space
    # 26-connected structuring element here, not physical-space. At T4 anisotropy
    # (Z=0.15, XY=0.045) a Z-voxel is ~3× an XY-voxel in physical distance, so a
    # 1-iteration closing can bridge slightly more in Z than XY. The gap being
    # bridged is a segmentation transition artifact between adjacent classes,
    # not a physical feature, so voxel-space is the right domain. If this causes
    # over-bridging in pathological cases, pass a spacing-aware structure.
    structure = ndimage.generate_binary_structure(3, 3)  # 26-connectivity

    n_bridged = 0
    n_voxels_added = 0
    n_still_split = 0
    still_split_labels = []

    for label in sorted(all_labels):
        # find_objects_streaming returns [z0, z1, y0, y1, x0, x1] lists.
        sp_bb = sp_bboxes.get(label)
        nk_bb = nk_bboxes.get(label)
        if sp_bb is None and nk_bb is None:
            continue

        if sp_bb is None:
            z0, z1, y0, y1, x0, x1 = nk_bb
        elif nk_bb is None:
            z0, z1, y0, y1, x0, x1 = sp_bb
        else:
            z0 = min(sp_bb[0], nk_bb[0]); z1 = max(sp_bb[1], nk_bb[1])
            y0 = min(sp_bb[2], nk_bb[2]); y1 = max(sp_bb[3], nk_bb[3])
            x0 = min(sp_bb[4], nk_bb[4]); x1 = max(sp_bb[5], nk_bb[5])

        # Expand bbox by close_iterations so closing has room to work at edges.
        pad = int(close_iterations)
        z0 = max(0, z0 - pad); z1 = min(shape[0], z1 + pad)
        y0 = max(0, y0 - pad); y1 = min(shape[1], y1 + pad)
        x0 = max(0, x0 - pad); x1 = min(shape[2], x1 + pad)
        dsl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

        sp_crop = np.asarray(spines_filtered[dsl])
        neck_crop = connected_necks[dsl]
        combined = (sp_crop == label) | (neck_crop == label)
        if not combined.any():
            continue

        # 26-connected component count on the combined spine+neck mask.
        _, n_cc = ndimage.label(combined, structure=structure)
        if n_cc <= 1:
            continue

        closed = ndimage.binary_closing(combined, structure=structure,
                                         iterations=close_iterations)
        new_voxels = closed & ~combined
        if not new_voxels.any():
            n_still_split += 1
            still_split_labels.append(int(label))
            continue

        # Never overwrite another spine, another neck, or dendrite.
        dend_crop = np.asarray(dendrites_mask[dsl]).astype(bool, copy=False)
        writable = new_voxels & (neck_crop == 0) & (sp_crop == 0) & (~dend_crop)
        if not writable.any():
            n_still_split += 1
            still_split_labels.append(int(label))
            continue

        # Write bridge voxels with this spine's label into connected_necks.
        # Re-read neck_crop as a mutable copy, modify, write back — avoids
        # fancy-index-on-view ambiguity when connected_necks came from a copy.
        neck_sub = neck_crop.copy()
        neck_sub[writable] = label
        connected_necks[dsl] = neck_sub

        n_bridged += 1
        n_voxels_added += int(writable.sum())

    logger.info(
        f"       Bridged spine<->neck gaps on {n_bridged} spines "
        f"({n_voxels_added} voxels added, {n_still_split} still split)")
    return connected_necks, sorted(still_split_labels)


# ============================================================================
# Streaming Dendrite Filtering
# ============================================================================

def filter_dendrites_streaming(dendrites, min_dendrite_vol, workspace, logger):
    """Label and volume-filter dendrites using streaming CC — no full-volume sort.

    Replaces filter_dendrites() for large volumes where scipy.ndimage.label +
    ndimage.sum_labels would OOM (np.unique on billions of elements).

    Parameters
    ----------
    dendrites : numpy.ndarray
        Binary dendrite mask.
    min_dendrite_vol : float
        Minimum volume in voxels to keep a dendrite component.
    workspace : ZarrWorkspace
        Zarr workspace for intermediate arrays.
    logger : logging.Logger

    Returns
    -------
    numpy.ndarray (int32)
        Relabeled dendrite volume with only large components.
    """
    shape = dendrites.shape
    slab_depth = 64

    # Step 1: Write dendrite mask to zarr
    mask_zarr = workspace.numpy_to_zarr('dend_binary', dendrites.astype(np.uint8))
    labels_zarr = workspace.create_array('dend_labels', shape, dtype=np.int32,
                                         chunks=STREAMING_CHUNKS)

    # Step 2: Streaming connected components (slab-by-slab, union-find)
    num_detected, _ = connected_components_streaming(
        mask_zarr, labels_zarr, min_volume=0, logger=logger)

    if num_detected == 0:
        logger.info(f"    Processing 0 of 0 detected dendrites")
        return np.zeros(shape, dtype=np.int32)

    # Step 3: Compute per-label volumes via slab scanning (no full-volume sort)
    label_volumes = {}
    for z in range(0, shape[0], slab_depth):
        end_z = min(z + slab_depth, shape[0])
        slab = np.array(labels_zarr[z:end_z])
        labels_in_slab, counts = np.unique(slab, return_counts=True)
        for lbl, cnt in zip(labels_in_slab, counts):
            if lbl == 0:
                continue
            label_volumes[lbl] = label_volumes.get(lbl, 0) + int(cnt)

    # Step 4: Filter by volume threshold
    keep_labels = sorted(lbl for lbl, vol in label_volumes.items()
                         if vol >= min_dendrite_vol)
    num_filtered = len(keep_labels)

    # Step 5: Build remap table and apply slab-by-slab
    label_remap = {old: new for new, old in enumerate(keep_labels, start=1)}
    # Pre-build lookup array sized for ALL labels (including filtered-out ones → 0)
    max_label_in_cc = max(label_volumes.keys()) if label_volumes else 0
    remap_lut = np.zeros(max_label_in_cc + 1, dtype=np.int32)
    for old_lbl, new_lbl in label_remap.items():
        remap_lut[old_lbl] = new_lbl

    result_zarr = workspace.create_array('labeled_dendrites', shape, dtype=np.int32,
                                          chunks=STREAMING_CHUNKS)
    for z in range(0, shape[0], slab_depth):
        end_z = min(z + slab_depth, shape[0])
        slab = np.array(labels_zarr[z:end_z])
        result_zarr[z:end_z] = remap_lut[slab]

    logger.info(f"    Processing {num_filtered} of {num_detected} detected dendrites "
                f"larger than minimum volume threshold of {min_dendrite_vol} voxels")
    return result_zarr


# ============================================================================
# Chunked Resize (avoids skimage convert_to_float full-volume allocation)
# ============================================================================
def chunked_resize_xy_pass(arr, target_shape, order=1, logger=None,
                            grid_mode=True):
    """First pass of split chunked resize: per-Z slice XY-resize.

    Returns a float32 intermediate of shape (in_Z, target_Y, target_X). The
    caller is responsible for ``del``-ing its reference to ``arr`` before
    invoking the Z pass — otherwise the input volume coexists with both the
    intermediate and the eventual output, breaking the memory budget.

    Used at T_LARGE upscale where input(17 GB) + intermediate(79 GB) +
    output(60 GB) = 156 GB exceeds 137 GB host. With the split, the caller
    does ``del input`` between passes, freeing 17 GB (more under
    multi-channel) before output allocation.
    """
    in_shape = arr.shape
    if len(in_shape) != 3:
        raise ValueError(f"chunked_resize_xy_pass expects 3D, got {in_shape}")
    zf_yx = (target_shape[1] / in_shape[1], target_shape[2] / in_shape[2])
    inter_shape = (in_shape[0], target_shape[1], target_shape[2])
    if logger:
        logger.info(f"     chunked_resize_xy: {in_shape} -> {inter_shape} "
                    f"(zoom={[round(z, 4) for z in zf_yx]}, order={order})")
    intermediate = np.empty(inter_shape, dtype=np.float32)
    _slice_log_every = max(1, in_shape[0] // 20)
    for z in range(in_shape[0]):
        slice_2d = np.asarray(arr[z], dtype=np.float32, order='C')
        resized_2d = ndimage.zoom(slice_2d, zf_yx, order=order, prefilter=False,
                                   grid_mode=grid_mode,
                                   mode='grid-constant' if grid_mode else 'constant')
        intermediate[z] = resized_2d
        del slice_2d, resized_2d
        if logger and (z + 1) % _slice_log_every == 0:
            logger.info(f"       chunked_resize XY: {z + 1}/{in_shape[0]} slices")
    return intermediate


def chunked_resize_z_pass(intermediate, target_shape, order=1, dtype=None,
                          logger=None, tile_yx=1024, grid_mode=True):
    """Second pass of split chunked resize: per-(Y,X) tile Z-resize.

    Reads from a float32 intermediate (output of ``chunked_resize_xy_pass``)
    and produces the final volume in the requested output dtype. Caller MUST
    have del'd the original input before calling this so the output
    allocation has room.
    """
    inter_shape = intermediate.shape
    if dtype is None:
        dtype = np.float32
    zf_z = target_shape[0] / inter_shape[0]
    if logger:
        logger.info(f"     chunked_resize_z: {inter_shape} -> {target_shape} "
                    f"(zoom_z={round(zf_z, 4)}, order={order}, dtype={dtype})")
    output = np.empty(target_shape, dtype=dtype)
    n_tiles_y = (inter_shape[1] + tile_yx - 1) // tile_yx
    n_tiles_x = (inter_shape[2] + tile_yx - 1) // tile_yx
    n_tiles_total = n_tiles_y * n_tiles_x
    _tile_log_every = max(1, n_tiles_total // 20)
    _tile_idx = 0
    for y_start in range(0, inter_shape[1], tile_yx):
        y_end = min(y_start + tile_yx, inter_shape[1])
        for x_start in range(0, inter_shape[2], tile_yx):
            x_end = min(x_start + tile_yx, inter_shape[2])
            tile = intermediate[:, y_start:y_end, x_start:x_end]
            resized_tile = ndimage.zoom(tile, (zf_z, 1.0, 1.0), order=order,
                                         prefilter=False, grid_mode=grid_mode,
                                         mode='grid-constant' if grid_mode else 'constant')
            output[:, y_start:y_end, x_start:x_end] = resized_tile.astype(dtype, copy=False)
            del resized_tile
            _tile_idx += 1
            if logger and _tile_idx % _tile_log_every == 0:
                logger.info(f"       chunked_resize Z: tile {_tile_idx}/{n_tiles_total}")
    return output


def chunked_resize_3d(arr, target_shape, order=1, dtype=None, logger=None,
                       tile_yx=1024, grid_mode=True):
    """Memory-bounded 3D resize via two-pass scipy.ndimage.zoom.

    skimage.transform.resize internally calls convert_to_float() which allocates
    a full-volume float64 buffer. At T_LARGE (29.8B voxels) that's 222 GB —
    instant OOM on a 137 GB host. This helper does:

      Pass 1: per-Z-slice XY resize (each slice float32 ≈ slice bytes × 4).
      Pass 2: per-(Y,X) tile Z resize (tile float32 ≈ tile bytes × 4).

    Peak transient = max(input, intermediate) + small per-slab/per-tile working.
    For T_LARGE this peaks around ~85 GB instead of 222 GB.

    Parameters
    ----------
    arr : np.ndarray
        Input 3D volume (Z, Y, X). Any numeric dtype.
    target_shape : tuple of int
        Output shape (Zt, Yt, Xt).
    order : int
        scipy.ndimage.zoom order. 0 = nearest (use for label volumes), 1 = linear.
    dtype : numpy dtype or None
        Output dtype. Default = arr.dtype.
    logger : logging.Logger, optional
    tile_yx : int
        Y-X tile size for the second pass. Larger = better cache locality but
        more peak memory per tile.
    grid_mode : bool
        Pass-through to scipy.ndimage.zoom. True matches skimage.transform.resize
        semantics (sample positions are interpreted as voxel-grid centers); False
        matches the legacy scipy.ndimage.zoom default. Default True for parity
        with skimage callers, especially order=0 label volumes where grid_mode
        controls boundary tie-breaking.

    Returns
    -------
    np.ndarray
        Resized volume of shape ``target_shape``.

    Notes
    -----
    No anti-aliasing pre-blur (skimage's anti_aliasing=True). For the typical
    nnU-Net-prep case (downscaling 0.0426→0.065 µm) the loss of aliasing
    suppression is negligible compared to model robustness; the model was
    trained with augmented data.
    """
    in_shape = arr.shape
    if len(in_shape) != 3:
        raise ValueError(f"chunked_resize_3d expects 3D input, got shape {in_shape}")
    if dtype is None:
        dtype = arr.dtype

    zf = (target_shape[0] / in_shape[0],
          target_shape[1] / in_shape[1],
          target_shape[2] / in_shape[2])

    if logger:
        logger.info(f"     chunked_resize: {in_shape} -> {target_shape} "
                    f"(zoom={[round(z, 4) for z in zf]}, order={order})")

    # --- Pass 1: per-Z XY resize ---
    # Intermediate kept as float32 (not the output dtype) to avoid per-pass
    # quantization. Linear/cubic interpolation is separable: 2-pass equals
    # 1-pass mathematically, but rounding to integer between passes accumulates
    # ~3% mean error vs skimage on uint16 data. Float32 intermediate doubles
    # bytes vs uint16 (still less than full float64 path) but keeps fidelity.
    inter_shape = (in_shape[0], target_shape[1], target_shape[2])
    intermediate = np.empty(inter_shape, dtype=np.float32)
    _slice_log_every = max(1, in_shape[0] // 20)
    for z in range(in_shape[0]):
        slice_2d = np.asarray(arr[z], dtype=np.float32, order='C')
        resized_2d = ndimage.zoom(slice_2d, (zf[1], zf[2]), order=order,
                                   prefilter=False, grid_mode=grid_mode,
                                   mode='grid-constant' if grid_mode else 'constant')
        intermediate[z] = resized_2d
        del slice_2d, resized_2d
        if logger and (z + 1) % _slice_log_every == 0:
            logger.info(f"       chunked_resize XY: {z + 1}/{in_shape[0]} slices")

    # --- Pass 2: per-tile Z resize over the intermediate ---
    # Note on memory: at this point the caller still holds a reference to `arr`
    # so it stays live through Pass 2. Peak transient = arr + intermediate +
    # output + per-tile working. For T_LARGE that's roughly
    # 60 + 52 + 17 + 4 = 133 GB (uint16 input; float32 intermediate is the
    # dominant new allocation). If the host can't take that, the caller should
    # crop / channel-extract before calling this helper, and let GC release
    # the parent buffer before invocation.
    output = np.empty(target_shape, dtype=dtype)
    n_tiles_y = (inter_shape[1] + tile_yx - 1) // tile_yx
    n_tiles_x = (inter_shape[2] + tile_yx - 1) // tile_yx
    n_tiles_total = n_tiles_y * n_tiles_x
    _tile_log_every = max(1, n_tiles_total // 20)
    _tile_idx = 0
    for y_start in range(0, inter_shape[1], tile_yx):
        y_end = min(y_start + tile_yx, inter_shape[1])
        for x_start in range(0, inter_shape[2], tile_yx):
            x_end = min(x_start + tile_yx, inter_shape[2])
            # intermediate is already float32; ndimage.zoom takes the slice
            # view and produces a contiguous output. No extra copy needed.
            tile = intermediate[:, y_start:y_end, x_start:x_end]
            resized_tile = ndimage.zoom(tile, (zf[0], 1.0, 1.0), order=order,
                                         prefilter=False, grid_mode=grid_mode,
                                         mode='grid-constant' if grid_mode else 'constant')
            output[:, y_start:y_end, x_start:x_end] = resized_tile.astype(dtype, copy=False)
            del resized_tile
            _tile_idx += 1
            if logger and _tile_idx % _tile_log_every == 0:
                logger.info(f"       chunked_resize Z: tile {_tile_idx}/{n_tiles_total}")

    del intermediate
    return output
