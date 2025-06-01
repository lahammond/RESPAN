# -*- coding: utf-8 -*-
"""
Dask modified Image Analysis tools and functions for spine analysis
==========


"""

__author__    = 'Luke Hammond <luke.hammond@osumc.edu>'
__license__   = 'GPL-3.0 License (see LICENSE)'
__copyright__ = 'Copyright © 2024 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/RESPAN'




import sitecustomize

import RESPAN.ImageAnalysis.IO as io
import RESPAN.ImageAnalysis.ImageAnalysis as imgan
import RESPAN.ImageAnalysis.Tables as tables

import os
import gc
import shutil
import time
import numpy as np
import pandas as pd
import re
import math, psutil, pathlib, tempfile
import numcodecs
import uuid
import warnings
from typing import Sequence
from functools import partial


from pathlib import Path
from contextlib import nullcontext

from collections import defaultdict

import cupy as cp
from cupyx.scipy import ndimage as cp_ndimage
import cupy.cuda.runtime as rt
from cupyx.scipy.ndimage import binary_dilation

from scipy.ndimage import distance_transform_edt
from scipy.ndimage import generate_binary_structure
from scipy import ndimage
from scipy.spatial import cKDTree

import tifffile
from tifffile import imread, imwrite

from skimage import measure, morphology, segmentation
from skimage.measure._regionprops import RegionProperties

from multiprocessing.pool import ThreadPool

import dask
import dask.dataframe as dd
import dask.array as da
import dask.config as dask_config
import dask_image.ndmeasure as ndm
from dask.distributed import get_client, wait, Client, LocalCluster
from dask_image.ndmeasure  import label as dask_label
from dask.diagnostics import ProgressBar


import zarr
from ome_zarr.writer import write_image
from ome_zarr.io import parse_url
from numcodecs import Blosc
from zarr.errors import ArrayNotFoundError
from ome_zarr.writer import write_multiscales_metadata

try:
    from dask_image.ndinterp import zoom as da_zoom
except ImportError:

    def da_zoom(arr, zoom, order=1):
        """blockwise linear zoom that stays 100 % lazy."""
        from scipy.ndimage import zoom as _zoom
        zoom = tuple(float(z) for z in zoom)
        out_chunks = tuple(int(c * zoom[i]) for i, c in enumerate(arr.chunksize))

        return arr.map_blocks(
            lambda blk, z=zoom: _zoom(blk, zoom=z, order=order),
            dtype=arr.dtype,
            chunks=out_chunks,
        )

warnings.filterwarnings(
    "ignore",
    message=r"Failed to get convex hull image",
    module=r"skimage\.measure\._regionprops",
)
warnings.filterwarnings(
    "ignore",
    message=r"divide by zero encountered in double_scalars",
    module=r"skimage\.measure\._regionprops",
)


GB = 1024 ** 3
_RAM_TOTAL      = psutil.virtual_memory().total          # physical RAM
_MEM_CLIENT = None
N_CPU = psutil.cpu_count(logical=False)
num_blosc_threads = min(N_CPU, 32)               # 32 is Blosc's hard cap
numcodecs.blosc.set_nthreads(num_blosc_threads)
os.environ["BLOSC_NUM_THREADS"] = str(num_blosc_threads)

_COMP = COMP = Blosc(cname="zstd", clevel=5, shuffle=2)
COMP = Blosc(cname="lz4", clevel=1, shuffle=0)
_SMALL_VOL  = 200 * 1024 ** 2
_MAX_THREADS = 16

LAZY_THRESHOLD  = max(512 * 1024 ** 2, int(_RAM_TOTAL * 0.10))

# need to update these into settings later..
default_chunks = {
    'distance': (32, 128, 128),
    'analysis': (64, 256, 256),
}
CHUNK_SETTINGS = {}
for key, val in default_chunks.items():
    env = os.getenv(f"DAKZARR_CHUNKS_{key.upper()}")
    if env:
        CHUNK_SETTINGS[key] = tuple(int(x) for x in env.split(','))
    else:
        CHUNK_SETTINGS[key] = val


def start_local_cluster(
        *,
        max_workers:  None,
        target_ram_frac: None,
        min_mem_per_worker: str = "2GB",
        worker_mem: None,        # NEW – e.g. "8GB"
        threads_per_worker: None,  # NEW – default 1/core
        tmp_dir: None,
        logger=None,
        simulate = None,
):
    """Spin up a sensible LocalCluster on the current machine.

    Parameters
    ----------
    max_workers        Limit the total number of workers (None → automatic).
    target_ram_frac    Fraction of *physical* RAM the cluster may consume.
                       If None, choose 0.60 … 0.85 depending on system size.
    min_mem_per_worker Hard floor below which a worker is never created.
    worker_mem         Optional *soft* goal per worker (e.g. "4GB", "8GB").
                       Overridden by `min_mem_per_worker` if smaller.
    threads_per_worker Force a fixed thread-count.  None → one per core.
    simulate           "low" → pretend the box has 32 GB / 8 cores (testing).
    """
    # ── hardware snapshot ──────────────────────────────────────────────────
    if simulate == "low":
        total_ram   = 32 * GB
        phys_cores  = 8
    else:
        total_ram   = psutil.virtual_memory().total
        phys_cores  = psutil.cpu_count(logical=False) or 1

    # ── target RAM fraction ────────────────────────────────────────────────
    if target_ram_frac is None:
        if total_ram >= 200 * GB:
            target_ram_frac = 0.85
        elif total_ram >= 128 * GB:
            target_ram_frac = 0.80
        elif total_ram >= 64 * GB:
            target_ram_frac = 0.75
        else:
            target_ram_frac = 0.60
    target_ram = int(total_ram * target_ram_frac)

    # ── worker count heuristic (memory-bound first, then CPU) ──────────────
    bytes_min    = dask.utils.parse_bytes(min_mem_per_worker)
    bytes_goal   = dask.utils.parse_bytes(worker_mem) if worker_mem else None
    bytes_per_w  = max(bytes_goal or bytes_min, bytes_min)

    n_by_mem     = max(1, target_ram // bytes_per_w)
    n_by_cpu     = phys_cores // (threads_per_worker or 1)
    n_workers    = min(max_workers or n_by_mem, n_by_mem, n_by_cpu)
    n_workers    = max(n_workers, 1)

    # ── threads per worker ────────────────────────────────────────────────
    threads = threads_per_worker or max(1, phys_cores // n_workers)

    # ── final memory limit per worker ──────────────────────────────────────
    mem_per_worker = max(int(target_ram / n_workers), bytes_min)
    mem_limit      = f"{mem_per_worker//GB}GB"

    # ── spill directory ────────────────────────────────────────────────────
    spill_dir = Path(tmp_dir or tempfile.gettempdir()) / "dask-spill"
    spill_dir.mkdir(exist_ok=True)

    # ── launch cluster ─────────────────────────────────────────────────────
    cluster = LocalCluster(
        processes=True,
        n_workers=n_workers,
        threads_per_worker=threads,
        memory_limit=mem_limit,
        local_directory=str(spill_dir),
        dashboard_address=":8787",
    )
    client = Client(cluster)

    client.run(lambda: dask.config.set({
        "distributed.worker.memory.target"   : target_ram_frac - 0.05,
        "distributed.worker.memory.pause"    : target_ram_frac + 0.10,
        "distributed.worker.memory.terminate": False,
    }))

    if logger:
        logger.info(
            f"Cluster: {n_workers} workers × {threads} threads, "
            f"{mem_limit} per worker "
            f"(target {target_ram_frac:.2f} · phys {total_ram//GB} GB)"
        )
    return client, cluster


def _write_chunk_to_zarr(chunk, zroot, group, block_id=None):
    ds = zarr.open(str(zroot / group), mode='a')['0']
    ds.store_chunk(chunk)

def _area(_img, lbls: da.Array, labs: np.ndarray) -> np.ndarray:
    """
    Cheap voxel‑count (area) for each label ID in `labs`.

    Parameters
    ----------
    _img : da.Array      (ignored – passed by the caller)
    lbls : da.Array      label image (same shape as the data)
    labs : np.ndarray    1‑D array of label IDs to measure
    """
    ones = da.ones_like(lbls, dtype=np.uint8)
    return ndm.sum(ones, lbls, labs)               # area per label

_FAST_PROP_FUNCS: dict[str, callable] = {
    "mean_intensity": lambda img, lbls, labs: ndm.mean(img.compute(), lbls.compute(), labs),
    "max_intensity" : lambda img, lbls, labs: ndm.maximum(img.compute(), lbls.compute(), labs),
    "area"          : _area,
}

_UNSAFE_PROPS = {
    "area_convex",          # needs convex hull
    "solidity", "extent",   # rely on convex hull
    "axis_major_length",    # rely on eigen‑analysis (sqrt(‑ve) crash)
    "axis_minor_length",
    "feret_diameter_max",
}

def _identity(x):
    return x                         # used by map_blocks

def _uid(p="a"):
    return f"{p}-{uuid.uuid4().hex[:8]}"

def _rekey(arr, p):
    """Return *arr* with a new graph name (zero copy/compute)."""
    return arr.map_blocks(lambda x: x, dtype=arr.dtype, name=_uid(p))

def _ensure_dask(arr):
    return arr if isinstance(arr, da.Array) else da.from_array(arr, chunks=arr.shape)

def rekey(arr: da.Array, tag: str = "rk") -> da.Array:
    """Return *arr* with a brand‑new graph key (O(1))."""
    uid = f"{tag}-{uuid.uuid4().hex}"
    return arr.map_blocks(lambda x: x, name=uid, dtype=arr.dtype)

def _print_mem(tag: str) -> None:
    """Print driver + worker RSS (GiB) so we can watch Dask usage grow."""

    drv = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    if _MEM_CLIENT is None:
        print(f"[MEM] {tag:36s} driver {drv:5.2f} GB")
        return
    try:
        wk = _MEM_CLIENT.run(lambda:
                             __import__('psutil').Process().memory_info().rss)
        tot_wk = sum(wk.values()) / 1024**3
        print(f"[MEM] {tag:36s} driver {drv:5.2f} GB   "
              f"{len(wk)} wkr {tot_wk:5.2f} GB   total {drv+tot_wk:5.2f} GB")
    except Exception:
        print(f"[MEM] {tag:36s} driver {drv:5.2f} GB  (workers n/a)")


# 1. monkey‑patch axis_major/minor_length so sqrt() never sees < 0
def _safe_axis_major(self):
    ev = self.inertia_tensor_eigvals  # ascending
    rad =  6 * ( ev[2] + ev[1] - ev[0])
    return math.sqrt(rad) if rad > 0 else 0.0

def _safe_axis_minor(self):
    ev = self.inertia_tensor_eigvals
    rad = 10 * (-ev[0] + ev[1] + ev[2])
    return math.sqrt(rad) if rad > 0 else 0.0

RegionProperties.axis_major_length = property(_safe_axis_major)
RegionProperties.axis_minor_length = property(_safe_axis_minor)

# 2. wrap regionprops_table so it quietly ignores unknown kwargs
_orig_rpt = measure.regionprops_table
def _rpt_compat(label_image, *, label_ids=None, **kwargs):
    try:
        if "label_ids" in _orig_rpt.__code__.co_varnames:
            return _orig_rpt(label_image, label_ids=label_ids, **kwargs)
        return _orig_rpt(label_image, **kwargs)          # old skimage
    except TypeError:                                    # unexpected kw
        return _orig_rpt(label_image, **kwargs)

measure.regionprops_table = _rpt_compat




def get_root(zarr_path: Path) -> zarr.Group:
    """Return the root group, creating the directory if needed."""
    return zarr.open_group(str(zarr_path), mode="a")

def _is_valid_array(obj) -> bool:
    """True when *obj* is a zarr Array *or* a group containing '0' Array."""
    if isinstance(obj, zarr.core.Array):
        return True
    if isinstance(obj, zarr.hierarchy.Group) and '0' in obj:      # NGFF case
        return isinstance(obj['0'], zarr.core.Array)
    return False

def _ensure_zarr_path(p: Path):
    """Create the directory and return it as POSIX str (helper)."""
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()

def exists(root: zarr.Group, key: str) -> bool:
    """Return **True only** when a readable array is present at *key*."""
    try:
        return _is_valid_array(root[key])
    except (KeyError, ArrayNotFoundError):
        return False


def load(root: zarr.Group, key: str) -> da.Array:
    """Lazy‐load the array regardless of whether it lives at *key* or *key/0*."""
    obj = root[key]
    if isinstance(obj, zarr.core.Array):
        return da.from_zarr(obj)
    return da.from_zarr(obj['0'])                # NGFF layout


def save(root: zarr.Group, key: str, arr: da.Array) -> None:
    """Overwrite any stale data and store *arr* at *key/0* (NGFF style)."""
    if key in root:          # nuke broken group/array first
        del root[key]
    save_volume_to_omezarr(arr, Path(root.store.path), key)


def get_or_compute(root: zarr.Group,
                   key: str,
                   fn):
    if exists(root, key):
        return load(root, key)
    root_root = Path(root.store.path)
    (root_root / key).mkdir(parents=True, exist_ok=True)
    arr = fn()
    return arr

def _read_block(path: str,
                block_id: tuple[int, ...],
                chunk_shape: tuple[int, ...]) -> np.ndarray:
    """
    Worker helper.  Opens *path*, returns the requested block.

    • Handles 3‑D (Z Y X) or 4‑D (C Z Y X) TIFFs.
    """
    with tifffile.TiffFile(path) as tf:
        arr = tf.series[0].asarray()

    if arr.ndim == 3:                                    # Z Y X
        arr = arr[np.newaxis, ...]                       # → C Z Y X with C=1

    # build per‑axis start / stop
    starts = [b * s for b, s in zip(block_id, chunk_shape)]
    stops  = [min(st + cs, sz) for st, cs, sz
              in zip(starts, chunk_shape, arr.shape)]
    slc = tuple(slice(st, en) for st, en in zip(starts, stops))
    return arr[slc]


def _local_to_zarr(arr, store, *, component="0", n_threads=None, show_pb=True, logger=None):
    """
    Store *arr* → *store* inside the current process.

    Parameters
    ----------
    n_threads : int or None
        Number of CPU threads that Dask + Blosc may use.
        • None  →  automatic  (min( #physical cores, 32 ))
    show_pb   : bool
        If True, draw a Dask progress bar in the terminal / log.
    """
    n_threads = n_threads or min(os.cpu_count() or 1, 32)
    if logger:
        logger.info(f"     [OME-Zarr] writing with {n_threads} thread(s)…")

    # Blosc honours BLOSC_NUM_THREADS  ➜  we set it just for this call
    os.environ.setdefault("BLOSC_NUM_THREADS", str(n_threads))

    pool = ThreadPool(n_threads)                # for the threaded scheduler
    sched_cfg = {"scheduler": "threads", "pool": pool}

    if show_pb and logger is not None:
        pb_ctx = ProgressBar(out=_LoggerWriter(logger))
    elif show_pb:
        pb_ctx = ProgressBar()  # stdout fallback
    else:
        pb_ctx = nullcontext()

    with dask_config.set(**sched_cfg), pb_ctx:       # <─ both contexts together
        da.to_zarr(
            arr,
            store,
            overwrite=True,
            compute=True,
            compressor=COMP,
            lock=False,
        )


def tiff_to_dask(path: str,
                 chunk_shape_in: tuple[int, ...]) -> da.Array:
    """
    Return a Dask array backed by per‑chunk TIFF reads.
    *chunk_shape_in* may be (Z,Y,X) or (C,Z,Y,X).  The function pads or
    trims it to match the TIFF dimensionality.
    """
    with tifffile.TiffFile(path) as tf:
        shape = tf.series[0].shape                    # (Z,Y,X) or (C,Z,Y,X)

    if len(shape) == 3:                               # ensure 4‑D shape
        shape = (1, *shape)

    # harmonise chunk spec
    if len(chunk_shape_in) == 3:                      # user gave Z,Y,X
        chunk_shape = (1, *chunk_shape_in)
    elif len(chunk_shape_in) == 4:
        chunk_shape = chunk_shape_in
    else:
        raise ValueError("chunk_shape must have 3 or 4 elements")

    nchunks = tuple(int(np.ceil(s / cs)) for s, cs in zip(shape, chunk_shape))

    name = "tiff-read-" + uuid.uuid4().hex
    dsk = {
        (name, c, z, y, x):
            (_read_block, path, (c, z, y, x), chunk_shape)
        for c in range(nchunks[0])
        for z in range(nchunks[1])
        for y in range(nchunks[2])
        for x in range(nchunks[3])
    }

    graph = dask.highlevelgraph.HighLevelGraph.from_collections(
        name, dsk, dependencies=[])
    return da.Array(graph, name, chunks=chunk_shape,
                    dtype=np.uint16, shape=shape)

def save_small(arr, zroot, group, *, compressor=COMP):
    """
    For volumes ≤ LAZY_THRESHOLD write immediately from the driver
    and return True.  Otherwise return False so the caller can fall
    back to the lazy-streaming path.
    """
    if arr.size * arr.dtype.itemsize <= LAZY_THRESHOLD:
        _local_to_zarr(
            arr,
            (Path(zroot) / group).as_posix(),
            component="0",
            n_threads=None
        )
        return True
    return False

def _run_with_progress(delayed_obj, logger=None, show_pb=True):
    """Compute *delayed_obj* with an optional Dask ProgressBar."""
    if show_pb:
        ctx = ProgressBar(out=_LoggerWriter(logger)) if logger else ProgressBar()
        with ctx:
            dask.compute(delayed_obj)
    else:
        dask.compute(delayed_obj)

def open_tiff_as_dask(
    tiff_path : str,
    *,
    client=None,
    chunks=(1, 64, 512, 512),
    pixel_sizes=None,
    resave=True,
    settings=None,
    logger=None,
):
    """
    Return a ``dask.array`` that points at an on-disk OME-Zarr store,
    converting the TIFF the first time we see it.  Any *incomplete* store
    is deleted and rebuilt automatically.
    """



    z_path = Path(tiff_path).with_suffix(".ome.zarr")

    if (z_path / "0" / ".zarray").exists():
        return da.from_zarr(str(z_path), component="0"), z_path


        # -- build cache on-demand ---------------------------------------------
    if resave:
        if resave:
            axes_format, success = tiff_to_ome_zarr(
                str(tiff_path), str(z_path),
                chunks=_flatten(chunks),
                pixel_sizes=pixel_sizes,
                logger=logger,
            )
            if success:
                # Load the data
                labels = da.from_zarr(str(z_path), component="0")

                # Handle either ZYX or CZYX format
                if axes_format == "czyx":
                    # If it has a channel dimension, remove it
                    labels = labels[0, :, :, :]  # remove channel dimension
                else:
                    # Already in ZYX format, no need to remove channel
                    pass
        return da.from_zarr(str(z_path), component="0"), z_path

    # -- no cache requested → lazy TIFF read -------------------------------
    with tifffile.TiffFile(tiff_path) as tf:
        z_in = zarr.open(tf.series[0].aszarr(), mode="r")
        img  = da.from_array(z_in, chunks=_flatten(chunks))

    return img, None  # keep the same 2-tuple contract


def tiff_to_ome_zarr(
    tiff_path,
    zarr_root,
    *,
    chunks: tuple[int, ...] = (1, 64, 512, 512),     # (C,Z,Y,X)
    pixel_sizes= None,
    compressor=COMP,
    overwrite: bool = False,
    logger=None,
    show_pb: bool = True,                             # progress-bar flag
) -> None:
    """
    Convert *tiff_path* → *zarr_root*/0 (OME-Zarr, level-0, no Dask).

    • ≤ _SMALL_VOL (200 MiB)  →  one-shot write.
    • larger volumes          →  streamed in blocks of *chunks[1]* Z-slices
                                  (per-channel when C present).

    Designed for Python 3.9, Windows/Linux, low-RAM boxes and workstations.
    """
    # ── paths & early-exit ──────────────────────────────────────────────
    tiff_path = Path(tiff_path)
    zarr_root = Path(zarr_root)

    # ── fast-exit ──────────────────────────────────────────────────────────
    if (zarr_root / "0" / ".zarray").exists() and not overwrite:
        if logger: logger.info("[OME-Zarr] exists – skip")
        # Try to determine the format from existing file
        try:
            with zarr.open(str(zarr_root), mode='r') as z:
                if "multiscales" in z.attrs:
                    axes_meta = z.attrs["multiscales"][0]["axes"]
                    axes = "".join(axis["name"] for axis in axes_meta)
                    has_c = "c" in axes
                    return axes, True
        except Exception:
            # If we can't determine, assume CZYX as default
            return "czyx", True

    if zarr_root.exists():
        shutil.rmtree(zarr_root)
    zarr_root.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------#
    #  Read TIFF header once
    # ---------------------------------------------------------------------#
    try:
        with tifffile.TiffFile(tiff_path) as tf:
            series = tf.series[0]
            shape = series.shape  # Z Y X  or  C Z Y X
            dtype = series.dtype
            nbytes = math.prod(shape) * dtype.itemsize

            # axis handling ----------------------------------------------------
            if len(shape) == 3:  # Z Y X
                axes, need_c = "zyx", False
            elif len(shape) == 4:  # C Z Y X
                axes, need_c = "czyx", True
            else:
                raise ValueError(f"unsupported TIFF dimensions {shape}")

            # -----------------------------------------------------------------#
            #  SMALL VOLUME – single writer call
            # -----------------------------------------------------------------#
            if nbytes <= _SMALL_VOL:
                data = series.asarray(out="memmap")
                root = zarr.group(store=parse_url(str(zarr_root), mode="w").store)
                # write_multiscales_metadata is not needed; writer makes it.
                from ome_zarr.writer import write_image
                write_image(
                    image=data,
                    group=root,
                    axes=axes,
                    chunks=chunks,
                    pixel_sizes=pixel_sizes,
                    storage_options={"compressor": compressor},
                    compute=True,
                    scaler=None,
                )
                if logger:
                    logger.info(f"      Created: {zarr_root}")
                return axes, True

            # -----------------------------------------------------------------#
            #  LARGE VOLUME – streamed copy
            # -----------------------------------------------------------------#
            root = zarr.group(store=parse_url(str(zarr_root), mode="w").store)
            z_arr = root.create_dataset(
                "0",
                shape=shape,
                chunks=chunks[:len(shape)],  # Ensure chunks match dimensionality
                dtype=dtype,
                compressor=compressor,
                overwrite=True,
            )

            # progress bar -----------------------------------------------------
            prog = None
            if show_pb:
                try:
                    from tqdm import tqdm
                    total = shape[0] * shape[1] if need_c else shape[0]
                    prog = tqdm(total=total, unit="slice", desc="OME-Zarr")
                except ModuleNotFoundError:
                    prog = None

            # choose Z-block size (<= chunks[1] to limit RAM)
            block_z = chunks[1] if need_c else chunks[0]
            block_z = max(1, block_z)

            if need_c:  # C Z Y X
                for c in range(shape[0]):
                    for z0 in range(0, shape[1], block_z):
                        z1 = min(z0 + block_z, shape[1])
                        block = series.asarray(key=(c, slice(z0, z1)))
                        z_arr[c, z0:z1, :, :] = block
                        if prog:
                            prog.update(z1 - z0)
            else:  # Z Y X
                for z0 in range(0, shape[0], block_z):
                    z1 = min(z0 + block_z, shape[0])
                    block = series.asarray(key=slice(z0, z1))
                    z_arr[z0:z1, :, :] = block
                    if prog:
                        prog.update(z1 - z0)

            if prog:
                prog.close()

            # -----------------------------------------------------------------#
            #  Attach NGFF metadata
            # -----------------------------------------------------------------#
            axes_meta, datasets_meta = _axes_and_transform(axes)
            write_multiscales_metadata(root, datasets_meta, axes=axes_meta)
            if pixel_sizes:
                # write_multiscales_metadata already stores pixel size if given
                root.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] = list(pixel_sizes)

            if logger:
                logger.info(f"[OME-Zarr] wrote {zarr_root}")

            return axes, True
    except Exception as e:
        if logger:
            logger.error(f"Error in tiff_to_ome_zarr: {str(e)}")
        # Clean up partial conversion
        if zarr_root.exists():
            try:
                shutil.rmtree(zarr_root)
            except Exception:
                pass
        return "czyx", False  # Return default format and failure flag


def _axes_and_transform(axes_str: str):
    if axes_str == "czyx":
        axes_meta = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]
    elif axes_str == "zyx":
        axes_meta = [
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]
    else:
        raise ValueError(f"unsupported axes '{axes_str}'")
    scale = [1.0] * len(axes_meta)
    datasets = [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": scale}]}]
    return axes_meta, datasets

def save_volume_to_omezarr(
        arr: da.Array,
        zroot: Path,
        group: str,                 # kept for signature compatibility
        *,
        compressor=COMP,
        logger=None,
        show_pb=True,
):
    """
    Persist *arr* in NGFF layout  <zroot>/0  (multiscales + dataset “0”).
    Logic mirrors `tiff_to_ome_zarr` – one-shot write for ≤200 MiB,
    Dask-delayed write otherwise.
    """
    try:
        # ── ensure (Z,Y,X) and choose axes ───────────────────────────────
        if arr.ndim == 4 and arr.shape[0] == 1:   # (1,Z,Y,X) → strip C
            arr  = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"array must be 3-D or (1,Z,Y,X); got {arr.shape}")
        axes = "zyx"

        zroot.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.group(
            store=zarr.DirectoryStore(str(zroot)),
            overwrite=True
        )

        SMALL = 200 * 1024 ** 2
        small = arr.nbytes <= SMALL
        if logger:
            note = "small" if small else f"{arr.nbytes/1024**2:,.1f} MB"
            logger.info(f"       save_volume_to_omezarr – {note}")

        # ── small volume: compute NumPy, single call ─────────────────────
        if small:
            data = arr.compute() if isinstance(arr, da.Array) else arr
            write_image(
                image          = data,
                group          = root,
                axes           = axes,
                chunks         = data.shape,          # one chunk
                storage_options={"compressor": compressor},
                compute        = True,
                scaler         = None,
            )
            return True

        # ── large volume: delayed write_image + single compute ───────────
        delayed = write_image(
            image          = arr,
            group          = root,
            axes           = axes,
            chunks         = _flatten(arr.chunks),
            storage_options={"compressor": compressor},
            compute        = False,
            scaler         = None,
        )

        ctx = (ProgressBar(out=_LoggerWriter(logger))
               if show_pb and logger else
               ProgressBar() if show_pb else nullcontext())

        with ctx:
            try:                                    # distributed client
                fut = get_client().compute(delayed, retries=0)
                wait(fut)
            except ValueError:                      # local threads
                dask.compute(delayed, scheduler="threads",
                              pool=ThreadPool(min(os.cpu_count(), 8)))
        return True

    except Exception as e:
        if logger:
            logger.error(f"save_volume_to_omezarr: {e}")
        return False

def distance_map_to_zarr(
        mask: da.Array,
        zarr_path: Path,
        voxel_size: Sequence[float],
        *,
        chunks=None,
        max_dist=None,
        global_scale=4,
        compressor=COMP,
        logger=None,
) -> da.Array:
    """
    Compute an Euclidean distance map for *mask* and save it as
    <zarr_path>/0.  For volumes ≤ 200 MiB the EDT is calculated
    completely with SciPy on NumPy data, bypassing the dask-overlap path
    that caused shape-broadcast errors on non-divisible dimensions.
    """
    SMALL_VOL = 200 * 1024 ** 2

    # ---------- make sure we have a Dask array --------------------------------
    if not isinstance(mask, da.Array):
        mask_da = da.from_array(mask, chunks=chunks or CHUNK_SETTINGS["distance"])
    else:
        mask_da = mask.rechunk(chunks or CHUNK_SETTINGS["distance"])

    nbytes = mask_da.size * mask_da.dtype.itemsize
    if logger:
        note = "small – SciPy" if nbytes <= SMALL_VOL else "large – dask EDT"
        logger.info(f"      Computing distance transform ({note}), shape {mask_da.shape}")

    # ---------- small volume: SciPy path --------------------------------------
    if nbytes <= SMALL_VOL:
        mask_np = mask_da.compute()           # NumPy boolean array
        dist_np = distance_transform_edt(~mask_np,
                                         sampling=voxel_size).astype("float32")
        dist_da = da.from_array(dist_np,
                                chunks=chunks or CHUNK_SETTINGS["distance"])
    # ---------- large volume: existing Dask EDT -------------------------------
    else:
        dist_da = distance_transform_edt_dask(
            mask_da,
            sampling=voxel_size,
            max_dist=max_dist,
            global_scale=global_scale,
        ).astype("float32")

    # ---------- save to NGFF ---------------------------------------------------
    zroot = Path(zarr_path)
    ok = save_volume_to_omezarr(
        dist_da,
        zroot,
        group="0",
        compressor=compressor,
        logger=logger,
    )
    if not ok:
        raise RuntimeError("     Failed to save distance map to zarr")

    return da.from_zarr(str(zroot / "0"))


def _filter_block(block: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """
    Vectorised per-chunk filter.
    `keep_mask[id] == True`  → keep original label
    `keep_mask[id] == False` → set to 0
    """
    return np.where(keep_mask[block], block, 0).astype("uint16", copy=False)


def filter_dendrites_dask(dend_da: da.Array, settings, logger):
    """
    Keep dendrites whose voxel-count ≥ settings.min_dendrite_vol.
    Returns a uint16 Dask array with ORIGINAL label IDs.
    Two passes only:
        1) connected components (26-conn)
        2) voxel-count reduction + block-wise masking
    """

    # ── uniform spatial chunks before labeling ──────────────────────────
    dend_da = dend_da.rechunk((64, 256, 256))

    # ── connected components ────────────────────────────────────────────
    lbl_da, n_labels = dask_label(dend_da)               # lazy
    max_label = int(n_labels.compute())
    if max_label == 0:
        logger.info("No dendrites found in volume.")
        return lbl_da.astype("uint16")

    # ── voxel counts per label (cheap reduction) ────────────────────────
    ones   = da.ones_like(lbl_da, dtype=np.uint8)
    idx    = np.arange(1, max_label + 1, dtype=np.uint32)
    counts = ndm.sum(ones, lbl_da, idx).compute()        # NumPy array
    keep   = idx[counts >= settings.min_dendrite_vol]

    logger.info(f"    Processing {len(keep)} of {max_label} dendrites "
                f"≥ {settings.min_dendrite_vol} vox")

    if keep.size == 0:
        return da.zeros_like(lbl_da, dtype="uint16")

    # ── build lookup mask once, broadcast to every worker ───────────────
    lut = np.zeros(max_label + 1, dtype=bool)
    lut[keep] = True
    lut_d = dask.delayed(lut)             # avoid large task graph literal

    # ── block-wise filtering (avoids global da.isin overhead) ───────────
    filtered = da.map_blocks(
        _filter_block,
        lbl_da,
        lut_d,
        dtype="uint16",
        meta=np.array((), dtype="uint16")
    )

    return filtered



class _LoggerWriter:
    """
    Thin stream-like adapter so Dask's ProgressBar can write to `logger`
    without flooding the log.

    • Only forwards a line when the percentage crosses the next threshold
      (20 % by default → 0 %, 20 %, 40 %, 60 %, 80 %, 100 %).
    • Any non-percentage text (errors, final summary) is forwarded verbatim.
    """

    _pct_re = re.compile(r"(\d+)%")

    def __init__(self, logger, level: str = "info",
                 step: float = 0.20, indent: str = "    "):
        self._logger = logger
        self._emit = getattr(logger, level.lower())
        self._step = max(step, 1e-6)
        self._next = 0.0  # next percentage to report
        self._indent = indent
        self._buf = ""  # accumulates partial lines

    # ---------------------------------------------------------------- write --
    def write(self, txt: str) -> None:
        if not txt:
            return
        self._buf += txt.replace("\r", "\n")     # treat CR like LF

        while "\n" in self._buf:                 # process *whole* lines
            line, self._buf = self._buf.split("\n", 1)
            if not line.strip():                 # skip blank line
                continue

            m = self._pct_re.search(line)
            if m:                                # -------- % progress -----
                pct = int(m.group(1)) / 100.0
                if pct + 1e-9 >= self._next:     # crossed the threshold
                    self._emit(f"{self._indent}{line.rstrip()}")
                    while self._next <= pct:
                        self._next += self._step
            # silently ignore any non-percentage chatter

    def flush(self):                             # file-like API
        pass


def add_singleton_c(arr):
    "Ensure arr is (C,Z,Y,X) with C=1 – needed when SAVING to NGFF."
    if arr.ndim == 3:
        new_chunks = ((1,),) + arr.chunks
        return arr.map_blocks(lambda b: b[np.newaxis, ...],
                              chunks=new_chunks,
                              dtype=arr.dtype)
    return arr        # already has C

def strip_singleton_c(arr):
    "Drop a trailing C=1 axis – preferred during ANALYSIS."
    return arr[:, 0, ...] if (arr.ndim == 4 and arr.shape[1] == 1) else arr

def _add_channel_axis(block: np.ndarray) -> np.ndarray:
    """Return a view with an explicit singleton channel axis C=1."""
    # incoming 3-D block  →  outgoing 4-D (1, Z, Y, X)
    return block[np.newaxis, ...]



def _flatten(chunkspec):
    if isinstance(chunkspec, tuple):
        return tuple(c[0] if isinstance(c, tuple) else c
                     for c in chunkspec)
    return chunkspec



def _zoom_linear_numpy(arr, zoom_factors, out_chunks):
    """CPU fallback: interpolate a *small* array in memory, re-wrap as dask."""
    arr_np = ndimage.zoom(arr.compute(),    # coarse → NumPy
                          zoom=zoom_factors,
                          order=1)          # linear
    return da.from_array(arr_np, chunks=out_chunks)



def distance_transform_edt_dask(
    binary,
    *,
    sampling=None,
    max_dist: None,
    global_scale: int = 4,
    _reduce=np.min,
):
    """
    Hybrid, scalable Euclidean distance transform (EDT).

    Parameters
    ----------
    binary : array-like (NumPy or Dask)
        Boolean mask *True = foreground*, distance is measured to the next
        *False* voxel (i.e. we call SciPy with the logical NOT).
    sampling : tuple[float] | None
        Voxel spacings (*dz, dy, dx, …*).  `None` → all 1.0 µm.
    max_dist : float | None, default 64
        *Finite value*  →  exact EDT inside a `max_dist` halo around every chunk
                           (fast and accurate near objects).
        *None*          →  **no cap** – skip the exact pass completely and
                           compute the coarse/global EDT only (lower RAM).
    global_scale : int, default 4
        Down-sampling factor for the global pass.  Larger ⇒ fewer pixels,
        less RAM, but coarser plateaus when up-scaled.
    _reduce : callable, default ``numpy.min``
        Function given to :py:meth:`dask.array.Array.coarsen` that reduces each
        *global_scale³* block to a single voxel.
        • `np.min` keeps *any* foreground voxel → preserves topology.
        • You could pass `np.max` or `np.mean` for different behaviour.

    Returns
    -------
    da.Array (float32)
        Distance map with the **same shape & chunks** as *binary*.

    Notes
    -----
    •  When *max_dist is None* we only run the *coarse* pass.
       That means distances will change in `global_scale`-voxel steps
       (a “terraced” look).  To **smooth** those plateaus you may feed the
       result through :pyfunc:`scipy.ndimage.zoom` or Dask’s
       :pyfunc:`dask_image.ndinterp.zoom` afterwards.
    """

    # ---------- 0 · pre-flight housekeeping ---------------------------------
    if binary.ndim == 4 and binary.shape[0] == 1:          # (1, Z, Y, X)
        binary = binary[0]

    if sampling is None:
        sampling = (1.0,) * binary.ndim
    elif len(sampling) != binary.ndim:                     # mimic SciPy rule
        sampling = (sampling + (1.0,) * binary.ndim)[: binary.ndim]

    if not isinstance(binary, da.Array):                   # pure NumPy
        return ndimage.distance_transform_edt(~binary, sampling=sampling)\
                     .astype(np.float32)

    # ---------- 1 · local exact EDT (unless max_dist is None) ---------------
    full_range = max_dist is None
    if not full_range:
        # halo depth limited by both chunk size and max_dist
        depth = [max(0, min(int(max_dist) + 2, min(ch) - 1))
                 for ch in binary.chunks]

        exact_fn = lambda b: ndimage.distance_transform_edt(~b, sampling=sampling)\
                                       .astype(np.float32)

        if any(d == 0 for d in depth):                     # very small chunks
            local_dt = binary.map_blocks(exact_fn, dtype=np.float32)
        else:
            local_dt = da.map_overlap(
                exact_fn, binary,
                depth=tuple(depth), boundary="reflect",
                dtype=np.float32, meta=np.array(())
            )
    else:
        local_dt = None                                    # skip exact path
        max_dist = -1.0                                    # sentinel

    # ---------- 2 · coarse/global EDT ---------------------------------------
    if global_scale < 2:
        coarse_dt = local_dt
    else:
        scale_map = {ax: global_scale for ax in range(binary.ndim)}

        # robust coarsen: method when available, else function
        if hasattr(binary, "coarsen"):  # Dask ≥2023.x
            coarse_mask = binary.coarsen(_reduce, scale_map, trim_excess=True)
        else:  # older builds
            coarse_mask = da.coarsen(_reduce, binary, scale_map,
                                     trim_excess=True)

        coarse_sampling = tuple(s * global_scale for s in sampling)
        coarse_dt = coarse_mask.map_blocks(
            lambda b: ndimage.distance_transform_edt(~b,
                                                     sampling=coarse_sampling
                                                     ).astype(np.float32),
            dtype=np.float32)

        # ----- up-sample back ------------------------------------------------
        zoom_factors = tuple(float(binary.shape[i]) / coarse_dt.shape[i]
                             for i in range(binary.ndim))

        if da_zoom is not None:  # , fully lazy
            coarse_dt = da_zoom(coarse_dt, zoom=zoom_factors, order=1)

        elif coarse_dt.npartitions == 1:  # single chunk → safe
            coarse_dt = coarse_dt.map_blocks(
                lambda blk: ndimage.zoom(blk, zoom=zoom_factors, order=1),
                dtype=np.float32,
                chunks=binary.chunks)

        else:  #  tiny in-memory fallback
            coarse_dt = _zoom_linear_numpy(coarse_dt,
                                           zoom_factors,
                                           out_chunks=binary.chunks)

        coarse_dt = coarse_dt[tuple(slice(0, s) for s in binary.shape)]

        if coarse_dt.chunks != binary.chunks:
            coarse_dt = coarse_dt.rechunk(binary.chunks)

    # ---------- 3 · merge & return ------------------------------------------
    final_dt = coarse_dt if full_range else \
               da.where(local_dt < max_dist, local_dt, coarse_dt)

    return final_dt.astype(np.float32, copy=False)


def distance_transform_edt_dask_previous(binary,
                                *,
                                sampling=(1,1,1),
                                max_dist=64):
    """
    Global EDT that *automatically clamps halo size* to each chunk.

    • If a chunk is smaller than 2×max_dist, halo = chunk‑1.
    • If the whole array is smaller than max_dist*2, fall back to SciPy
      on the entire array (no overlap needed).
    """
    if binary.ndim == 4 and binary.shape[0] == 1:  # C Z Y X, C=1
        binary = binary[0]  # → Z Y X

    if len(sampling) != binary.ndim:
        sampling = sampling[-binary.ndim:]  # keep last dims

    if not isinstance(binary, da.Array):
        # already NumPy – just run SciPy once
        return distance_transform_edt(binary, sampling=sampling).astype(np.float32)

    # ---- choose per‑axis halo, never exceeding chunk‑size‑1 ------------
    depth = []
    for c in binary.chunks:
        depth.append(tuple(min(int(max_dist)+2, int(ch[0])-1) for ch in [c]))
    depth = tuple(d[0] for d in depth)

    # if *any* axis would get depth<=0, compute whole‑array directly
    if any(d <= 0 for d in depth):
        return binary.map_blocks(
            lambda b: distance_transform_edt(b, sampling=sampling).astype(np.float32),
            dtype=np.float32)

    fn = lambda b, **kw: distance_transform_edt(b, sampling=sampling).astype(np.float32)

    return da.map_overlap(fn, binary,
                          depth      =depth,
                          boundary   ='reflect',
                          dtype      =np.float32,
                          meta       =np.array(()))


def spine_and_whole_neuron_processing(image, labels_vol, spine_summary, raw_zarr_root,  settings, locations, filename, log, logger):

    time_initial = time.time()
    root = get_root(raw_zarr_root)

    _print_mem("TIFF loaded")
    # log_memory_usage(logger)

    neuron = image[:, settings.neuron_channel - 1, :, :]

    if neuron.size > 1e8:
        logger.info(
            f"   The neuron channel for this image is ~{neuron.size / 1e9:.2f} GB in size. This may take considerable time to process.")
        logger.info(
            f"   Estimated processing time is ~{round(neuron.size * 2.5e-8, 0)} minutes, depending on computational resources.")

        # temp size limits
    if neuron.size > 1e9:
        logger.info(
            f"    *Note, as the dataset is over 1GB, full 3D validation data export has been disabled (as these volumes can be 20x raw input)."
            f"\n    To generate 3D validation datasets, please isolate specific regions of the dataset and process separately.")
        logger.info(
            f"    Due to the size of this image volume, neck generation features are currently unavailable.")
        settings.neck_analysis = False
        settings.mesh_analysis = True
        settings.save_val_data = False
    else:
        settings.neck_analysis = True
        settings.mesh_analysis = True

    if settings.model_type == 1:
        spines = (labels_vol == 1)
        dendrites = (labels_vol == 2)
        soma = (labels_vol == 3)
    elif settings.model_type == 2:
        dendrites = (labels_vol == 1)
        soma = (labels_vol == 2)
        spines = (labels_vol == 10)  # create an empty volume for spines
    elif settings.model_type == 3:
        spines = (labels_vol == 1)
        dendrites = (labels_vol == 2)
        soma = (labels_vol == 3)
        necks = (labels_vol == 4)

    elif settings.model_type == 4:
        dendrites = (labels_vol == 1)
        spine_cores = (labels_vol == 2)
        spine_membranes = (labels_vol == 3)
        necks = (labels_vol == 4)
        soma = (labels_vol == 5)
        axons = (labels_vol == 6)
        spines = spine_cores + spine_membranes

    del labels_vol
    gc.collect()


    # filter dendrites
    logger.info("   Filtering dendrites...")
    labeled_dendrites = filter_dendrites_dask(dendrites, settings, logger).squeeze()

    if labeled_dendrites.max().compute() == 0:
        logger.info("  *No dendrites were analyzed for this image.")
        return spine_summary


    print(f' shape of labeled_dendrites = {labeled_dendrites.shape}')
    _print_mem("dendrites filtered + persisted")

    spatial_chunks = labeled_dendrites.chunks

    if soma.max().compute() == 0:
        soma_distance = da.zeros_like(labeled_dendrites, dtype="uint8")

    else:
        logger.info("   Calculating soma distance...")

        soma_distance = get_or_compute(
            root,
            "derived/soma_distance",
            lambda: distance_map_to_zarr(
                soma,
                Path(root.store.path) / "derived" / "soma_distance",
                voxel_size=(settings.input_resZ,
                            settings.input_resXY,
                            settings.input_resXY),
                chunks=(32, 128, 128),
                max_dist=None,
                global_scale=4,
            )
        ).squeeze().rechunk(spatial_chunks).persist()

    soma_distance = _rekey(soma_distance, "sd")

    print(f' shape of soma_distance = {soma_distance.shape}')
    _print_mem("soma_distance persisted")

    # Create Distance Map
    logger.info("   Calculating distances from dendrites...")

    with dask_config.set(**{"array.chunk-size": "64 MiB"}):
        dendrite_distance = get_or_compute(
            root,
            "derived/dendrite_distance",
            lambda: distance_map_to_zarr(
                (labeled_dendrites > 0),
                Path(root.store.path) / "derived" / "dendrite_distance",
                voxel_size=(settings.input_resZ,
                            settings.input_resXY,
                            settings.input_resXY),
                chunks=(32, 128, 128),
                max_dist=settings.neuron_spine_dist + 8,
                global_scale=4, logger = logger
            ).squeeze().rechunk(spatial_chunks).persist()
        ).squeeze()

    dendrite_distance = _rekey(dendrite_distance, "dd")
    #save as a tiff
    imwrite(locations.tables + 'Dendrite_distance.tif', dendrite_distance.compute().astype(np.float32), imagej=True, photometric='minisblack',
              metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})


    _print_mem("dendrite_distance persisted")

    print(f' shape of dendrite_distance = {dendrite_distance.shape}')

    logger.info("   Calculating dendrite skeleton...")

    print( f' shape of labeled dendrites = {labeled_dendrites.shape}')
    skeleton = get_or_compute(
        root,
        "derived/skeleton",
        lambda: (labeled_dendrites > 0).map_blocks(
            lambda blk: morphology.skeletonize(blk).astype("uint8"),
            dtype="uint8",
        )
    ).squeeze().rechunk(spatial_chunks).persist()
    skeleton = _rekey(skeleton, "sk")

    print(f' shape of skeleton = {skeleton.shape}')

    _print_mem("skeleton persisted")
    # if settings.save_val_data == True:
    #    save_3D_tif(neuron_distance.astype(np.uint16), locations.validation_dir+"/Neuron_Mask_Distance_3D"+file, settings)

    # Create Neuron MIP for validation - include distance map too
    # neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines, dendrites, skeleton, dendrite_distance], locations.MIPs+"MIP_"+files[file], 'float', settings)
    # neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, neuron_mask, soma_mask, soma_distance, skeleton, neuron_distance, density_image], locations.analyzed_images+"/Neuron/Neuron_MIP_"+file, 'float', settings)
    logger.info(f"    Time taken for initial processing: {time.time() - time_initial:.2f} seconds\n")
    # Spine Detection

    logger.info("   Analyzing spines...")
    model_options = ["Spines, Dendrites, and Soma", "Dendrites and Soma Only", "Necks, Spines, Dendrites, and Soma",
                     "Dendrites, Spine Cores, Spine Membranes, Necks, Soma, and Axons"]
    logger.info(f"    Using model type {model_options[settings.model_type - 1]}")

    if settings.model_type <= 3:
        spine_labels = spine_detection_dask(spines, settings.erode_shape, settings.remove_touching_boarders,
                                       logger)  # binary image, erosion value (0 for no erosion)

        spine_labels = spine_labels.rechunk(spatial_chunks).persist()


    else:
        logger.info("    Analyzing spines using cores and membranes...")
        spine_labels = imgan.spine_detection_cores_and_membranes(spine_cores, spine_membranes,
                                                           settings.remove_touching_boarders, logger,
                                                           settings)  # binary image, erosion value (0 for no erosion)

    # now we have spine_labels and connected_necks for all spines for futher processing and measurements

    # logger.info(f" {np.max(spine_labels)}.")
    # max_label = np.max(spine_labels)

    # Measurements
    # Create 4D Labels
    # imwrite(locations.tables + 'Detected_spines.tif', spine_labels.astype(np.uint16), imagej=True, photometric='minisblack',
    #        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})
    time_spine = time.time()
    _print_mem("spines")
    # spine_table, spines_filtered = spine_measurementsV2(image, spine_labels, 1, 0, settings.neuron_channel, dendrite_distance, soma_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, filename, logger)

    spine_table, spines_filtered = initial_spine_measurements_dask(image, spine_labels, 1, 0, settings.neuron_channel,
                                                              dendrite_distance,
                                                              settings.neuron_spine_size,
                                                              settings.neuron_spine_dist,
                                                              settings, locations, filename, logger)

    spines_filtered = spines_filtered.rechunk(spatial_chunks).persist()

    logger.info(f"     Time taken for initial spine detection: {time.time() - time_spine:.2f} seconds")
    # spine_table, spines_filtered = spine_measurementsV1(image, spine_labels, settings.neuron_channel, dendrite_distance, soma_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, filename, logger)

    # Create 4D Labels
    # imwrite(locations.tables + 'Detected_spines_filtered.tif', spines_filtered.astype(np.uint16), imagej=True, photometric='minisblack',
    #        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})                                                  #soma_mask, soma_distance, )

    logger.info("\n    Calculating spine necks...")
    time_neck_connection = time.time()

    dendrites_mask = (labeled_dendrites > 0).rechunk(spatial_chunks)
    # disable neck analysis for very large datasets
    if settings.neck_analysis == False:
        logger.info(f"     Image shape is {spines_filtered.shape}. Neck analysis has been disabled.")
        connected_necks = da.zeros_like(spines_filtered)
    else:

        if settings.model_type >= 3:

            #neck_labels = measure.label(necks)
            neck_labels = dask_label(necks)[0].rechunk(spatial_chunks).persist()

            # associate spines with necks
            logger.info("     Associating spines with necks...")
            neck_labels_updated = associate_spines_with_necks_gpu_dask(spines_filtered, neck_labels).persist()
            # save this as a tif
            # imwrite(locations.Vols + 'Detected_necks.tif', neck_labels_updated.astype(np.uint16), imagej=True,
            #           photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX'})

            # unassociated neck voxels
            #remaining_necks = (neck_labels) & (neck_labels_updated == 0)
            remaining_necks = (neck_labels > 0) & (neck_labels_updated == 0)

            # imwrite(locations.Vols + 'remaining_necks.tif', remaining_necks.astype(np.uint16), imagej=True,
            #        photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX'})
            # extend spines with necks to dendrite
            # background = label == 0
            # traversable_for_necks = remaining_necks | labels == 0
            # = (remaining_necks + dendrites) == 0
            #target = (dendrites_mask + (remaining_necks > 0)) - neck_labels_updated > 0
            target_mask = ((dendrites_mask > 0) | (remaining_necks > 0)) & \
                          (neck_labels_updated == 0)

            # Extend connected necks to neck on dendrite or directly to dendrite
            # currently can have paths crossing existing labels - need ensure these paths go around existing labels
            #
            logger.info("     Extending necks to dendrites...")
            extended_necks = extend_objects_GPU_dask(neck_labels_updated, target_mask, neuron, settings, locations,
                                                logger).persist()
            extended_necks = np.where(neck_labels_updated == 0, extended_necks, 0)
            # imwrite(locations.Vols + 'extended_necks.tif', extended_necks.astype(np.uint16), imagej=True,
            #        photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX'})

            # associate extended_necks with  remaining_necks
            logger.info("     Associating extended necks with remaining necks...")
            connected_necks = associate_spines_with_necks_gpu_dask((extended_necks + neck_labels_updated),
                                                              remaining_necks).persist()

            connected_necks = connected_necks - spines_filtered
            #connected_necks = np.where(dendrites_mask == 0, connected_necks, 0)
            connected_necks = da.where(dendrites_mask == 0,
                       connected_necks,
                       0)

            # imwrite(locations.Vols + 'connected_necks.tif', connected_necks.astype(np.uint16), imagej=True,
            #        photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX'})

            # extend spines without necks
            # spines_without_necks = spine_labels.copy()
            # spines_without_necks[extended_necks > 0] = 0  # Remove spines that have necks

            # Update traversable mask to include extended B objects
            # traversable_for_necks = remaining_necks | dendrites
            # extended_spines = extend_objects(spines_without_necks, dendrites, traversable_for_necks)
            # extended_spines = extend_objects_GPU(spines_filtered, dendrites, neuron, settings, locations,
            #                   logger)
            # Combine extended A and B objects
            # connected_necks = np.maximum(extended_spines, extended_necks)

        else:
            # traversable_for_necks = dendrites == 0
            logger.info("     Extending spines to dendrites...")
            #connected_necks = extend_objects_GPU_dask(spines_filtered, dendrites_mask, neuron, settings, locations,
            #                                     logger)
            connected_necks = extend_objects_GPU_dask(spines_filtered, dendrites_mask, neuron,
                        settings, locations, logger).persist()
        save(root, "derived/connected_necks", connected_necks)
    # time for neck connection
    logger.info(f"     Time taken for neck generation: {time.time() - time_neck_connection:.2f} seconds")
    _print_mem("necks")
    # logger.info(connected_necks.shape)
    # Remove spines that are connected to necks
    # connected_necks[spine_labels > 0] = 0

    # Second pass analysis
    # take spines(necks and spines filtered) create subvolumes, mask with a 5 pixel dilation and pass through the 2nd pass model

    if settings.second_pass:
        logger.info("     Running spine refinement...")
        if settings.refinement_model_path != None:
            spines_filtered, connected_necks = imgan.second_pass_annotation(spines_filtered, connected_necks,
                                                                      dendrites_mask, neuron,
                                                                      locations, settings, logger)
        else:
            logger.info(
                "Spine refinement model not found. Skipping second pass annotation. Please update location of second pass model in settings file to enable second pass annotation.")

    # log_memory_usage(logger)

    # calulating dendrite statistics ###### CLEAN UP WHOLE NEURON AND KEEP DEND STATS
    # logger.info("\n    Calculating dendrite statistics...")
    # originally whole neuron stats but now calculating dendrite specific

    logger.info("     Calculating neuron statistics...")
    if isinstance(skeleton, da.Array):
        neuron_length = skeleton.astype("uint8").sum().compute()
    else:
        neuron_length = np.sum(skeleton == 1)

    if isinstance(dendrites_mask, da.Array):
        neuron_volume = labeled_dendrites.astype("uint8").sum().compute()
    else:
        neuron_volume = np.sum(dendrites_mask == 1)

    del dendrites_mask

    logger.info("     Calculating dendrite statistics...")
    # get dendrite lengths and volumes as dictoinaries
    dendrite_lengths, dendrite_volumes, skeleton_coords, labeled_skeletons = calculate_dendrite_length_and_volume_fast_dask(
        labeled_dendrites, skeleton, logger)

    # finished calculating dendrite statistics
    logger.info("      Complete.")

    logger.info("     Calculating spine dendrite ID and geodesic distance...")
    if np.max(soma) == 0:
        spine_dendID_and_geodist, geodesic_distance_image = calculate_dend_ID_and_geo_distance_dask(labeled_dendrites,
                                                                                               spines_filtered,
                                                                                               skeleton_coords,
                                                                                               labeled_skeletons,
                                                                                               filename, locations,
                                                                                               soma_vol=None,
                                                                                               settings=settings,
                                                                                                    logger =logger)
    else:
        spine_dendID_and_geodist, geodesic_distance_image = calculate_dend_ID_and_geo_distance_dask(labeled_dendrites,
                                                                                               spines_filtered,
                                                                                               skeleton_coords,
                                                                                               labeled_skeletons,
                                                                                               filename, locations,
                                                                                               soma_vol=soma,
                                                                                               settings=settings, logger=logger)

    # save spine dendrite ID and geodesic distance as csv using pands
    # spine_dendID_and_geodist.to_csv(locations.tables + 'Detected_spines_dendrite_ID_and_geodesic_distance_' + filename + '.csv', index=False)

    logger.info("      Complete.")
    _print_mem("geodesic distance")

    # analyze whole spines
    logger.info("\n     Performing additional mesh measurements on spines in batches on GPU...")

    # combine connected_necks and Spines_filtered
    # print max id for spines fileterd and connected_necks
    if settings.additional_logging:
        logger.info(f"Max ID for spines filtered is {np.max(spines_filtered)}")
        logger.info(f"Max ID for connected necks is {np.max(connected_necks)}")

    if settings.mesh_analysis == True:
        spine_mesh_results = imgan.analyze_spines_batch(((connected_necks * ~(spines_filtered > 0)) + spines_filtered),
                                                  spines_filtered, labeled_dendrites, neuron, locations, settings,
                                                  logger,
                                                  [settings.input_resZ, settings.input_resXY, settings.input_resXY])
    # spine_mesh_results.to_csv(locations.tables + 'Detected_spines_mesh_measurements_' + filename + '.csv', index=False)

    # analyze spine necks
    # logger.info("      Analyzing spine necks...")
    # neck_results = analyze_spine_necks_batch(connected_necks, logger, [settings.input_resZ,  settings.input_resXY,  settings.input_resXY])
    # neck_results.to_csv(locations.tables + 'Detected_necks_mesh_measurements_' + filename + '.csv', index=False)

    if len(spine_table) == 0:
        logger.info(f"  *No spines were analyzed for this image")

    else:
        if settings.save_val_data == True:
            logger.info("    Saving validation MIP image...")
            io.create_mip_and_save_multichannel_tiff(
                [neuron, spines, spines_filtered, connected_necks, labeled_dendrites, skeleton, dendrite_distance,
                 geodesic_distance_image], locations.MIPs + "MIP_" + filename+".tif", 'float', settings)

        if settings.save_intermediate_data == True:
            logger.info("    Saving validation volume image...")
            io.create_and_save_multichannel_tiff(
                [neuron, spines, spines_filtered, connected_necks, labeled_dendrites, skeleton, dendrite_distance,
                 geodesic_distance_image], locations.Vols + filename +".tif", 'float', settings)

        del neuron, spines, labeled_dendrites, skeleton
        gc.collect()
        # logger.info("\n   Creating spine arrays on GPU...")
        # Extract MIPs for each spine

        # spine_MIPs, spine_slices, spine_vols = create_spine_arrays_in_blocks(image, labels_vol, spines_filtered, spine_table, settings.roi_volume_size, settings, locations, filename,  logger, settings.GPU_block_size)

        ##### We now have refined labels for all spines - so we should remeasure intensities and any voxel measurements

        # perform final vox based measurements on spines for morophology and intensity
        logger.info("\n    Calculating final measurements...")
        logger.info("     Calculating additional spine head measurements...")
        spine_head_table, spines_filtered = spine_vox_measurements_dask_chunk(image, spines_filtered, 1, 0,
                                                                   settings.neuron_channel, 'head',
                                                                   dendrite_distance, soma_distance,
                                                                   settings.neuron_spine_size,
                                                                   settings.neuron_spine_dist,
                                                                   settings, locations, filename, logger)
        logger.info("     Calculating additional whole spine measurements...")
        spine_whole_table, spines_filtered = spine_vox_measurements_dask_chunk(image, spines_filtered + connected_necks, 1, 0,
                                                                    settings.neuron_channel, 'spine',
                                                                    dendrite_distance, soma_distance,
                                                                    settings.neuron_spine_size,
                                                                    settings.neuron_spine_dist,
                                                                    settings, locations, filename, logger)

        logger.info("     Calculating additional neck measurements...")
        # now measure in necks (what about if neck label doesn't exist (ensure has value 0)
        neck_table, spines_filtered = spine_vox_measurements_dask_chunk(image, connected_necks, 1, 0,
                                                             settings.neuron_channel, 'neck',
                                                             dendrite_distance, soma_distance,
                                                             settings.neuron_spine_size,
                                                             settings.neuron_spine_dist,
                                                             settings, locations, filename, logger)

        # merge
        del connected_necks, dendrite_distance, soma_distance, spines_filtered
        gc.collect()
        _print_mem("final calcs")

        # multiply geodesic_distance by settings.input_resXY to get in microns
        spine_dendID_and_geodist['geodesic_dist'] = spine_dendID_and_geodist['geodesic_dist'] * settings.input_resXY

        spine_table = tables.merge_spine_measurements(spine_table, spine_dendID_and_geodist, settings, logger)

        spine_table = tables.merge_spine_measurements(spine_table, spine_head_table, settings, logger)
        spine_table = tables.merge_spine_measurements(spine_table, spine_whole_table, settings, logger)
        spine_table = tables.merge_spine_measurements(spine_table, neck_table, settings, logger)

        if settings.mesh_analysis == True:
            # drop 'start_coords' from spine_mesh_results
            spine_mesh_results.drop(['start_coords'], axis=1, inplace=True)

            if settings.additional_logging:
                pd.set_option('display.max_columns', None)
                logger.info(spine_table.columns)
                logger.info(spine_mesh_results.columns)
                logger.info(spine_table.head())
                logger.info(spine_mesh_results.head())

            # save these dfs as csvs
            # spine_table.to_csv(locations.tables + 'Detected_spines_vox_measurements_' + filename + '.csv', index=False)
            # spine_mesh_results.to_csv(locations.tables + 'Detected_spines_mesh_measurements_' + filename + '.csv', index=False)

            spine_table = tables.merge_spine_measurements(spine_table, spine_mesh_results, settings, logger)

        logger.info("     Calculating final complete spine measurements...")
        #spine_table.insert(10, f'spine_vol_calc',
        #                   spine_table['head_vol'] + spine_table['neck_vol'])
        # Intensity now directly measured for whole spine when performing whole spine morp measurements
        # spine_table.insert(9, f'spine_C1_int_density',
        #                   spine_table['head_C1_int_density'] + spine_table['neck_C1_int_density'])
        # spine_table.insert(10, f'spine_C1_max_int', np.maximum(spine_table['head_C1_max_int'], spine_table['neck_C1_max_int']))
        # spine_table.insert(11, f'spine_C1_mean_int', spine_table['spine_C1_int_density']/ spine_table['spine_vol'])

        spine_table['spine_type'] = spine_table.apply(
            lambda row: imgan.categorize_spine(
                row['spine_length'],
                row['head_width_mean'],
                row['neck_width_mean']
            ),
            axis=1
        )

        spine_table = tables.move_column(spine_table, 'neck_vol', 6)

        if settings.mesh_analysis == True:
            # adjust columns for mesh data
            spine_table = tables.move_column(spine_table, 'dendrite_id', 4)

            spine_table = tables.move_column(spine_table, 'spine_area', 4)
            spine_table = tables.move_column(spine_table, 'spine_vol', 5)  # now from mesh
            # spine_table = tables.move_column(spine_table, 'spine_vol_m', 6)
            spine_table = tables.move_column(spine_table, 'spine_surf_area', 7)
            spine_table = tables.move_column(spine_table, 'spine_length', 8)

            spine_table = tables.move_column(spine_table, 'head_area', 9)
            spine_table = tables.move_column(spine_table, 'head_vol', 10)
            # spine_table = tables.move_column(spine_table, 'head_vol_m', 11)
            spine_table = tables.move_column(spine_table, 'head_surf_area', 12)
            spine_table = tables.move_column(spine_table, 'head_length', 13)
            spine_table = tables.move_column(spine_table, 'head_width_mean', 14)
            spine_table = tables.move_column(spine_table, 'neck_area', 15)
            spine_table = tables.move_column(spine_table, 'neck_vol', 16)
            # spine_table = tables.move_column(spine_table, 'neck_vol_m', 17)
            spine_table = tables.move_column(spine_table, 'neck_surf_area', 17)
            spine_table = tables.move_column(spine_table, 'neck_length', 18)
            spine_table = tables.move_column(spine_table, 'neck_width_mean', 19)
            spine_table = tables.move_column(spine_table, 'neck_width_min', 20)
            spine_table = tables.move_column(spine_table, 'neck_width_max', 21)
            spine_table = tables.move_column(spine_table, 'dendrite_id', 4)
            spine_table = tables.move_column(spine_table, 'geodesic_dist', 5)
            spine_table = tables.move_column(spine_table, 'spine_type', 22)

        # Loop for C2 to C5 measurements

        '''
        for i in range(2, 6):
            c_label = f'C{i}'
            head_col = f'head_{c_label}_mean_int'
            neck_col = f'neck_{c_label}_mean_int'

            if head_col in spine_table.columns and neck_col in spine_table.columns:
                #logger.info(f"    Creating columns for {c_label} measurements...")

                # Calculate integrated density
                spine_table.insert(len(spine_table.columns), f'spine_{c_label}_int_density',
                                   spine_table[f'head_{c_label}_int_density'] + spine_table[
                                       f'neck_{c_label}_int_density'])

                # Calculate max intensity
                spine_table.insert(len(spine_table.columns),  f'spine_{c_label}_max_int',
                                   np.maximum(spine_table[f'head_{c_label}_max_int'],  spine_table[f'neck_{c_label}_max_int']))
                #spine_table.insert(len(spine_table.columns), f'spine_{c_label}_max_int',
                 #                  spine_table[[f'head_{c_label}_max_int', f'neck_{c_label}_max_int']].np.maximum(axis=1))

                # Calculate mean intensity
                spine_table.insert(len(spine_table.columns), f'spine_{c_label}_mean_int',
                                   spine_table[f'spine_{c_label}_int_density'] / spine_table['spine_vol'])

                #logger.info(f"    Columns for {c_label} measurements created successfully.")
        '''
        '''
        #use the spine_MIPs to measure spine head area - move this to the mesh section to get areas for neck head and spine
        label_areas = spine_MIPs[:, 1, :, :]
        spine_areas = np.sum(label_areas > 0, axis=(1, 2))
        spine_masks = label_areas > 0
        spine_ids = np.nan_to_num(np.sum(label_areas * spine_masks, axis=(1, 2)) / np.sum(spine_masks, axis=(1, 2), where=spine_masks))

        df_spine_areas = pd.DataFrame({'head_area_v': spine_areas})
        df_spine_areas['label'] = spine_ids

        spine_table = spine_table.merge(df_spine_areas, on='label', how='left')
        spine_table.insert(5, 'head_area', spine_table['head_area_v'] * (settings.input_resXY **2))
        spine_table.drop(['head_area_v'], axis=1, inplace=True)
        '''

        # df_spine_areas['label'] = spine_table['label'].values
        # Reindex df_spine_areas to match the index of spine_table
        # df_spine_areas_reindex = df_spine_areas.reindex(spine_table.index)
        # df_spine_areas_reindex.to_csv(locations.tables + 'Detected_spines_'+filename+'reindex.csv',index=False)
        # spine_table.insert(5, 'spine_area', spine_table.pop('spine_area')) #pops and inserts
        # spine_table.insert(5, 'spine_area', df_spine_areas['spine_area'])

        # spine_table.drop(['dendrite_id'], axis=1, inplace=True)
        # update label column to id
        spine_table.rename(columns={'label': 'spine_id'}, inplace=True)

        # drop some metrics that need furth optimization width measuremnts
        drop_columns = ['head_width_mean', 'neck_width_mean', 'neck_width_min', 'neck_width_max']
        spine_table.drop(columns=drop_columns, inplace=True, errors='ignore')

        spine_table.to_csv(locations.tables + filename + '_detected_spines.csv', index=False)

        tables.create_spine_summary_dendrite(spine_table, filename, dendrite_lengths, dendrite_volumes, settings,
                                      locations)

        # create summary
        summary = tables.create_spine_summary_neuron(spine_table, filename, neuron_length, neuron_volume, settings)

        # Append to the overall summary DataFrame
        spine_summary = pd.concat([spine_summary, summary], ignore_index=True)


    logger.info("     Complete.\n")
    logger.info(f" Processing complete for file {filename}\n---")

    return spine_summary


def _regionprops_single_v1(
    lbl_chunk: np.ndarray,
    int_chunk: np.ndarray,
    labels_wanted: Sequence[int],
    props: Sequence[str],
) -> pd.DataFrame:
    """Compute *props* for labels in *lbl_chunk*.
    Masks convex‑hull failures by filling with NaNs so the parent call
    never explodes."""
    if labels_wanted is not None:
        mask = np.isin(lbl_chunk, labels_wanted)
        if not mask.any():
            return pd.DataFrame(columns=["label", *props])
    try:
        tbl = measure.regionprops_table(lbl_chunk, intensity_image=int_chunk, properties=list(props))
        return pd.DataFrame(tbl)
    except ValueError:
        # convex‑hull or eigenvalue failure → retry with safe props only
        safe_props = [p for p in props if p not in _UNSAFE_PROPS]

        tbl = measure.regionprops_table(
            lbl_chunk,
            intensity_image=int_chunk,
            properties=["label", *safe_props],  # keep the label column
        )
        df = pd.DataFrame(tbl)

        # add any missing columns so every chunk has identical layout
        for p in props:
            if p not in df.columns:
                df[p] = np.nan

        # enforce a consistent column order for Dask’s metadata validator
        df = df[["label", *props]]
        return df

# ------------------------------------------------------------------- #

def _props_blockx(
        labels_block: np.ndarray,
        image_block:  np.ndarray,
        props:        tuple[str, ...],
        depth:        tuple[int, int, int],
        block_info=None
) -> np.ndarray:
    """
    Called by `map_overlap` on ONE (bz, by, bx) NumPy chunk **with halo**.

    •  We “own” an object iff its MIN‑voxel lies inside this chunk’s *core*
       (i.e. not inside the halo).  Objects cut by the chunk boundary are
       therefore measured **exactly once**, no matter how many chunks they
       span.

    •  Returns a *record array* so Dask treats it like a 2‑D table.
    """
    # ------------------------------------------------------------------ set‑up
    dz, dy, dx = depth

    loc = block_info[0]["chunk-location"]  # (bz, by, bx)
    nblocks = block_info[0]["num-chunks"]  # total blocks per axis

    # only trim on a side that actually has a neighbour
    trim_z0 = dz if loc[0] > 0 else 0  # front‑Z
    trim_z1 = dz if loc[0] < nblocks[0] - 1 else 0  # back‑Z
    trim_y0 = dy if loc[1] > 0 else 0
    trim_y1 = dy if loc[1] < nblocks[1] - 1 else 0
    trim_x0 = dx if loc[2] > 0 else 0
    trim_x1 = dx if loc[2] < nblocks[2] - 1 else 0

    core_slices = (
        slice(trim_z0, -trim_z1 or None),
        slice(trim_y0, -trim_y1 or None),
        slice(trim_x0, -trim_x1 or None),
    )
    # which labels start inside the core?
    owned = np.unique(labels_block[core_slices])
    owned = owned[owned != 0]           # drop background
    if owned.size == 0:
        return np.empty((0,), dtype=[("label", "i8")])

    # ---------------------------------------------------------------- measure
    tbl = measure.regionprops_table(
        labels_block,
        intensity_image=image_block,
        properties=["label", *props],
        label_ids=owned,
    )
    # return as record array (row‑oriented) → map_overlap friendly
    return np.rec.fromrecords(pd.DataFrame(tbl).to_records(index=False))

def regionprops_table_dask_v3(
    labels_da   : da.Array,
    intensity_da: da.Array,
    properties  : list[str],
    *,
    rechunk     : tuple[int, int, int] = (64, 256, 256),
    halo        : tuple[int, int, int] = (8, 8, 8),
) -> pd.DataFrame:
    """
    Chunk‑wise `regionprops_table` with overlapping blocks (`halo`)
    so objects crossing a block boundary are measured just once.
    """

    # ---------- checks -----------------------------------------------------
    if labels_da.shape != intensity_da.shape:
        raise ValueError("label / intensity shapes differ")
    if len(halo) != 3:
        raise ValueError("halo must be a 3‑tuple, e.g. (8, 8, 8)")

    props_no_label = [p for p in dict.fromkeys(properties) if p != "label"]

    # ---------- helper run on every block ---------------------------------
    def _props_block(
        lbl_chunk:  np.ndarray,
        int_chunk:  np.ndarray,
        props:      tuple[str, ...],
        depth:      tuple[int, int, int],
        *,
        block_info=None,             # injected by Dask
    ) -> pd.DataFrame:
        """
        Robust per‑chunk wrapper for skimage 0.24.x (no `label_ids` kwarg).

        • Trims the halo so each chunk measures only the labels it **owns**.
        • Patches `axis_major_length` / `axis_minor_length` inside the worker
          to prevent “math domain error”.
        • If a label still fails (degenerate hull), that single label is
          skipped; all others are analysed.
        """


        # ---------- one‑time patch (per worker) -----------------------------
        if not hasattr(RegionProperties, "_safe_len_patch"):
            def _safe_major(self):
                ev  = self.inertia_tensor_eigvals
                rad = 6 * (ev[2] + ev[1] - ev[0])
                return math.sqrt(rad) if rad > 0 else 0.0

            def _safe_minor(self):
                ev  = self.inertia_tensor_eigvals
                rad = 10 * (-ev[0] + ev[1] + ev[2])
                return math.sqrt(rad) if rad > 0 else 0.0

            RegionProperties.axis_major_length = property(_safe_major)
            RegionProperties.axis_minor_length = property(_safe_minor)
            RegionProperties._safe_len_patch   = True

        # ---------- determine which voxels belong to *this* chunk ----------
        dz, dy, dx = depth
        loc     = block_info[0]["chunk-location"]
        nblocks = block_info[0]["num-chunks"]

        trim_z0 = dz if loc[0] > 0            else 0
        trim_z1 = dz if loc[0] < nblocks[0]-1 else 0
        trim_y0 = dy if loc[1] > 0            else 0
        trim_y1 = dy if loc[1] < nblocks[1]-1 else 0
        trim_x0 = dx if loc[2] > 0            else 0
        trim_x1 = dx if loc[2] < nblocks[2]-1 else 0

        core = (
            slice(trim_z0, lbl_chunk.shape[0] - trim_z1 or None),
            slice(trim_y0, lbl_chunk.shape[1] - trim_y1 or None),
            slice(trim_x0, lbl_chunk.shape[2] - trim_x1 or None),
        )

        owned = np.unique(lbl_chunk[core])
        owned = owned[owned != 0]
        if owned.size == 0:
            return pd.DataFrame(columns=["label", *props])

        # ---------- make a mask that keeps only *owned* labels -------------
        mask      = np.isin(lbl_chunk, owned)
        lbl_local = np.where(mask, lbl_chunk, 0).astype(np.int32)

        # ---------- run regionprops_table (no label_ids in 0.24) -----------
        try:
            tbl = measure.regionprops_table(
                lbl_local,
                intensity_image=int_chunk,
                properties=["label", *props],
            )
            return pd.DataFrame(tbl)

        except ValueError:
            # Rare geometry failure — fall back to per‑label loop
            rows = []
            for lab in owned:
                try:
                    sub_tbl = measure.regionprops_table(
                        (lbl_local == lab).astype(np.int32) * lab,
                        intensity_image=int_chunk,
                        properties=["label", *props],
                    )
                    if sub_tbl["label"]:
                        rows.append({k: v[0] for k, v in sub_tbl.items()})
                except Exception:
                    continue          # skip this one bad label
            return pd.DataFrame(rows, columns=["label", *props])

    fn = partial(_props_block, props=tuple(props_no_label), depth=halo)

    # ---------- map across the grid with overlap --------------------------
    rec = da.map_overlap(
        fn,
        labels_da, intensity_da,
        dtype=object,
        depth=halo,
        trim=False,       # trimming handled inside helper
        boundary="none",
    )

    # ---------- gather & deduplicate --------------------------------------
    base_cols = ["label", *props_no_label]

    delayed_blocks = rec.to_delayed().ravel().tolist()

    # (1) no blocks at all
    if not delayed_blocks:
        return pd.DataFrame(columns=base_cols)

    # (2) compute each block → list[DataFrame]
    dfs_list = dask.compute(*delayed_blocks)

    # (3) concatenate – ignore completely empty frames
    dfs_list = [df for df in dfs_list if df is not None and not df.empty]
    if not dfs_list:
        return pd.DataFrame(columns=base_cols)

    dfs = pd.concat(dfs_list, ignore_index=True)

    # (4) drop duplicates & return
    dfs = dfs.drop_duplicates(subset="label", keep="first")
    return dfs.reset_index(drop=True)

def spine_detection_dask(spines, erode, remove_borders, logger):
    """
    Dask‐based spine detection, returns a 3D Dask array of uint32 labels.

    Parameters
    ----------
    spines : da.Array[bool]
        Binary mask of spines.
    erode : (z,y,x)
        Ellipsoid radii for a preliminary erosion (in voxels).
    remove_borders : bool
        If True, clear any labels touching the volume border.
    logger : any
        Logger for messages (unused here but kept for signature).

    Returns
    -------
    da.Array[uint32]
        A label map, same shape as `spines`.
    """
    # 1) Optional erosion + distance transform → seeds
    if erode[0] > 0:
        ellip = imgan.create_ellipsoidal_element(*erode)
        depth = tuple(s // 2 for s in ellip.shape)

        spines_eroded = da.map_overlap(
            ndimage.binary_erosion,
            spines,
            depth=depth,
            boundary='none',
            dtype=bool,
            structure=ellip
        )

        distance = da.map_overlap(
            lambda b: ndimage.distance_transform_edt(b).astype('float32'),
            spines_eroded,
            depth=3,
            dtype='float32'
        )

        thr = distance.max().compute() * 0.10
        seed_mask = distance > thr

        # **Unpack** the tuple here:
        seeds, _ = dask_label(seed_mask)

        labels = da.map_blocks(
            lambda d, s, m: segmentation.watershed(-d, s, mask=m),
            distance, seeds, spines,
            dtype='uint32'
        )

    else:
        # **Unpack** here too:
        labels, _ = dask_label(spines)

    # 2) Optionally clear any labels touching the volume border
    if remove_borders:
        labels = labels.map_blocks(
            lambda blk: segmentation.clear_border(blk).astype('uint32'),
            dtype='uint32'
        )

    return labels

def filter_invalid_objects_dask(
        labels_da, *, min_volume=10, min_dims=3,          # ⬅ default now 3 axes
        flat_ratio=0.40, rechunk=(64, 256, 256), halo=(8, 8, 8)
):
    """
    Pure‑Dask filter that discards
        • tiny objects              (voxel count < `min_volume`)
        • nearly‑planar objects     (occupy < `min_dims` axes)
        • extremely flat objects    (short/long extent < `flat_ratio`)
    Returns (filtered_labels_da, num_valid).
    """
    import dask.array as da, numpy as np
    import dask_image.ndmeasure as ndm

    # -- prepare label volume -----------------------------------------------
    lbl = labels_da.squeeze()
    if lbl.ndim != 3:
        raise ValueError("`labels_da` must be 3‑D after squeeze().")

    # -- unique labels -------------------------------------------------------
    labs = da.unique(lbl).compute()
    labs = labs[labs != 0]                           # drop background
    if labs.size == 0:
        return da.zeros_like(lbl), 0

    # -- voxel counts --------------------------------------------------------
    vols = ndm.sum(da.ones_like(lbl, dtype=np.uint8), lbl, labs).compute()
    keep = labs[vols >= min_volume]
    if keep.size == 0:
        return da.zeros_like(lbl), 0

    # -- bounding boxes for geometry checks ---------------------------------
    pmin = ndm.minimum_position(lbl > 0, lbl, keep).compute()
    pmax = ndm.maximum_position(lbl > 0, lbl, keep).compute()
    dims = (pmax - pmin).astype(np.int64)            # (N, 3) – safe dtype

    good_dims = (dims > 1).sum(axis=1) >= min_dims
    with np.errstate(divide="ignore", invalid="ignore"):
        flatness = dims.min(axis=1) / np.maximum(dims.max(axis=1), 1)
    not_flat = flatness >= flat_ratio

    keep = keep[good_dims & not_flat]
    if keep.size == 0:
        return da.zeros_like(lbl), 0

    # -- lazily build filtered label array ----------------------------------
    keep_set = set(map(int, np.asarray(keep)))
    mask = lbl.map_blocks(np.isin, keep_set, dtype=bool)
    filtered = da.where(mask, lbl, 0)

    return filtered, int(keep.size)


def initial_spine_measurements_dask(image, labels, dendrite, max_label, neuron_ch, dendrite_distance, sizes, dist,
                         settings, locations, filename, logger):
    """ measures intensity of each channel, as well as distance to dendrite
    """


    print(f" {labels.shape}, {image.shape}")
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=1)

    # Measure channel 1:
    logger.info("    Making initial morphology and intensity measurements for channel 1...")
    # logger.info(f" {labels.shape}, {image.shape}")
    props = [
        'label', 'centroid', 'area'
    ]

    main_table = regionprops_table_dask_v3(
        labels,
        image[:, 0, :, :],
        props,
        rechunk=settings.dask_block,
        halo=settings.dask_halo
    )

    main_table.rename(columns={'centroid-0': 'z'}, inplace=True)
    main_table.rename(columns={'centroid-1': 'y'}, inplace=True)
    main_table.rename(columns={'centroid-2': 'x'}, inplace=True)
    # measure distance to dendrite
    logger.info("    Measuring distances to dendrite/s...")
    # logger.info(f" {labels.shape}, {dendrite_distance.shape}")

    props = [
        'label', 'min_intensity', 'max_intensity'
    ]

    distance_table = regionprops_table_dask_v3(
        labels,
        dendrite_distance,
        props,
        rechunk=settings.dask_block,
        halo=settings.dask_halo
    )
    # rename distance column
    distance_table.rename(columns={'min_intensity': 'dist_to_dendrite'}, inplace=True)
    distance_table.rename(columns={'max_intensity': 'spine_length'}, inplace=True)

    distance_col = distance_table["dist_to_dendrite"]
    main_table = main_table.join(distance_col)
    distance_col = distance_table["spine_length"]
    main_table = main_table.join(distance_col)

    # filter out small objects
    volume_min = sizes[0]  # 3
    volume_max = sizes[1]  # 1500?

    # logger.info(f" Filtering spines between size {volume_min} and {volume_max} voxels...")

    # filter based on volume
    # logger.info(f"  filtered table before area = {len(main_table)}")
    spinebefore = len(main_table)

    filtered_table = main_table[(main_table['area'] > volume_min) & (main_table['area'] < volume_max)]

    logger.info(f"     Total putative spines: {spinebefore}")
    logger.info(f"     Spines after volume filtering = {len(filtered_table)} ")
    # logger.info(f"  filtered table after area = {len(filtered_table)}")

    # filter based on distance to dendrite
    spinebefore = len(filtered_table)
    # logger.info(f" Filtering spines less than {dist} voxels from dendrite...")
    # logger.info(f"  filtered table before dist = {len(filtered_table)}. and distance = {dist}")
    filtered_table = filtered_table[(filtered_table['dist_to_dendrite'] < dist)]
    logger.info(f"     Spines after distance filtering = {len(filtered_table)} ")

    if not settings.Track:
        # ---- bump label IDs lazily & safely ------------------------------
        if isinstance(labels, da.Array):
            labels = da.where(labels > 0, labels + max_label, 0)
            labels = _rekey(labels, "lbladd")
        else:
            labels[labels > 0] += max_label

        # ---- keep only objects still present in main_table --------------
        keep = main_table['label'].astype(labels.dtype, copy=False).values
        labels = (
            create_filtered_labels_image_dask(labels, keep)
            if isinstance(labels, da.Array)
            else imgan.create_filtered_labels_image(labels, main_table, logger)
        )

    else:

        # Clean up label image to remove objects from image.
        ids_to_keep = set(filtered_table['label'])  # Extract IDs to keep from your filtered DataFrame
        # Create a mask
        mask_to_keep = np.isin(labels, list(ids_to_keep))
        # Apply the mask: set pixels not in `ids_to_keep` to 0
        labels = np.where(mask_to_keep, labels, 0)

    # update to included dendrite_id
    filtered_table.insert(4, 'dendrite_id', dendrite)

    # create vol um measurement
    filtered_table.insert(6, 'spine_vol',
                          filtered_table['area'] * (settings.input_resXY * settings.input_resXY * settings.input_resZ))
    # drop filtered_table['area']
    filtered_table = filtered_table.drop(['area'], axis=1)
    # filtered_table.rename(columns={'area': 'spine_vol'}, inplace=True)

    # create dist um cols

    filtered_table = tables.move_column(filtered_table, 'spine_length', 7)
    # replace multiply column spine_length by settings.input_resXY
    filtered_table['spine_length'] *= settings.input_resXY
    # filtered_table.insert(8, 'spine_length_um', filtered_table['spine_length'] * (settings.input_resXY))
    filtered_table = tables.move_column(filtered_table, 'dist_to_dendrite', 9)
    filtered_table['dist_to_dendrite'] *= settings.input_resXY
    # filtered_table.insert(10, 'dist_to_dendrite_um', filtered_table['dist_to_dendrite'] * (settings.input_resXY))
    #filtered_table = tables.movecolumn(filtered_table, 'dist_to_soma', 11)
    #filtered_table['dist_to_soma'] *= settings.input_resXY
    # filtered_table.insert(12, 'dist_to_soma_um', filtered_table['dist_to_soma'] * (settings.input_resXY))

    # logger.info(f"  filtered table before image filter = {len(filtered_table)}. ")
    # logger.info(f"  image labels before filter = {np.max(labels)}.")
    # integrated_density
    #filtered_table['C1_int_density'] = filtered_table['spine_vol'] * filtered_table['C1_mean_int']

    # measure remaining channels
    #for ch in range(image.shape[1] - 1):
    #    filtered_table['C' + str(ch + 2) + '_int_density'] = filtered_table['spine_vol'] * filtered_table[
    #        'C' + str(ch + 2) + '_mean_int']

    # Drop unwanted columns
    # filtered_table = filtered_table.drop(['spine_vol','spine_length', 'dist_to_dendrite', 'dist_to_soma'], axis=1)
    #logger.info(
    #    f"     After filtering {len(filtered_table)} spines were analyzed from a total of {len(main_table)} putative spines")
    #create a subset of filtered table using columns label, x, y, z
    filtered_table_subset = filtered_table[['label', 'x', 'y', 'z']]

    return filtered_table_subset, labels

def spine_vox_measurements_dask_chunk(
        image, labels, dendrite, max_label, neuron_ch, prefix,
        dendrite_distance, soma_distance, sizes, dist,
        settings, locations, filename, logger):
    """
    Final morphology / intensity measurements – Dask‑optimised (chunk + halo).
    Returns (table, filtered_labels).
    """

    if image.ndim == 3:
        image = np.expand_dims(image, 1)       # → (z, c, y, x)



    # ---------------- channel 1 ---------------------------------------------
    props = ("label", "area", "mean_intensity", "max_intensity")
    main_table = regionprops_table_dask_v3(
        labels, image[:, 0, :, :], props,
        rechunk=settings.dask_block, halo=settings.dask_halo)

    main_table = (main_table
                  .rename(columns={"mean_intensity": f"{prefix}_C1_mean_int",
                                   "max_intensity":  f"{prefix}_C1_max_int",
                                   "area":           f"{prefix}_vol_vox"}))
    main_table[f"{prefix}_C1_int_density"] = \
        main_table[f"{prefix}_vol_vox"] * main_table[f"{prefix}_C1_mean_int"]

    # ---------------- remaining channels ------------------------------------
    for ch in range(1, image.shape[1]):
        logger.info(f"    Measuring channel {ch+1} …")
        props = ("label", "mean_intensity", "max_intensity")
        tbl = regionprops_table_dask_v3(
            labels, image[:, ch, :, :], props,
            rechunk=settings.dask_block, halo=settings.dask_halo)

        tbl = tbl.rename(columns={
            "mean_intensity": f"{prefix}_C{ch+1}_mean_int",
            "max_intensity":  f"{prefix}_C{ch+1}_max_int",
        })
        main_table = main_table.join(tbl.set_index("label"),
                                     on="label", how="left")
        main_table[f"{prefix}_C{ch+1}_int_density"] = \
            main_table[f"{prefix}_vol_vox"] * \
            main_table[f"{prefix}_C{ch+1}_mean_int"]

    # ---------------------------------------------------------------- head / spine extra metrics
    def _extra_metrics(src_img, rename_map):
        """
        Compute additional (often hull‑based) region‑props safely.

        • 1st pass → try full label set.
        • If any NaN/Inf/zero‑only hull metrics appear, fall back to a geometry
          pre‑filter and recompute only for the surviving labels.
        • Returns a DataFrame indexed by 'label'.
        """

        def _compute(lbl):
            """Core call – returns renamed table."""
            tbl = regionprops_table_dask_v3(
                lbl, src_img, tuple(rename_map.keys()),
                rechunk=settings.dask_block, halo=settings.dask_halo
            ).rename(columns=rename_map)
            return tbl

        # ---------- 1st try: everything ----------------------------------------
        tbl = _compute(labels)

        # If *everything* came back NaN/0 or the DF is empty, retry after
        # a light geometry pre‑filter.  Otherwise leave rows untouched.
        if tbl.empty or not tbl.select_dtypes("number").to_numpy().any():
            labels_filt, _ = filter_invalid_objects_dask(
                labels,
                min_volume=settings.neuron_spine_size[0],
                min_dims=3,  # span all 3 axes
                flat_ratio=0.25,  # trim extremely flat objects
                rechunk=settings.dask_block,
                halo=settings.dask_halo,
            )
            tbl = _compute(labels_filt)

        return tbl.set_index("label")

    if prefix == "head":
        rename = {
            "min_intensity":   "head_euclidean_dist_to_dendrite",
            "area_bbox":       "head_bbox",
            "extent":          "head_extent",
            "solidity":        "head_solidity",
            "area_convex":     "head_vol_convex",
            "axis_major_length": "head_major_length",
            "axis_minor_length": "head_minor_length",
        }
        extra = _extra_metrics(dendrite_distance, rename)
        main_table = main_table.join(extra, on="label", how="left")

        # convert to µm
        for col in (  "head_major_length", "head_minor_length"):
            if col in main_table:
                main_table[col] *= settings.input_resXY
        for col in ("head_bbox", "head_vol_convex"):
            if col in main_table:
                main_table[col] *= (settings.input_resXY *
                                    settings.input_resXY *
                                    settings.input_resZ)

        if soma_distance.max() > 0:
            soma_tbl = regionprops_table_dask_v3(
                labels, soma_distance, ("label", "min_intensity"),
                rechunk=settings.dask_block, halo=settings.dask_halo
            ).rename(columns={"min_intensity": "head_euclidean_dist_to_soma"}
            ).set_index("label")
            #soma_tbl["head_euclidean_dist_to_soma"] *= settings.input_resXY
            main_table = main_table.join(soma_tbl, how="left")
        else:
            main_table["head_euclidean_dist_to_soma"] = pd.NA

    elif prefix == "spine":
        rename = {
            "area_bbox":        "spine_bbox",
            "extent":           "spine_extent",
            "solidity":         "spine_solidity",
            "axis_major_length":"spine_major_length",
            "axis_minor_length":"spine_minor_length",
        }
        extra = _extra_metrics(dendrite_distance, rename)
        main_table = main_table.join(extra, on="label", how="left")

        # convert to µm
        for col in ("spine_major_length", "spine_minor_length"):
            if col in main_table:
                main_table[col] *= settings.input_resXY
        if "spine_bbox" in main_table:
            main_table["spine_bbox"] *= (settings.input_resXY *
                                         settings.input_resXY *
                                         settings.input_resZ)

    # ---------------------------------------------------------------- label housekeeping
    if not settings.Track:
        if isinstance(labels, da.Array):
            labels = da.where(labels > 0, labels + max_label, 0)
            labels = _rekey(labels, "lbladd")
        else:
            labels[labels > 0] += max_label

        keep = main_table["label"].astype(labels.dtype, copy=False).values
        labels = (create_filtered_labels_image_dask(labels, keep)
                  if isinstance(labels, da.Array)
                  else imgan.create_filtered_labels_image(labels, main_table, logger))

    # ---------------------------------------------------------------- cleanup & return
    main_table = main_table.drop(columns=[f"{prefix}_vol_vox"], errors="ignore")
    return main_table.reset_index(drop=True), labels

def associate_spines_with_necks_gpu_dask(spines, necks):
    """
    Dask‑aware wrapper: maps the GPU kernel to every chunk, then returns
    a *Dask Array*.
    """
    if not isinstance(spines, da.Array):
        # fall back to old behaviour on whole volume
        return _associate_block(spines, necks)

    result = da.map_blocks(
        _associate_block,
        spines, necks,
        dtype=spines.dtype,
        meta=np.array((), dtype=spines.dtype)
    )
    return result




def pathfinding_dask(object_subvolume_gpu, target_subvolume_gpu, intensity_image_gpu):
    """
    Pathfinding that combines distance map and intensity image to prioritize brighter voxels,
    with optimized start and end point selection.
    """
    # Compute the distance map from the target subvolume
    distance_map_gpu = cp_ndimage.distance_transform_edt(1 - target_subvolume_gpu)

    # Normalize the intensity image to match the scale of the distance map
    intensity_image_gpu = cp.asarray(intensity_image_gpu)
    # Create a mask for the object (1 where there's no object, 0 where there is)
    object_mask_gpu = 1 - object_subvolume_gpu

    # Apply the object mask to the intensity image
    intensity_image_gpu = intensity_image_gpu * object_mask_gpu

    # Check if intensity image has any non-zero values
    if cp.max(intensity_image_gpu) > 0:
        normalized_intensity_gpu = intensity_image_gpu / cp.max(intensity_image_gpu)
    else:
        normalized_intensity_gpu = intensity_image_gpu  # If all zeros, keep as is
        print("        Intensity image is all zeros, skipping normalization")

    # Apply gamma correction to increase the influence of brighter voxels
    gamma = 0.4  # Adjust this value to fine-tune the brightness influence
    gamma_corrected_intensity = cp.power(normalized_intensity_gpu, gamma)

    # Create blurred intensity image
    blurred_intensity_gpu = cp_ndimage.gaussian_filter(gamma_corrected_intensity, sigma=[0.25, 10, 10])

    # Normalize the distance map to [0, 1]
    max_distance = cp.max(distance_map_gpu)
    if max_distance > 0:
        normalized_distance_map_gpu = distance_map_gpu / (max_distance + 1e-6)
    else:
        normalized_distance_map_gpu = distance_map_gpu
        print("        Distance map is all zeros, skipping normalization")

    # Invert the intensity so that brighter voxels have lower values (to minimize)
    intensity_weight = 0.2  # Adjust this weight to balance distance and intensity
    blurred_intensity_weight = 0.3
    epsilon = 1e-6
    augmented_map_gpu = normalized_distance_map_gpu / (
            intensity_weight * gamma_corrected_intensity +
            blurred_intensity_weight * blurred_intensity_gpu +
            epsilon
    )

    # Apply Gaussian smoothing to reduce noise and create a smoother path
    sigma = [0.5, 0.5, 0.5]  # Adjust these values for each dimension
    augmented_map_gpu = cp_ndimage.gaussian_filter(augmented_map_gpu, sigma)

    # Convert augmented map, object, and target to NumPy for processing
    augmented_map = augmented_map_gpu.get()
    object_volume = object_subvolume_gpu.get()
    target_volume = target_subvolume_gpu.get()

    # Find potential start points (near the object)
    dilated_object = ndimage.binary_dilation(object_volume, iterations=1)
    start_candidates = np.argwhere(dilated_object & ~object_volume)

    # Check if we have valid start points
    if len(start_candidates) == 0:
        print("        No valid start candidates found in pathfinding_v3b")
        return None, augmented_map_gpu.get()

    # Find potential end points (on the target)
    modified_target = np.logical_xor((ndimage.binary_dilation(target_volume, iterations=1)), target_volume)
    end_candidates = np.argwhere(modified_target)

    # Check if we have valid end points
    if len(end_candidates) == 0:
        print("        No valid end candidates found in pathfinding_v3b")
        return None, augmented_map_gpu.get()

    # Find the best start and end points
    best_start = min(start_candidates, key=lambda p: augmented_map[tuple(p)])
    best_end = min(end_candidates, key=lambda p: augmented_map[tuple(p)])

    # Initialize the path
    path = []
    current_point = tuple(best_start)
    path.append(current_point)

    max_iterations = 200  # Increased to allow for longer paths
    reached_target = False
    for _ in range(max_iterations):
        z, y, x = current_point

        # Get possible moves
        neighbors = [
            (int(z + dz), int(y + dy), int(x + dx))
            for dz in [-1, 0, 1] for dy in [-1, 0, 1] for dx in [-1, 0, 1]
            if (0 <= z + dz < augmented_map.shape[0] and
                0 <= y + dy < augmented_map.shape[1] and
                0 <= x + dx < augmented_map.shape[2] and
                not target_volume[z + dz, y + dy, x + dx])
        ]

        # Check if we have valid neighbors
        if not neighbors:
            # No valid moves available, terminate the path
            break

        # Sort neighbors by the augmented map value
        neighbors.sort(key=lambda p: augmented_map[p])

        # Find the best move
        best_move = neighbors[0]

        # Try to create a z_only_move if possible
        z_only_move = (int(z + np.sign(best_move[0] - z)), y, x)

        # Check if z_only_move is valid
        z_move_valid = (0 <= z_only_move[0] < augmented_map.shape[0] and
                        not target_volume[z_only_move])

        # If moving only in Z is better or the same and valid, prefer that
        if z_move_valid and augmented_map[z_only_move] <= augmented_map[best_move]:
            next_point = z_only_move
        else:
            next_point = best_move

        # Stop if the next point is at the modified target
        if modified_target[next_point]:
            path.append(next_point)
            reached_target = True
            break

        # Update current point and path
        current_point = next_point
        path.append(current_point)

    if reached_target:
        return path, augmented_map_gpu.get()
    else:
        print("      Path finding failed: did not reach the target.")
        return None, augmented_map_gpu.get()


def extend_single_object_GPU_v2_dask(label_value, object_subvolume, target_subvolume, intensity_subvolume, settings):
    # Convert numpy arrays to CuPy arrays
    # logger.info(f"Label {label_value} - Subvolume shapes: subvolume {object_subvolume.shape}, target {target_subvolume.shape}, traversable {traversable_subvolume.shape}")

    pad_width = 1

    object_subvolume_gpu = imgan.pad_subvolume_gpu(cp.asarray(object_subvolume), pad_width)
    target_subvolume_gpu = imgan.pad_subvolume_gpu(cp.asarray(target_subvolume), pad_width)
    # traversable_subvolume_gpu = cp.asarray(traversable_subvolume)
    intensity_subvolume_gpu = imgan.pad_subvolume_gpu(cp.asarray(intensity_subvolume), pad_width)
    # Use the enhanced simple pathfinding method - below method works well but seeing if we can use intensity as well
    # path, distance_map = strict_z_first_pathfinding(object_subvolume_gpu, target_subvolume_gpu)

    path, distance_map = pathfinding_dask(object_subvolume_gpu, target_subvolume_gpu, intensity_subvolume_gpu
                                               )

    #imwrite distance map
    #ast ype float
    #distance_map = distance_map.astype(np.float32)
    #imwrite(f"D:/Project_Data/RESPAN/Testing/_2024_08_Test_with_Spines/1/Validation_Data/distance_map{label_value}.tif", distance_map.astype(np.float32), imagej=True, photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX'})
    # if not path:
    #    logger.info(f"No path found for label {label_value}.")
    # else:
    #    logger.info(f"Path length: {len(path)}")
    '''
    #logger.info(f"Start point: {start_point}, End point: {end_point}")
    #logger.info(f"Cost array slice: {cost_array[start_point[0], start_point[1], start_point[2]]}")
    #logger.info(f"Path found: {path}")
    # Create extended subvolume

    path_volume = cp.copy(object_subvolume_gpu)
    if path:
        path_array = np.array(path)
        logger.info(f"Updating path_volume at indices: {path_array.T}")
        path_volume[tuple(path_array.T)] = label_value
    #print label value
    logger.info(f"Label value: {label_value}")

    path_volume = cp.logical_xor(object_subvolume_gpu, path_volume)
    path_volume = path_volume * label_value
    '''
    # Create the path volume
    path_volume_gpu = cp.zeros_like(object_subvolume_gpu)
    if path is not None:
        path_array = np.array(path, dtype=int)  # Ensure path array is of integer type
        # logger.info(f"Updating path_volume at indices: {path_array.T}")

        # Limit the extension to 10 voxels - added to prevent long paths
        max_distance = 20
        path_array = path_array[distance_map[tuple(path_array.T)] <= max_distance]
        path_volume_gpu[tuple(path_array.T)] = 1
    else:
        return label_value, cp.zeros_like(object_subvolume), cp.zeros_like(object_subvolume)

    # Subtract the path from the object subvolume using logical XOR
    final_path_volume_gpu = path_volume_gpu * (1 - object_subvolume_gpu)

    # Multiply the final path volume by the label value
    final_path_volume_gpu = final_path_volume_gpu * label_value

    # Convert to NumPy after the operation
    path_volume = imgan.unpad_subvolume_gpu(cp.asnumpy(final_path_volume_gpu), pad_width)
    distance_map = imgan.unpad_subvolume_gpu(cp.asnumpy(distance_map), pad_width)


    return label_value, path_volume, distance_map


def extend_objects_GPU_single_dask(objects, target_objects, intensity, settings, locations):
    print("     Finding spine necks using GPU...")

    if not (objects.shape == target_objects.shape):
        raise ValueError("Input array shapes do not match")

    cp_objects = cp.asarray(objects)
    cp_necks = cp.zeros_like(objects)
    unique_labels = cp.unique(cp_objects)
    unique_labels = unique_labels[unique_labels != 0]

    full_shape = objects.shape
    for label_value in unique_labels:
        label_value = int(label_value.item())
        object_mask = cp_objects == label_value

        bbox = imgan.get_bounding_box_cupy(object_mask, 6, full_shape, settings)
        #bbox = get_bounding_box_with_target_cupy(object_mask, target_objects, full_shape, settings)
        bbox = tuple(map(int, bbox))

        object_sub_volume = cp.asnumpy(object_mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]])
        target_subvolume = cp.asnumpy(cp.asarray(target_objects)[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]])
        intensity_subvolume = cp.asnumpy(cp.asarray(intensity)[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]])

        try:
            path_volume = 0
            result_label, path_volume, distance_vol = extend_single_object_GPU_v2_dask(
                label_value, object_sub_volume, target_subvolume, intensity_subvolume, settings)



            #print the max val in path volume
            #logger.info(f"Max value in path volume: {path_volume.max()}")

            cp_necks[bbox[0]:bbox[1],
            bbox[2]:bbox[3],
            bbox[4]:bbox[5]] = cp.maximum(
                cp_necks[bbox[0]:bbox[1],
                bbox[2]:bbox[3],
                bbox[4]:bbox[5]],
                cp.asarray(path_volume)
            )

            #logger.info(f"Extended object with label {result_label}. Bounding box: {bbox}")


            #logger.info(f"Subvolume shape: {object_sub_volume.shape}")
            #logger.info(f"Extended subvolume shape: {path_volume.shape}")
            #logger.info(f"Target subvolume shape: {target_subvolume.shape}")
            #logger.info(f"Traversable subvolume shape: {traversable_subvolume.shape}")
            save_neck_val_tifs = False
            if save_neck_val_tifs:
                tiff_filename = os.path.join(locations.Vols, f"subvol_{result_label}.tif")
                multichannel_subvolume = np.stack([
                    object_sub_volume,
                    distance_vol,
                    path_volume,
                    intensity_subvolume,
                    target_subvolume
                ], axis=1)
                imwrite(tiff_filename, multichannel_subvolume.astype(np.uint16), compression=('zlib', 1), imagej=True,
                        photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX'})


        except Exception as e:
            print(f"Error processing object with label {label_value}: {str(e)}")
            # Clean up memory for each loop iteration
            del object_mask, object_sub_volume, target_subvolume, intensity_subvolume, path_volume, distance_vol
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()

            # Clean up memory for variables that are no longer needed
    del cp_objects, unique_labels
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    return cp.asnumpy(cp_necks)

def _extend_block(obj_blk, tgt_blk, int_blk, settings=None, locations=None):
    """Runs existing extend_objects_GPU on a single NumPy chunk."""
    return extend_objects_GPU_single_dask(obj_blk, tgt_blk, int_blk, settings, locations)

def extend_objects_GPU_dask(objects_da, target_da, intensity_da,
                            settings, locations, logger):
    """
    • If inputs are NumPy → fall back to original GPU function.
    • If Dask → apply chunk‑wise on the worker, keeping memory low.
    """
    if not isinstance(objects_da, da.Array):
        return imgan.extend_objects_GPU(objects_da, target_da, intensity_da,
                                  settings, locations, logger)

    # keep same chunks across all arrays
    tgt_da  = target_da.rechunk(objects_da.chunks)
    int_da  = intensity_da.rechunk(objects_da.chunks)

    out = da.map_blocks(
        _extend_block,
        objects_da, tgt_da, int_da,
        dtype=objects_da.dtype,
        settings=settings, locations=locations,
        meta=np.array((), dtype=objects_da.dtype)
    )
    return out


def _associate_block(sp_blk, nk_blk):
    """Runs entirely on GPU for one chunk."""


    lbl = cp.asarray(sp_blk, dtype=cp.uint32)
    nks = cp.asarray(nk_blk > 0)

    struct = cp.ones((3, 3, 3), dtype=bool)
    while True:
        dil = binary_dilation(lbl > 0, structure=struct)
        growth = dil & nks & (lbl == 0)
        if not growth.any():
            break
        for lab in cp.unique(lbl):
            if lab == 0:
                continue
            mask = (lbl == lab)
            lbl[growth & binary_dilation(mask, structure=struct)] = lab
    return lbl.get()                            # back to NumPy


def calculate_dendrite_length_and_volume_fast_dask(labeled_dendrites,
                                                   skeleton, logger):
    """
    Returns:
      dendrite_lengths  – {label: voxels along skeleton}
      dendrite_volumes  – {label: voxels in dendrite}
      skeleton_coords   – N×3 NumPy array of (z,y,x)
      skeleton_labels   – length‑N NumPy array of dendrite ID at each coord
    """
    t0 = time.time()

    # --- ensure both arrays share chunks ------------------------------------
    if isinstance(skeleton, da.Array):
        skeleton = skeleton.rechunk(labeled_dendrites.chunks)
    else:
        skeleton = da.from_array(skeleton, chunks=labeled_dendrites.chunks)

    # voxels where skeleton crosses dendrite labels
    labeled_skel = labeled_dendrites * skeleton      # still lazy

    unique = da.unique(labeled_skel).compute()
    unique = unique[unique != 0]
    if unique.size == 0:
        return {}, {}, np.empty((0, 3), int), np.empty((0,), int)

    # ---- lengths (# skeleton voxels) per dendrite ---------------------------
    ones = da.ones_like(labeled_skel, dtype=np.uint32)
    lengths = ndm.sum_labels(ones, labeled_skel, index=unique).compute()

    # ---- volumes (# voxels) per dendrite ------------------------------------
    ones_all = da.ones_like(labeled_dendrites, dtype=np.uint32)
    volumes = ndm.sum_labels(ones_all, labeled_dendrites,
                             index=unique).compute()

    dend_lengths = dict(zip(unique.astype(int), lengths))
    dend_volumes = dict(zip(unique.astype(int), volumes))

    # ---- skeleton point list + labels  (small) -----------------------------
    sk_bool = skeleton.astype(bool).compute()
    skeleton_coords = np.column_stack(np.nonzero(sk_bool))
    skeleton_labels = labeled_skel[sk_bool].compute()   # only voxels on skeleton

    logger.info(f"     Dendrite stats done in {time.time()-t0:.2f}s "
                f"({len(unique)} dendrites).")

    return dend_lengths, dend_volumes, skeleton_coords, skeleton_labels


def calculate_dend_ID_and_geo_distance_dask(labeled_dendrites,
                                            labeled_spines,
                                            skeleton_coords,
                                            skeleton_labels,
                                            filename, locations,
                                            soma_vol=None,
                                            settings=None, logger=None):
    # ---------- ensure NumPy vols (uint16/8 – small enough to fit RAM) -------
    if isinstance(labeled_dendrites, da.Array):
        labeled_dendrites = labeled_dendrites.astype('uint16').compute()
    if isinstance(labeled_spines, da.Array):
        labeled_spines = labeled_spines.astype('uint16').compute()
    if isinstance(soma_vol, da.Array):
        soma_vol = soma_vol.astype('uint8').compute()

    # fall‑back if skeleton_labels missing
    if skeleton_labels is None or skeleton_labels.size == 0:
        skeleton_labels = labeled_dendrites[
            skeleton_coords[:, 0],
            skeleton_coords[:, 1],
            skeleton_coords[:, 2]
        ]

    # ----------------------- build KD‑tree on down‑sampled skeleton ----------
    t0 = time.time()
    kd_tree, skeleton_labels = imgan.create_kdtree_from_skeleton(
        skeleton_coords, skeleton_labels,
        sampling_method="systematic", sampling_param=4)
    logger.info(f"      KD‑tree built in {time.time()-t0:.2f}s")

    # -------- map every spine head to nearest skeleton voxel -----------------
    rel_sp, sp_ids, sp_dict, idxs, dists, dendIDs = \
        imgan.match_and_relabel_objects_geo(kd_tree, skeleton_labels, labeled_spines)

    # ------------------- group data by dendrite --------------------------------
    mapping = defaultdict(list)
    for i, sp_id in enumerate(sp_ids):
        dend_lab = sp_dict.get(sp_id, 0)
        if dend_lab:
            mapping[dend_lab].append({
                "label_B": sp_id,
                "coord_A": kd_tree.data[idxs[i]],
            })

    # ------------------- compute geodesic distances per dendrite --------------
    geo_img = np.full(labeled_dendrites.shape, np.nan, np.float32)
    geo_dict = {}

    for dend_lab, items in mapping.items():
        mask = (labeled_dendrites == dend_lab)
        if not mask.any():
            continue

        # pick start voxel
        if soma_vol is not None and soma_vol.any():
            cKD = cKDTree(np.argwhere(soma_vol))
            obj = np.argwhere(mask)
            start = tuple(obj[cKD.query(obj)[1].argmin()])
        else:
            start = tuple(np.argwhere(mask)[0])

        dist_map = compute_geodesic_distance_map_dask(mask, [start])
        geo_img[mask] = dist_map[mask]

        for itm in items:
            coord = tuple(np.round(itm["coord_A"]).astype(int))
            geo_dict[itm["label_B"]] = dist_map[coord]

    # write distances back to dataframe
    for lbl, gdist in geo_dict.items():
        dendIDs.loc[dendIDs['label'] == lbl, 'geodesic_dist'] = gdist

    geo_img = np.nan_to_num(geo_img).astype(np.float32)
    return dendIDs, geo_img


def compute_geodesic_distance_map_dask(object_mask, starting_points):
    # Initialize distance map with infinity
    distance_map = np.full(object_mask.shape, np.inf)

    # Set distance at starting points to zero
    for pt in starting_points:
        distance_map[pt] = 0

    # Use a queue for BFS
    from collections import deque
    queue = deque(starting_points)

    # Define neighborhood (6-connected for 3D)
    struct = generate_binary_structure(3, 1)

    while queue:
        current = queue.popleft()
        current_distance = distance_map[current]

        # Iterate over neighbors
        for offset in zip(*np.where(struct)):
            neighbor = tuple(np.array(current) + np.array(offset) - 1)
            if (0 <= neighbor[0] < object_mask.shape[0] and
                    0 <= neighbor[1] < object_mask.shape[1] and
                    0 <= neighbor[2] < object_mask.shape[2]):
                if object_mask[neighbor]:
                    if distance_map[neighbor] > current_distance + 1:
                        distance_map[neighbor] = current_distance + 1
                        queue.append(neighbor)
    return distance_map




def create_filtered_labels_image_dask(labels_da: da.Array,
                                      keep_ids: np.ndarray):
    """Zero‑out every voxel whose label is *not* in keep_ids."""
    mask  = da.isin(labels_da, keep_ids, assume_unique=True)
    mask  = _rekey(mask, "isin")
    out   = da.where(mask, labels_da, 0)
    return _rekey(out, "filterlbl").astype(labels_da.dtype, copy=False)



def regionprops_table_dask(
    labels_da      : da.Array,
    intensity_da   : da.Array,
    props          : list[str],
    *,
    rechunk        : tuple[int,int,int] = (64,64,64),
):
    """
    Lightweight dask‑aware replacement for skimage.measure.regionprops_table
    that only materialises *small* sub‑volumes per object when needed.

    Parameters
    ----------
    labels_da      : dask.array, int
        3‑D label image with **unique** positive IDs.
    intensity_da   : dask.array, float/int or None
        Matching intensity image.  Pass None if not required.
    props          : list[str]
        Same property names accepted by skimage.  Only those are computed.
    rechunk        : tuple[int,int,int], optional
        How to rechunk `labels_da`/`intensity_da` for faster I/O.
    """
    # ---------- preparation --------------------------------------------------
    labels_da    = labels_da.squeeze().rechunk(rechunk)
    if intensity_da is None:
        intensity_da = da.zeros_like(labels_da, dtype=float)
    else:
        intensity_da = intensity_da.squeeze().rechunk(rechunk)

    props = list(dict.fromkeys(props))

    if labels_da.ndim != 3:
        raise ValueError("labels_da must be 3‑D")

    labels = da.unique(labels_da).compute()
    labels = labels[labels != 0]                 # drop background
    if labels.size == 0:
        return pd.DataFrame(columns=["label"] + props)
    # ---------- cheap statistics done fully in dask -------------------------
    results = {"label": labels}

    cheap_props = [p for p in props if p in _FAST_PROP_FUNCS]
    expensive   = [p for p in props if p not in cheap_props]

    for p in cheap_props:
        fn = _FAST_PROP_FUNCS[p]
        res = fn(intensity_da if "intensity" in p else labels_da,
                 labels_da, labels) if labels.size else np.empty(0)
        results[p] = res.compute() if hasattr(res, "compute") else res

    # ---------- expensive metrics – one delayed task per object -------------
    delayed_frames = []
    if expensive:
        # bounding boxes via ndmeasure (two passes but still lazy) ----------
        pmin = ndm.minimum_position(labels_da>0, labels_da, labels).compute()
        pmax = ndm.maximum_position(labels_da>0, labels_da, labels).compute()

        for lab, lo, hi in zip(labels, pmin, pmax):
            z0,y0,x0 = lo
            z1,y1,x1 = hi
            z1+=1; y1+=1; x1+=1           # slice end is exclusive
            sub_lbl = labels_da[z0:z1, y0:y1, x0:x1]
            sub_int = intensity_da[z0:z1, y0:y1, x0:x1]
            d = dask.delayed(_regionprops_single)(
                    sub_lbl, sub_int, int(lab), expensive
                )
            delayed_frames.append(d)

        heavy_df = dd.from_delayed(delayed_frames).compute() \
                   if delayed_frames else pd.DataFrame()

    # ---------- merge and return --------------------------------------------
    easy_df = pd.DataFrame(results)
    if expensive:
        out = easy_df.merge(heavy_df, on="label", how="left")
    else:
        out = easy_df
    return out.reset_index(drop=True)



def _regionprops_single_v2(lbl_sub, int_sub, props):
    """
    Run skimage.regionprops_table on one object.
    Gracefully degrades when convex‑hull based metrics (e.g. feret_diameter_max)
    raise ValueError on flat / tiny objects.
    """
    try:
        tbl = measure.regionprops_table(
            lbl_sub, intensity_image=int_sub,
            properties=["label"] + props
        )
    except ValueError:                            # planar or zero‑volume hull
        safe = [p for p in props if p not in (
            "area_convex", "feret_diameter_max",
            "axis_major_length", "axis_minor_length")]
        tbl  = measure.regionprops_table(
            lbl_sub, intensity_image=int_sub,
            properties=["label"] + safe
        )
        for miss in set(props) - set(safe):
            tbl[miss] = [np.nan] * len(tbl["label"])

    return pd.DataFrame(tbl)


def distance_map_to_zarr_prev(mask: da.Array,
                         zarr_path: Path,
                         voxel_size: Sequence[float],
                         *,
                         chunks=None,
                         max_dist=None,
                         global_scale=4,
                         compressor=COMP, logger = None) -> da.Array:
    zroot = Path(zarr_path)
    mask = mask.rechunk(chunks or CHUNK_SETTINGS['distance'])

    dist_da = distance_transform_edt_dask(
        mask,
        sampling=voxel_size,
        max_dist=max_dist,
        global_scale=global_scale
    ).astype("float32")

    #future = save_volume_to_omezarr(
    #    dist_da,
    #    zroot,
     #   group="0",
      #  compressor=compressor
    #)
    #wait(future)
    save_volume_to_omezarr(dist_da, zroot, group="0", compressor=compressor, logger=logger)

    return da.from_zarr(str(zroot), component="0")



def tiff_to_ome_zarr_dask(tiff_path: str,
                     zarr_root: str,
                     chunks=(1, 64, 512, 512),
                     pixel_sizes=None,
                     client=None,               # kept for signature-compatibility
                     settings=None,
                     logger=None):
    """
    Convert TIFF/OME-TIFF → NGFF level-0 OME-Zarr.

    * All TIFF reads occur inside workers – cluster-friendly.
    * The resulting write task is scheduled on the existing Dask client
      when available, otherwise with a local threaded scheduler.
    """
    # ---------- normalise chunk spec ---------------------------------------
    if isinstance(chunks, tuple):
        flat = tuple(c[0] if isinstance(c, tuple) else c for c in chunks)
    else:
        flat = tuple(chunks)

    # ---------- build Dask array without shared locks ----------------------
    arr = tiff_to_dask(tiff_path, flat)           # (C Z Y X)
    if arr.dtype.kind == "O":
        arr = arr.astype("uint16")

    if arr.ndim == 4:
        axes = "czyx"
    elif arr.ndim == 3:
        axes = "zyx"
    else:
        raise ValueError(f"unsupported ndim {arr.ndim} – expected 3 or 4")                            # NGFF still fine

    SMALL_VOL = 200 * 1024 ** 2
    if arr.nbytes <= SMALL_VOL:
        if logger:
                      logger.info("     Small volume – writing synchronously")

        write_image(
                arr, zarr_root,
                axes = axes, chunks = arr.shape,  # ONE chunk → one file
                pixel_sizes = pixel_sizes,
                compressor = None,  # change to FAST_BLOSC if wanted
                compute = True  # do it *now*, no Dask graph
                                  )
        return


    # ---------- temporary target -------------------------------------------
    tmp_root = f"{zarr_root}.tmp-{uuid.uuid4().hex}"
    store    = zarr.DirectoryStore(tmp_root)
    root     = zarr.group(store=store)

    # ---------- progress bar + threads -------------------------------------
    n_threads = getattr(settings, "zarr_threads", None)
    if not isinstance(n_threads, int) or n_threads <= 0:
        n_threads = min(os.cpu_count() or 1, 32)
    os.environ.setdefault("BLOSC_NUM_THREADS", str(n_threads))

    pool = ThreadPool(n_threads)
    sched_cfg = {"scheduler": "threads", "pool": pool}

    show_pb = getattr(settings, "zarr_show_progress", True)
    pb_ctx = (
        ProgressBar(out=_LoggerWriter(logger))
        if show_pb and logger is not None else
        (ProgressBar() if show_pb else nullcontext())
    )
    if logger:
        logger.info(f"     [OME-Zarr] writing with {n_threads} thread(s)…")

    # ---------- create delayed store task ----------------------------------
    with dask_config.set(**sched_cfg), pb_ctx:
        store_task = write_image(
            arr,
            root,
            axes=axes,
            chunks=flat,
            pixel_sizes=pixel_sizes,
            compute=False)           # always delayed

    # ---------- run on cluster when available ------------------------------
    with pb_ctx:
        try:
            client = get_client()  # distributed present
            fut = client.compute(store_task, retries=0)
            wait(fut)    # <-- block until finished
        except ValueError:  # no client → local threads
            dask.compute(store_task, scheduler="threads", pool=pool)

    # ---------- move into final place --------------------------------------
    shutil.move(tmp_root, zarr_root)
    return store_task

def save_volume_to_omezarr_dask(arr: da.Array,
                           zroot: Path,
                           group: str,
                           *,
                           compressor=COMP,
                           lazy_threshold=LAZY_THRESHOLD, logger=None, show_pb = True):
    if arr.ndim == 3:
        arr = arr.map_blocks(_add_channel_axis,
                             dtype=arr.dtype,
                             chunks=((1,),) + arr.chunks)
    elif arr.ndim != 4:
        raise ValueError("array must be 3-D or 4-D (C Z Y X)")

    tgt_dir = _ensure_zarr_path(zroot / group)

    SMALL_VOL = 200 * 1024 ** 2
    if arr.nbytes <= SMALL_VOL:
        _local_to_zarr(arr, tgt_dir, component="0", n_threads = min(os.cpu_count(), 16),
             show_pb = show_pb, logger = logger)
        return

    store_task = da.to_zarr(
        arr,
        tgt_dir,
        component="0",
        overwrite=True,
        compressor=compressor,
        lock=False,
        compute=False
    )

    #future = get_client().persist(store_task)
    #get_client().wait(future)
    #return future
    pb_ctx = (
        ProgressBar(out=_LoggerWriter(logger))
        if show_pb and logger is not None else
        (ProgressBar() if show_pb else nullcontext())
    )
    with pb_ctx:
        try:  # distributed present
            fut = get_client().compute(store_task, retries=0)
            wait(fut)  # <-- block until finished
            return fut
        except ValueError:  # local threads fallback
            return dask.compute(
                store_task,
                scheduler="threads",
                pool=ThreadPool(min(os.cpu_count() or 1, 8))
            )[0]


def filter_dendrites_dask_old(dend_da, settings, logger):
    """
    Keep dendrites whose voxel-count ≥ settings.min_dendrite_vol.
    Returns a uint16 Dask array with ORIGINAL label IDs.  Two passes only:
    1) connected components  2) voxel-count reduction + Boolean mask.
    """

    # ── ensure uniform spatial chunks before labeling ─────────────────
    dend_da = dend_da.rechunk((64, 256, 256))

    # ── connected components (26-conn) ─────────────────────────────────
    lbl_da, n_labels = dask_label(dend_da)         # lazy; returns (array, scalar)
    max_label = int(n_labels.compute())
    if max_label == 0:
        logger.info("No dendrites found in volume.")
        return lbl_da.astype("uint16")

    # ── voxel counts per label (pure reduction, low RAM) ───────────────
    ones   = da.ones_like(lbl_da, dtype=np.uint8)
    index  = np.arange(1, max_label + 1, dtype=np.uint32)   # skip 0
    counts = ndm.sum(ones, lbl_da, index=index).compute()   # 1-D NumPy
    keep   = index[counts >= settings.min_dendrite_vol]

    logger.info(f"    Processing {len(keep)} of {max_label} dendrites "
                f"≥ {settings.min_dendrite_vol} vox")
    if keep.size == 0:
        return da.zeros_like(lbl_da, dtype="uint16")

    # ── fast Boolean mask then cast to uint16 ──────────────────────────
    mask_da   = da.isin(lbl_da, keep, assume_unique=True)
    filtered  = da.where(mask_da, lbl_da, 0).astype("uint16")

    return filtered

def to_zarr_cache(arr, name, store_dir):
    path = pathlib.Path(store_dir) / f"{name}.zarr"
    if not path.exists():
        da.to_zarr(arr, path, overwrite=True)
    return da.from_zarr(path)


def _local_to_zarr_dask(arr, store, *, component="0", n_threads=None, show_pb=True, logger=None):
    """
    Store *arr* → *store* inside the current process.

    Parameters
    ----------
    n_threads : int or None
        Number of CPU threads that Dask + Blosc may use.
        • None  →  automatic  (min( #physical cores, 32 ))
    show_pb   : bool
        If True, draw a Dask progress bar in the terminal / log.
    """
    n_threads = n_threads or min(os.cpu_count() or 1, 32)
    if logger:
        logger.info(f"     [OME-Zarr] writing with {n_threads} thread(s)…")

    # Blosc honours BLOSC_NUM_THREADS  ➜  we set it just for this call
    os.environ.setdefault("BLOSC_NUM_THREADS", str(n_threads))

    pool = ThreadPool(n_threads)                # for the threaded scheduler
    sched_cfg = {"scheduler": "threads", "pool": pool}

    if show_pb and logger is not None:
        pb_ctx = ProgressBar(out=_LoggerWriter(logger))
    elif show_pb:
        pb_ctx = ProgressBar()  # stdout fallback
    else:
        pb_ctx = nullcontext()

    with dask_config.set(**sched_cfg), pb_ctx:       # <─ both contexts together
        da.to_zarr(
            arr,
            store,
            component=component,
            overwrite=True,
            compute=True,
            compressor=COMP,
            lock=False,
        )
