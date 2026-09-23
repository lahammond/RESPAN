# -*- coding: utf-8 -*-
"""
Image Analysis tools and functions for spine analysis
==========


"""

__author__    = 'Luke Hammond <luke.hammond@osumc.edu>'
__license__   = 'GPL-3.0 License (see LICENSE)'
__copyright__ = 'Copyright © 2024 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/RESPAN'


# CuPy crashes enumerating a malformed .dist-info; this applies the filter.
# Lives in the package so the published source carries it - see the module
# docstring for why it used to be a bare `import sitecustomize`.
from RESPAN.ImageAnalysis import _cupy_metadata_fix  # noqa: F401
from RESPAN.ImageAnalysis.label_semantics import (
    LABEL_CHANNEL_MAP, soma_label_from_channel_map)
import RESPAN.ImageAnalysis.Segmentation_and_Restoration as sr
import RESPAN.ImageAnalysis.IO as io
import RESPAN.ImageAnalysis.DaskImageAnalysis as dia
import RESPAN.ImageAnalysis.ChunkedProcessing as chunked
import RESPAN.ImageAnalysis.Tables as tables
import RESPAN.ImageAnalysis.MemProfiler as mp

import time
import os
import zarr
import numpy as np
import pandas as pd
import math
import re
import shutil
import gc
import sys

import memory_profiler
import psutil
import subprocess

from collections import defaultdict
from pathlib import Path

from tifffile import imread, imwrite

from skimage.graph import route_through_array
from skimage import measure, morphology, segmentation, exposure # graph #util,  color, data, filters,  exposure, restoration
from skimage.transform import resize
from skimage.measure import marching_cubes
import trimesh

from scipy import ndimage
from scipy.ndimage import generate_binary_structure, distance_transform_edt, gaussian_filter
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

import cupy as cp
import cupy.cuda.runtime as rt
from cupyx.scipy import ndimage as cp_ndimage
from cupyx.scipy.ndimage import binary_dilation

import dask as dask
from dask.distributed import Client, LocalCluster

import numba
from numba import jit
from scipy.spatial import ConvexHull

GB = 1024 ** 3


##############################################################################
# GPU Memory Management Utilities
##############################################################################

class VRAMTracker:
    """Track peak GPU memory usage across a pipeline run."""
    def __init__(self):
        self.peak_used = 0
        self.peak_label = ""

    def checkpoint(self, label, logger):
        """Log GPU memory state and track peak usage."""
        cp.cuda.Device().synchronize()
        free, total = cp.cuda.runtime.memGetInfo()
        used = total - free
        pool = cp.get_default_memory_pool()
        pool_used = pool.used_bytes()

        if used > self.peak_used:
            self.peak_used = used
            self.peak_label = label

        logger.info(f"    [VRAM] {label}: {used / 1e9:.2f}/{total / 1e9:.2f} GB used "
                    f"(free: {free / 1e9:.2f} GB, pool: {pool_used / 1e9:.2f} GB)")
        return free, total

    def report(self, logger):
        """Report peak GPU memory usage."""
        logger.info(f"    [VRAM PEAK] {self.peak_used / 1e9:.2f} GB at '{self.peak_label}'")


def vram_guard(nbytes_or_array, name, logger, safety_factor=0.8):
    """Check if uploading an array to GPU would exceed available VRAM.

    Returns True if safe to upload, False if would overflow.
    """
    if hasattr(nbytes_or_array, 'nbytes'):
        nbytes = nbytes_or_array.nbytes
    else:
        nbytes = int(nbytes_or_array)

    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    free, total = cp.cuda.runtime.memGetInfo()

    pct_of_free = nbytes / free * 100 if free > 0 else float('inf')

    if nbytes > free * safety_factor:
        logger.warning(
            f"    [VRAM GUARD] {name}: {nbytes / 1e9:.2f} GB needed, "
            f"only {free / 1e9:.2f} GB free. Would use {pct_of_free:.0f}% of free VRAM. "
            f"Blocked — using chunked approach.")
        return False

    logger.info(f"    [VRAM] Allocating {name}: {nbytes / 1e9:.2f} GB "
                f"({pct_of_free:.1f}% of {free / 1e9:.2f} GB free)")
    return True


def flush_gpu():
    """Aggressively free GPU memory pools."""
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


##############################################################################
# Main Processing Functions
##############################################################################


def _workspace_restore(workspace, primary_name, fallback_name):
    """Restore array from workspace as numpy (legacy — full materialization)."""
    primary_path = os.path.join(workspace.base_dir, primary_name + ".zarr", ".zarray")
    if os.path.isfile(primary_path):
        return workspace.array_to_numpy(primary_name)
    return workspace.array_to_numpy(fallback_name)


def _workspace_restore_zarr(workspace, primary_name, fallback_name):
    """Return zarr handle from workspace without materialization.

    Downstream consumers that accept
    both zarr and numpy (via slab slicing, .nbytes, .shape) should call this
    instead of `_workspace_restore`. Avoids the ~27.7 GB necks numpy alloc
    at T_LARGE (and 22 GB skeleton, etc.) when the consumer never needs
    contiguous numpy.
    """
    primary_path = os.path.join(workspace.base_dir, primary_name + ".zarr", ".zarray")
    if os.path.isfile(primary_path):
        return workspace.open_array(primary_name)
    return workspace.open_array(fallback_name)


def _cleanup_workspace(workspace, logger):
    if workspace is not None:
        try:
            workspace.cleanup()
        except Exception as e:
            if logger:
                logger.warning(f"     Workspace cleanup failed: {e}")


def analyze_spines(settings, locations, log, logger):
    if settings.dask_enabled:
        try:  # already started elsewhere?
            client = dask.distributed.get_client()
        except ValueError:  # no client yet
            client, cluster = dia.start_local_cluster(
                max_workers = None,
                target_ram_frac = None,
                worker_mem="8GB",  # many small workers
                threads_per_worker=2,  # light processes
                  # optional upper bound
                tmp_dir = None,
                logger=logger,
                simulate = None, # or "low" for low mem
            )

    else:
        logger.info("Analyzing spines...")
    #spines = 1
    #dendrites = 2
    #soma = 3

    files = [file_i
             for file_i in os.listdir(locations.input_dir)
             if file_i.endswith('.tif')]
    files = sorted(files)

    label_files = [file_i
                   for file_i in os.listdir(locations.labels)
                   if file_i.endswith('.tif')]

    label_files = sorted(label_files)

    if len(files) != len(label_files):
        #logger.info(log)
        raise RuntimeError("Number of raw and label images are not the same - check data.")



    spine_summary = pd.DataFrame()

    n_failed = 0  # images whose processing raised; reported at the end and returned to the caller
    for file in range(len(files)):
        logger.info(f' Analyzing image {file+1} of {len(files)} \n  Raw Image: {files[file]} & Label Image: {label_files[file]}')

        settings.filename = files[file].replace(".tif", "")

        if settings.dask_enabled == True:

            if settings.resave_omezarr:
                    logger.info(f"  Resaving files as OME.Zarr for quick access using Dask...")
            logger.info(f"  Resaving raw image data...")

            image, raw_zarr_root  = dia.open_tiff_as_dask(
                locations.input_dir + files[file],
                client=client,
                chunks=settings.dask_block,
                pixel_sizes=(
                    None,  # C – unknown
                    settings.input_resZ,
                    settings.input_resXY,
                    settings.input_resXY,
                ),
                resave=settings.resave_omezarr,
                settings=settings,
                logger =  logger
            )
            logger.info(f"  Resaving labels data...")
            #image = dia.open_tiff_as_dask(locations.input_dir + files[file])
            labels, _  = dia.open_tiff_as_dask(
                locations.labels + files[file],
                client=client,
                chunks=image.chunks,  # match raw grid *now*
                pixel_sizes=None,  # voxels only
                resave=settings.resave_omezarr,
                settings=settings,
                logger=logger
            )
            #labels = dia.open_tiff_as_dask(locations.labels + files[file]).rechunk(image.chunks)
            if len(labels.shape) == 4:  # Check if it has a channel dimension (CZYX)
                labels = labels[0, :, :, :]  # remove channel dimension
                logger.info(f"  Removed channel dimension from labels, new shape: {labels.shape}") # remove channel dimension
            image = image.persist()
            labels = labels.persist()

        else:
            # Phase L1: metadata-first streaming entry. Read shape/dtype/axes
            # for both image and labels without loading data; pick streaming
            # vs imread per file based on raw byte estimate. Above the 5 GiB
            # threshold, imread risks contiguous-numpy MemoryError (T_LARGE
            # 55 GB raw + 27.7 GB labels failed at 82 GB demand on a 137 GB
            # host). Falls back to legacy imread on unusual axes or when
            # metadata read fails — preserves T2 byte-identical regression.
            _STREAM_INGEST_THRESHOLD_BYTES = 5 * (1024 ** 3)  # 5 GiB
            _img_path = locations.input_dir + files[file]
            _lbl_path = locations.labels + files[file]

            try:
                _img_meta = chunked.tiff_metadata(_img_path)
                _lbl_meta = chunked.tiff_metadata(_lbl_path)
            except Exception as _meta_err:
                logger.warning(
                    f"  tiff_metadata failed ({_meta_err}); falling back to imread")
                _img_meta = None
                _lbl_meta = None

            _use_stream_image = False
            _use_stream_labels = False
            _select_channel_img = None
            if _img_meta is not None:
                _src_axes = _img_meta['axes']
                _src_shape = _img_meta['shape']
                _itemsize = int(_img_meta['dtype'].itemsize)
                if _src_axes == 'ZYX':
                    _neuron_bytes = int(np.prod(np.array(_src_shape, dtype=np.int64))) * _itemsize
                    _use_stream_image = _neuron_bytes > _STREAM_INGEST_THRESHOLD_BYTES
                    _select_channel_img = None
                elif _src_axes == 'ZCYX':
                    _z, _c, _y, _x = _src_shape
                    _neuron_bytes = int(_z) * int(_y) * int(_x) * _itemsize
                    _use_stream_image = _neuron_bytes > _STREAM_INGEST_THRESHOLD_BYTES
                    _select_channel_img = settings.neuron_channel - 1
                elif _src_axes == 'CZYX':
                    _c, _z, _y, _x = _src_shape
                    _neuron_bytes = int(_z) * int(_y) * int(_x) * _itemsize
                    _use_stream_image = _neuron_bytes > _STREAM_INGEST_THRESHOLD_BYTES
                    _select_channel_img = settings.neuron_channel - 1

            if _lbl_meta is not None:
                # Labels must be 3D ZYX (RESPAN convention). Stream when raw
                # byte count exceeds threshold.
                if _lbl_meta['axes'] == 'ZYX':
                    _lbl_bytes = _lbl_meta['n_bytes_estimate']
                    _use_stream_labels = _lbl_bytes > _STREAM_INGEST_THRESHOLD_BYTES

            # Stream BOTH or NEITHER — peak host RAM is dominated by whichever
            # path goes through imread, so a one-streamed-one-imread compromise
            # gives the worst of both worlds at T_LARGE (still allocates one
            # contiguous numpy of the larger volume). Either go fully chunked
            # at entry, or stay legacy.
            _use_stream_entry = _use_stream_image or _use_stream_labels
            # Test override: settings.force_stream_entry forces the streaming
            # path on small data so we can validate end-to-end zarr semantics
            # without a 4-hour T7 cycle. T_LARGE-scale runs trigger via the
            # 5 GiB threshold above and don't need the override.
            if (_img_meta is not None and _lbl_meta is not None
                    and getattr(settings, 'force_stream_entry', False)):
                _use_stream_entry = True
                if _select_channel_img is None and _img_meta['axes'] != 'ZYX':
                    _select_channel_img = (settings.neuron_channel - 1)

            if _use_stream_entry:
                # If streaming fires but
                # MemoryBudget at L405 then evaluates `neuron.shape` and
                # decides use_chunked=False (e.g., 5-15 GiB on a 137 GB host),
                # zarr image would flow through legacy numpy-only branches.
                # Streaming at entry implies chunked downstream — force it.
                settings.force_chunked = True

                # Hoist workspace creation here. spine_and_whole_neuron_processing
                # detects an existing settings._active_workspace and reuses it
                # instead of creating a second cache directory.
                _entry_filename = files[file].replace('.tif', '')
                _entry_workspace = chunked.ZarrWorkspace(
                    locations, _entry_filename, logger)
                settings._active_workspace = _entry_workspace
                logger.info(
                    f"  Phase L1 streaming entry "
                    f"(image bytes={_img_meta['n_bytes_estimate']/1e9:.1f} GB, "
                    f"labels bytes={_lbl_meta['n_bytes_estimate']/1e9:.1f} GB)")

                # Stream raw → 3D zarr (just neuron channel). Skip the (Z,1,Y,X)
                # 4D-singleton-C representation — downstream check_image_shape
                # is bypassed for zarr image, and L399 has a zarr-3D guard.
                image = chunked.tiff_to_zarr_3d(
                    _img_path, _entry_workspace, 'image_neuron',
                    select_channel=_select_channel_img, logger=logger)

                # Stream labels → single 3D zarr (handles compressed
                # multi-IFD labels via per-page; ImageJ big-TIFF via memmap).
                # The existing labels_to_zarr_channels() at L483 then slab-
                # iterates this zarr to produce per-class output zarrs.
                labels = chunked.tiff_to_zarr_3d(
                    _lbl_path, _entry_workspace, 'labels_full',
                    select_channel=None, logger=logger)
            else:
                image = imread(_img_path)
                labels = imread(_lbl_path) # use original file name to ensure correct image regardless of sorting

        logger.info(f"  Raw shape: {image.shape} & Labels shape: {labels.shape}")
        #print image size in GB, with comma between thousands
        image_GB = image.nbytes / 1e9

        logger.info(f"  Raw image size: {image_GB:.3f} GB")
        if image_GB > 2 and settings.resave_omezarr == False:
            logger.info(f"  *Note, as the dataset is over 2GB, we recommend enabling Dask for efficient processing. ")

        if image_GB > 2 and settings.Vaa3d == True:
                    logger.info(f"\n\n  Automated Vaa3D SWC generation is disabled for images over 2GB. We recommend using Simple Neurite Tracer in ImageJ with"
                        f"\n  the neuron segmentation masks from RESPAN to generate SWC files.\n")

        # if using axial resotration - image needs to be scaled to match the labels
        if settings.axial_restore == True:
            #rescale image to match labels
            if len(image.shape)==3:
                image = resize(image, labels.shape, order=0, mode='constant', preserve_range=True, anti_aliasing=True)
            elif len(image.shape)==4:
                # Broadcast the labels to have the same shape in axis 1 as the image
                labels_reshaped = labels[:, np.newaxis, ...]  # Add a new axis at position 1
                # Use np.tile to match the channel dimension of  image
                labels_reshaped = np.tile(labels_reshaped, (1, image.shape[1], 1, 1))

                # Resize the image to match the reshaped labels' shape
                image = resize(image, labels_reshaped.shape, order=0, mode='constant', preserve_range=True,
                               anti_aliasing=True)
                del labels_reshaped

            logger.info(f"  Due to axial restoration, image rescaled to match labels: {image.shape}")

        # rescale labels back up if required

        if settings.axial_restore == False:
            if settings.original_shape[0] != None and settings.input_resZ != settings.model_resZ or settings.input_resXY != settings.model_resXY and settings.prev_labels == False:
                logger.info(f"  orignal settings shape: {settings.original_shape[file]}")
                # Gate same as Segmentation_and_Restoration.py:395 — skimage's
                # resize float64 alloc is fatal at T_LARGE upscale (29.8B uint8
                # output, 238 GB float64 transient otherwise). order=0 nearest
                # neighbor for label volumes; chunked path uses scipy.ndimage.zoom
                # internally with the same semantics.
                _n_out_vox = (int(settings.original_shape[file][0])
                              * int(settings.original_shape[file][1])
                              * int(settings.original_shape[file][2]))
                if _n_out_vox > 1_000_000_000:
                    import RESPAN.ImageAnalysis.ChunkedProcessing as _chunked
                    import gc as _gc
                    # Split-pass to free `labels` between passes — at T_LARGE,
                    # input(17 GB) + float32 intermediate(79 GB) + output(60 GB)
                    # = 156 GB exceeds 137 GB host if all coexist. Explicit del
                    # after Pass 1 drops input bytes before output allocation.
                    _label_dtype = labels.dtype
                    _intermediate = _chunked.chunked_resize_xy_pass(
                        labels, settings.original_shape[file],
                        order=0, logger=logger)
                    del labels
                    _gc.collect()
                    labels = _chunked.chunked_resize_z_pass(
                        _intermediate, settings.original_shape[file],
                        order=0, dtype=_label_dtype, logger=logger)
                    del _intermediate
                    _gc.collect()
                else:
                    labels = resize(labels, settings.original_shape[file], order=0, mode='constant', preserve_range=True, anti_aliasing=None)
                logger.info(f"  labels resized: {labels.shape}")

                imwrite(locations.labels + label_files[file], labels.astype(np.uint8), compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
                        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
                        resolution=(settings.input_resXY, settings.input_resXY))


        #process images through Vaa3d
        if settings.Vaa3d and image_GB < 2 and os.path.exists(settings.Vaa3Dpath) :
            logger.info(f"  Creating SWC file using Vaa3D...")
            create_dir(locations.swcs)
            if os.path.exists(locations.swcs + files[file]+".swc"):
                logger.info(f"   {files[file]}.swc already exists, delete this file to regenerate.")
            else:
                # can no longer use                 vaa3D_neuron = (labels >= 2)*255 as necks are labeled with 4 and spines with 1
                # vaa3D_neuron = (labels >= 2)*255
                vaa3D_mask = np.where((labels >= 2) & (labels <= 3), 1, 0)
                neuron = image[:, settings.neuron_channel - 1, :, :]
                masked_neuron = neuron * vaa3D_mask

                imwrite(locations.swcs + files[file], masked_neuron.astype(np.uint8), compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
                        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})
                #run Vaa3D on image:
                input_path = os.path.join(locations.swcs, files[file])
                output_path = os.path.join(locations.swcs, files[file] + ".swc")
                cmd_list = [
                    str(settings.Vaa3Dpath),
                    "/x", "vn2", "/f", "app2",
                    "/i", input_path,
                    "/o", output_path,
                    "/p", "NULL", "0", "1", "1", "1", "1", "0", "5", "1", "0", "0"
                ]
                # Run the command (list-based, handles spaces in paths)
                process = subprocess.Popen(cmd_list)
                stdout, stderr = process.communicate()
                #logger.info(stdout + stderr)

                os.remove(locations.swcs + files[file])
                # Clear variables to free up memory
                del vaa3D_mask
                del neuron
                del masked_neuron
                # Force garbage collection
                gc.collect()

        # Phase L1: skip check_image_shape for zarr-streamed image. The function
        # at L5453 calls np.expand_dims/np.moveaxis which materializes the full
        # zarr (BAD at T_LARGE 55 GB). For chunked-streamed image: shape is
        # already normalized at ingest (3D ZYX = pre-extracted neuron channel,
        # the same data that check_image_shape+image[:,c,:,:] would have given
        # for numpy). Downstream consumers detect 3D zarr and treat as neuron
        # directly (see L399 zarr guard, spine_vox_measurements L5036-5038).
        if not hasattr(image, 'chunks'):
            image = check_image_shape(image, logger)

        #dendrite specific superceeded by whole neuron analysis
        #if settings.analysis_method == "Dendrite Specific":
        #    spine_summary = spine_and_dendrite_processing(image, labels, spine_summary, settings, locations, files[file], log, logger)
        #else:
        try:
            if settings.dask_enabled == True:
                spine_summary = dia.spine_and_whole_neuron_processing(image, labels, spine_summary, raw_zarr_root, settings, locations,
                                                                      files[file].replace(".tif",""), log, logger)
            else:
                spine_summary = spine_and_whole_neuron_processing(image, labels, spine_summary, settings, locations,
                                                                  files[file].replace(".tif", ""), log, logger)
        except Exception as e:
            logger.error(f"  ERROR processing image {files[file]}: {e}")
            logger.error(f"  Partial results from prior images (if any) will be saved.")
            import traceback
            logger.error(traceback.format_exc())
            # Ensure workspace is cleaned up even on crash
            _cleanup_workspace(getattr(settings, '_active_workspace', None), logger)
            settings._active_workspace = None
            n_failed += 1

        if settings.shape_error == True:
            logger.info(f"!!! One or more images moved to \\Not_Processed due to having\nless than 5 Z slices. Please modify these files before reprocessing.\n")

        # Save after each image so partial results survive crashes
        spine_summary.to_csv(locations.tables + 'Detected_spines_summary.csv',index=False)

    # 2026-09-19: a per-image failure is caught above so the batch can go on,
    # but it used to end in the same "analysis complete" line as a clean run
    # (seen on a GUI run whose only image died in neck association: empty
    # tables, "complete" banner). Say so, and hand the count to the caller.
    if n_failed:
        logger.error(f"RESPAN analysis finished with ERRORS: {n_failed} of {len(files)} image(s) failed "
                     f"(see 'ERROR processing image' above). Their tables and MIPs are missing or incomplete.")
    else:
        logger.info("RESPAN analysis complete.")
    return n_failed

@mp.profile_mem()
def spine_and_whole_neuron_processing(image, labels_vol, spine_summary, settings, locations, filename, log, logger):
    time_initial = time.time()
    vram = VRAMTracker()
    vram.checkpoint("pipeline_start", logger)

    # Filopodia recovery state — initialized here so it's defined regardless of
    # whether neck generation runs. Populated by recover_filopodia_from_orphan_necks
    # if settings.recover_filopodia is True and model_type >= 3.
    _filopodia_labels = set()
    _filopodia_metadata = {}
    # Partial-spine detection — populated by detect_partial_spines after all
    # neck processing (drop_disconnected, spurious, filopodia) if neck_generation
    # is on. These labels get spine_type='partial-spine' and neck metrics zeroed.
    _partial_spine_labels = set()

    # Phase L1: zarr 3D image is the chunked-streamed pre-extracted neuron
    # channel (single-channel single-volume; multi-channel deferred to post-L5
    # For numpy 4D image (legacy path) extract the
    # neuron channel as a strided view. spine_vox_measurements L5036-5038
    # already handles the 3D zarr case for measurement reads.
    if hasattr(image, 'chunks'):
        if image.ndim == 3:
            neuron = image
        elif image.ndim == 4:
            # Fail fast for 4D zarr until that path lands
            # multi-channel zarr support. Without L5, image[:, c, :, :] on a
            # 4D zarr at T_LARGE materializes 55 GB numpy — defeats Phase L1.
            raise NotImplementedError(
                f"Phase L1: 4D zarr image not supported until L5 (multi-channel "
                f"not yet supported). Got ndim=4 zarr shape={image.shape}. "
                f"L1 only produces 3D zarr (single neuron channel) at the entry.")
        else:
            raise ValueError(
                f"Phase L1: unexpected zarr image.ndim={image.ndim} (shape={image.shape})")
    else:
        neuron = image[:, settings.neuron_channel - 1, :, :]

    # RAM pre-flight check
    budget = chunked.MemoryBudget()
    estimated_ram = neuron.nbytes * 8
    available_ram = psutil.virtual_memory().available
    use_chunked = budget.needs_chunked_processing(neuron.shape) or getattr(settings, 'force_chunked', False)
    workspace = None
    if use_chunked:
        # Phase L1: reuse entry-side workspace if analyze_spines hoisted one
        # for streaming TIFF→zarr ingest. Creating a second ZarrWorkspace
        # pointed at the same base_dir would (a) double-log the cache path
        # and (b) risk one instance's cleanup() wiping the other's arrays.
        _entry_ws = getattr(settings, '_active_workspace', None)
        # An isinstance check alone is insufficient.
        # A failed prior file's run can leave _active_workspace pointing at a
        # workspace whose base_dir was already cleaned up, OR whose filename
        # corresponds to a different image. Verify both before reuse.
        _can_reuse = (
            _entry_ws is not None
            and isinstance(_entry_ws, chunked.ZarrWorkspace)
            and os.path.isdir(_entry_ws.base_dir)
            and os.path.basename(_entry_ws.base_dir) == filename
        )
        if _can_reuse:
            workspace = _entry_ws
            logger.info(f"   Reusing entry-side ZarrWorkspace (Phase L1 streaming)")
        else:
            if _entry_ws is not None:
                # Stale state — clear so a future entry can see a clean slot.
                logger.info(
                    f"   Discarding stale settings._active_workspace "
                    f"(base_dir exists={os.path.isdir(_entry_ws.base_dir) if hasattr(_entry_ws, 'base_dir') else 'n/a'}, "
                    f"filename match={os.path.basename(_entry_ws.base_dir) == filename if hasattr(_entry_ws, 'base_dir') else 'n/a'})")
                settings._active_workspace = None
            workspace = chunked.ZarrWorkspace(locations, filename, logger)
            settings._active_workspace = workspace
        budget.log_summary(logger)
        logger.info(f"   Large image detected (est. peak {estimated_ram / 1e9:.1f} GB). "
                     f"GPU-chunked processing will be used for distance transforms.")

        # Phase F — spill `image` to zarr at entry so its ~11 GB (T7) of RAM
        # is freed before spine detection, Pass 1 neck assignment, and all
        # downstream stages. `image = zarr_handle` preserves `.shape`,
        # `.ndim`, `.dtype`, and slice semantics (slice → numpy slab).
        # Downstream consumers check `hasattr(image, 'chunks')` to branch
        # into per-spine zarr-safe regionprops paths; numpy-only paths
        # still work via zarr slice materialization per read.
        if isinstance(image, np.ndarray):
            _pre_image_nbytes_gb = image.nbytes / 1e9
            _img_for_spill = image
            if image.ndim == 3:
                # Promote to (Z, 1, Y, X) so downstream `image[:, C, :, :]`
                # indexing is uniform; downstream `np.expand_dims` branches
                # become no-ops (len(image.shape) == 4 already).
                _img_for_spill = image[:, np.newaxis, :, :]
                logger.info(
                    f"   Promoting image 3D → 4D before zarr spill (shape {image.shape} → {_img_for_spill.shape}).")
            try:
                # Chunks: full Z per chunk in C axis (size 1), STREAMING_CHUNKS for Z/Y/X.
                _img_chunks = (chunked.STREAMING_CHUNKS[0], 1,
                               chunked.STREAMING_CHUNKS[1], chunked.STREAMING_CHUNKS[2])
                workspace.numpy_to_zarr('image', _img_for_spill, chunks=_img_chunks)
                del _img_for_spill, image
                gc.collect()
                image = workspace.open_array('image')
                logger.info(
                    f"   Phase F: image spilled to zarr at entry ({_pre_image_nbytes_gb:.1f} GB freed). "
                    f"Handle shape={tuple(image.shape)}, dtype={image.dtype}.")
                vram.checkpoint("after_image_zarr_spill", logger)
            except Exception as _e:
                logger.warning(
                    f"   Phase F image spill failed ({_e}); keeping image in RAM. "
                    "T7-class runs may hit OOM without the spill.")
                image = _img_for_spill if _img_for_spill is not image else image
    if estimated_ram > available_ram * 0.8:
        logger.warning(
            f"   WARNING: Estimated RAM usage ({estimated_ram / 1e9:.1f} GB) may exceed 80% of available RAM "
            f"({available_ram / 1e9:.1f} GB). Processing may be slow or fail due to memory pressure.")

    if neuron.size > 1e8:
        logger.info(
            f"   The neuron channel for this image is ~{neuron.size / 1e9:.2f} GB in size. This may take considerable time to process.")
        logger.info(
            f"   Estimated processing time is ~{round(neuron.size * 2.5e-8, 0)} minutes, depending on computational resources.")

        # temp size limits
    if neuron.size > 1e9:
        logger.info(
            f"    *Note, as the dataset is over 1GB, full 3D validation data export has been disabled (as these volumes can be 20x raw input)."
            f"\n    If using Dask, OME-Zarr files can be viewed in a Zarr Viewer. Otherwise, to generate 3D validation datasets, please isolate specific regions of the dataset and process separately.")
        if not hasattr(settings, 'mesh_analysis') or settings.mesh_analysis is not False:
            settings.mesh_analysis = True
        settings.save_val_volumes = False  # Skip 3D validation volumes for large images
        # MIPs are always generated (2D projections, small memory footprint)
    else:
        if not hasattr(settings, 'mesh_analysis') or settings.mesh_analysis is not False:
            settings.mesh_analysis = True

    # Extract label channels — zarr slab-by-slab for large volumes, numpy for small
    channel_map = LABEL_CHANNEL_MAP[settings.model_type]

    if use_chunked:
        logger.info("   Extracting label channels to zarr (slab-by-slab)...")
        _label_zarrs = chunked.labels_to_zarr_channels(
            labels_vol, channel_map, workspace, logger=logger)
        dendrites = _label_zarrs['dendrites']
        soma = _label_zarrs['soma']
        spines = _label_zarrs.get('spines', None)
        necks = _label_zarrs.get('necks', None)
        if settings.model_type == 4:
            spine_cores = _label_zarrs['spine_cores']
            spine_membranes = _label_zarrs['spine_membranes']
            spines = _label_zarrs['spines']  # combined cores+membranes written by helper
        if spines is None:
            spines = workspace.create_array('spines', labels_vol.shape, dtype=np.uint8,
                                             chunks=chunked.STREAMING_CHUNKS)
        if necks is None and settings.model_type in (3, 4):
            pass  # necks already in _label_zarrs
        elif necks is None:
            necks = np.zeros(1, dtype=np.uint8)
        # Precompute soma_present from slab scan (avoids np.max on full zarr).
        # 2026-09-20: this indexed channel_map BY POSITION -
        # `channel_map.get(3, channel_map.get(5, -1))` - but the map is
        # {label value: class name}, so it returned a NAME ('spine_membranes' for
        # model type 4, 'soma' for types 1 and 3, -1 for type 2) and the slab was
        # compared against a STRING. NumPy answers all-False, so soma_present was
        # False on EVERY chunked run for EVERY model type. Measured on T4 with
        # Dataset226: 1,289,594 soma voxels present, soma is reported absent.
        # The cost was not only euclidean_dist_to_soma (0.0 before v1.5.1, NaN
        # after): soma_present=False also nulls _soma_arg, so no soma KDTree is
        # built and geodesic distances are measured from the corner-most dendrite
        # voxel while still being named geodesic_dist_to_soma.
        # Derive the label from the map instead of assuming its position.
        _soma_label = soma_label_from_channel_map(channel_map)
        soma_present = False
        if _soma_label is not None:
            for _z in range(0, labels_vol.shape[0], 128):
                _ze = min(_z + 128, labels_vol.shape[0])
                if np.any(labels_vol[_z:_ze] == _soma_label):
                    soma_present = True
                    break
        if soma_present:
            logger.info(f"    Soma present (label {_soma_label}).")
        del labels_vol
        gc.collect()
    else:
        if settings.model_type == 1:
            spines = (labels_vol == 1)
            dendrites = (labels_vol == 2)
            soma = (labels_vol == 3)
        elif settings.model_type == 2:
            dendrites = (labels_vol == 1)
            soma = (labels_vol == 2)
            spines = (labels_vol == 10)
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
            spines = spine_cores + spine_membranes
        del labels_vol
        gc.collect()
        soma_present = bool(np.max(soma) > 0)

    flush_gpu()
    # filter dendrites
    logger.info("   Filtering dendrites...")

    if use_chunked:
        labeled_dendrites = chunked.filter_dendrites_streaming(
            dendrites, settings.min_dendrite_vol, workspace, logger)
    else:
        labeled_dendrites = filter_dendrites(dendrites, settings, logger)
    # Free the raw dendrite binary — labeled_dendrites is all we need now
    del dendrites
    gc.collect()

    # Dendrite repair: bridge fragments that should be one dendrite
    # but were split by discontinuous nnU-Net predictions. Operates in XY only,
    # only fills background voxels along the bridge path (no morphological
    # dilation), and remaps merged fragments to the parent (largest CC) label.
    if getattr(settings, 'dendrite_repair', False):
        _max_dist = float(getattr(settings, 'dendrite_repair_max_dist_um', 2.0))
        logger.info(
            f"   Dendrite repair (max gap {_max_dist:.2f} µm, XY only)...")
        labeled_dendrites, _ = dendrite_repair(
            labeled_dendrites, _max_dist, settings.input_resXY, logger)

    vram.checkpoint("after_filter_dendrites", logger)

    if isinstance(labeled_dendrites, np.ndarray):
        _has_dendrites = bool(np.max(labeled_dendrites) > 0)
    else:
        _has_dendrites = False
        for _z in range(0, labeled_dendrites.shape[0], 128):
            if np.any(np.array(labeled_dendrites[_z:min(_z+128, labeled_dendrites.shape[0])]) > 0):
                _has_dendrites = True
                break
    if _has_dendrites:

        if not soma_present:
            # label and filter soma
            soma_distance = soma
        else:
            # Soma distance is a global metric (spines can be 50-200+ µm from soma).
            # Always use full-image adaptive EDT — never chunk this, as halo-limited
            # chunks would silently underestimate distances for distant spines.
            logger.info("   Calculating soma distance...")
            # Plumb workspace + dest_name when
            # chunked, so soma EDT writes the bool intermediate to the
            # workspace (not system temp) AND streams the upsample directly
            # to a workspace zarr (avoids 120 GB float32 numpy peak at T_LARGE).
            if use_chunked and workspace is not None:
                soma_distance = adaptive_distance_transform(
                    soma, logger,
                    workspace=workspace, dest_name='soma_distance')
            else:
                soma_distance = adaptive_distance_transform(soma, logger)

        # For large volumes, free arrays not needed until after spine detection.
        # soma is done (soma_distance captured); necks/spines not needed until post-detection.
        _soma_spilled = False
        _necks_spilled = False
        _spines_spilled = False
        if use_chunked:
            if isinstance(soma, np.ndarray):
                if soma_present:
                    workspace.numpy_to_zarr('soma_spill', soma)
                    del soma
                    _soma_spilled = True
                workspace.numpy_to_zarr('necks_spill', necks)
                del necks
                _necks_spilled = True
                workspace.numpy_to_zarr('spines_spill', spines)
                del spines
                _spines_spilled = True
                gc.collect()
                logger.info("    Spilled soma/necks/spines to zarr (freed ~15 GB for EDT/skeleton/spine detection)")
            else:
                # Already zarr-backed from labels_to_zarr_channels — just mark as spilled
                # so restore logic uses workspace.open_array with the original channel names
                _soma_spilled = True
                _necks_spilled = True
                _spines_spilled = True
                logger.info("    soma/necks/spines already zarr-backed from label extraction")

        # Create Distance Map
        # dendrite_distance is used for BOTH filtering (spine_dist threshold)
        # AND measurement (spine_length = max distance within spine from dendrite).
        # The full distance field is required — do NOT cap with max_dist, as that
        # would silently truncate spine_length for long spines.
        if use_chunked:
            # Build dendrites_mask as zarr slab-by-slab (avoids 5 GB numpy temp)
            vol_shape = labeled_dendrites.shape
            dendrites_mask_zarr = workspace.create_array('dendrites_mask', vol_shape,
                                                          dtype=np.uint8, chunks=chunked.STREAMING_CHUNKS)
            for _z in range(0, vol_shape[0], 128):
                _ze = min(_z + 128, vol_shape[0])
                dendrites_mask_zarr[_z:_ze] = (np.array(labeled_dendrites[_z:_ze]) > 0).astype(np.uint8)
            dendrites_mask = dendrites_mask_zarr

            # Distance transform — GPU-chunked EDT for accuracy (no downsampling),
            # with scipy fallback only when volume is too small to benefit from GPU.
            logger.info("   Calculating distances from dendrites...")
            spacing = (settings.input_resZ, settings.input_resXY, settings.input_resXY)
            dend_dist_zarr = workspace.create_array('dendrite_distance', vol_shape, dtype=np.float32,
                                                      chunks=chunked.spine_access_chunks(settings))
            # max_dist must cover beyond spine_dist so distant spines get correct
            # (large) distances and are properly filtered OUT. Default min(shape)*min(spacing)/2
            # is catastrophically wrong for anisotropic volumes (T4: 80*0.045/2 = 1.8 µm caps
            # ALL distances, making the spine filter useless).
            _edt_max_dist = settings.neuron_spine_dist * settings.input_resXY * 3
            _dd_max_reliable_vox = chunked.distance_transform_gpu_chunked(
                dendrites_mask_zarr, dend_dist_zarr, spacing=spacing,
                max_dist=_edt_max_dist,
                logger=logger)
            dendrite_distance = dend_dist_zarr
            # Stash for MIP/Vol save zero-out (visualization only; raw values
            # remain in dend_dist_zarr for measurement code)
            settings._dd_viz_clamp_vox = _dd_max_reliable_vox
            vram.checkpoint("after_distance_transforms", logger)

            # Per-object skeletonization
            logger.info("   Calculating dendrite skeleton...")
            skeleton_zarr = workspace.create_array('skeleton', vol_shape, dtype=np.uint8,
                                                    chunks=chunked.STREAMING_CHUNKS)
            _skel_coords = chunked.skeletonize_per_object(
                labeled_dendrites, skeleton_zarr, logger=logger)
            gc.collect()
        else:
            dendrites_mask = (labeled_dendrites > 0).astype(np.uint8)
            logger.info("   Calculating distances from dendrites...")
            dendrite_distance = adaptive_distance_transform(labeled_dendrites > 0, logger)
            vram.checkpoint("after_distance_transforms", logger)
            logger.info("   Calculating dendrite skeleton...")
            skeleton = morphology.skeletonize(dendrites_mask)

        logger.info(f"    Time taken for initial processing: {time.time() - time_initial:.2f} seconds\n")
        # Spine Detection — spill remaining large arrays to zarr to free RAM
        # for the zarr-backed CC + watershed in spine_detection_cores_and_membranes.
        _soma_dist_spilled = False
        _dend_labels_spilled = use_chunked  # labeled_dendrites is zarr from filter_dendrites_streaming
        _dend_mask_spilled = use_chunked    # dendrites_mask is zarr, built above
        _skeleton_deferred = use_chunked    # skeleton stays in zarr
        if use_chunked:
            if soma_present:
                workspace.numpy_to_zarr('soma_distance_spill', soma_distance)
                del soma_distance
                _soma_dist_spilled = True
            gc.collect()

        logger.info("   Analyzing spines...")
        model_options = ["Spines, Dendrites, and Soma", "Dendrites and Soma Only", "Necks, Spines, Dendrites, and Soma",
                         "Dendrites, Spine Cores, Spine Membranes, Necks, Soma, and Axons"]
        logger.info(f"    Using model type {model_options[settings.model_type - 1]}")
        if settings.model_type <= 3:
            spine_labels = spine_detection(spines, settings.erode_shape, settings.remove_touching_boarders,
                                           logger)  # binary image, erosion value (0 for no erosion)
        else:
            logger.info("    Analyzing spines using cores and membranes...")
            # For zarr cores/membranes + per-cluster
            # path (volume_bytes > 2 GB inside the function), pass the zarr
            # handles directly. The function's per-cluster branch uses bbox
            # crops via `cores[sl]` / `membranes[sl]` which is zarr-safe — no
            # need to materialize. Prior code did np.array(zarr).astype(bool)
            # which allocates 27.7 GB bool numpy each at T_LARGE (55+ GB peak).
            _cn_voxels = int(np.prod(np.array(spine_cores.shape, dtype=np.int64)))
            _cores_will_per_cluster = _cn_voxels * 8 > 2 * (1024 ** 3)
            _cores_is_zarr = hasattr(spine_cores, 'chunks') or hasattr(spine_membranes, 'chunks')
            if _cores_will_per_cluster and _cores_is_zarr:
                # Zarr handles flow straight through. function's small-volume
                # branch (cores | membranes) won't fire because we cleared the
                # 2 GB gate above.
                _cores_np = spine_cores
                _membranes_np = spine_membranes
                logger.info("    Passing zarr cores+membranes through to per-cluster watershed (no materialization)")
            else:
                _cores_np = (np.array(spine_cores) if not isinstance(spine_cores, np.ndarray) else spine_cores).astype(bool)
                _membranes_np = (np.array(spine_membranes) if not isinstance(spine_membranes, np.ndarray) else spine_membranes).astype(bool)
            # Phase H — return spine_labels as a zarr handle on the chunked
            # path. Saves ~22 GB int32 numpy materialization at T7 (and is
            # the only viable form at 55 GB+ where the materialized
            # spine_labels would exceed available host RAM). The downstream
            # per_spine_regionprops bbox compute that previously hung 4+
            # hours on dual-zarr inputs is now O(N_voxels) per slab via
            # find_objects_streaming — see ChunkedProcessing.per_spine_regionprops.
            # Numpy return is preserved for the non-chunked path so T2/T4
            # numpy regressions stay byte-identical.
            _ph_workspace = workspace if use_chunked else None
            _ph_result_name = 'spine_labels' if use_chunked else None
            spine_labels = spine_detection_cores_and_membranes(
                _cores_np, _membranes_np,
                settings.remove_touching_boarders, logger, settings,
                workspace=_ph_workspace,
                result_name=_ph_result_name)
            # Phase H state: zarr return only when use_chunked AND the
            # function-internal use_per_cluster threshold (>2 GB int64) hit.
            # Logging here separates "Phase H attempted" from "Phase H active".
            if hasattr(spine_labels, 'chunks'):
                logger.info(f"    Phase H ACTIVE — spine_labels returned as zarr handle "
                            f"(shape={tuple(spine_labels.shape)}, dtype={spine_labels.dtype}).")
            elif _ph_workspace is not None:
                logger.info(f"    Phase H attempted but volume below per-cluster "
                            f"threshold — spine_labels is numpy ({spine_labels.nbytes / 1e9:.2f} GB).")
            del _cores_np, _membranes_np

        # now we have spine_labels — free large intermediates no longer needed
        if settings.model_type == 4:
            del spine_cores, spine_membranes
            gc.collect()

        # Defer labeled_dendrites restore until after neck processing (saves 20 GB during necks).
        # dendrites_mask is still needed for neck processing — keep as zarr
        # handle. All consumers below
        # use slab/bbox reads (L1127 _dm_slab[_z:_ze], L1558 dendrites_mask
        # [_sz:_sze], L3230 dendrites_mask[dsl]) — none require contiguous
        # numpy. array_to_numpy here previously alloc'd ~520 MB at T4 stream-
        # low / 27.7 GB at T_LARGE for no functional benefit.
        if _dend_mask_spilled:
            dendrites_mask = _workspace_restore_zarr(workspace, 'dendrites_mask', 'dendrites_mask')
        # skeleton consumers (L1554 skeleton[_sz:_sze], L2433 skeleton[_z:_ze])
        # also use slab reads — keep as zarr.
        if _skeleton_deferred:
            skeleton = _workspace_restore_zarr(workspace, 'skeleton', 'skeleton')
        if _soma_dist_spilled:
            soma_distance = workspace.open_array('soma_distance_spill')
        # Defer restore of necks/spines/soma — only load when actually needed
        # to avoid holding 15 GB of arrays that may not be used (e.g., neck_generation=False)
        # necks consumers: nbytes check + connected_components_streaming(zarr) — both zarr-safe.
        if _necks_spilled and settings.neck_generation:
            necks = _workspace_restore_zarr(workspace, 'necks', 'necks_spill')
        elif _necks_spilled:
            necks = np.zeros(1, dtype=np.uint8)  # placeholder — not used
        # soma consumer at L1558+ uses slab reads — keep zarr.
        if _soma_spilled:
            soma = _workspace_restore_zarr(workspace, 'soma', 'soma_spill')

        # logger.info(f" {np.max(spine_labels)}.")
        # max_label = np.max(spine_labels)

        # Measurements
        # Create 4D Labels
        # imwrite(locations.tables + 'Detected_spines.tif', spine_labels.astype(np.uint16), imagej=True, photometric='minisblack',
        #        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})
        time_spine = time.time()

        # spine_table, spines_filtered = spine_measurementsV2(image, spine_labels, 1, 0, settings.neuron_channel, dendrite_distance, soma_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, filename, logger)

        spine_table, spines_filtered = initial_spine_measurements(image, spine_labels, 1, 0, settings.neuron_channel,
                                                                  dendrite_distance,
                                                                  settings.neuron_spine_size,
                                                                  settings.neuron_spine_dist,
                                                                  settings, locations, filename, logger)

        del spine_labels
        gc.collect()
        logger.info(f"     Time taken for initial spine detection: {time.time() - time_spine:.2f} seconds")

        # Tiny-head backstop: drop stray 1-voxel spine-head labels before neck
        # association. Observed failure mode (T2, neck 19/29): a 1-voxel head
        # sitting along a real neck corridor claimed half the neck in Pass 1,
        # splitting one real neck into two labels. The µm³ filter in
        # initial_spine_measurements doesn't always catch these (watershed,
        # orphan-membrane recovery, and erosion paths can leave 1-voxel
        # residuals). Neck channel is never touched.
        _min_head_vox = int(getattr(settings, 'min_spine_head_voxels', 2))
        _n_tiny_dropped = 0
        _tiny_dropped_records = []
        if _min_head_vox > 1:
            spines_filtered, spine_table, _n_tiny_dropped, _tiny_dropped_records = filter_tiny_spine_heads(
                spines_filtered, spine_table, _min_head_vox, logger)
            if _tiny_dropped_records:
                _tiny_csv = pd.DataFrame(
                    _tiny_dropped_records, columns=['label', 'voxel_count'])
                _tiny_csv['min_voxels_threshold'] = _min_head_vox
                _tiny_csv.to_csv(
                    locations.tables + filename + '_tiny_head_dropped_labels.csv',
                    index=False)

        # F1 invariant: capture label-count baseline after all pre-measurement
        # filters that can remove labels from `spines_filtered`. Downstream,
        # only spurious-necks mode='both' removes spine labels before
        # spine_vox_measurements; see pre-measurement check below.
        try:
            _f1_baseline = len(chunked.safe_unique(spines_filtered)) - 1  # exclude label 0
        except Exception:
            _f1_baseline = None

        # Create 4D Labels
        # imwrite(locations.tables + 'Detected_spines_filtered.tif', spines_filtered.astype(np.uint16), imagej=True, photometric='minisblack',
        #        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})                                                  #soma_mask, soma_distance, )

        vram.checkpoint("before_neck_analysis", logger)
        logger.info("\n    Calculating spine necks...")
        time_neck_connection = time.time()

        # For large volumes with neck generation enabled, spill arrays not
        # needed during neck processing to free RAM for the heavy temporaries
        # (neck_labels + associate + extend = ~65 GB).
        # Skip spilling when neck_generation is disabled — no heavy ops to free for.
        _neck_soma_dist_spilled = False
        _neck_skeleton_spilled = False
        _neck_soma_spilled = False
        _neck_dend_labels_spilled = False
        _neck_spines_spilled = False
        if use_chunked and settings.neck_generation:
            if soma_present and isinstance(soma_distance, np.ndarray):
                workspace.numpy_to_zarr('soma_distance_neck_spill', soma_distance)
                del soma_distance
                _neck_soma_dist_spilled = True
            if isinstance(skeleton, np.ndarray):
                del skeleton
                _neck_skeleton_spilled = True
            if isinstance(soma, np.ndarray):
                del soma
                _neck_soma_spilled = True
            # labeled_dendrites not needed during neck processing (only dendrites_mask)
            if isinstance(labeled_dendrites, np.ndarray):
                workspace.numpy_to_zarr('labeled_dendrites_neck_spill', labeled_dendrites)
                del labeled_dendrites
                _neck_dend_labels_spilled = True
            # spines + spine_labels not needed during neck processing
            if isinstance(spines, np.ndarray):
                del spines
                _neck_spines_spilled = True
            gc.collect()
            # Force Windows to release freed pages to OS
            try:
                import ctypes
                ctypes.windll.kernel32.SetProcessWorkingSetSize(
                    ctypes.windll.kernel32.GetCurrentProcess(),
                    ctypes.c_size_t(-1), ctypes.c_size_t(-1))
            except Exception:
                pass

        # disable neck analysis for very large datasets
        if settings.neck_generation == False:
            logger.info(f"     Image shape is {spines_filtered.shape}. Neck analysis has been disabled.")
            # Allocate as zarr when
            # spines_filtered is zarr-backed (chunked path). For T_LARGE the
            # legacy np.zeros_like would alloc 27.7 GB numpy on the disabled-
            # neck branch — same dtype as spines_filtered.
            if hasattr(spines_filtered, 'chunks') and workspace is not None:
                connected_necks = workspace.create_array(
                    'connected_necks', spines_filtered.shape,
                    dtype=spines_filtered.dtype,
                    chunks=chunked.STREAMING_CHUNKS)
            else:
                connected_necks = np.zeros_like(spines_filtered)
        else:
            if settings.model_type >= 3:
                # L2 testing: settings.neck_streaming_threshold_bytes lets
                # profiles lower this gate so T4/T6 exercise the streaming-CC
                # neck branch without a synthetic fixture. Production default
                # is 2e9 (matches T7+ natively).
                _neck_thresh = int(getattr(settings, 'neck_streaming_threshold_bytes', int(2e9)))
                if use_chunked and necks.nbytes > _neck_thresh:
                    # Write CC output directly to a
                    # workspace zarr and keep `neck_labels` as a zarr handle.
                    # Downstream consumers (the slab-loop at the if use_chunked
                    # branch below, and associate_spines_with_necks_gpu via the
                    # streaming threshold) already accept zarr — no need to
                    # materialize. Prior code allocated a full-volume int32
                    # numpy here (~1 GB at T4 stream-low, ~111 GB at T_LARGE).
                    _neck_zarr = workspace.create_array(
                        'neck_labels_cc', necks.shape, dtype=np.int32,
                        chunks=chunked.STREAMING_CHUNKS)
                    _n_necks, _ = chunked.connected_components_streaming(
                        necks, _neck_zarr, logger=logger)
                    del necks  # binary no longer needed — neck_labels replaces it
                    gc.collect()
                    # Force Windows to release freed pages — gc.collect() frees Python
                    # objects but doesn't return virtual memory to the OS, blocking
                    # the next large allocation even when RAM is nominally available.
                    try:
                        import ctypes
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(
                            ctypes.windll.kernel32.GetCurrentProcess(),
                            ctypes.c_size_t(-1), ctypes.c_size_t(-1))
                    except Exception:
                        pass
                    neck_labels = _neck_zarr  # zarr handle, slab-iterated downstream
                else:
                    neck_labels = measure.label(necks)

                # associate spines with necks
                logger.info("     Associating spines with necks...")
                neck_labels_updated = associate_spines_with_necks_gpu(
                    spines_filtered, neck_labels, logger,
                    workspace=workspace, use_chunked=use_chunked,
                    settings=settings)
                # save this as a tif
                # imwrite(locations.Vols + 'Detected_necks.tif', neck_labels_updated.astype(np.uint16), imagej=True,
                #           photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX'})

                # unassociated neck voxels — build slab-by-slab into zarr when chunked
                # (otherwise 5 GB bool on T7, 50 GB on 100 GB images)
                if use_chunked:
                    _nk_shape = neck_labels.shape
                    _nk_row = int(_nk_shape[1]) * int(_nk_shape[2])
                    _nk_slab = max(1, int(2 * 1024**3 / _nk_row))
                    _neck_mask = workspace.create_array(
                        'neck_mask_zarr', _nk_shape, dtype=np.uint8,
                        chunks=chunked.STREAMING_CHUNKS)
                    remaining_necks = workspace.create_array(
                        'remaining_necks_zarr', _nk_shape, dtype=np.uint8,
                        chunks=chunked.STREAMING_CHUNKS)
                    _neck_total_voxels = 0
                    for _z in range(0, _nk_shape[0], _nk_slab):
                        _ze = min(_z + _nk_slab, _nk_shape[0])
                        _nm_slab = (neck_labels[_z:_ze] > 0).astype(np.uint8)
                        _neck_mask[_z:_ze] = _nm_slab
                        _neck_total_voxels += int(_nm_slab.sum())
                        remaining_necks[_z:_ze] = (
                            _nm_slab.astype(bool) & (neck_labels_updated[_z:_ze] == 0)
                        ).astype(np.uint8)
                        del _nm_slab
                else:
                    _neck_mask = neck_labels > 0
                    _neck_total_voxels = int(_neck_mask.sum())
                    remaining_necks = _neck_mask & (neck_labels_updated == 0)
                # Free neck_labels (22 GB), labeled_dendrites (20 GB), image (10 GB)
                # before creating target — saves ~52 GB before extend_objects_GPU.
                del neck_labels
                _image_freed = False
                _image_spilled = False
                _labeled_dend_freed = False
                if use_chunked:
                    if isinstance(labeled_dendrites, np.ndarray):
                        del labeled_dendrites
                        _labeled_dend_freed = True
                    if isinstance(image, np.ndarray):
                        # Spill image to zarr so we can restore it as a zarr handle
                        # after neck generation. Avoids TIF re-read (~1 hr at 100 GB)
                        # and keeps downstream per_spine_regionprops reading
                        # per-spine bbox crops directly from zarr.
                        # Shape normalization: we want the spilled zarr to be
                        # 3D for single-channel (so a zarr `arr[z0:z1, y0:y1, x0:x1]`
                        # is a per-spine crop), or 4D for multi-channel.
                        try:
                            if image.ndim == 4 and image.shape[1] == 1:
                                workspace.numpy_to_zarr(
                                    'image', image[:, 0, :, :],
                                    chunks=chunked.STREAMING_CHUNKS)
                            elif image.ndim == 3:
                                workspace.numpy_to_zarr(
                                    'image', image,
                                    chunks=chunked.STREAMING_CHUNKS)
                            else:
                                # 4D multi-channel — spill full 4D; downstream
                                # materializes per channel (regression at 100 GB
                                # multi-channel; rare for spine datasets).
                                workspace.numpy_to_zarr(
                                    'image', image,
                                    chunks=(chunked.STREAMING_CHUNKS[0], 1,
                                            chunked.STREAMING_CHUNKS[1],
                                            chunked.STREAMING_CHUNKS[2]))
                            _image_spilled = True
                        except Exception as _img_spill_err:
                            logger.warning(f"    image spill to zarr failed: {_img_spill_err}. "
                                           "Will fall back to TIF re-read.")
                        del image
                        _image_freed = True
                gc.collect()

                # Target for neck-path extension: voxels reachable from dendrite or
                # remaining necks, excluding already-assigned neck voxels.
                # FIX: explicit boolean logic avoids int32 arithmetic overflow.
                if use_chunked:
                    target = workspace.create_array(
                        'neck_target_zarr', _nk_shape, dtype=np.uint8,
                        chunks=chunked.STREAMING_CHUNKS)
                    for _z in range(0, _nk_shape[0], _nk_slab):
                        _ze = min(_z + _nk_slab, _nk_shape[0])
                        _dm_slab = (dendrites_mask[_z:_ze] > 0)
                        _rn_slab = remaining_necks[_z:_ze] > 0
                        _nl_slab = (neck_labels_updated[_z:_ze] == 0)
                        target[_z:_ze] = ((_dm_slab | _rn_slab) & _nl_slab).astype(np.uint8)
                        del _dm_slab, _rn_slab, _nl_slab
                else:
                    target = ((dendrites_mask > 0) | (remaining_necks > 0)) & (neck_labels_updated == 0)

                gc.collect()

                logger.info("     Extending necks to dendrites...")
                # Pass spines_filtered as obstacle reference so OTHER spine
                # heads are forbidden during this neck's path extension. Without
                # this, the path can route alongside a neighbour's head because
                # `objects` (neck_labels_updated) doesn't contain head voxels.
                extended_necks = extend_objects_GPU(neck_labels_updated, target, neuron, settings, locations,
                                                    logger,
                                                    workspace=workspace if use_chunked else None,
                                                    result_array_name='_extend_necks_result',
                                                    obstacle_labels=spines_filtered)

                # Slab size targeting ~2 GB per slab for int32 full-volume arrays.
                _ncshape = extended_necks.shape
                _row_bytes = int(_ncshape[1]) * int(_ncshape[2]) * 4
                _slab = max(1, int(2 * 1024**3 / _row_bytes))

                # Zero extended_necks wherever neck_labels_updated has a value.
                # Slab-by-slab np.where (slab-sized temp, not full-volume temp).
                for _z in range(0, _ncshape[0], _slab):
                    _ze = min(_z + _slab, _ncshape[0])
                    extended_necks[_z:_ze] = np.where(
                        neck_labels_updated[_z:_ze] == 0,
                        extended_necks[_z:_ze], 0)

                # Cap extension length at spine_dist * 1.5 from owning spine
                _spine_dist_um = settings.neuron_spine_dist * settings.input_resXY
                _max_neck_ext_um = getattr(settings, 'max_neck_length', _spine_dist_um * 1.5)
                _max_neck_px = int(_max_neck_ext_um / settings.input_resXY)

                if use_chunked:
                    # Chunked GPU EDT with isotropic voxel halo (spacing=(1,1,1)) matches
                    # original voxel-unit ndimage.distance_transform_edt semantics.
                    _spine_mask_zarr = workspace.create_array(
                        'spine_mask_for_cap', _ncshape, dtype=np.uint8,
                        chunks=chunked.STREAMING_CHUNKS)
                    for _z in range(0, _ncshape[0], _slab):
                        _ze = min(_z + _slab, _ncshape[0])
                        _spine_mask_zarr[_z:_ze] = (spines_filtered[_z:_ze] > 0).astype(np.uint8)
                    _spine_dist_zarr = workspace.create_array(
                        'spine_dist_for_cap', _ncshape, dtype=np.float32,
                        chunks=chunked.STREAMING_CHUNKS)
                    chunked.distance_transform_gpu_chunked(
                        _spine_mask_zarr, _spine_dist_zarr,
                        spacing=(1.0, 1.0, 1.0),
                        max_dist=float(_max_neck_px), logger=logger)
                    _n_before = 0
                    _n_after = 0
                    for _z in range(0, _ncshape[0], _slab):
                        _ze = min(_z + _slab, _ncshape[0])
                        _ext_slab = extended_necks[_z:_ze]
                        _n_before += int(np.count_nonzero(_ext_slab))
                        _dist_slab = np.asarray(_spine_dist_zarr[_z:_ze])
                        _ext_slab[_dist_slab > _max_neck_px] = 0
                        _n_after += int(np.count_nonzero(_ext_slab))
                        del _dist_slab
                else:
                    _spine_dist_map = ndimage.distance_transform_edt(spines_filtered == 0)
                    _n_before = np.count_nonzero(extended_necks)
                    extended_necks[_spine_dist_map > _max_neck_px] = 0
                    _n_after = np.count_nonzero(extended_necks)
                    del _spine_dist_map

                if _n_before > _n_after:
                    logger.info(f"       Neck extension cap: {_max_neck_ext_um:.1f} \u00b5m ({_max_neck_px} px), "
                                f"removed {_n_before - _n_after} voxels ({(_n_before-_n_after)/max(1,_n_before)*100:.0f}%)")

                # Merge extended_necks into neck_labels_updated in-place slab-by-slab
                # so the combined source can be passed to the next associate call
                # without a 20 GB (extended_necks + neck_labels_updated) temporary.
                # Regions are already disjoint after the zeroing above, so np.maximum
                # is equivalent to + but avoids any risk of label-ID addition collisions.
                for _z in range(0, _ncshape[0], _slab):
                    _ze = min(_z + _slab, _ncshape[0])
                    neck_labels_updated[_z:_ze] = np.maximum(
                        neck_labels_updated[_z:_ze], extended_necks[_z:_ze])
                # Free 20 GB before the next associate call.
                del extended_necks
                gc.collect()

                # Associate remaining unassigned necks with nearest extended/associated neck
                logger.info("     Associating extended necks with remaining necks...")
                _remaining_assigned = associate_spines_with_necks_gpu(
                    neck_labels_updated,
                    remaining_necks, logger,
                    workspace=workspace, use_chunked=use_chunked,
                    settings=settings)

                # Spill _remaining_assigned (20 GB numpy on T7) to zarr so we can
                # free the numpy buffer before allocating connected_necks. Saves
                # 20 GB of peak RAM at the critical pre-mesh point.
                if use_chunked:
                    _ra_zarr = workspace.create_array(
                        'remaining_assigned_zarr', _ncshape,
                        dtype=_remaining_assigned.dtype,
                        chunks=chunked.STREAMING_CHUNKS)
                    for _z in range(0, _ncshape[0], _slab):
                        _ze = min(_z + _slab, _ncshape[0])
                        _ra_zarr[_z:_ze] = _remaining_assigned[_z:_ze]
                    del _remaining_assigned
                    gc.collect()
                    _remaining_assigned = _ra_zarr

                # Build connected_necks slab-by-slab. Avoids two 20 GB np.maximum
                # temporaries + the spine-overlap zeroing temporary. Source for
                # _remaining_assigned is zarr (chunked) or numpy (non-chunked).
                # np.zeros_like on a zarr-backed
                # neck_labels_updated would alloc full-volume int32 numpy
                # (~111 GB at T_LARGE). When chunked, write into a workspace
                # zarr instead. Downstream consumers (bridges, drop filters,
                # final measurements) already accept zarr for connected_necks.
                if hasattr(neck_labels_updated, 'chunks') and workspace is not None:
                    connected_necks = workspace.create_array(
                        'connected_necks', neck_labels_updated.shape,
                        dtype=neck_labels_updated.dtype,
                        chunks=chunked.STREAMING_CHUNKS)
                else:
                    connected_necks = np.zeros_like(neck_labels_updated)
                for _z in range(0, _ncshape[0], _slab):
                    _ze = min(_z + _slab, _ncshape[0])
                    _slab_out = np.maximum(np.asarray(neck_labels_updated[_z:_ze]),
                                            np.asarray(_remaining_assigned[_z:_ze]))
                    _slab_out[np.asarray(spines_filtered[_z:_ze]) > 0] = 0
                    connected_necks[_z:_ze] = _slab_out
                    del _slab_out
                del _remaining_assigned
                # Free neck_labels_updated BEFORE the bridge call — the bridge
                # collects per-label bboxes and keeps a few per-spine crops in
                # RAM, so freeing the ~22 GB numpy buffer (on T7) here avoids
                # stacking two full-volume int32 arrays simultaneously.
                # (This is a T7 OOM risk.)
                del neck_labels_updated
                gc.collect()
                # Do NOT mask with dendrites_mask — necks must touch the dendrite
                # surface to form a complete spine→neck→dendrite connection.

                # Bridge 1-voxel gaps between spine heads and their assigned
                # necks (nnU-Net transition-class artifact; ~10% of spines).
                # Phase 1 — morphological closing (cheap, handles 1-vox gaps).
                # Writes only into background voxels — never overwrites other
                # labels or dendrite.
                connected_necks, _still_split = chunked.bridge_spine_neck_gaps(
                    connected_necks, spines_filtered, dendrites_mask, logger)

                # Phase 2 — signal-aware pathfinding for remaining gaps. Reuses
                # pathfinding_v3b (distance + gamma-weighted intensity) per
                # still-split spine to bridge multi-voxel gaps only where the
                # neuron fluorescence signal supports a connection.
                # `neuron` is a view into image (see L362); `del image` at L806
                # doesn't invalidate it because the view holds the memory alive.
                if _still_split:
                    connected_necks = bridge_spine_neck_via_pathfinding(
                        connected_necks, spines_filtered, dendrites_mask,
                        _still_split, neuron,
                        settings, locations, logger)

                # Slab-by-slab diagnostic (avoids 5 GB bool temp)
                _nnunet_preserved = 0
                for _z in range(0, _ncshape[0], _slab):
                    _ze = min(_z + _slab, _ncshape[0])
                    _nnunet_preserved += int(np.sum(
                        _neck_mask[_z:_ze] & (connected_necks[_z:_ze] > 0)))
                _nnunet_total = _neck_total_voxels
                logger.info(f"       nnU-Net neck preservation: {_nnunet_preserved}/{_nnunet_total} "
                            f"({_nnunet_preserved/max(1,_nnunet_total)*100:.1f}%)")

                # Drop disconnected neck fragments: after both bridges, if a
                # label still has multiple connected components in connected_necks,
                # Pass 2 assigned a dendrite-side fragment by proximity without
                # a real corridor to the spine. Keep only the component that is
                # 26-adjacent to the spine head.
                if getattr(settings, 'drop_disconnected_neck_fragments', True):
                    connected_necks = drop_disconnected_neck_fragments(
                        connected_necks, spines_filtered, logger)

                # Drop wrong-direction neck fragments: sibling case — a label
                # has multiple neck CCs that are all adjacent to the spine
                # head (so the combined object is 1 CC and the function above
                # leaves it alone), but one of them goes AWAY from the dendrite.
                # Keep only the CC that reaches closest to the dendrite. Biol.
                # necks monotonically approach the dendrite; wrong-direction
                # fragments are artifacts of tiny heads picking up stray
                # Pass-2 neck assignments from occluded neighbours.
                if getattr(settings, 'drop_wrong_direction_neck_fragments', True):
                    connected_necks = drop_wrong_direction_neck_fragments(
                        connected_necks, dendrite_distance, settings, logger)

                # Spurious-neck filter: detect (and optionally remove) necks
                # that have (a) >length_min_um extent, (b) <support_min fraction
                # of nnU-Net backing, and (c) <intensity_ratio_min fluorescence
                # vs. head. YAML-configurable thresholds; on by default.
                connected_necks, _spurious_flagged_labels = flag_spurious_necks(
                    connected_necks, spines_filtered, _neck_mask, neuron,
                    settings, locations, logger)

                # _neck_mask deletion deferred — needed by recover_filopodia_from_orphan_necks
                # after second_pass_annotation. See filopodia recovery block below.
                # neck_labels_updated was freed earlier (before bridge call).
                gc.collect()

                # The extend_objects_GPU pathfinding should bridge neck→dendrite gaps.
                # If gaps remain, the extension voxels exist in extended_necks but may
                # overlap with dendrite voxels. Since we no longer mask with dendrites_mask,
                # these bridge voxels should be preserved in connected_necks.
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
                connected_necks = extend_objects_GPU(spines_filtered, dendrites_mask, neuron, settings, locations,
                                                     logger,
                                                     workspace=workspace if use_chunked else None,
                                                     result_array_name='_extend_spines_result')

        # time for neck connection
        logger.info(f"     Time taken for neck generation: {time.time() - time_neck_connection:.2f} seconds")
        flush_gpu()
        vram.checkpoint("after_neck_analysis", logger)

        # Restore arrays needed for post-neck analysis
        # Phase A: on chunked path keep labeled_dendrites & skeleton as zarr handles.
        # Full-volume numpy would peak at 22 GB + 5 GB which OOMs on T7 (10 GB image).
        _labeled_dend_freed = locals().get('_labeled_dend_freed', False)
        if _labeled_dend_freed or (_dend_labels_spilled and
                                    not isinstance(labeled_dendrites, np.ndarray)):
            if use_chunked:
                labeled_dendrites = workspace.open_array('labeled_dendrites')
            else:
                labeled_dendrites = workspace.array_to_numpy('labeled_dendrites')
        # dendrites_mask: on chunked path we never rebuild the 5 GB bool mask; the
        # only consumer (neuron-volume slab loop below) reads labeled_dendrites > 0
        # slab-by-slab instead. Setting to None keeps the later `del dendrites_mask`
        # compatible (guarded below).
        _dend_mask_freed = locals().get('_dend_mask_freed', False)
        if _dend_mask_freed:
            if use_chunked:
                dendrites_mask = None
            else:
                dendrites_mask = (labeled_dendrites > 0).astype(np.uint8)
        # image: freed before extend. Phase D spilled to zarr when use_chunked —
        # restore as zarr handle (downstream per_spine_regionprops reads crops).
        # Fall back to TIF re-read if the spill was skipped or failed.
        if locals().get('_image_freed', False):
            if use_chunked and locals().get('_image_spilled', False):
                image = workspace.open_array('image')
                logger.info("    Restored image as zarr handle from workspace")
            else:
                from tifffile import imread as _tif_read
                _tif_path = os.path.join(locations.input_dir, filename + '.tif')
                image = _tif_read(_tif_path)
                image = check_image_shape(image, logger)
                logger.info(f"    Reloaded image from {_tif_path}")
        elif _neck_dend_labels_spilled:
            _neck_spill_path = os.path.join(workspace.base_dir, 'labeled_dendrites_neck_spill.zarr', '.zarray')
            _spill_exists = os.path.isfile(_neck_spill_path)
            if use_chunked:
                if _spill_exists:
                    labeled_dendrites = workspace.open_array('labeled_dendrites_neck_spill')
                else:
                    labeled_dendrites = workspace.open_array('labeled_dendrites')
            else:
                if _spill_exists:
                    labeled_dendrites = workspace.array_to_numpy('labeled_dendrites_neck_spill')
                else:
                    labeled_dendrites = workspace.array_to_numpy('labeled_dendrites')
        if _neck_spines_spilled:
            # On chunked path keep spines as zarr — 5 GB numpy at T7, 50 GB at
            # 100 GB scale. MIP (via io.create_mip_and_save_multichannel_tiff)
            # reads zarr slab-by-slab; _workspace_restore is only used for non-
            # chunked callers.
            if use_chunked:
                _spill_path = os.path.join(workspace.base_dir, 'spines_spill.zarr', '.zarray')
                if os.path.isfile(_spill_path):
                    spines = workspace.open_array('spines_spill')
                else:
                    spines = workspace.open_array('spines')
            else:
                spines = _workspace_restore(workspace, 'spines', 'spines_spill')
        if _neck_soma_dist_spilled:
            soma_distance = workspace.open_array('soma_distance_neck_spill')
        if _neck_skeleton_spilled:
            if use_chunked:
                skeleton = workspace.open_array('skeleton')
            else:
                skeleton = workspace.array_to_numpy('skeleton')
        # soma restore deferred to just before geodesic distance calc (after
        # dendrites_mask freed) — restoring everything eagerly OOMs on T7.
        # logger.info(connected_necks.shape)
        # Remove spines that are connected to necks
        # connected_necks[spine_labels > 0] = 0

        # Second pass analysis
        # take spines(necks and spines filtered) create subvolumes, mask with a 5 pixel dilation and pass through the 2nd pass model
        if settings.second_pass:
            logger.info("     Running spine refinement...")
            if settings.refinement_model_path != None:
                spines_filtered, connected_necks = second_pass_annotation(spines_filtered, connected_necks,
                                                                          dendrites_mask, neuron,
                                                                          locations, settings, logger)
            else:
                logger.info(
                    "Spine refinement model not found. Skipping second pass annotation. Please update location of second pass model in settings file to enable second pass annotation.")

        # Filopodia recovery — runs AFTER all spine→neck association is complete
        # (Pass 1, Pass 2, pathfinding bridges, drop-disconnected, spurious-flag,
        # second-pass refinement). Any nnU-Net neck CC with zero overlap in
        # connected_necks at this point is a genuine orphan — preserve as filopodium
        # unless it's adjacent to an existing spine head. Off by default; enable
        # via settings.recover_filopodia for the new behaviour.
        if ('_neck_mask' in locals() and settings.model_type >= 3
                and getattr(settings, 'recover_filopodia', False)):
            logger.info("\n    Recovering filopodia from orphan nnU-Net neck CCs...")
            spines_filtered, connected_necks, _filopodia_metadata = (
                recover_filopodia_from_orphan_necks(
                    spines_filtered, connected_necks, _neck_mask,
                    dendrite_distance, settings.neuron_spine_size,
                    settings.neuron_spine_dist,
                    settings, workspace, use_chunked, logger))
            _filopodia_labels = set(_filopodia_metadata.keys())
            # Append filopodia rows to spine_table with label + head centroid
            # (x, y, z) so they match normal spines' schema. Downstream merges
            # (dendrite_id, head/whole/neck measurements, mesh) will populate
            # the remaining columns via outer join on label.
            if _filopodia_metadata:
                _filo_rows = pd.DataFrame([
                    {'label': lbl, 'x': int(round(meta['x'])),
                     'y': int(round(meta['y'])), 'z': int(round(meta['z']))}
                    for lbl, meta in _filopodia_metadata.items()
                ])
                spine_table = pd.concat([spine_table, _filo_rows], ignore_index=True)
                logger.info(
                    f"       Appended {len(_filo_rows)} filopodia rows to spine_table "
                    f"(new total: {len(spine_table)}).")
            vram.checkpoint("after_filopodia_recovery", logger)

        # Partial-spine detection: find spines whose neck exists but does NOT
        # reach the dendrite (most commonly when a nearby spine occludes the
        # neck path). Runs AFTER all neck processing (drop_disconnected,
        # spurious, filopodia) so we screen the final connected_necks state.
        if settings.model_type >= 3 and 'connected_necks' in locals():
            _partial_thresh = float(getattr(
                settings, 'partial_spine_dist_threshold_vox', 2.0))
            _partial_spine_labels = detect_partial_spines(
                connected_necks, spines_filtered, dendrite_distance,
                _partial_thresh, logger)
            # Exclude filopodia from partial-spine set — filopodia are their
            # own category and already have correct neck semantics.
            _partial_spine_labels -= _filopodia_labels

            # Unite spurious-flagged spines (when neck was dropped but head
            # retained) into the partial-spine set. These are structurally
            # the same case: head with no reachable neck. Without this, they
            # fall through to categorize_spine and get misleading 'stubby'
            # classifications despite having originally had a (rejected) neck.
            _sp_mode = getattr(settings, 'spurious_exclude_mode', 'neck_only')
            _sp_keep_head = bool(getattr(
                settings, 'spurious_keep_head_if_flagged', False))
            _sp_head_retained = (_sp_mode == 'neck_only' or
                                  (_sp_mode == 'both' and _sp_keep_head))
            if _sp_head_retained and '_spurious_flagged_labels' in locals():
                _sp_as_partial = (set(_spurious_flagged_labels)
                                   - _filopodia_labels
                                   - _partial_spine_labels)
                if _sp_as_partial:
                    _partial_spine_labels |= _sp_as_partial
                    logger.info(
                        f"       Classified {len(_sp_as_partial)} "
                        f"spurious-flagged spines as partial-spine "
                        f"(neck dropped, head retained).")

            if _partial_spine_labels:
                logger.info(
                    f"       Partial spines detected: {len(_partial_spine_labels)} "
                    f"spines with necks not reaching dendrite "
                    f"(min dd > {_partial_thresh:.1f} vox, after filopodia "
                    f"exclusion). Will be kept with neck metrics zeroed "
                    f"(keep_partial_spines=True) or dropped (False).")
            else:
                logger.info("       No partial spines detected.")

        # 3D overlap audit: report (do NOT auto-fix) any voxel-level overlaps
        # between spines_filtered / connected_necks / labeled_dendrites. These
        # should all be disjoint by construction — any non-zero count indicates
        # a pipeline bug worth investigating. Reported only; user explicitly
        # requested no auto-correction to avoid over-trimming gaps.
        if getattr(settings, 'audit_3d_overlaps', True):
            try:
                _audit_3d_label_overlaps(
                    spines_filtered, connected_necks, labeled_dendrites, logger)
            except Exception as _aud_err:
                logger.warning(f"    3D overlap audit failed: {_aud_err}")

        # Multi-head spine detection — group physically connected heads under
        # one parent ID. Runs AFTER all neck processing (drop_disconnected,
        # spurious, filopodia, partial-spine flag) so the final state of
        # connected_necks is screened. _neck_mask is still alive here (we
        # delete it below). Default OFF — opt-in via settings.
        _multi_head_parent_map = {}
        _multi_head_groups = {}
        if getattr(settings, 'detect_multi_head_spines', False):
            logger.info("   Multi-head spine detection (physical connectivity)...")
            _nm_for_mh = locals().get('_neck_mask', None)
            try:
                _multi_head_parent_map, _multi_head_groups = detect_multi_head_spines(
                    spines_filtered, connected_necks, _nm_for_mh, logger,
                    workspace=workspace if use_chunked else None)
            except Exception as _mh_err:
                logger.warning(f"    Multi-head detection failed: {_mh_err}; continuing.")
                _multi_head_parent_map = {}
                _multi_head_groups = {}

        # Release _neck_mask now that filopodia recovery + multi-head
        # detection are done with it.
        if '_neck_mask' in locals():
            del _neck_mask
            gc.collect()

        # log_memory_usage(logger)

        # calulating dendrite statistics ###### CLEAN UP WHOLE NEURON AND KEEP DEND STATS
        # logger.info("\n    Calculating dendrite statistics...")
        # originally whole neuron stats but now calculating dendrite specific

        logger.info("    Calculating neuron statistics...")
        # Slab-by-slab sum. When chunked, dendrites_mask is None — derive the
        # mask on the fly from labeled_dendrites > 0 to avoid holding 5 GB.
        neuron_length = 0
        neuron_volume = 0
        for _sz in range(0, skeleton.shape[0], 64):
            _sze = min(_sz + 64, skeleton.shape[0])
            neuron_length += int(np.sum(np.asarray(skeleton[_sz:_sze]) == 1))
            if dendrites_mask is None:
                neuron_volume += int(np.sum(np.asarray(labeled_dendrites[_sz:_sze]) > 0))
            else:
                neuron_volume += int(np.sum(dendrites_mask[_sz:_sze] == 1))

        if dendrites_mask is not None:
            del dendrites_mask

        logger.info("    Calculating dendrite statistics...")
        # get dendrite lengths and volumes as dictoinaries
        dendrite_lengths, dendrite_volumes, skeleton_coords, labeled_skeletons = calculate_dendrite_length_and_volume_fast(
            labeled_dendrites, skeleton, logger)

        # finished calculating dendrite statistics
        logger.info("      Complete.")

        logger.info("    Calculating spine dendrite ID and geodesic distance...")
        # For large volumes, skip topology-preserving skeleton downsampling
        # (it does a full-volume int32 convolution = 20 GB for T7). The simpler
        # KDTree sampling path works well enough for spine-to-dendrite matching.
        if _neck_soma_spilled:
            # On chunked path keep soma as zarr; calculate_dend_ID_and_geo_distance
            # now slab-iterates np.argwhere on zarr soma_vol (Phase E).
            if use_chunked:
                _soma_spill_path = os.path.join(workspace.base_dir, 'soma_spill.zarr', '.zarray')
                if os.path.isfile(_soma_spill_path):
                    soma = workspace.open_array('soma_spill')
                else:
                    soma = workspace.open_array('soma')
            else:
                soma = _workspace_restore(workspace, 'soma', 'soma_spill')
        _skel_vol = skeleton if not use_chunked else None
        _soma_arg = None if not soma_present else soma
        spine_dendID_and_geodist, geodesic_distance_image = calculate_dend_ID_and_geo_distance(
            labeled_dendrites, spines_filtered, skeleton_coords, labeled_skeletons,
            filename, locations, soma_vol=_soma_arg, settings=settings,
            skeleton_volume=_skel_vol, logger=logger)

        # save spine dendrite ID and geodesic distance as csv using pands
        # spine_dendID_and_geodist.to_csv(locations.tables + 'Detected_spines_dendrite_ID_and_geodesic_distance_' + filename + '.csv', index=False)

        logger.info("     Complete.")

        # analyze whole spines
        flush_gpu()
        vram.checkpoint("before_mesh_analysis", logger)
        logger.info("\n    Performing additional mesh measurements on spines in batches on GPU...")

        # combine connected_necks and Spines_filtered
        # print max id for spines fileterd and connected_necks
        if settings.additional_logging:
            logger.info(f"      Max ID for spines filtered is {chunked.safe_max(spines_filtered)}")
            logger.info(f"      Max ID for connected necks is {chunked.safe_max(connected_necks)}")

        if settings.mesh_analysis == True:
            try:
                # When connected_necks is all zeros (neck_gen disabled), skip the
                # expensive volume arithmetic that creates 20+ GB temporaries.
                # safe_max handles zarr-backed connected_necks without materializing.
                _necks_empty = chunked.safe_max(connected_necks) == 0
                if _necks_empty:
                    _mesh_combined = spines_filtered
                elif use_chunked and (hasattr(spines_filtered, 'chunks')
                                      or hasattr(connected_necks, 'chunks')):
                    # Slab-build into zarr — mesh = spines_filtered where >0 else connected_necks.
                    # Avoids the 22 GB int32 transient at T7 and the impossible 220 GB at 100 GB.
                    _mc_shape = spines_filtered.shape
                    _mc_dtype = np.int32
                    _mesh_combined = workspace.create_array(
                        '_mesh_combined_zarr', _mc_shape, dtype=_mc_dtype,
                        chunks=chunked.STREAMING_CHUNKS)
                    _mc_bpz = int(_mc_shape[1]) * int(_mc_shape[2]) * 4
                    _mc_slz = max(1, min(64, int(2 * 1024**3 / _mc_bpz)))
                    for _mz in range(0, _mc_shape[0], _mc_slz):
                        _mze = min(_mz + _mc_slz, _mc_shape[0])
                        _sf_slab = np.asarray(spines_filtered[_mz:_mze]).astype(np.int32, copy=False)
                        _cn_slab = np.asarray(connected_necks[_mz:_mze]).astype(np.int32, copy=False)
                        _mesh_combined[_mz:_mze] = np.where(_sf_slab > 0, _sf_slab, _cn_slab)
                        del _sf_slab, _cn_slab
                else:
                    _mesh_combined = (connected_necks * ~(spines_filtered > 0)) + spines_filtered
                spine_mesh_results = analyze_spines_batch(_mesh_combined,
                                                          spines_filtered, labeled_dendrites, neuron, locations, settings,
                                                          logger,
                                                          [settings.input_resZ, settings.input_resXY, settings.input_resXY])
                if _mesh_combined is not spines_filtered:
                    del _mesh_combined
            except Exception as e:
                import traceback
                logger.error(f"    Mesh analysis failed: {e}. Continuing with voxel-based measurements only.")
                logger.error(traceback.format_exc())
                spine_mesh_results = pd.DataFrame()
                settings.mesh_analysis = False  # skip mesh merge downstream
        # spine_mesh_results.to_csv(locations.tables + 'Detected_spines_mesh_measurements_' + filename + '.csv', index=False)

        # analyze spine necks
        # logger.info("      Analyzing spine necks...")
        # neck_results = analyze_spine_necks_batch(connected_necks, logger, [settings.input_resZ,  settings.input_resXY,  settings.input_resXY])
        # neck_results.to_csv(locations.tables + 'Detected_necks_mesh_measurements_' + filename + '.csv', index=False)

        ##2025_09 - labeled_dendrites check to ensure saving for dendrite only images - but may prolong processing time or cause downstream issues
        # confirm if this is most elegant solution
        # np.max on a zarr array silently materializes the full volume (22 GB at T7).
        # Reuse the _has_dendrites flag computed earlier in this function.
        _ld_empty = (not _has_dendrites) if hasattr(labeled_dendrites, 'chunks') \
                    else (np.max(labeled_dendrites) == 0)
        if len(spine_table) == 0 and _ld_empty:
            logger.info(f"  *No spines or dendrites were analyzed for this image")

        else:
            # Restore spines from zarr for MIP (deferred to avoid holding 5 GB during measurements)
            if _spines_spilled:
                spines = _workspace_restore(workspace, 'spines', 'spines_spill')

            # MIPs are always saved (2D projections, small memory footprint).
            # dendrite_distance channel: zero out far-field beyond halo-reach to
            # hide chunk-boundary noise (raw zarr unchanged for measurements).
            _dd_clamp = getattr(settings, '_dd_viz_clamp_vox', None)
            # Multi-head grouped channel — inserted at position 5 (after
            # connected_necks, before labeled_dendrites) when feature ON.
            # Each spine head + its neck is relabeled with parent_spine_id so
            # multi-head groups appear as a single connected object in Fiji.
            # Solitary spines map to themselves (visually identical to C3/C4).
            _grouped_spines = None
            if (getattr(settings, 'detect_multi_head_spines', False)
                    and _multi_head_parent_map):
                _grouped_spines = _build_grouped_spine_volume(
                    spines_filtered, connected_necks, _multi_head_parent_map,
                    workspace if use_chunked else None, logger)
            # Main MIP — limited to 7 channels for ImageJ composite mode. Both
            # distance maps (dendrite_distance, geodesic_distance) are split
            # off into separate single-channel files alongside the main MIP.
            # Without multi-head: 6-channel main; with multi-head: 7-channel.
            logger.info("    Saving validation MIP image...")
            if _grouped_spines is not None:
                _mip_arrays = [neuron, spines, spines_filtered, connected_necks,
                               _grouped_spines, labeled_dendrites, skeleton]
            else:
                _mip_arrays = [neuron, spines, spines_filtered, connected_necks,
                               labeled_dendrites, skeleton]
            io.create_mip_and_save_multichannel_tiff(
                _mip_arrays, locations.MIPs + "MIP_" + filename+".tif", 'float', settings)
            # Separate single-channel MIPs for the two distance maps. Apply the
            # dendrite-distance viz clamp here so the standalone file shows the
            # same far-field zeroing as before.
            io.create_mip_and_save_multichannel_tiff(
                [dendrite_distance],
                locations.MIPs + "MIP_" + filename + "_dendrite_distance.tif",
                'float', settings,
                channel_zero_above={0: _dd_clamp} if _dd_clamp else None)
            io.create_mip_and_save_multichannel_tiff(
                [geodesic_distance_image],
                locations.MIPs + "MIP_" + filename + "_geodesic_distance.tif",
                'float', settings)

            # 3D volumes only when explicitly enabled and not disabled for large images
            if settings.save_intermediate_data == True and getattr(settings, 'save_val_volumes', True):
                if _grouped_spines is not None:
                    _vol_arrays = [neuron, spines, spines_filtered, connected_necks,
                                   _grouped_spines, labeled_dendrites, skeleton]
                else:
                    _vol_arrays = [neuron, spines, spines_filtered, connected_necks,
                                   labeled_dendrites, skeleton]
                # By default skip 3D save when any array is zarr-backed (too large to
                # materialize at T7+ scale). force_val_volume_materialize=True bypasses
                # the gate — useful for debugging chunked-path artifacts on smaller volumes.
                _vol_arrays_check = _vol_arrays + [dendrite_distance, geodesic_distance_image]
                _all_numpy = all(isinstance(a, np.ndarray) for a in _vol_arrays_check)
                _force = getattr(settings, 'force_val_volume_materialize', False)
                if _all_numpy or _force:
                    if not _all_numpy:
                        # Estimate materialized size at uint16 for caller awareness
                        n_vox = int(_vol_arrays[0].shape[0]) * int(_vol_arrays[0].shape[1]) * int(_vol_arrays[0].shape[2])
                        est_gb = n_vox * 2 * len(_vol_arrays_check) / 1024**3
                        logger.info(f"    Saving validation volume image (force_val_volume_materialize=True, ~{est_gb:.1f} GB uint16)...")
                    else:
                        logger.info("    Saving validation volume image...")
                    _dd_clamp_vol = getattr(settings, '_dd_viz_clamp_vox', None)
                    io.create_and_save_multichannel_tiff(
                        _vol_arrays, locations.Vols + filename+".tif", 'float', settings)
                    # Distance volumes split out the same way as the MIPs.
                    io.create_and_save_multichannel_tiff(
                        [dendrite_distance],
                        locations.Vols + filename + "_dendrite_distance.tif",
                        'float', settings,
                        channel_zero_above={0: _dd_clamp_vol} if _dd_clamp_vol else None)
                    io.create_and_save_multichannel_tiff(
                        [geodesic_distance_image],
                        locations.Vols + filename + "_geodesic_distance.tif",
                        'float', settings)
                else:
                    logger.info("    Skipping 3D volume save (zarr-backed arrays in chunked mode; set settings.force_val_volume_materialize=True to override)")

            del neuron, spines, labeled_dendrites, skeleton
            gc.collect()
            # logger.info("\n   Creating spine arrays on GPU...")
            # Extract MIPs for each spine

            # spine_MIPs, spine_slices, spine_vols = create_spine_arrays_in_blocks(image, labels_vol, spines_filtered, spine_table, settings.roi_volume_size, settings, locations, filename,  logger, settings.GPU_block_size)

            ##### We now have refined labels for all spines - so we should remeasure intensities and any voxel measurements

            # F1 invariant check: verify label count stays within expected bounds.
            # Between baseline (post-tiny-head) and pre-measurement the label
            # set can change in two ways: spurious-flagged labels may lose their
            # spine voxels (only when mode='both' AND not keep_head_if_flagged),
            # and filopodia recovery may add new labels. We accept
            # [baseline - n_spurious_drops, baseline + n_filopodia_added].
            if _f1_baseline is not None:
                try:
                    _f1_mode = getattr(settings, 'spurious_exclude_mode', 'neck_only')
                    _f1_keep_head = bool(getattr(
                        settings, 'spurious_keep_head_if_flagged', False))
                    _f1_spurious_labels = locals().get('_spurious_flagged_labels', None)
                    _f1_filopodia_labels = locals().get('_filopodia_labels', set())
                    # Only mode='both' without keep_head_if_flagged can zero spine vox.
                    _f1_max_drops = (
                        len(_f1_spurious_labels)
                        if _f1_spurious_labels and _f1_mode == 'both' and not _f1_keep_head
                        else 0)
                    _f1_max_adds = len(_f1_filopodia_labels)
                    _f1_actual = len(chunked.safe_unique(spines_filtered)) - 1
                    _f1_lo = _f1_baseline - _f1_max_drops
                    _f1_hi = _f1_baseline + _f1_max_adds
                    if _f1_actual < _f1_lo or _f1_actual > _f1_hi:
                        logger.warning(
                            f"     F1 invariant: label count out of range "
                            f"[{_f1_lo}, {_f1_hi}] (baseline={_f1_baseline}, "
                            f"max-spurious-drops={_f1_max_drops}, "
                            f"max-filopodia-adds={_f1_max_adds}), got {_f1_actual}.")
                except Exception as _f1_err:
                    logger.warning(f"     F1 invariant check failed: {_f1_err}")

            # perform final vox based measurements on spines for morophology and intensity
            logger.info("\n    Calculating final measurements...")
            logger.info("     Calculating additional spine head measurements...")
            spine_head_table, spines_filtered = spine_vox_measurements(image, spines_filtered, 1, 0,
                                                                       settings.neuron_channel, 'head',
                                                                       dendrite_distance, soma_distance,
                                                                       settings.neuron_spine_size,
                                                                       settings.neuron_spine_dist,
                                                                       settings, locations, filename, logger,
                                                                       soma_present=soma_present)
            logger.info("     Calculating additional whole spine measurements...")
            # Avoid spines_filtered + connected_necks (20 GB temp) when necks are empty.
            # On chunked + zarr inputs, slab-build to zarr; native arithmetic otherwise.
            _ws_necks_empty = chunked.safe_max(connected_necks) == 0
            if _ws_necks_empty:
                _whole_spine_vol = spines_filtered
            elif use_chunked and (hasattr(spines_filtered, 'chunks')
                                   or hasattr(connected_necks, 'chunks')):
                _ws_shape = spines_filtered.shape
                _whole_spine_vol = workspace.create_array(
                    '_whole_spine_vol_zarr', _ws_shape, dtype=np.int32,
                    chunks=chunked.STREAMING_CHUNKS)
                _ws_bpz = int(_ws_shape[1]) * int(_ws_shape[2]) * 4
                _ws_slz = max(1, min(64, int(2 * 1024**3 / _ws_bpz)))
                for _wz in range(0, _ws_shape[0], _ws_slz):
                    _wze = min(_wz + _ws_slz, _ws_shape[0])
                    _sf = np.asarray(spines_filtered[_wz:_wze]).astype(np.int32, copy=False)
                    _cn = np.asarray(connected_necks[_wz:_wze]).astype(np.int32, copy=False)
                    _whole_spine_vol[_wz:_wze] = _sf + _cn
                    del _sf, _cn
            else:
                _whole_spine_vol = spines_filtered + connected_necks
            spine_whole_table, spines_filtered = spine_vox_measurements(image, _whole_spine_vol, 1, 0,
                                                                        settings.neuron_channel, 'spine',
                                                                        dendrite_distance, soma_distance,
                                                                        settings.neuron_spine_size,
                                                                        settings.neuron_spine_dist,
                                                                        settings, locations, filename, logger,
                                                                        soma_present=soma_present)
            del _whole_spine_vol

            logger.info("     Calculating additional neck measurements...")
            # now measure in necks (what about if neck label doesn't exist (ensure has value 0)
            neck_table, spines_filtered = spine_vox_measurements(image, connected_necks, 1, 0,
                                                                 settings.neuron_channel, 'neck',
                                                                 dendrite_distance, soma_distance,
                                                                 settings.neuron_spine_size,
                                                                 settings.neuron_spine_dist,
                                                                 settings, locations, filename, logger,
                                                                 soma_present=soma_present)

            # merge
            del connected_necks, dendrite_distance, soma_distance, spines_filtered
            gc.collect()

            # geodesic distances already in microns (Dijkstra uses physical spacing)


            spine_table = tables.merge_spine_measurements(spine_table, spine_dendID_and_geodist, settings, logger)

            spine_table = tables.merge_spine_measurements(spine_table, spine_head_table, settings, logger)
            spine_table = tables.merge_spine_measurements(spine_table, spine_whole_table, settings, logger)
            spine_table = tables.merge_spine_measurements(spine_table, neck_table, settings, logger)


            if settings.mesh_analysis == True:
                # drop 'start_coords' from spine_mesh_results (may not exist if 0 spines)
                spine_mesh_results.drop(['start_coords'], axis=1, inplace=True,
                                        errors='ignore')

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

            if spine_head_table.empty:
                logger.info('   Note some spine head meshes couldn\'t be generated, reverting those to voxel based')

            #Coorection if meshes are not generated
            missing_cols = [
                # head metrics
                'head_vol', 'head_area', 'head_surf_area', 'head_length', 'head_width_mean',
                # spine metrics
                'spine_vol', 'spine_area', 'spine_surf_area', 'spine_length',
                # neck metrics
                'neck_vol', 'neck_area', 'neck_surf_area', 'neck_length',
                'neck_width_mean', 'neck_width_min', 'neck_width_max'
            ]

            voxel_vol = settings.input_resXY ** 2 * settings.input_resZ
            for col in missing_cols:
                if col not in spine_table.columns:
                    if col.endswith('_vol') and f'{col}_vox' in spine_table.columns:
                        spine_table[col] = spine_table.pop(f'{col}_vox') * voxel_vol
                    else:
                        spine_table[col] = np.nan




            # Classify spine_type BEFORE dropping head_width_mean / neck_width_mean
            # (L1427 drops those columns). Priority:
            #   1. filopodia (recovered from orphan nnU-Net neck CC)
            #   2. partial-spine (head detected but neck doesn't reach dendrite)
            #   3. categorize_spine by morphology (currently collapses to 'spine')
            #
            # Filopodia + partial-spines use the SAME length/distance columns as
            # normal spines; spine_type is the only distinguishing mark. Partial
            # spines get their neck_* metrics zeroed below (see partial-spine
            # post-processing block).
            def _classify_row(row):
                lbl = row.get('label')
                try:
                    lbl_int = int(lbl)
                except (ValueError, TypeError):
                    lbl_int = None
                if lbl_int is not None and lbl_int in _filopodia_labels:
                    return 'filopodia'
                if lbl_int is not None and lbl_int in _partial_spine_labels:
                    return 'partial-spine'
                return categorize_spine(
                    row.get('spine_length', 0) or 0,
                    row.get('head_width_mean', 0) or 0,
                    row.get('neck_width_mean', 0) or 0,
                )
            spine_table['spine_type'] = spine_table.apply(_classify_row, axis=1)

            # F2 invariant: capture pre-drop row count for post-CSV check.
            _f2_pre_drops_len = len(spine_table)
            _f2_n_partial_dropped = 0
            _f2_n_spurious_rows_dropped = 0

            # Partial-spine post-processing.
            # If keep_partial_spines=True (default): zero the neck metrics for
            # these rows since the neck measurement is unreliable (neck doesn't
            # reach dendrite). Distance metrics (head_euclidean_dist_to_dend,
            # dist_to_dendrite, spine_length_euclidean) are kept — those are
            # valid morphological measurements for the detected head.
            # If False: drop the rows entirely.
            if _partial_spine_labels:
                _keep_partial = bool(getattr(settings, 'keep_partial_spines', True))
                _partial_mask = spine_table['label'].isin(_partial_spine_labels)
                _n_partial = int(_partial_mask.sum())
                if _keep_partial:
                    # Zero out neck metrics for partial spines.
                    # 2026-09-20: except the two DIMENSIONLESS shape ratios. Zero is
                    # a real solidity/extent value - a maximally non-convex object -
                    # so writing it here dragged avg_neck_solidity down in the
                    # summary tables (24 of 84 rows on the T2 image) and contradicted
                    # the NaN batch_mesh_analysis now writes for an object with no
                    # measurable shape. The dimensional columns keep 0, where it
                    # reads as "no neck" consistently with neck_vol = 0 flagging
                    # exactly these rows.
                    _neck_shape_cols = ('neck_extent', 'neck_solidity')
                    _neck_cols = [c for c in spine_table.columns
                                  if c.startswith('neck_') and c != 'spine_type']
                    for _c in _neck_cols:
                        if _c in _neck_shape_cols:
                            spine_table[_c] = spine_table[_c].astype('float64')
                            spine_table.loc[_partial_mask, _c] = float('nan')
                        else:
                            spine_table.loc[_partial_mask, _c] = 0
                    logger.info(
                        f"       Zeroed neck metrics for {_n_partial} partial-spine "
                        f"rows ({len(_neck_cols)} neck_* columns; neck_extent and "
                        f"neck_solidity set to NaN - a ratio has no zero-valued "
                        f"meaning); distance + head measurements retained.")
                else:
                    _before = len(spine_table)
                    spine_table = spine_table[~_partial_mask].reset_index(drop=True)
                    _f2_n_partial_dropped = _before - len(spine_table)
                    logger.info(
                        f"       Dropped {_n_partial} partial-spine rows "
                        f"(keep_partial_spines=False); {_before} -> {len(spine_table)}.")

            # update label column to id
            spine_table.rename(columns={'label': 'spine_id'}, inplace=True)

            # drop some metrics that need furth optimization width measuremnts
            drop_columns = ['head_width_mean', 'neck_width_mean', 'neck_width_min', 'neck_width_max']
            spine_table.drop(columns=drop_columns, inplace=True, errors='ignore')

            # calcuate integrated density
            for prefix in ("head", "neck", "spine"):
                vol_col = f"{prefix}_vol"  # eg. head_vol
                if vol_col not in spine_table.columns:
                    continue  # nothing to do for this prefix

                # loop over every mean-intensity column for this prefix
                for col in spine_table.columns:
                    if col.startswith(f"{prefix}_C") and col.endswith("_mean_int"):
                        id_col = col.replace("_mean_int", "_int_density")
                        if id_col not in spine_table.columns:  # avoid overwriting
                            spine_table[id_col] = spine_table[vol_col] * spine_table[col]

            #reorder for readability
            spine_table = tables.reorder_spine_table_columns(spine_table)

            # Drop rows for spines flagged as spurious by flag_spurious_necks
            # (when mode='both'). Filtering happens here — after all per-spine
            # measurements have joined into spine_table — so the CSV omits
            # both the spurious spine row and all its derived columns.
            try:
                _drop_labels = _spurious_flagged_labels  # set from filter call
            except NameError:
                _drop_labels = set()
            _mode = getattr(settings, 'spurious_exclude_mode', 'both')
            if _drop_labels and _mode == 'both':
                _label_col = 'spine_id' if 'spine_id' in spine_table.columns else 'label'
                _before = len(spine_table)
                spine_table = spine_table[~spine_table[_label_col].isin(_drop_labels)].reset_index(drop=True)
                _after = len(spine_table)
                _f2_n_spurious_rows_dropped = _before - _after
                logger.info(
                    f"       Dropped {_before - _after} spurious spine rows "
                    f"from {filename}_detected_spines.csv ({_before} → {_after})")

            # Multi-head spine columns (only when feature ON). Inserted right
            # after spine_id / dendrite_id columns so they're easy to find.
            # Solitary spines have parent_spine_id == spine_id, head_index=0,
            # multi_head_group_size=1.
            if (getattr(settings, 'detect_multi_head_spines', False)
                    and '_multi_head_parent_map' in locals()):
                _label_col = 'spine_id' if 'spine_id' in spine_table.columns else 'label'
                _pmap = _multi_head_parent_map
                _gmeta = _multi_head_groups
                # Default solitary semantics
                _parents = []
                _indices = []
                _sizes = []
                # head_index ordering: sort group members by spine_id
                # ascending so primary (smallest) is index 0.
                _ordering_cache = {}
                for _lbl in spine_table[_label_col].astype(int).tolist():
                    _parent = int(_pmap.get(_lbl, _lbl))
                    if _parent not in _ordering_cache:
                        _members = (_gmeta.get(_parent, {}).get('member_ids')
                                    or [_parent])
                        _ordering_cache[_parent] = list(sorted(_members))
                    _members = _ordering_cache[_parent]
                    _idx = _members.index(_lbl) if _lbl in _members else 0
                    _parents.append(_parent)
                    _indices.append(_idx)
                    _sizes.append(len(_members))
                spine_table['parent_spine_id'] = _parents
                spine_table['head_index'] = _indices
                spine_table['multi_head_group_size'] = _sizes

            spine_table.to_csv(locations.tables + filename + '_detected_spines.csv', index=False)

            # Multi_Head_Spine_Groups.csv — one row per multi-head group
            # (size >= 2). Aggregates per-group stats by summing member rows.
            if (getattr(settings, 'detect_multi_head_spines', False)
                    and '_multi_head_groups' in locals()
                    and _multi_head_groups):
                _groups_rows = []
                _label_col = 'spine_id' if 'spine_id' in spine_table.columns else 'label'
                for _parent, _meta in sorted(_multi_head_groups.items()):
                    _members = _meta['member_ids']
                    _rows = spine_table[spine_table[_label_col].isin(_members)]
                    if len(_rows) == 0:
                        continue
                    _row = {
                        'parent_spine_id': int(_parent),
                        'n_heads': int(_meta['n_heads']),
                        'member_spine_ids': ';'.join(str(int(m)) for m in _members),
                    }
                    for _col, _agg in (('head_vol', 'sum'), ('spine_vol', 'sum'),
                                        ('neck_vol', 'sum'), ('neck_length', 'sum'),
                                        ('head_area', 'sum'), ('spine_surf_area', 'sum')):
                        if _col in _rows.columns:
                            _row[f'total_{_col}'] = float(_rows[_col].sum())
                    if 'dendrite_id' in _rows.columns:
                        _row['dendrite_id'] = int(_rows['dendrite_id'].iloc[0])
                    if 'geodesic_dist_to_soma' in _rows.columns:
                        _row['geodesic_dist_to_soma'] = float(
                            _rows['geodesic_dist_to_soma'].min())
                    _groups_rows.append(_row)
                if _groups_rows:
                    pd.DataFrame(_groups_rows).to_csv(
                        locations.tables + filename + '_Multi_Head_Spine_Groups.csv',
                        index=False)
                    logger.info(
                        f"       Multi-head groups CSV: {len(_groups_rows)} group(s) "
                        f"saved to {filename}_Multi_Head_Spine_Groups.csv")

            # F2 invariant: verify final row count matches baseline minus
            # tracked drops. Catches accounting gaps between a filter's
            # reported drop count and its actual effect on spine_table.
            try:
                _f2_expected = (
                    _f2_pre_drops_len
                    - _f2_n_partial_dropped
                    - _f2_n_spurious_rows_dropped)
                if len(spine_table) != _f2_expected:
                    logger.warning(
                        f"     F2 invariant: spine_table row count mismatch — "
                        f"expected {_f2_expected} (baseline={_f2_pre_drops_len}, "
                        f"partial-dropped={_f2_n_partial_dropped}, "
                        f"spurious-rows-dropped={_f2_n_spurious_rows_dropped}), "
                        f"got {len(spine_table)}. "
                        f"Delta={len(spine_table) - _f2_expected}.")
            except Exception as _f2_err:
                logger.warning(f"     F2 invariant check failed: {_f2_err}")

            tables.create_spine_summary_dendrite(spine_table, filename, dendrite_lengths, dendrite_volumes, settings,
                                          locations)

            # create summary
            summary = tables.create_spine_summary_neuron(spine_table, filename, neuron_length, neuron_volume, settings)

            # Append to the overall summary DataFrame
            spine_summary = pd.concat([spine_summary, summary], ignore_index=True)
    else:
        logger.info("  *No dendrites were analyzed for this image.")
    logger.info("     Complete.\n")
    total_time = (time.time() - time_initial) / 60
    logger.info(f" Processing complete for file {filename}. Total spine analysis time: {round(total_time, 2)} minutes.\n---")
    vram.checkpoint("pipeline_end", logger)
    vram.report(logger)
    _cleanup_workspace(workspace, logger)
    settings._active_workspace = None
    flush_ram_and_gpu_memory(settings, logger)

    return spine_summary



@mp.profile_mem()
def spine_detection(spines, erode, remove_borders, logger):
    # with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    spines_clean = morphology.remove_small_holes(spines, holes)

    if erode[0] > 0:
        # Erode
        # element = morphology.ball(erode)
        ellipsoidal_element = create_ellipsoidal_element(erode[0], erode[1], erode[2])

        spines_eroded = ndimage.binary_erosion(spines, ellipsoidal_element)

        # Distance Transform to mark centers
        distance = ndimage.distance_transform_edt(spines_eroded)
        seeds = ndimage.label(distance > 0.1 * distance.max())[0]

        # Watershed
        labels = segmentation.watershed(-distance, seeds, mask=spines)

    else:
        labels = measure.label(spines)

    # Remove objects touching border — use chunked_clear_border for large volumes
    # to avoid the 2x np.pad transient (40 GB int32 on T7).
    if remove_borders == True:
        _small_numpy = (isinstance(labels, np.ndarray)
                        and labels.nbytes < 8 * 1024**3)
        if _small_numpy:
            padded = np.pad(
                labels,
                ((1, 1), (0, 0), (0, 0)),
                mode='constant',
                constant_values=0,
            )
            labels = segmentation.clear_border(padded)[1:-1]
        else:
            labels = chunked.chunked_clear_border(labels, faces='yx', logger=logger)

    # add new axis for labels
    # labels = labels[:, np.newaxis, :, :]

    return labels



@mp.profile_mem()
def spine_detection_cores_and_membranes(cores, membranes, remove_borders, logger, settings,
                                          workspace=None, result_name=None):
    """
    Perform spine detection using cores and membranes.

    Parameters:
    -----------
    cores : ndarray
        Binary mask of spine cores/centers
    membranes : ndarray
        Binary mask of spine membranes/boundaries
    erode : tuple or None
        Erosion parameters (not used when cores are directly provided)
    remove_borders : bool
        Whether to remove objects touching image borders
    logger : object or None
        Logger object for debug information
    workspace : chunked.ZarrWorkspace, optional
        When provided together with `result_name` AND the per-cluster (zarr)
        path is active, `labeled_spines` is allocated directly in the
        workspace instead of being materialized to a numpy array at the
        end. This is Phase H — saves ~22 GB host RAM at T7 (int32 × 5.5B
        voxels). The returned value is a zarr.Array handle in that case.
    result_name : str, optional
        Name of the workspace zarr array to create. Must be provided alongside
        `workspace`. `workspace.create_array` removes any existing array
        directory before creating a fresh zarr, so re-runs start from a clean
        slate (no stale chunk files from a prior pipeline invocation).

    Returns:
    --------
    labels : ndarray or zarr.Array
        Labeled segmentation of spines. Type is numpy for small-volume path
        or when `workspace` is None; zarr when workspace+result_name provided
        AND the per-cluster path is active.
    """

    shape = cores.shape
    n_voxels = int(shape[0]) * int(shape[1]) * int(shape[2])
    volume_bytes = n_voxels * 8  # int64 for scipy label output
    use_per_cluster = volume_bytes > 2 * GB
    # Zarr-backed path for all per-cluster volumes — medium volumes previously
    # held labeled_cores + cluster_labels + combined_mask + labeled_spines
    # simultaneously (~6 full-volume arrays). Unifying eliminates that peak.
    use_zarr = use_per_cluster

    _cache_dir = None
    labeled_cores_zarr = None
    labeled_spines_zarr = None
    cluster_zarr = None
    try:
        # ------------------------------------------------------------------
        # Step 1: Label cores (markers for watershed)
        # ------------------------------------------------------------------
        if use_zarr:
            import tempfile
            _cache_dir = tempfile.mkdtemp(prefix='respan_spine_det_')
            logger.info(f"    Zarr-backed spine detection (cache: {_cache_dir})")
            logger.info(f"    Streaming CC for core labeling ({cores.nbytes / 1e9:.1f} GB)")
            labeled_cores_zarr = zarr.open(
                os.path.join(_cache_dir, 'labeled_cores'), mode='w',
                shape=shape, dtype=np.int32, chunks=chunked.STREAMING_CHUNKS)
            # Pass cores numpy directly — CC reads via slicing
            n_cores, _ = chunked.connected_components_streaming(
                cores, labeled_cores_zarr, logger=logger)
            labeled_cores = None  # stays in zarr, never materialized
        else:
            labeled_cores_zarr = None
            labeled_cores = measure.label(cores)

        # ------------------------------------------------------------------
        # Step 2a: Small volume — all in-memory
        # ------------------------------------------------------------------
        if not use_per_cluster:
            combined_mask = cores | membranes
            if labeled_cores is None:
                labeled_cores = np.array(labeled_cores_zarr)

            # Filter out tiny nnU-Net core predictions before watershed.
            # Isolated 1-2 voxel cores are typically prediction noise (e.g.,
            # label 19 in T2 @ (z=18,y=97,x=466) — a 1-vox core that split off
            # from the real spine next to it). Default ≥3 voxels matches the
            # minimum meaningful object in 3D.
            _min_core = int(getattr(settings, 'min_core_voxels', 3))
            if _min_core > 1 and labeled_cores.max() > 0:
                _core_sizes = np.bincount(labeled_cores.ravel())
                _tiny = np.where(_core_sizes[1:] < _min_core)[0] + 1
                if _tiny.size:
                    _tiny_set = set(int(x) for x in _tiny)
                    _max_cl = int(labeled_cores.max())
                    _lut = np.arange(_max_cl + 1, dtype=labeled_cores.dtype)
                    for _t in _tiny_set:
                        _lut[_t] = 0
                    labeled_cores = _lut[labeled_cores]
                    logger.info(
                        f"    Filtered {len(_tiny_set)} tiny nnU-Net cores "
                        f"(<{_min_core} vox) before watershed.")

            distance = ndimage.distance_transform_edt(~membranes)
            labeled_spines = segmentation.watershed(
                -distance, markers=labeled_cores, mask=combined_mask)
            # Recovery: some nnU-Net spine heads are predicted as membrane-only
            # (no core voxels inside). Watershed produces label=0 for those
            # clusters, silently losing them. Re-label the orphan-membrane
            # clusters (mask voxels that are 0 in labeled_spines) with fresh
            # labels above the core-label range.
            orphan_mask = combined_mask & (labeled_spines == 0)
            if orphan_mask.any():
                orphan_labels, n_orphan = ndimage.label(orphan_mask)
                if n_orphan > 0:
                    max_core = int(labeled_cores.max()) if labeled_cores.size else 0
                    # Shift orphan labels so they don't collide with cores.
                    orphan_labels = np.where(
                        orphan_labels > 0, orphan_labels + max_core, 0
                    ).astype(labeled_spines.dtype)
                    labeled_spines = np.maximum(labeled_spines, orphan_labels)
                    logger.info(f"    Recovered {n_orphan} orphan-membrane spines "
                                f"(nnU-Net predicted membrane without core).")

        # ------------------------------------------------------------------
        # Step 2b: Large/medium volume — zarr-backed per-cluster watershed
        # ------------------------------------------------------------------
        else:
            logger.info(f"    Large volume ({volume_bytes / GB:.1f} GB) "
                         f"— zarr-backed per-cluster watershed...")

            # Lazy combined_mask — generates cores|membranes per slab on demand
            # so we never allocate a full-volume boolean array
            class _LazyOrMask:
                """Array-like that returns cores|membranes slabs on demand."""
                def __init__(self, a, b):
                    self.shape = a.shape
                    self.dtype = np.dtype(np.uint8)
                    self._a, self._b = a, b
                def __getitem__(self, idx):
                    return (self._a[idx] | self._b[idx]).astype(np.uint8)

            # Stream cluster CC without materializing combined_mask or cluster_labels
            cluster_zarr = zarr.open(
                os.path.join(_cache_dir, 'cluster_labels'), mode='w',
                shape=shape, dtype=np.int32, chunks=chunked.STREAMING_CHUNKS)
            n_clusters, _ = chunked.connected_components_streaming(
                _LazyOrMask(cores, membranes), cluster_zarr, logger=logger)

            # Get bboxes + sizes from zarr in one pass (no materialization)
            bboxes, cluster_sizes = chunked._streaming_bboxes_and_sizes(
                cluster_zarr, n_clusters, logger=logger)
            # cluster_zarr stays alive through the per-cluster loop below.
            # We need to read per-cluster crops `(cluster_zarr[sl] == cl)` to
            # build a true cluster mask — without it, the padded crop's halo
            # leaks neighbour clusters into the orphan check and the orphan
            # fallback paints across the entire padded bbox (T_LARGE-6 bug:
            # ~22% of labels rendered as bbox-fill rectangles in the MIP).
            gc.collect()

            # Output: write per-cluster crops to zarr, materialize at end UNLESS
            # Phase H is active (workspace + result_name provided) — then the
            # zarr lives in the caller's workspace and is returned as a handle.
            _phase_h = workspace is not None and result_name is not None
            if _phase_h:
                labeled_spines_zarr = workspace.create_array(
                    result_name, shape=shape, dtype=np.int32,
                    chunks=chunked.STREAMING_CHUNKS)
                logger.info(
                    f"    Phase H: labeled_spines stored in workspace "
                    f"('{result_name}') — will return zarr handle, "
                    f"saves ~{int(shape[0]) * int(shape[1]) * int(shape[2]) * 4 / 1e9:.1f} GB RAM materialization.")
            else:
                labeled_spines_zarr = zarr.open(
                    os.path.join(_cache_dir, 'labeled_spines'), mode='w',
                    shape=shape, dtype=np.int32, chunks=chunked.STREAMING_CHUNKS,
                    fill_value=0)

            pad = 3
            min_cluster_vox = (int(getattr(settings, 'neuron_spine_size', [10])[0])
                               if hasattr(settings, 'neuron_spine_size') else 10)
            n_skipped = 0
            n_processed = 0
            n_orphan_membrane = 0
            # Orphan-membrane clusters (nnU-Net predicted spine membrane with
            # no core inside) will be assigned labels above the core-label
            # range so they never collide with watershed labels from real cores.
            try:
                _next_orphan_label = int(chunked.safe_max(labeled_cores_zarr)) + 1
            except Exception:
                _next_orphan_label = 1

            for cl, (z0, z1, y0, y1, x0, x1) in sorted(bboxes.items()):
                if cluster_sizes[cl] < min_cluster_vox:
                    n_skipped += 1
                    continue

                # Expand bounding box with padding
                sl = (slice(max(0, z0 - pad), min(shape[0], z1 + pad)),
                      slice(max(0, y0 - pad), min(shape[1], y1 + pad)),
                      slice(max(0, x0 - pad), min(shape[2], x1 + pad)))

                # Read small crops: membranes/cores from numpy, labeled_cores from zarr.
                # Force bool dtype: when cores/membranes are zarr-backed (Phase L4
                # zarr pass-through), `cores[sl]` returns uint8 — uint8 mask used
                # in `crop_ws[mask] = label` triggers NumPy advanced INTEGER
                # indexing (not boolean), silently overwriting axis-0 planes.
                crop_mem = np.asarray(membranes[sl], dtype=bool)
                crop_cores_labeled = np.array(labeled_cores_zarr[sl])
                crop_mask = np.asarray(cores[sl], dtype=bool) | crop_mem

                # True per-cluster mask — the padded `crop_mask` includes voxels
                # from neighbour clusters in the 3-voxel halo. cluster_mask
                # restricts every decision and write to THIS cluster's voxels.
                cluster_mask = (np.asarray(cluster_zarr[sl]) == cl)
                # Drop neighbour cores that leak into the padded halo before
                # they bias the orphan check or seed our watershed.
                if not cluster_mask.all():
                    crop_cores_labeled = crop_cores_labeled.copy()
                    crop_cores_labeled[~cluster_mask] = 0

                # Filter tiny cores inside this cluster — identical rule to the
                # small-volume path. Prevents a single-voxel nnU-Net core from
                # becoming its own spine via watershed. Restrict filtering to
                # cores fully INSIDE this cluster (ignore neighbour labels that
                # leak into the padded halo) and use np.unique+counts so the
                # LUT stays proportional to labels actually present, not
                # max_label_in_crop.
                _min_core = int(getattr(settings, 'min_core_voxels', 3))
                if _min_core > 1 and crop_cores_labeled.any():
                    # Only count cores inside the raw cluster bbox (unpadded).
                    # Padding includes neighbours whose own cluster will
                    # process them with a full count.
                    _inner_cores = np.array(labeled_cores_zarr[
                        z0:z1, y0:y1, x0:x1])
                    if _inner_cores.any():
                        _present, _counts = np.unique(_inner_cores, return_counts=True)
                        _tiny_global = set(int(l) for l, c in zip(_present, _counts)
                                            if l != 0 and c < _min_core)
                        if _tiny_global:
                            # Apply to padded crop via np.isin (fast, no big LUT).
                            _mask_tiny = np.isin(crop_cores_labeled,
                                                 list(_tiny_global))
                            if _mask_tiny.any():
                                crop_cores_labeled = crop_cores_labeled.copy()
                                crop_cores_labeled[_mask_tiny] = 0

                if not np.any(crop_cores_labeled):
                    # Orphan-membrane cluster (nnU-Net predicted spine membrane
                    # but no core). Watershed produces no labels without markers,
                    # silently losing these real spines. Fallback: assign THIS
                    # cluster's voxels (cluster_mask, NOT the padded crop_mask)
                    # a single new label above the core-label range.
                    crop_ws = np.zeros_like(crop_cores_labeled, dtype=np.int32)
                    crop_ws[cluster_mask] = _next_orphan_label
                    _next_orphan_label += 1
                    n_orphan_membrane += 1
                else:
                    crop_dist = ndimage.distance_transform_edt(~crop_mem)
                    # Watershed bounded by cluster_mask so neighbour-cluster
                    # markers (if any survived the cleanup above) cannot bleed
                    # into this cluster's voxels.
                    crop_ws = segmentation.watershed(
                        -crop_dist, markers=crop_cores_labeled, mask=cluster_mask)

                # Write back to zarr (np.maximum handles overlapping padded bboxes)
                existing = np.array(labeled_spines_zarr[sl])
                labeled_spines_zarr[sl] = np.maximum(existing, crop_ws)

                n_processed += 1
                if n_processed % 500 == 0:
                    logger.info(f"      Processed {n_processed}/{len(bboxes)} clusters...")

            if n_skipped > 0:
                logger.info(f"    Skipped {n_skipped} clusters smaller than {min_cluster_vox} voxels.")
            if n_orphan_membrane > 0:
                logger.info(f"    Recovered {n_orphan_membrane} orphan-membrane spines "
                            f"(nnU-Net predicted membrane without core — fallback-labeled).")
            if _phase_h:
                # Phase H: skip materialization — return zarr handle. The
                # workspace-backed zarr survives `_cache_dir` rmtree in the
                # finally block (its data lives in `workspace.base_dir`, not
                # `_cache_dir`). labeled_cores_zarr remains in _cache_dir and
                # is released/cleaned up as usual.
                labeled_spines = labeled_spines_zarr
                logger.info(f"    Processed {n_processed} clusters. Returning zarr handle (Phase H).")
            else:
                logger.info(f"    Processed {n_processed} clusters. Materializing result...")
                # Slab-by-slab materialization — np.array(zarr_array) loads all
                # compressed chunks simultaneously (~1.5x transient spike). For
                # T7-scale int32 labeled_spines (~40 GB) that spike OOMs.
                labeled_spines = np.empty(shape, dtype=np.int32)
                slab_depth = labeled_spines_zarr.chunks[0] if labeled_spines_zarr.chunks else 64
                for _z in range(0, shape[0], slab_depth):
                    _ze = min(_z + slab_depth, shape[0])
                    labeled_spines[_z:_ze] = labeled_spines_zarr[_z:_ze]
                del labeled_spines_zarr, labeled_cores_zarr
                gc.collect()

    finally:
        # Release zarr handles before rmtree (Windows locks open stores).
        # Phase H: only clear the local `_cache_dir` references here — the
        # workspace-backed zarr lives in `workspace.base_dir` and must survive.
        # Setting the local `labeled_spines_zarr` ref to None is fine: the
        # returned `labeled_spines` already holds the handle for the caller.
        labeled_cores_zarr = None
        labeled_spines_zarr = None
        cluster_zarr = None
        if _cache_dir is not None:
            import shutil
            shutil.rmtree(_cache_dir, ignore_errors=True)

    # Slab-iterated "voxels unassigned" count — works for both numpy and zarr
    # labeled_spines without materializing the full volume. Factor in dtype
    # bytes (int32 labels + bool membranes) so the slab-size budget reflects
    # actual transient RAM; at T7 this keeps each iteration under ~2 GB.
    _unassigned = 0
    _lb_shape = labeled_spines.shape
    _bytes_per_z = int(_lb_shape[1]) * int(_lb_shape[2]) * (
        np.dtype(labeled_spines.dtype).itemsize + np.dtype(membranes.dtype).itemsize)
    _lb_slab = max(1, min(64, int(2 * 1024**3 / max(_bytes_per_z, 1))))
    for _z in range(0, _lb_shape[0], _lb_slab):
        _ze = min(_z + _lb_slab, _lb_shape[0])
        _lb_s = np.asarray(labeled_spines[_z:_ze])
        _mm_s = np.asarray(membranes[_z:_ze])
        _unassigned += int(np.sum(_mm_s & ~(_lb_s > 0)))
        del _lb_s, _mm_s
    logger.info(f"    Voxels unassigned to spines: {_unassigned}.")

    # Remove objects touching border if requested. The legacy code padded Z
    # with 1 voxel of zeros then ran clear_border — effectively only scrubbing
    # labels touching the Y or X boundary faces. For T7+ the pad allocates a
    # second 40 GB int32 transient; chunked_clear_border reproduces the
    # semantics in O(slab) memory.
    if remove_borders == True:
        _small_numpy = (isinstance(labeled_spines, np.ndarray)
                        and labeled_spines.nbytes < 8 * 1024**3)
        if _small_numpy:
            padded = np.pad(
                labeled_spines,
                ((1, 1), (0, 0), (0, 0)),
                mode='constant',
                constant_values=0,
            )
            labeled_spines = segmentation.clear_border(padded)[1:-1]
        else:
            labeled_spines = chunked.chunked_clear_border(
                labeled_spines, faces='yx', logger=logger)

    return labeled_spines

@mp.profile_mem()
def spine_detection_4d(spines, erode, remove_borders, logger):
    # with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    spines_clean = morphology.remove_small_holes(spines, holes)
    if erode[0] > 0:

        # Erode
        # element = morphology.ball(erode)
        ellipsoidal_element = create_ellipsoidal_element(erode[0], erode[1], erode[2])
        ellipsoidal_element = ellipsoidal_element[np.newaxis, :, :, :]

        spines_eroded = ndimage.binary_erosion(spines, ellipsoidal_element)

        # Distance Transform to mark centers
        distance = ndimage.distance_transform_edt(spines_eroded)
        seeds = ndimage.label(distance > 0.1 * distance.max())[0]

        # Watershed
        labels = segmentation.watershed(-distance, seeds, mask=spines)

    else:
        labels = measure.label(spines)

    if remove_borders == True:
        spines_list = []
        # Remove objects touching border
        for t in range(labels.shape[0]):
            labels_3d = labels[t, :, :, :]
            padded = np.pad(
                labels_3d,
                ((1, 1), (0, 0), (0, 0)),
                mode='constant',
                constant_values=0,
            )
            labels_3d = segmentation.clear_border(padded)[1:-1]
            spines_list.append(labels_3d)

        labels = np.stack(spines_list, axis=0)
    # Check if the sequence of labels is continuous
    unique_labels = np.unique(labels)
    if np.all(np.diff(unique_labels) == 1) != True:
        labels = measure.label(labels > 0, background=0)

    return labels

@mp.profile_mem()
def calculate_dendrite_length_and_volume_fast(labeled_dendrites, skeleton, logger):
    # Slab-based skeleton-coord + label extraction. Works for numpy and zarr inputs.
    # Avoids the previous np.nonzero(skeleton) + labeled_dendrites[fancy_index] path
    # which materialized the full skeleton volume and then scanned all of
    # labeled_dendrites (22 GB at T7 scale).
    _bpz = int(labeled_dendrites.shape[1]) * int(labeled_dendrites.shape[2]) * labeled_dendrites.dtype.itemsize
    _slz = max(1, min(64, int(2 * 1024**3 / _bpz)))

    _coords_list = []
    _labels_list = []
    dend_max = 0
    dend_counts_list = []

    for _z in range(0, labeled_dendrites.shape[0], _slz):
        _ze = min(_z + _slz, labeled_dendrites.shape[0])
        sk_slab = np.asarray(skeleton[_z:_ze])
        ld_slab = np.asarray(labeled_dendrites[_z:_ze])
        # Fused: skeleton coords + labeled_dendrites bincount in the same slab read.
        local = np.array(np.nonzero(sk_slab)).T  # (N_slab, 3) in slab-frame
        if local.size:
            _labs = ld_slab[tuple(local.T)]
            # Offset Z column to global frame.
            local[:, 0] += _z
            _coords_list.append(local)
            _labels_list.append(_labs)
        # dend_max + bincount for whole-volume dendrite volumes.
        if ld_slab.size:
            slab_max = int(ld_slab.max())
            if slab_max > dend_max:
                dend_max = slab_max
        dend_counts_list.append(np.bincount(ld_slab.ravel()))

    if _coords_list:
        skeleton_coords = np.concatenate(_coords_list, axis=0)
        labeled_skeletons = np.concatenate(_labels_list, axis=0)
    else:
        skeleton_coords = np.empty((0, 3), dtype=np.int64)
        labeled_skeletons = np.empty((0,), dtype=np.int64)

    # Reconcile dend_counts across slabs — each slab's bincount may be shorter
    # than dend_max + 1. Sum into a uniform-length array.
    dend_counts = np.zeros(dend_max + 1, dtype=np.int64)
    for _bc in dend_counts_list:
        dend_counts[:len(_bc)] += _bc[:dend_max + 1]

    # Get unique labels, excluding background (0)
    unique_dendrite_labels = np.unique(labeled_skeletons)
    unique_dendrite_labels = unique_dendrite_labels[unique_dendrite_labels != 0]

    dendrite_lengths = {label: 0 for label in unique_dendrite_labels}

    # Bincount on the (small) 1D labeled_skeletons array — always numpy now.
    skel_max = int(labeled_skeletons.max()) if labeled_skeletons.size > 0 else 0
    skel_counts = np.zeros(skel_max + 1, dtype=np.int64)
    if labeled_skeletons.size > 0:
        _bc = np.bincount(labeled_skeletons.ravel(), minlength=skel_max + 1)
        skel_counts[:len(_bc)] += _bc[:skel_max + 1]
    for label in unique_dendrite_labels:
        if label < len(skel_counts):
            dendrite_lengths[label] = int(skel_counts[label])

    dendrite_volumes = {int(label): int(dend_counts[label]) if label < len(dend_counts) else 0
                        for label in unique_dendrite_labels}

    return dendrite_lengths, dendrite_volumes, skeleton_coords, labeled_skeletons

def filter_tiny_spine_heads(spines_filtered, spine_table, min_voxels, logger):
    """Backstop filter: drop spine-head labels with voxel count <= `min_voxels`.

    Runs AFTER initial_spine_measurements (which applies the µm³ floor from
    neuron_spine_size) and BEFORE associate_spines_with_necks_per_component.
    A stray 1-voxel head sitting along a real neck corridor will claim half
    of that neck in Pass 1 (nearest-spine assignment), splitting the neck
    into two labels. Dropping these tiny heads first lets the full corridor
    go to the real spine head.

    Only `spines_filtered` and the matching `spine_table` rows are modified.
    The neck mask, `connected_necks`, and raw nnU-Net predictions are never
    touched — no neck information is lost.

    Parameters
    ----------
    spines_filtered : numpy or zarr int32
        Labeled spine-head volume. Modified in place for numpy; zarr writes
        are slab-scoped.
    spine_table : pandas.DataFrame
        Initial per-spine measurements with a `label` column.
    min_voxels : int
        Threshold. Labels with voxel count <= `min_voxels` are zeroed.
        `min_voxels <= 1` disables the filter (only label 0 would match).
    logger : logging.Logger

    Returns
    -------
    spines_filtered : same type/shape as input (in-place update).
    spine_table : pandas.DataFrame with offending rows dropped.
    n_dropped : int — labels removed.
    dropped_records : list of (label, voxel_count) tuples for CSV audit.
    """
    if min_voxels is None or int(min_voxels) <= 1:
        return spines_filtered, spine_table, 0, []

    min_voxels = int(min_voxels)
    # Slab-iterate bincount. For zarr inputs this avoids materializing the
    # full int32 volume (up to ~20 GB at T7). For numpy it's equivalent to
    # a single bincount but keeps the same code path.
    is_zarr = hasattr(spines_filtered, 'chunks')
    shape = spines_filtered.shape
    if len(shape) < 3:
        # Unexpected — fall through to numpy ravel
        counts = np.bincount(np.asarray(spines_filtered).ravel())
    else:
        row_bytes = int(shape[1]) * int(shape[2]) * spines_filtered.dtype.itemsize
        slab_depth = max(1, min(64, int(2 * 1024 ** 3 / max(row_bytes, 1))))
        counts = None
        for _z in range(0, shape[0], slab_depth):
            _ze = min(_z + slab_depth, shape[0])
            slab = np.asarray(spines_filtered[_z:_ze])
            if slab.size == 0:
                continue
            _bc = np.bincount(slab.ravel())
            if counts is None:
                counts = _bc
            elif len(_bc) > len(counts):
                _bc[:len(counts)] += counts
                counts = _bc
            else:
                counts[:len(_bc)] += _bc
            del slab, _bc

    if counts is None or len(counts) <= 1:
        logger.info(f"     Tiny spine-head filter: no labels present (min_voxels={min_voxels}).")
        return spines_filtered, spine_table, 0, []

    # Label 0 is background — never drop. Only flag labels that ACTUALLY exist
    # (count > 0) and are at or below threshold — a phantom index from gaps in
    # the label space would have count == 0 and should be ignored in reporting.
    _fg = counts[1:]
    small_labels = np.where((_fg > 0) & (_fg <= min_voxels))[0] + 1
    small_labels = small_labels.astype(np.int64).tolist()
    n_total_labels = int(np.count_nonzero(_fg))

    if not small_labels:
        logger.info(
            f"     Tiny spine-head filter: 0 labels with <={min_voxels} voxels "
            f"(kept {n_total_labels}).")
        return spines_filtered, spine_table, 0, []

    dropped_records = [(int(_lbl), int(_fg[_lbl - 1])) for _lbl in small_labels]
    small_set = set(small_labels)
    # Zero small labels in `spines_filtered` slab-by-slab. For zarr, read slab,
    # mask the matching labels to 0, write the slab back. For numpy, in-place.
    if is_zarr:
        for _z in range(0, shape[0], slab_depth):
            _ze = min(_z + slab_depth, shape[0])
            slab = np.asarray(spines_filtered[_z:_ze])
            if slab.size == 0:
                continue
            _mask = np.isin(slab, small_labels)
            if _mask.any():
                slab[_mask] = 0
                spines_filtered[_z:_ze] = slab
            del slab, _mask
    else:
        _mask = np.isin(spines_filtered, small_labels)
        spines_filtered[_mask] = 0
        del _mask

    # Drop matching rows from spine_table. The `label` column may be int, str,
    # or mixed depending on upstream paths — coerce to int64 for comparison.
    before = len(spine_table)
    if 'label' in spine_table.columns:
        try:
            _lbl_int = spine_table['label'].astype(np.int64, errors='ignore')
        except Exception:
            _lbl_int = spine_table['label']
        spine_table = spine_table.loc[~_lbl_int.isin(small_set)].reset_index(drop=True)
    after = len(spine_table)

    # Log sample dropped labels for debugging the neck 19/29-style case.
    _sample = small_labels[:10]
    _sample_str = ",".join(str(x) for x in _sample)
    _suffix = "" if len(small_labels) <= 10 else f", ... (+{len(small_labels)-10} more)"
    logger.info(
        f"     Tiny spine-head filter: removed {len(small_labels)} labels with "
        f"<={min_voxels} voxels (kept {n_total_labels - len(small_labels)}). "
        f"Dropped sample: [{_sample_str}{_suffix}]. "
        f"spine_table: {before} -> {after} rows.")

    return spines_filtered, spine_table, len(small_labels), dropped_records


@mp.profile_mem()
def initial_spine_measurements(image, labels, dendrite, max_label, neuron_ch, dendrite_distance, sizes, dist,
                         settings, locations, filename, logger):
    """ measures intensity of each channel, as well as distance to dendrite
    """

    # Only `np.expand_dims` on a numpy image — zarr image is pre-promoted to 4D
    # by the Phase F entry spill (see spine_and_whole_neuron_processing L~380).
    if not hasattr(image, 'chunks') and len(image.shape) == 3:
        image = np.expand_dims(image, axis=1)

    # Measure channel 1:
    logger.info("    Making initial morphology and intensity measurements for channel 1...")
    # Compute bboxes once and reuse across both per_spine_regionprops calls
    # below — recomputing forces a second decompression of the labels zarr,
    # which is costly under Phase H even with the streaming-bbox helper.
    # Skip the precompute when no zarr branch will fire (T2/T4 all-numpy
    # path uses measure.regionprops_table directly and never reads
    # _shared_bboxes).
    _zarr_path_active = (hasattr(labels, 'chunks')
                         or hasattr(image, 'chunks')
                         or hasattr(dendrite_distance, 'chunks'))
    if _zarr_path_active:
        if hasattr(labels, 'chunks'):
            _shared_bboxes = chunked.find_objects_streaming(labels, logger=logger)
        elif isinstance(labels, np.ndarray) and labels.max() > 0:
            _shared_bboxes = ndimage.find_objects(labels, int(labels.max()))
        else:
            _shared_bboxes = None
    else:
        _shared_bboxes = None
    if hasattr(labels, 'chunks') or hasattr(image, 'chunks'):
        # Zarr-safe path — either labels OR image is zarr. per_spine_regionprops
        # reads per-spine bboxes from zarr without materializing full volume.
        # For this label/centroid/area call we don't need intensity from image
        # (centroid is computed from the label mask alone).
        main_table = chunked.per_spine_regionprops(
            labels, {}, ['label', 'centroid', 'area'],
            spine_bboxes=_shared_bboxes, logger=logger)
        main_table.rename(columns={'centroid-0': 'z', 'centroid-1': 'y', 'centroid-2': 'x'},
                          inplace=True)
    else:
        main_table = pd.DataFrame(
            measure.regionprops_table(
                labels,
                intensity_image=image[:, 0, :, :],
                properties=['label', 'centroid', 'area'],
            )
        )
        main_table.rename(columns={'centroid-0': 'z', 'centroid-1': 'y', 'centroid-2': 'x'},
                          inplace=True)
    # measure distance to dendrite
    logger.info("    Measuring distances to dendrite/s...")
    if hasattr(dendrite_distance, 'chunks'):
        # Zarr-safe path: per-spine bbox reads instead of full-volume regionprops.
        # Bboxes already computed above — reuse to avoid re-decompressing labels.
        distance_table = chunked.per_spine_regionprops(
            labels, {'dd': dendrite_distance},
            ['label', 'min_intensity', 'max_intensity'],
            spine_bboxes=_shared_bboxes, logger=logger)
        distance_table.rename(columns={'dd_min_intensity': 'dist_to_dendrite',
                                        'dd_max_intensity': 'spine_length'}, inplace=True)
    else:
        distance_table = pd.DataFrame(
            measure.regionprops_table(
                labels,
                intensity_image=dendrite_distance,
                properties=['label', 'min_intensity', 'max_intensity'],
            )
        )
        distance_table.rename(columns={'min_intensity': 'dist_to_dendrite'}, inplace=True)
        distance_table.rename(columns={'max_intensity': 'spine_length'}, inplace=True)

    # Merge on label column to ensure correct alignment (per_spine_regionprops
    # sorts by Z for I/O locality, which may differ from label order)
    main_table = pd.merge(main_table, distance_table[['label', 'dist_to_dendrite', 'spine_length']],
                          on='label', how='left')

    # filter out small objects
    volume_min = sizes[0]  # 3
    volume_max = sizes[1]  # 1500?

    # logger.info(f" Filtering spines between size {volume_min} and {volume_max} voxels...")

    # filter based on volume
    # logger.info(f"  filtered table before area = {len(main_table)}")
    spinebefore = len(main_table)

    # Inclusive bounds (>=, <=). Previous strict inequalities silently dropped
    # spines whose volume exactly matched the configured min/max — at low
    # thresholds (e.g., spine_vol=(0.0005, 15) → min=1 voxel) this erased
    # every 1-voxel spine.
    filtered_table = main_table[(main_table['area'] >= volume_min) & (main_table['area'] <= volume_max)]

    logger.info(f"     Total putative spines: {spinebefore}")
    logger.info(f"     Spines after volume filtering = {len(filtered_table)} ")
    # logger.info(f"  filtered table after area = {len(filtered_table)}")

    # filter based on distance to dendrite
    spinebefore = len(filtered_table)
    logger.info(f"     Distance threshold: {dist:.1f} voxels. "
                f"dist_to_dendrite range: [{filtered_table['dist_to_dendrite'].min():.2f}, "
                f"{filtered_table['dist_to_dendrite'].max():.2f}]")
    filtered_table = filtered_table[(filtered_table['dist_to_dendrite'] < dist)]
    logger.info(f"     Spines after distance filtering = {len(filtered_table)} ")

    if settings.Track != True:
        # update label numbers based on offset
        filtered_table['label'] += max_label
        # Zarr-safe slab-iterated version of `labels[labels > 0] += max_label`.
        # In practice max_label=0 in the test-runner/GUI path so this is a
        # no-op; guard kept for non-zero legacy callers.
        if max_label:
            _ly, _lx = int(labels.shape[1]), int(labels.shape[2])
            _dtbytes = np.dtype(labels.dtype).itemsize
            _slab_z = max(1, min(64, int(2 * 1024**3 / max(_ly * _lx * _dtbytes, 1))))
            for _z in range(0, labels.shape[0], _slab_z):
                _ze = min(_z + _slab_z, labels.shape[0])
                _slab = np.asarray(labels[_z:_ze])
                _slab[_slab > 0] += max_label
                labels[_z:_ze] = _slab
                del _slab
        labels = create_filtered_labels_image(labels, filtered_table, logger)
    else:

        # Clean up label image — LUT remap, slab-by-slab to avoid
        # allocating a full-volume copy (20+ GB for T7-class images)
        ids_to_keep = filtered_table['label'].astype(labels.dtype).values
        # safe_max handles zarr labels under Phase H; falls through to
        # arr.max() for numpy. Direct labels.max() would materialize
        # the entire zarr volume.
        max_lbl = int(chunked.safe_max(labels))
        lut = np.zeros(max_lbl + 1, dtype=labels.dtype)
        lut[ids_to_keep] = ids_to_keep
        _bpz = int(labels.shape[1]) * int(labels.shape[2]) * labels.dtype.itemsize
        _slz = max(1, min(64, int(2 * 1024**3 / _bpz)))
        for _z in range(0, labels.shape[0], _slz):
            _ze = min(_z + _slz, labels.shape[0])
            labels[_z:_ze] = lut[labels[_z:_ze]]

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
    #filtered_table = tables.move_column(filtered_table, 'dist_to_soma', 11)
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

@mp.profile_mem()
def associate_spines_with_necks_gpu(spines, necks, logger, workspace=None,
                                     use_chunked=False, settings=None):
    """Dispatcher: assign target fragments to their nearest reference label.

    Delegates to `chunked.associate_spines_with_necks_per_component`, which
    uses CC-of-targets + nearest-reference-mode-vote + distance cap. This
    preserves nnU-Net neck fragments disconnected from any spine (the prior
    full-volume GPU/CPU paths silently dropped them via CC(spines|necks)).

    In Pass 1 of the RESPAN neck pipeline, `spines` is the spine label volume
    and `necks` is the nnU-Net neck labels. In Pass 2, `spines` is the
    assigned/extended neck corridor and `necks` is the remaining unassigned
    neck voxels — the same "assign target fragments to nearest reference"
    semantics in both cases.
    """
    input_resXY = float(getattr(settings, 'input_resXY', 0.065)) if settings is not None else 0.065
    input_resZ = float(getattr(settings, 'input_resZ', 0.15)) if settings is not None else 0.15
    # Settings.neuron_spine_dist is stored in voxels (XY) after GUI/batch
    # conversion `spine_dist_um / input_resXY`. Convert back to µm for the cap.
    spine_dist_vox = float(getattr(settings, 'neuron_spine_dist', 2.0 / input_resXY)) if settings is not None else 2.0 / input_resXY
    spine_dist_um = spine_dist_vox * input_resXY
    # L2 testing: settings.association_streaming_threshold_voxels lets profiles
    # lower the streaming gate so T4/T6 exercise the streaming path without a
    # synthetic fixture. Production default is 5e8 (matches T7 natively).
    _assoc_thresh = int(getattr(settings, 'association_streaming_threshold_voxels', 500_000_000)) if settings is not None else 500_000_000
    # When chunked, hand a result_name so the
    # per-component helper allocates `result` as a workspace zarr instead of
    # a 22 GB / 111 GB int32 numpy. Pass 1 result is needed by Pass 2 and
    # downstream extend_objects_GPU; both already accept zarr inputs.
    _result_name = ('associate_result_' + str(int(time.time() * 1000) % 1_000_000)
                     if (use_chunked and workspace is not None) else None)
    return chunked.associate_spines_with_necks_per_component(
        spines, necks, logger,
        spine_dist_um=spine_dist_um,
        input_resXY=input_resXY,
        input_resZ=input_resZ,
        workspace=workspace,
        streaming_threshold_voxels=_assoc_thresh,
        result_name=_result_name,
    )


def _associate_spines_with_necks_legacy_tiled(spines, necks, logger):
    """LEGACY: Tiled GPU version using iterative dilation (less accurate than EDT).

    Kept as fallback until per-component EDT (associate_spines_with_necks_per_component)
    is fully validated. Produces approximate nearest-spine assignments via dilation
    rather than exact EDT distances.

    Processes spatial tiles with overlap on GPU. Each tile runs the iterative
    dilation independently. Overlap ensures spines near tile boundaries can
    reach nearby neck voxels.
    """
    shape = spines.shape
    result = np.zeros_like(spines)

    # Calculate tile size to fit in ~20% of GPU VRAM (with 5x overhead)
    flush_gpu()
    free, total = cp.cuda.runtime.memGetInfo()
    bytes_per_voxel = spines.dtype.itemsize * 5  # 5 arrays simultaneously
    target_voxels = int(free * 0.20 / bytes_per_voxel)
    # Use aspect-ratio-aware tile sizing
    aspect = shape[1] / max(1, shape[0])
    tz = max(1, int((target_voxels / (aspect * aspect)) ** (1 / 3)))
    ty = tx = max(1, int(tz * aspect))
    tz = min(tz, shape[0])
    ty = min(ty, shape[1])
    tx = min(tx, shape[2])

    # Overlap must cover maximum expected neck growth distance
    overlap = 30  # voxels — generous for neck association

    logger.info(f"     Tiled neck association: tile=({tz},{ty},{tx}), overlap={overlap}")

    for z in range(0, shape[0], max(1, tz - overlap)):
        for y in range(0, shape[1], max(1, ty - overlap)):
            for x in range(0, shape[2], max(1, tx - overlap)):
                z_end = min(z + tz, shape[0])
                y_end = min(y + ty, shape[1])
                x_end = min(x + tx, shape[2])

                tile_spines = spines[z:z_end, y:y_end, x:x_end]
                tile_necks = necks[z:z_end, y:y_end, x:x_end]

                # Skip empty tiles
                if np.max(tile_spines) == 0 and np.max(tile_necks) == 0:
                    continue

                # Process tile on GPU
                labeled_cp = cp.array(tile_spines)
                binary_cp = cp.array(tile_necks > 0)
                struct = cp.ones((3, 3, 3), dtype=cp.bool_)

                growth = True
                while growth:
                    dilated = binary_dilation(labeled_cp > 0, structure=struct)
                    new_growth = dilated & binary_cp & (labeled_cp == 0)

                    expanded = cp.zeros_like(labeled_cp)
                    for lid in cp.unique(labeled_cp):
                        if lid == 0:
                            continue
                        lmask = (labeled_cp == lid)
                        dilated_lmask = binary_dilation(lmask, structure=struct)
                        expanded = cp.where(dilated_lmask & new_growth, lid, expanded)

                    labeled_cp = cp.where(new_growth, expanded, labeled_cp)
                    growth = new_growth.any()

                tile_result = labeled_cp.get()
                del labeled_cp, binary_cp, expanded, new_growth, dilated
                flush_gpu()

                # Paste core region (exclude overlap margins)
                half_ov = overlap // 2
                core_z = z + half_ov if z > 0 else z
                core_y = y + half_ov if y > 0 else y
                core_x = x + half_ov if x > 0 else x
                core_z_end = z_end - half_ov if z_end < shape[0] else z_end
                core_y_end = y_end - half_ov if y_end < shape[1] else y_end
                core_x_end = x_end - half_ov if x_end < shape[2] else x_end

                local_z = core_z - z
                local_y = core_y - y
                local_x = core_x - x
                local_z_end = core_z_end - z
                local_y_end = core_y_end - y
                local_x_end = core_x_end - x

                result[core_z:core_z_end, core_y:core_y_end, core_x:core_x_end] = \
                    tile_result[local_z:local_z_end, local_y:local_y_end, local_x:local_x_end]

    logger.info(f"     Tiled neck association complete.")
    return result

@mp.profile_mem()
def _adaptive_flush_cadence():
    """Return the number of per-spine pathfinder iterations between GPU flushes,
    scaled to the detected GPU size. More frequent flushes keep peak VRAM low
    on constrained GPUs; the <1 ms cost per flush is negligible compared with
    a pathfinding iteration (~0.3 s), so throughput is effectively unchanged.

    Tiers (total GPU memory):
        ≤4 GB   → flush every 2   (laptop / integrated)
        ≤8 GB   → flush every 5   (entry-level discrete)
        ≤16 GB  → flush every 10  (RTX 30/40 mid-range)
        ≤32 GB  → flush every 25  (TITAN RTX / RTX 6000)
        >32 GB  → flush every 50  (ADA RTX 6000 48 GB and similar)
    """
    try:
        gb = cp.cuda.runtime.memGetInfo()[1] / (1024 ** 3)
    except Exception:
        return 10  # conservative default if detection fails
    if gb <= 4.5:
        return 2
    if gb <= 8.5:
        return 5
    if gb <= 16.5:
        return 10
    if gb <= 32.5:
        return 25
    return 50


def extend_objects_GPU(objects, target_objects, intensity, settings, locations, logger,
                         workspace=None, result_array_name=None,
                         obstacle_labels=None):
    """Extend spine labels toward dendrites via GPU pathfinding.

    # TODO: PERF — batch 8-16 spines per GPU launch to reduce transfer overhead.
    # Currently each spine is processed independently with GPU sync between each.
    # Expected 2-3x speedup at 100K spines.

    Uses CPU-orchestrated per-label subvolume extraction to avoid uploading
    the full volume to GPU. Each spine's bounding box is extracted from CPU,
    processed on GPU, and results are accumulated on CPU (or zarr when
    `workspace` is provided — Phase G RAM saving at T7 scale, ~22 GB per call).
    Per-spine subvolume writes use np.maximum semantics against the current
    slab read from the accumulator, which works identically for numpy and zarr.

    Parameters
    ----------
    workspace : chunked.ZarrWorkspace, optional
        When provided, the full-volume `result_necks` accumulator is allocated
        as a zarr-backed array via `workspace.create_array()`. Saves ~22 GB at
        T7 (int32 × 5.5B voxels). Writes are slab-scoped (read-modify-write per
        spine bbox), same pattern as numpy.
    result_array_name : str, optional
        When `workspace` is provided, the zarr array name to use. Default:
        '_extend_result_necks'. Two calls in spine_and_whole_neuron_processing
        must use distinct names to avoid collision.
    """
    logger.info("     Finding spine necks using GPU...")

    if not (objects.shape == target_objects.shape):
        raise ValueError("Input array shapes do not match")

    # Get unique labels via slab-scan (avoids 20 GB flatten from np.unique on T7)
    _unique_set = set()
    for _z in range(0, objects.shape[0], 64):
        _ze = min(_z + 64, objects.shape[0])
        _unique_set.update(np.unique(objects[_z:_ze]).tolist())
    _unique_set.discard(0)
    unique_labels_np = np.array(sorted(_unique_set))

    # Use find_objects for O(1) bounding box lookup. For zarr inputs (Phase H),
    # route to find_objects_streaming — bare ndimage.find_objects would
    # materialize the full volume via np.asarray (22 GB int32 at T7).
    max_label = int(unique_labels_np.max()) if len(unique_labels_np) > 0 else 0
    if max_label == 0:
        slices_lookup = []
    elif hasattr(objects, 'chunks'):
        _bbox_map = chunked.find_objects_streaming(objects, logger=logger)
        slices_lookup = [None] * max_label
        for _lbl, (_z0, _z1, _y0, _y1, _x0, _x1) in _bbox_map.items():
            if 1 <= _lbl <= max_label:
                slices_lookup[_lbl - 1] = (slice(_z0, _z1), slice(_y0, _y1), slice(_x0, _x1))
    else:
        slices_lookup = ndimage.find_objects(objects, max_label)

    # Sort labels by Z-origin for spatial locality (better cache hit rate)
    label_z_pairs = []
    for lv in unique_labels_np:
        sl = slices_lookup[int(lv) - 1]
        z_origin = sl[0].start if sl is not None else 0
        label_z_pairs.append((int(lv), z_origin))
    label_z_pairs.sort(key=lambda x: x[1])

    # Accumulate results on CPU (numpy) OR zarr (Phase G) when workspace given.
    # zarr-backed accumulator saves ~22 GB RAM at T7 per call; the per-spine
    # read-modify-write pattern below is identical either way.
    full_shape = objects.shape
    _accum_dtype = objects.dtype if hasattr(objects, 'dtype') else np.int32
    if workspace is not None:
        _rname = result_array_name if result_array_name else '_extend_result_necks'
        result_necks = workspace.create_array(
            _rname, full_shape, dtype=_accum_dtype, chunks=chunked.STREAMING_CHUNKS)
        logger.info(f"       extend_objects_GPU: result accumulator is zarr "
                    f"('{_rname}', shape={tuple(full_shape)}, dtype={_accum_dtype}) — "
                    f"Phase G saves ~{full_shape[0] * full_shape[1] * full_shape[2] * _accum_dtype.itemsize / 1e9:.1f} GB RAM.")
    else:
        result_necks = np.zeros_like(objects)

    # Margin in voxels (6 microns converted to voxels)
    margin_y = margin_x = int(6 / settings.input_resXY)
    margin_z = int(6 / settings.input_resZ)

    n_skipped_no_target = 0
    n_processed = 0
    n_attempted = 0
    n_total = len(label_z_pairs)
    _milestone_every = max(100, n_total // 20) if n_total else 0
    gpu_flush_counter = 0
    # Adaptive flush cadence — scales from every 2 spines on 4 GB GPUs up to
    # every 50 on 48 GB GPUs. Keeps peak VRAM manageable on constrained
    # hardware without slowing down throughput on larger cards.
    flush_every = _adaptive_flush_cadence()
    try:
        _gpu_total_gb = cp.cuda.runtime.memGetInfo()[1] / (1024 ** 3)
    except Exception:
        _gpu_total_gb = 0.0
    logger.info(f"       extend_objects_GPU flush cadence: every {flush_every} "
                f"spines (GPU={_gpu_total_gb:.1f} GB, {n_total} total spines)")
    time_start = time.time()

    for label_value, _ in label_z_pairs:
        n_attempted += 1
        if _milestone_every and n_attempted % _milestone_every == 0:
            logger.info(f"       Extend objects GPU: {n_attempted}/{n_total} attempted "
                        f"({n_processed} processed, {n_skipped_no_target} skipped)...")

        # Get bounding box from find_objects (0-indexed)
        sl = slices_lookup[label_value - 1]
        if sl is None:
            continue

        # Expand bounding box with margin
        z_min = max(0, sl[0].start - margin_z)
        z_max = min(full_shape[0], sl[0].stop + margin_z)
        y_min = max(0, sl[1].start - margin_y)
        y_max = min(full_shape[1], sl[1].stop + margin_y)
        x_min = max(0, sl[2].start - margin_x)
        x_max = min(full_shape[2], sl[2].stop + margin_x)

        # Early exit: skip if no target voxels in the expanded bbox.
        # Avoids expensive GPU EDT + gaussian_filter for spines with no
        # reachable dendrite in their neighborhood.
        target_subvolume = target_objects[z_min:z_max, y_min:y_max, x_min:x_max]
        if not np.any(target_subvolume):
            n_skipped_no_target += 1
            continue

        # Extract subvolumes from CPU arrays
        _spines_crop = np.asarray(objects[z_min:z_max, y_min:y_max, x_min:x_max])
        object_sub_volume = (_spines_crop == label_value).astype(objects.dtype)
        intensity_subvolume = intensity[z_min:z_max, y_min:y_max, x_min:x_max]

        # Other spine heads in this bbox are FORBIDDEN for this neck's path.
        # Without this, the cost grid (which favours bright voxels) routes
        # paths through neighbouring heads — biologically wrong and visible
        # at MIP scale as ~21% of T2 necks skirting/clipping adjacent heads.
        # When objects is e.g. neck_labels_updated (call #1), spine HEADS are
        # NOT in `objects` and the default obstacle derived from `objects` is
        # blind to them. The caller can pass `obstacle_labels` (typically
        # spines_filtered) to make all heads obstacles regardless of which
        # label volume is being extended.
        if obstacle_labels is not None:
            _obs_crop = np.asarray(obstacle_labels[z_min:z_max, y_min:y_max, x_min:x_max])
            obstacle_subvolume = (_obs_crop > 0) & (_obs_crop != label_value)
            # Also forbid any non-self labels in the source `objects` (e.g.
            # other neck assignments) — same contamination class, just from
            # a different label volume.
            obstacle_subvolume |= ((_spines_crop > 0) & (_spines_crop != label_value))
        else:
            obstacle_subvolume = (_spines_crop > 0) & (_spines_crop != label_value)

        try:
            result_label, path_volume, distance_vol = extend_single_object_GPU_v2(
                label_value, object_sub_volume, target_subvolume, intensity_subvolume, settings, logger,
                obstacle_subvolume=obstacle_subvolume)

            # Accumulate using np.maximum — same semantics for numpy (view)
            # and zarr (read slab, merge, write back). `np.asarray` on a zarr
            # slice materializes only that subvolume (~bbox-size MB).
            _dsl = (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))
            if workspace is not None:
                _existing = np.asarray(result_necks[_dsl])
                result_necks[_dsl] = np.maximum(_existing, path_volume)
                del _existing
            else:
                result_necks[_dsl] = np.maximum(result_necks[_dsl], path_volume)
            n_processed += 1

        except Exception as e:
            logger.error(f"Error processing object with label {label_value}: {str(e)}")

        gpu_flush_counter += 1

        # Periodic GPU cleanup. Flush cadence is GPU-size adaptive (see above).
        # Verified: batching does not affect path quality (64-pixel difference
        # persists with per-spine flush — difference is from GPU non-determinism).
        if gpu_flush_counter >= flush_every:
            flush_gpu()
            gpu_flush_counter = 0

    # Final GPU cleanup
    if gpu_flush_counter > 0:
        flush_gpu()

    elapsed = time.time() - time_start
    per_spine = elapsed / max(n_processed, 1)
    logger.info(f"      Processed {n_processed}/{n_total} objects in {elapsed:.1f}s "
                f"({per_spine:.3f}s/spine)")
    if n_skipped_no_target > 0:
        logger.info(f"      Skipped {n_skipped_no_target} objects with no target in bbox")

    return result_necks


def bridge_spine_neck_via_pathfinding(connected_necks, spines_filtered,
                                         dendrites_mask, still_split_labels,
                                         intensity, settings, locations, logger):
    """Second-pass spine<->neck bridge using signal-aware pathfinding.

    For each still-split spine (where morph closing couldn't bridge the gap),
    calls `extend_single_object_GPU_v2` with the spine mask as `object` and
    the spine's assigned-neck mask as `target`, using the neuron intensity
    image. The pathfinder follows bright signal through a distance+gamma-
    intensity augmented map — bridging gaps of up to ~20 voxels where real
    fluorescence supports the connection, skipping gaps with no signal.

    Path voxels are written into `connected_necks` with the spine's label,
    respecting existing labels and dendrite (never overwrites them).
    """
    if not still_split_labels:
        return connected_necks

    logger.info(f"     Signal-aware pathfinding bridge for "
                f"{len(still_split_labels)} still-split spines...")

    # Same 6 µm margin as the main extend_objects_GPU — no accuracy compromise.
    # Scalability is achieved via adaptive flush cadence (below), not by
    # shrinking the subvolume: smaller margins could miss multi-voxel gaps.
    margin_y = margin_x = max(1, int(6 / settings.input_resXY))
    margin_z = max(1, int(6 / settings.input_resZ))

    _flush_every = _adaptive_flush_cadence()
    try:
        _gpu_total_gb = cp.cuda.runtime.memGetInfo()[1] / (1024 ** 3)
    except Exception:
        _gpu_total_gb = 0.0
    logger.info(f"       Bridge GPU policy: margin=6.0 µm, "
                f"flush every {_flush_every} spines (GPU={_gpu_total_gb:.1f} GB)")

    full_shape = connected_necks.shape

    # Per-label bboxes via streaming scan (never materializes zarr spines).
    from RESPAN.ImageAnalysis import ChunkedProcessing as _chunked
    sp_bboxes = _chunked.find_objects_streaming(spines_filtered, logger=None)
    nk_bboxes = _chunked.find_objects_streaming(connected_necks, logger=None)

    n_added = 0
    n_bridged_labels = 0
    n_skipped = 0
    gpu_flush_counter = 0
    t_start = time.time()

    for label_value in still_split_labels:
        sp_bb = sp_bboxes.get(int(label_value))
        nk_bb = nk_bboxes.get(int(label_value))
        if sp_bb is None and nk_bb is None:
            n_skipped += 1
            continue

        # Union bbox of spine + assigned-neck extents.
        if sp_bb is None:
            z0, z1, y0, y1, x0, x1 = nk_bb
        elif nk_bb is None:
            z0, z1, y0, y1, x0, x1 = sp_bb
        else:
            z0 = min(sp_bb[0], nk_bb[0]); z1 = max(sp_bb[1], nk_bb[1])
            y0 = min(sp_bb[2], nk_bb[2]); y1 = max(sp_bb[3], nk_bb[3])
            x0 = min(sp_bb[4], nk_bb[4]); x1 = max(sp_bb[5], nk_bb[5])

        # Expand bbox by margin to give the pathfinder room to route around.
        z_min = max(0, z0 - margin_z); z_max = min(full_shape[0], z1 + margin_z)
        y_min = max(0, y0 - margin_y); y_max = min(full_shape[1], y1 + margin_y)
        x_min = max(0, x0 - margin_x); x_max = min(full_shape[2], x1 + margin_x)
        dsl = (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))

        # Per-label masks.
        sp_crop = np.asarray(spines_filtered[dsl])
        nk_crop = connected_necks[dsl]
        object_sub = (sp_crop == label_value).astype(connected_necks.dtype)
        target_sub = (nk_crop == label_value).astype(connected_necks.dtype)
        if not target_sub.any() or not object_sub.any():
            n_skipped += 1
            continue

        # Pathfind intensity from neuron channel.
        intensity_sub = np.asarray(intensity[dsl])

        # Other spine heads in this bbox are FORBIDDEN for the bridge path
        # (Same routing failure mode as extend_objects_GPU.)
        obstacle_sub = (sp_crop > 0) & (sp_crop != label_value)

        try:
            _, path_volume, _ = extend_single_object_GPU_v2(
                int(label_value), object_sub, target_sub, intensity_sub,
                settings, logger, obstacle_subvolume=obstacle_sub)
        except Exception as e:
            logger.error(f"       Pathfinder bridge error on label {label_value}: {e}")
            n_skipped += 1
            continue

        # Write path voxels into connected_necks only where:
        #  - path has a value (non-zero)
        #  - destination is currently background (0) — don't overwrite spines,
        #    other necks, or this label's own existing voxels
        #  - not inside dendrite
        dend_crop = np.asarray(dendrites_mask[dsl]).astype(bool, copy=False)
        writable = (path_volume > 0) & (nk_crop == 0) & (sp_crop == 0) & (~dend_crop)
        if writable.any():
            neck_sub = nk_crop.copy()
            neck_sub[writable] = label_value
            connected_necks[dsl] = neck_sub
            n_added += int(writable.sum())
            n_bridged_labels += 1

        gpu_flush_counter += 1
        if gpu_flush_counter >= _flush_every:
            flush_gpu()
            gpu_flush_counter = 0

    if gpu_flush_counter > 0:
        flush_gpu()

    elapsed = time.time() - t_start
    logger.info(
        f"       Pathfinder bridge: {n_bridged_labels}/{len(still_split_labels)} "
        f"gapped spines connected ({n_added} voxels added, "
        f"{n_skipped} skipped) in {elapsed:.1f}s")
    return connected_necks


def drop_disconnected_neck_fragments(connected_necks, spines_filtered, logger):
    """Keep only the spine-connected component of each label in connected_necks.

    Pass 2 neck assignment uses nearest-reference-label semantics within a
    distance cap. A neck fragment can legitimately be assigned label L if it
    is near an extension corridor that also carries label L — even if the
    fragment is not itself 26-adjacent to spine L. This produces "floating"
    dendrite-side neck chunks unconnected to the spine head.

    This cleanup walks per-label bboxes, computes connected components of
    (spine ∪ neck) for each label, and — when there are multiple components —
    keeps only the voxels in the component that contains the spine head.
    Voxels in orphan fragments are zeroed in connected_necks. Spines with no
    neck voxels (no original neck assigned) are untouched.
    """
    from RESPAN.ImageAnalysis import ChunkedProcessing as _chunked
    from scipy import ndimage as _ndi

    # Use the streaming bbox helper — works for numpy and zarr, never
    # materializes full volumes.
    sp_bboxes = _chunked.find_objects_streaming(spines_filtered, logger=None)
    nk_bboxes = _chunked.find_objects_streaming(connected_necks, logger=None)
    structure = _ndi.generate_binary_structure(3, 3)  # 26-connectivity

    n_labels_checked = 0
    n_labels_fixed = 0
    n_voxels_dropped = 0
    dropped_label_ids = []

    for label, nk_bb in nk_bboxes.items():
        sp_bb = sp_bboxes.get(label)
        if sp_bb is None:
            # No spine head for this neck label — nothing to anchor to. Leave
            # as-is (caller decides whether to drop via spurious filter).
            continue

        # Union bbox
        z0 = min(sp_bb[0], nk_bb[0]); z1 = max(sp_bb[1], nk_bb[1])
        y0 = min(sp_bb[2], nk_bb[2]); y1 = max(sp_bb[3], nk_bb[3])
        x0 = min(sp_bb[4], nk_bb[4]); x1 = max(sp_bb[5], nk_bb[5])
        dsl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

        sp_crop = np.asarray(spines_filtered[dsl])
        nk_crop = connected_necks[dsl]
        spine_mask = (sp_crop == label)
        neck_mask = (nk_crop == label)
        combined = spine_mask | neck_mask

        n_labels_checked += 1
        cc, n_cc = _ndi.label(combined, structure=structure)
        if n_cc <= 1:
            continue

        # Find the CC(s) that contain spine voxels — those are the real
        # head-attached components. Anything else is orphan.
        spine_cc_ids = set(int(x) for x in np.unique(cc[spine_mask]) if x != 0)
        if not spine_cc_ids:
            # Spine mask empty in bbox (shouldn't happen given sp_bb exists).
            continue

        # Orphan mask: neck voxels whose CC is NOT in the spine's CC set.
        cc_is_spine = np.zeros(n_cc + 1, dtype=bool)
        for cid in spine_cc_ids:
            cc_is_spine[cid] = True
        orphan = neck_mask & ~cc_is_spine[cc]
        n_orphan_vox = int(orphan.sum())
        if n_orphan_vox == 0:
            continue

        # Zero out the orphan neck voxels for this label only.
        neck_sub = nk_crop.copy()
        neck_sub[orphan] = 0
        connected_necks[dsl] = neck_sub
        n_labels_fixed += 1
        n_voxels_dropped += n_orphan_vox
        dropped_label_ids.append(int(label))

    _sample = sorted(dropped_label_ids)[:10]
    _sample_str = ",".join(str(x) for x in _sample)
    _suffix = "" if len(dropped_label_ids) <= 10 else f", ... (+{len(dropped_label_ids) - 10} more)"
    logger.info(
        f"       Dropped disconnected neck fragments: {n_labels_fixed}/{n_labels_checked} "
        f"labels cleaned ({n_voxels_dropped} orphan voxels removed). "
        f"Sample: [{_sample_str}{_suffix}].")
    return connected_necks


def drop_wrong_direction_neck_fragments(connected_necks, dendrite_distance,
                                          settings, logger):
    """For each label with ≥2 neck CCs, keep only the CC whose closest voxel
    reaches nearest to the dendrite — drop the rest.

    Catches the sibling case of `drop_disconnected_neck_fragments`. That function
    uses CC analysis of `(spine ∪ neck)` and leaves a neck-side fragment alone
    if it happens to touch the spine head (so combined = 1 CC). Spines with
    small / noisy heads can have TWO disconnected-in-neck-space fragments
    connected via the shared head — one going toward the dendrite (the real
    neck), one going away (a stray Pass-2 assignment from an occluded neighbour).

    Biological necks monotonically approach the dendrite. A neck CC whose
    MINIMUM distance-to-dendrite never drops below the candidate winner's
    minimum is going the wrong direction and is dropped.

    Only `connected_necks` is modified. Spines with a single neck CC are
    untouched. Spines with no neck voxels are untouched.
    """
    from RESPAN.ImageAnalysis import ChunkedProcessing as _chunked
    from scipy import ndimage as _ndi

    nk_bboxes = _chunked.find_objects_streaming(connected_necks, logger=None)
    structure = _ndi.generate_binary_structure(3, 3)  # 26-connectivity

    # Tolerance: a real neck CC's min_dd sometimes differs from the winner
    # by a fraction of a voxel due to discretization. Accept CCs whose min_dd
    # is within TOLERANCE µm of the winner's min_dd. Default: 0.2 µm ≈ 3 vox
    # at 65 nm XY. Tightens a stringent rule into a pragmatic one without
    # opening the door to obvious wrong-direction fragments.
    tol_um = float(getattr(settings, 'neck_direction_tolerance_um', 0.2))
    res_xy = float(getattr(settings, 'input_resXY', 0.065))
    # dendrite_distance is in VOXELS (not µm) when produced by chunked EDT
    # without sampling, or in µm when sampling is applied. Check magnitude:
    # voxel-based EDTs at 65 nm XY produce values O(1-100); µm-based produce
    # O(0.1-10). A safer rule: convert tolerance to voxels via res_xy.
    tol_vox = tol_um / res_xy

    n_labels_checked = 0
    n_labels_fixed = 0
    n_cc_dropped = 0
    n_voxels_dropped = 0
    dropped_label_ids = []

    for label, bb in nk_bboxes.items():
        z0, z1, y0, y1, x0, x1 = bb
        dsl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

        nk_crop = np.asarray(connected_necks[dsl])
        nk_mask = (nk_crop == label)
        if nk_mask.sum() == 0:
            continue

        cc, n_cc = _ndi.label(nk_mask, structure=structure)
        if n_cc <= 1:
            continue

        n_labels_checked += 1

        # Compute min-dist-to-dendrite for each CC.
        dd_crop = np.asarray(dendrite_distance[dsl])
        min_dds = []
        cc_sizes = []
        for c in range(1, n_cc + 1):
            m = (cc == c)
            min_dds.append(float(dd_crop[m].min()))
            cc_sizes.append(int(m.sum()))

        winner_dd = min(min_dds)
        keep = [(d <= winner_dd + tol_vox) for d in min_dds]

        if all(keep):
            # All CCs reach the dendrite within tolerance — biologically
            # unusual (branched neck) but not a clear error. Leave alone.
            continue

        # Drop losing CCs.
        drop_mask = np.zeros_like(nk_mask)
        dropped_here = 0
        for c_idx, keep_flag in enumerate(keep):
            if not keep_flag:
                drop_mask |= (cc == (c_idx + 1))
                dropped_here += cc_sizes[c_idx]

        if dropped_here == 0:
            continue

        nk_crop_out = nk_crop.copy()
        nk_crop_out[drop_mask] = 0
        connected_necks[dsl] = nk_crop_out

        n_labels_fixed += 1
        n_cc_dropped += sum(1 for k in keep if not k)
        n_voxels_dropped += dropped_here
        dropped_label_ids.append(int(label))

    _sample = sorted(dropped_label_ids)[:10]
    _sample_str = ",".join(str(x) for x in _sample)
    _suffix = "" if len(dropped_label_ids) <= 10 else f", ... (+{len(dropped_label_ids) - 10} more)"
    logger.info(
        f"       Dropped wrong-direction neck fragments: "
        f"{n_labels_fixed}/{n_labels_checked} labels cleaned "
        f"({n_cc_dropped} CCs, {n_voxels_dropped} voxels — kept CC closest to dendrite). "
        f"Sample: [{_sample_str}{_suffix}].")
    return connected_necks


def trim_necks_touching_other_spines(connected_necks, spines_filtered, logger):
    """Drop neck voxels that are 26-adjacent to a different spine's head.

    Biologically, a spine's neck should terminate at the dendrite, never on
    another spine head. Pass 2 proximity assignment + pathfinder bridges can
    route a neck through a neighbouring spine's territory, producing the
    "chained spines" topology (Head_A → Neck_A → Head_B → Neck_B → Dendrite).
    This function restores the pre-pathfinding behaviour by dropping any neck
    voxel of label X that touches spine label Y ≠ X.

    Acts slab-by-slab (zarr-safe), uses 26-connectivity.
    """
    from RESPAN.ImageAnalysis import ChunkedProcessing as _chunked
    from scipy import ndimage as _ndi

    structure = _ndi.generate_binary_structure(3, 3)  # 26-conn

    # Per-label bboxes — streaming helper (works for zarr or numpy).
    nk_bboxes = _chunked.find_objects_streaming(connected_necks, logger=None)
    sp_bboxes = _chunked.find_objects_streaming(spines_filtered, logger=None)

    full_shape = connected_necks.shape
    n_labels_trimmed = 0
    n_voxels_dropped = 0

    for label, nk_bb in nk_bboxes.items():
        if int(label) == 0:
            continue
        z0, z1, y0, y1, x0, x1 = nk_bb
        # Halo by 1 voxel so 26-conn neighbours from adjacent spines are visible.
        hz0 = max(0, z0 - 1); hz1 = min(full_shape[0], z1 + 1)
        hy0 = max(0, y0 - 1); hy1 = min(full_shape[1], y1 + 1)
        hx0 = max(0, x0 - 1); hx1 = min(full_shape[2], x1 + 1)
        dsl = (slice(hz0, hz1), slice(hy0, hy1), slice(hx0, hx1))

        nk_crop = np.asarray(connected_necks[dsl])
        sp_crop = np.asarray(spines_filtered[dsl])

        neck_of_label = (nk_crop == int(label))
        if not neck_of_label.any():
            continue

        # Other spines' head voxels within this bbox (label != this one, != 0).
        other_heads = (sp_crop > 0) & (sp_crop != int(label))
        if not other_heads.any():
            continue

        # A neck voxel is "bad" if it is 26-adjacent to another spine's head.
        # Dilate other_heads by 1 voxel 26-conn and AND with neck_of_label.
        dilated_other = _ndi.binary_dilation(
            other_heads, structure=structure, iterations=1)
        bad_neck = neck_of_label & dilated_other
        n_bad = int(bad_neck.sum())
        if n_bad == 0:
            continue

        # Zero out the bad neck voxels for this label only.
        nk_sub = nk_crop.copy()
        nk_sub[bad_neck] = 0
        connected_necks[dsl] = nk_sub
        n_labels_trimmed += 1
        n_voxels_dropped += n_bad

    logger.info(
        f"       Trimmed necks touching other spines: {n_labels_trimmed} labels "
        f"({n_voxels_dropped} neck voxels zeroed)")
    return connected_necks


def _audit_3d_label_overlaps(spines_filtered, connected_necks, labeled_dendrites,
                              logger):
    """Diagnostic: count 3D voxel overlaps between the three label channels.

    By construction, spines_filtered / connected_necks / labeled_dendrites
    should be mutually exclusive at the voxel level. Any non-zero count
    reported here indicates a pipeline bug (e.g. a pathfinder bridge wrote
    into a dendrite voxel, or a per-cluster watershed spilled over).

    Runs slab-by-slab (zarr-safe). Reports only — does NOT modify the volumes.
    """
    shape = spines_filtered.shape
    row_bytes = int(shape[1]) * int(shape[2]) * 4
    # Cap slab at ~2 GB.
    slab = max(1, min(64, int(2 * 1024**3 / max(row_bytes, 1))))

    def _safe_slab(arr, z0, z1):
        return np.asarray(arr[z0:z1])

    n_sf_cn_same = 0          # same label in both spines_filtered and connected_necks
    n_sf_cn_diff = 0          # different labels in the same voxel (real overlap)
    n_sf_dend = 0             # spine head voxel inside dendrite
    n_cn_dend = 0             # neck voxel inside dendrite
    total = int(shape[0]) * int(shape[1]) * int(shape[2])

    for z0 in range(0, shape[0], slab):
        z1 = min(z0 + slab, shape[0])
        sf = _safe_slab(spines_filtered, z0, z1)
        cn = _safe_slab(connected_necks, z0, z1)
        dn = _safe_slab(labeled_dendrites, z0, z1)

        sf_pos = sf > 0
        cn_pos = cn > 0
        dn_pos = dn > 0

        both_sfcn = sf_pos & cn_pos
        if both_sfcn.any():
            same = (sf == cn) & both_sfcn
            n_sf_cn_same += int(same.sum())
            n_sf_cn_diff += int((both_sfcn & ~same).sum())
        n_sf_dend += int((sf_pos & dn_pos).sum())
        n_cn_dend += int((cn_pos & dn_pos).sum())

    logger.info("    3D overlap audit (all should be 0 for disjoint labels):")
    logger.info(f"      spines_filtered ∩ connected_necks, same label : {n_sf_cn_same} vox")
    logger.info(f"      spines_filtered ∩ connected_necks, diff label : {n_sf_cn_diff} vox")
    logger.info(f"      spines_filtered ∩ labeled_dendrites           : {n_sf_dend} vox")
    logger.info(f"      connected_necks ∩ labeled_dendrites           : {n_cn_dend} vox")
    worst = max(n_sf_cn_same, n_sf_cn_diff, n_sf_dend, n_cn_dend)
    if worst == 0:
        logger.info("      ✓ No 3D overlaps — label channels are disjoint.")
    else:
        logger.warning(
            f"      ⚠ 3D overlap detected — {worst} voxels in worst pair. "
            "Investigate pipeline write logic (see bridge_spine_neck_via_pathfinding, "
            "second_pass_annotation, recover_filopodia_from_orphan_necks).")


def _build_grouped_spine_volume(spines_filtered, connected_necks, parent_map,
                                  workspace, logger):
    """Build a per-voxel grouped-spine label volume for the validation MIP.

    Each spine-head voxel and its assigned-neck voxel are relabelled with
    `parent_map[label]` (the multi-head group's parent_spine_id). Solitary
    spines map to themselves. Slab-iterates so it works for both numpy and
    zarr inputs. When `workspace` is provided, the output is a workspace
    zarr; otherwise a numpy array.

    Returns the grouped volume (same shape as spines_filtered).
    """
    shape = spines_filtered.shape
    is_zarr = (hasattr(spines_filtered, 'chunks')
               or hasattr(connected_necks, 'chunks'))
    out_dtype = np.int32

    # Build LUT covering all observed labels.
    max_label = 0
    if is_zarr:
        for z in range(0, shape[0], 64):
            ze = min(z + 64, shape[0])
            sf_max = int(np.asarray(spines_filtered[z:ze]).max())
            cn_max = int(np.asarray(connected_necks[z:ze]).max())
            max_label = max(max_label, sf_max, cn_max)
    else:
        max_label = max(int(np.asarray(spines_filtered).max()),
                        int(np.asarray(connected_necks).max()))
    if max_label == 0:
        return None

    lut = np.arange(max_label + 1, dtype=out_dtype)
    for k, v in parent_map.items():
        if 0 < k <= max_label:
            lut[k] = int(v)

    if is_zarr and workspace is not None:
        out = workspace.create_array(
            'grouped_spines', shape, dtype=out_dtype,
            chunks=chunked.STREAMING_CHUNKS)
        is_out_zarr = True
    else:
        out = np.zeros(shape, dtype=out_dtype)
        is_out_zarr = False

    row_bytes = int(shape[1]) * int(shape[2])
    slab = max(1, min(64, int(2 * (1024 ** 3) / max(row_bytes, 1))))
    for z in range(0, shape[0], slab):
        ze = min(z + slab, shape[0])
        sf = np.asarray(spines_filtered[z:ze])
        cn = np.asarray(connected_necks[z:ze])
        # Relabel spine voxels first, then OR-in relabelled neck voxels.
        # Where both are nonzero (shouldn't happen by construction, but
        # defensive), spines win — gives the head its own parent.
        sub = lut[sf].astype(out_dtype, copy=False)
        cn_relabelled = lut[cn].astype(out_dtype, copy=False)
        sub = np.where(sub > 0, sub, cn_relabelled)
        if is_out_zarr:
            out[z:ze] = sub
        else:
            out[z:ze] = sub
    return out


def detect_multi_head_spines(spines_filtered, connected_necks, neck_mask, logger,
                              workspace=None):
    """Detect multi-headed spines via physical connectivity.

    Two spine heads belong to the same multi-head group when their territories
    (head + assigned neck + raw nnU-Net neck mask) are 26-connected. The
    nnU-Net mask term catches the common case where each head is partial-spine
    (no assigned neck) but the network predicted a continuous neck region
    bridging them — without it those groups are missed.

    Two-tier representation matching the literature standard
    (Imaris FilamentTracer, Spinifel 2024, NeuTu, NeuronStudio): each head
    keeps its individual spine_id; a `parent_spine_id` (smallest spine_id in
    the group) marks group membership, and `head_index` orders heads within
    the group (0 = primary). Solitary spines have parent_spine_id == spine_id,
    head_index == 0, group_size == 1.

    Parameters
    ----------
    spines_filtered : numpy.ndarray or zarr.Array
        Per-head spine label volume.
    connected_necks : numpy.ndarray or zarr.Array
        Per-head neck label volume (label-aligned with spines_filtered).
    neck_mask : numpy.ndarray or zarr.Array, optional
        Raw nnU-Net neck binary mask. May be None when neck_generation=False.
    logger : logging.Logger

    Returns
    -------
    parent_map : dict[int, int]
        Mapping spine_label -> parent_spine_id. Solitary spines map to self.
    group_metadata : dict[int, dict]
        Per-parent-id aggregate:
        {parent_id: {'n_heads': int, 'member_ids': sorted list[int]}}.
        Only includes groups with n_heads >= 2.
    """
    shape = spines_filtered.shape
    if len(shape) != 3:
        logger.info("    Multi-head spine detection: expected 3D labels, skipping.")
        return {}, {}

    # Slab-iterating territory union — supports both numpy and zarr inputs.
    is_zarr = (hasattr(spines_filtered, 'chunks')
               or hasattr(connected_necks, 'chunks')
               or (neck_mask is not None and hasattr(neck_mask, 'chunks')))
    row_bytes = int(shape[1]) * int(shape[2])
    slab = max(1, min(64, int(2 * (1024 ** 3) / max(row_bytes, 1))))

    # Build territory union slab-by-slab. For numpy inputs we keep it as a
    # bool numpy array (fits because spines_filtered itself fits as int32).
    # For zarr inputs we route through the workspace zarr — a numpy territory
    # at T_LARGE scale (29.8B voxels = 30 GB bool) fits, but the subsequent
    # ndimage.label allocates an int64 cc array (~238 GB) which OOMs. The
    # workspace path streams CC via connected_components_streaming.
    if is_zarr and workspace is not None:
        territory = workspace.create_array(
            '_multihead_territory', shape, dtype=np.uint8,
            chunks=chunked.STREAMING_CHUNKS)
        any_set = False
        for z in range(0, shape[0], slab):
            ze = min(z + slab, shape[0])
            sf = np.asarray(spines_filtered[z:ze]) > 0
            cn = np.asarray(connected_necks[z:ze]) > 0
            t = sf | cn
            if neck_mask is not None:
                nm = np.asarray(neck_mask[z:ze])
                if nm.dtype != bool:
                    nm = nm > 0
                t |= nm
            if t.any():
                any_set = True
            territory[z:ze] = t.astype(np.uint8)
        if not any_set:
            logger.info("    Multi-head spine detection: no spine territory — nothing to group.")
            return {}, {}
        # Streaming 26-conn CC. Output zarr is int32 (also workspace-backed).
        cc_zarr = workspace.create_array(
            '_multihead_cc', shape, dtype=np.int32,
            chunks=chunked.STREAMING_CHUNKS)
        n_cc, _ = chunked.connected_components_streaming(
            territory, cc_zarr, connectivity=3, logger=None)
        cc = cc_zarr
    else:
        if is_zarr:
            # Zarr but no workspace — fall back to numpy bool territory and
            # ndimage.label. Acceptable for T6 and below; will OOM at T_LARGE.
            territory = np.zeros(shape, dtype=bool)
            for z in range(0, shape[0], slab):
                ze = min(z + slab, shape[0])
                sf = np.asarray(spines_filtered[z:ze]) > 0
                cn = np.asarray(connected_necks[z:ze]) > 0
                t = sf | cn
                if neck_mask is not None:
                    nm = np.asarray(neck_mask[z:ze])
                    if nm.dtype != bool:
                        nm = nm > 0
                    t |= nm
                territory[z:ze] = t
        else:
            sf = np.asarray(spines_filtered) > 0
            cn = np.asarray(connected_necks) > 0
            territory = sf | cn
            if neck_mask is not None:
                nm = np.asarray(neck_mask)
                if nm.dtype != bool:
                    nm = nm > 0
                territory |= nm
        if not territory.any():
            logger.info("    Multi-head spine detection: no spine territory — nothing to group.")
            return {}, {}
        cc, n_cc = ndimage.label(
            territory, structure=ndimage.generate_binary_structure(3, 3))
        del territory

    # For each CC, collect unique spine head labels present in spines_filtered.
    parent_map = {}     # head_label -> parent_label
    group_metadata = {}  # parent_label -> {'n_heads': N, 'member_ids': [...]}
    n_groups = 0
    n_grouped_heads = 0
    n_solitary = 0
    cc_heads = {}  # cc_id -> set(head_labels)
    cc_is_zarr = hasattr(cc, 'chunks')
    if cc_is_zarr or is_zarr:
        for z in range(0, shape[0], slab):
            ze = min(z + slab, shape[0])
            sf_slab = np.asarray(spines_filtered[z:ze])
            cc_slab = np.asarray(cc[z:ze]) if cc_is_zarr else cc[z:ze]
            mask = sf_slab > 0
            if not mask.any():
                continue
            for cc_id, lbl in zip(cc_slab[mask].ravel(), sf_slab[mask].ravel()):
                cc_heads.setdefault(int(cc_id), set()).add(int(lbl))
        del cc
    else:
        sf_full = np.asarray(spines_filtered)
        mask = sf_full > 0
        flat_cc = cc[mask].ravel()
        flat_lbl = sf_full[mask].ravel()
        for cc_id, lbl in zip(flat_cc, flat_lbl):
            cc_heads.setdefault(int(cc_id), set()).add(int(lbl))
        del cc, sf_full

    for cc_id, heads in cc_heads.items():
        heads_sorted = sorted(heads)
        if len(heads_sorted) >= 2:
            parent = heads_sorted[0]
            for h in heads_sorted:
                parent_map[h] = parent
            group_metadata[parent] = {
                'n_heads': len(heads_sorted),
                'member_ids': heads_sorted,
            }
            n_groups += 1
            n_grouped_heads += len(heads_sorted)
        else:
            # Solitary: still record self-mapping so callers can default
            # parent_spine_id = parent_map.get(label, label).
            n_solitary += 1
            for h in heads_sorted:
                parent_map[h] = h

    logger.info(
        f"    Multi-head spine detection: {n_groups} multi-head group(s) "
        f"covering {n_grouped_heads} head(s); {n_solitary} solitary spine(s).")
    return parent_map, group_metadata


def detect_partial_spines(connected_necks, spines_filtered, dendrite_distance,
                           threshold_vox, logger):
    """Identify spines whose neck exists but does NOT reach the dendrite.

    A neck "reaches" the dendrite when at least one of its voxels has
    `dendrite_distance <= threshold_vox` (default 2 voxels ≈ 0.13 µm). Spines
    whose neck's minimum dendrite distance exceeds the threshold are flagged
    as 'partial-spine' — typically caused by the neck path being occluded by
    a neighbouring spine head.

    Returns a set of label ints classified partial. Does NOT modify the
    label volumes — downstream post-processing zeroes neck metrics for these
    labels (if keep_partial_spines=True) or drops them entirely (False).
    """
    from RESPAN.ImageAnalysis import ChunkedProcessing as _chunked

    nk_bboxes = _chunked.find_objects_streaming(connected_necks, logger=None)
    sp_bboxes = _chunked.find_objects_streaming(spines_filtered, logger=None)

    partial = set()
    for label, nk_bb in nk_bboxes.items():
        if int(label) == 0:
            continue
        if label not in sp_bboxes:
            continue  # no spine head — handled by spurious/filopodia paths
        z0, z1, y0, y1, x0, x1 = nk_bb
        dsl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        nk_crop = np.asarray(connected_necks[dsl])
        neck_mask = (nk_crop == int(label))
        if not neck_mask.any():
            continue
        dd_crop = np.asarray(dendrite_distance[dsl])
        min_dd = float(dd_crop[neck_mask].min())
        if min_dd > threshold_vox:
            partial.add(int(label))

    # Note: caller may further subtract filopodia_labels from this set.
    # Logging happens there after the final count to avoid over-reporting.
    return partial


def flag_spurious_necks(connected_necks, spines_filtered, original_nnunet_neck_mask,
                          neuron_intensity, settings, locations, logger):
    """Detect necks with no nnU-Net support AND biologically implausibly long.

    Two AND-gated criteria (both must trigger for a neck to be flagged):
      1. path_length_um > settings.spurious_path_length_min_um (default 1.0 µm)
      2. nnunet_support  < settings.spurious_nnunet_support_min (default 0.2)

    `intensity_ratio` is recorded for visibility but is not part of the gate —
    pathfinder-generated paths inherently follow bright voxels, so intensity
    is a poor discriminator.

    Acts on `settings.spurious_exclude_mode`:
      "neck_only" — removes only the neck voxels; spine head kept. Default.
                     Head is then reclassified as partial-spine by the
                     downstream classifier and retained or dropped via
                     `keep_partial_spines`.
      "both"      — removes both the neck voxels AND the spine voxels of each
                     flagged label. Use when you are confident flagged spines
                     are always false positives.
      "flag_only" — no removal; just writes neck_path_confidence.csv for review.

    Emits Tables/neck_path_confidence.csv with columns:
       label, path_length_um, nnunet_support, intensity_ratio, confidence,
       flagged_spurious
    """
    import pandas as pd

    t0 = time.time()
    # Backward compat: honour the legacy exclude_spurious_necks bool if present.
    mode = getattr(settings, 'spurious_exclude_mode', None)
    if mode is None:
        legacy = bool(getattr(settings, 'exclude_spurious_necks', True))
        mode = "neck_only" if legacy else "flag_only"
    if mode not in {"both", "neck_only", "flag_only"}:
        mode = "neck_only"
    # When keep_head_if_flagged=True, escalate "both" to "neck_only" behaviour
    # (remove neck but keep the spine head). Users enable this to review heads
    # of spines whose necks are spurious.
    if mode == "both" and bool(getattr(settings, 'spurious_keep_head_if_flagged', False)):
        mode = "neck_only"
    support_min = float(getattr(settings, 'spurious_nnunet_support_min', 0.2))
    intensity_min = float(getattr(settings, 'spurious_intensity_ratio_min', 0.4))
    length_min_um = float(getattr(settings, 'spurious_path_length_min_um', 1.0))

    # Per-label bboxes — streaming helper avoids materializing zarr-backed
    # spines_filtered at T7 scale (would otherwise peak RAM by 22 GB).
    # For numpy inputs the helper just runs find_objects on reasonable-sized
    # slabs internally, so throughput is comparable to direct find_objects.
    nk_bboxes = chunked.find_objects_streaming(connected_necks, logger=None)
    sp_bboxes = chunked.find_objects_streaming(spines_filtered, logger=None)

    res_xy = float(settings.input_resXY)
    res_z = float(settings.input_resZ)

    rows = []
    n_flagged = 0
    n_voxels_removed = 0
    all_labels = sorted(set(nk_bboxes.keys()) | set(sp_bboxes.keys()))

    for label in all_labels:
        nk_bb = nk_bboxes.get(label)
        sp_bb = sp_bboxes.get(label)
        if nk_bb is None:
            continue  # no neck → nothing to flag
        nk_sl = (slice(nk_bb[0], nk_bb[1]), slice(nk_bb[2], nk_bb[3]), slice(nk_bb[4], nk_bb[5]))

        nk_crop = connected_necks[nk_sl]
        neck_mask = (nk_crop == label)
        neck_vox_count = int(neck_mask.sum())
        if neck_vox_count == 0:
            continue

        # Physical path length — use the neck bbox principal extent in µm as a
        # cheap proxy (true path length would require skeletonisation per spine).
        dz_um = (nk_bb[1] - nk_bb[0]) * res_z
        dy_um = (nk_bb[3] - nk_bb[2]) * res_xy
        dx_um = (nk_bb[5] - nk_bb[4]) * res_xy
        path_length_um = float(np.sqrt(dz_um * dz_um + dy_um * dy_um + dx_um * dx_um))

        # nnU-Net support: fraction of neck voxels that are also in the
        # original (pre-association) nnU-Net neck mask.
        nnunet_crop = np.asarray(original_nnunet_neck_mask[nk_sl]).astype(bool, copy=False)
        nnunet_support = float((neck_mask & nnunet_crop).sum()) / float(neck_vox_count)

        # Intensity ratio: mean neuron along neck vs mean within spine head.
        neck_int = np.asarray(neuron_intensity[nk_sl])
        mean_neck_int = float(neck_int[neck_mask].mean()) if neck_vox_count > 0 else 0.0

        if sp_bb is not None:
            sp_sl = (slice(sp_bb[0], sp_bb[1]), slice(sp_bb[2], sp_bb[3]), slice(sp_bb[4], sp_bb[5]))
            sp_crop = np.asarray(spines_filtered[sp_sl])
            spine_mask = (sp_crop == label)
            if spine_mask.any():
                sp_int = np.asarray(neuron_intensity[sp_sl])
                mean_spine_int = float(sp_int[spine_mask].mean())
            else:
                mean_spine_int = 0.0
        else:
            mean_spine_int = 0.0

        intensity_ratio = (mean_neck_int / mean_spine_int) if mean_spine_int > 0 else 0.0

        # AND-gated flag — three criteria, ALL must trigger:
        #   1. Long path (> length_min_um, default 1.5 µm)
        #   2. Low nnU-Net neck support (< support_min, default 0.2)
        #   3. Low intensity vs spine head (< intensity_min, default 0.4)
        # Intensity is the discriminator that separates real necks with
        # missing nnU-Net segmentation (bright fluorescence, intensity ≈ head)
        # from truly spurious synthesis (dim path through background).
        # Validated on T2 (flags 2 user-confirmed spurious; preserves all 5
        # real-but-flagged previously). T4's "synthetic" long necks follow
        # bright structures (likely other-neuron features) and are NOT caught
        # by this filter — they require spatial/annotation-based rejection.
        is_long = path_length_um > length_min_um
        is_unsupported = nnunet_support < support_min
        is_dim = intensity_ratio < intensity_min
        # Gate: long AND nnU-Net unsupported. The intensity criterion was
        # found to be unreliable for pathfinder-synthesised bridges (which
        # by design follow bright voxels → high intensity_ratio), so it's
        # opt-in via `settings.spurious_require_intensity`. Default OFF
        # correctly flags the large class of pathfinder-bridge-only necks
        # (nnU-Net support = 0) that previous AND gate missed.
        if bool(getattr(settings, 'spurious_require_intensity', False)):
            flagged = bool(is_long and is_unsupported and is_dim)
        else:
            flagged = bool(is_long and is_unsupported)

        # Continuous confidence score [0, 1] — higher = more trustworthy. Weighted
        # mostly on nnU-Net support (the meaningful signal); intensity ratio kept
        # as a minor contribution so the field isn't dead weight.
        confidence = (0.7 * min(1.0, nnunet_support / max(support_min, 1e-6))
                      + 0.3 * min(1.0, intensity_ratio / max(intensity_min, 1e-6)))
        confidence = float(max(0.0, min(1.0, confidence)))

        rows.append({
            'label': int(label),
            'path_length_um': round(path_length_um, 3),
            'nnunet_support': round(nnunet_support, 4),
            'intensity_ratio': round(intensity_ratio, 4),
            'confidence': round(confidence, 4),
            'flagged_spurious': flagged,
        })

        if flagged:
            n_flagged += 1
            if mode in ("both", "neck_only"):
                # Null the neck voxels of this label in connected_necks.
                neck_sub = nk_crop.copy()
                neck_sub[neck_mask] = 0
                connected_necks[nk_sl] = neck_sub
                n_voxels_removed += neck_vox_count
            if mode == "both" and sp_bb is not None:
                # Also null the spine voxels — a spurious neck implies a
                # spurious spine. Avoids orphan "spine head with no neck"
                # which is biologically meaningless.
                sp_sl_local = (slice(sp_bb[0], sp_bb[1]),
                                slice(sp_bb[2], sp_bb[3]),
                                slice(sp_bb[4], sp_bb[5]))
                if isinstance(spines_filtered, np.ndarray):
                    sp_crop = spines_filtered[sp_sl_local].copy()
                    sp_crop[sp_crop == label] = 0
                    spines_filtered[sp_sl_local] = sp_crop
                else:
                    # zarr path: read, mutate, write slab-by-slab within bbox.
                    sp_crop = np.asarray(spines_filtered[sp_sl_local])
                    sp_crop[sp_crop == label] = 0
                    spines_filtered[sp_sl_local] = sp_crop

    # Persist per-spine confidence scores.
    try:
        df = pd.DataFrame(rows)
        out_path = os.path.join(locations.tables, 'neck_path_confidence.csv')
        df.to_csv(out_path, index=False)
        logger.info(f"       Wrote neck confidence scores to {out_path} ({len(rows)} rows).")
    except Exception as e:
        logger.warning(f"       Could not write neck_path_confidence.csv: {e}")

    flagged_labels = {row['label'] for row in rows if row['flagged_spurious']}
    elapsed = time.time() - t0
    logger.info(
        f"       Spurious-neck filter: {n_flagged}/{len(rows)} necks flagged "
        f"(mode={mode}; thresholds length>{length_min_um}µm, nnU-Net<{support_min}; "
        f"{n_voxels_removed} neck voxels removed) in {elapsed:.1f}s")
    return connected_necks, flagged_labels


def _grow_head_from_tip(cc_mask, dd_crop, tip_local, target_size):
    """Region-grow from tip voxel in 26-conn BFS, ordered by dendrite_distance desc.

    Used by `recover_filopodia_from_orphan_necks` to carve a spine head from a
    filopodium's nnU-Net neck CC. Grows a connected head starting at the tip
    (max-dd voxel), preferring voxels with higher dendrite_distance at each
    step, until `target_size` voxels are collected.

    Uses `seen` to deduplicate heap entries; voxels are added to the returned
    head ONLY when popped from the priority queue, guaranteeing the head grows
    along the highest-dd frontier (not in insertion order).

    Returns a bool mask (same shape as cc_mask) with the carved head True, or
    None if growth can't reach `target_size`.
    """
    import heapq
    if not cc_mask[tip_local]:
        return None
    head = np.zeros_like(cc_mask, dtype=bool)
    seen = np.zeros_like(cc_mask, dtype=bool)
    seen[tip_local] = True
    heap = [(-float(dd_crop[tip_local]), tip_local)]
    shape = cc_mask.shape
    n_head = 0
    while heap and n_head < target_size:
        _, (z, y, x) = heapq.heappop(heap)
        if head[z, y, x]:
            continue
        head[z, y, x] = True
        n_head += 1
        if n_head >= target_size:
            break
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]
                            and cc_mask[nz, ny, nx] and not seen[nz, ny, nx]):
                        seen[nz, ny, nx] = True
                        heapq.heappush(heap, (-float(dd_crop[nz, ny, nx]),
                                               (nz, ny, nx)))
    return head if n_head >= target_size else None


def recover_filopodia_from_orphan_necks(spines_filtered, connected_necks,
                                          neck_mask, dendrite_distance,
                                          neuron_spine_size, spine_dist_voxels,
                                          settings, workspace, use_chunked, logger):
    """Recover orphan nnU-Net neck CCs as filopodia (no synthesis, voxels carved from CC).

    Runs AFTER all spine→neck association is complete. Any nnU-Net neck CC with
    zero overlap with `connected_necks > 0` is a true orphan — an nnU-Net neck
    prediction that never got assigned to a spine head. If it's near a dendrite
    (min dendrite_distance within `spine_dist_voxels`) and NOT adjacent (26-conn)
    to an existing spine head, we recover it as a filopodium:
      - Carve a "head" from the CC by region-growing from the tip voxel
        (max dendrite_distance) in 26-connected BFS ordered by dd-descending,
        until we collect `MIN_HEAD_VOXELS` voxels.
      - Remaining CC voxels form the neck.
      - Assign a new label = max(spines_filtered) + k.
      - Write head voxels to spines_filtered, neck voxels to connected_necks.

    No voxels are synthesized; the entire filopodium is grounded in nnU-Net
    predictions. Returns a metadata dict so downstream code can mark these
    objects with spine_type='filopodia' and emit base-vs-tip dendrite distance.

    Parameters
    ----------
    spines_filtered : numpy or zarr int32 — spine head labels (1..N).
    connected_necks : numpy or zarr int32 — neck labels (same label space).
    neck_mask       : numpy bool or zarr uint8 — original nnU-Net neck binary.
    dendrite_distance : numpy or zarr float32 — EDT to dendrite (voxels).
    neuron_spine_size : tuple (min_vol_um3, max_vol_um3) — from settings.
    spine_dist_voxels : float — max base-to-dendrite distance (voxels).
    settings, workspace, use_chunked, logger.

    Returns
    -------
    spines_filtered_updated : same type as input (modified in place + returned).
    connected_necks_updated : same type as input.
    filopodia_metadata : dict {label: {'base_dist_um', 'tip_dist_um',
                                       'cc_voxels', 'head_voxels', 'neck_voxels'}}
    """
    t0 = time.time()
    res_xy = float(settings.input_resXY)

    # `settings.neuron_spine_size` is already in VOXELS by the time it reaches
    # the pipeline (GUI/test-runner convert from µm³ via the voxel-volume factor
    # in apply_profile / _ConfigureRun). Use the value as-is, with a hard floor
    # of 3 voxels so the carved head is always measurable.
    MIN_HEAD_VOXELS = max(3, int(neuron_spine_size[0]))
    MIN_NECK_VOXELS = 3
    MIN_CC_VOXELS = MIN_HEAD_VOXELS + MIN_NECK_VOXELS
    # Elongation gate: dendrite_distance should span at least ~0.3 µm within CC
    # (excludes ball-shaped nnU-Net artifacts but preserves short filopodia).
    # 0.3 µm ≈ 5 voxels at 65 nm XY; scales with resolution. Can be overridden
    # by settings.filopodia_min_extent_um (µm) if user wants stricter gating.
    _min_extent_um = float(getattr(settings, 'filopodia_min_extent_um', 0.3))
    MIN_EXTENT_VOX = max(3, int(round(_min_extent_um / res_xy)))

    logger.info(
        f"     Filopodia recovery starting: MIN_HEAD={MIN_HEAD_VOXELS} vox, "
        f"MIN_NECK={MIN_NECK_VOXELS} vox, MIN_CC={MIN_CC_VOXELS} vox, "
        f"MIN_EXTENT={MIN_EXTENT_VOX} vox, max_base_dist={spine_dist_voxels:.1f} vox")

    # Step 1: Build ORPHAN mask — nnU-Net neck voxels that are currently
    # unassigned (connected_necks == 0). This captures both wholly-unassigned
    # nnU-Net CCs AND the partial fragments that `drop_disconnected_neck_fragments`
    # zeroed out (dropped from a spine's neck because they were 26-disconnected
    # from the spine head). Those dropped fragments are exactly what the user
    # wants preserved as filopodia.
    #
    # CC-label the orphan mask with 26-connectivity (filopodia are thin and
    # often diagonal — 6-conn would split them).
    struct26 = ndimage.generate_binary_structure(3, 3)
    if hasattr(neck_mask, 'chunks'):
        # Build orphan mask slab-by-slab into a zarr array (cheap at T7).
        orphan_mask = workspace.create_array(
            '_filopodia_orphan_mask', neck_mask.shape, dtype=np.uint8,
            chunks=chunked.STREAMING_CHUNKS)
        _shape = neck_mask.shape
        _bpz = int(_shape[1]) * int(_shape[2])
        _slz = max(1, min(64, int(2 * 1024**3 / max(_bpz, 1))))
        _total_orphan = 0
        for _z in range(0, _shape[0], _slz):
            _ze = min(_z + _slz, _shape[0])
            _nm_slab = np.asarray(neck_mask[_z:_ze]).astype(bool, copy=False)
            _cn_slab = np.asarray(connected_necks[_z:_ze])
            _om = (_nm_slab & (_cn_slab == 0)).astype(np.uint8)
            orphan_mask[_z:_ze] = _om
            _total_orphan += int(_om.sum())
            del _nm_slab, _cn_slab, _om
        if _total_orphan == 0:
            logger.info("       No orphan neck voxels — nothing to recover.")
            return spines_filtered, connected_necks, {}

        cc_zarr = workspace.create_array(
            '_filopodia_cc', neck_mask.shape, dtype=np.int32,
            chunks=chunked.STREAMING_CHUNKS)
        # connected_components_streaming returns (n_components, bboxes_dict).
        n_cc_total, _ = chunked.connected_components_streaming(
            orphan_mask, cc_zarr, connectivity=3, logger=None)
        neck_cc_labels = cc_zarr
    else:
        _nm_np = np.asarray(neck_mask)
        if _nm_np.dtype != bool:
            _nm_np = _nm_np > 0
        _cn_np = np.asarray(connected_necks)
        _orphan_np = _nm_np & (_cn_np == 0)
        _total_orphan = int(_orphan_np.sum())
        if _total_orphan == 0:
            logger.info("       No orphan neck voxels — nothing to recover.")
            return spines_filtered, connected_necks, {}
        neck_cc_labels, n_cc_total = ndimage.label(_orphan_np, structure=struct26)
        del _orphan_np, _nm_np, _cn_np

    if n_cc_total == 0:
        logger.info("       No orphan neck CCs found — nothing to recover.")
        return spines_filtered, connected_necks, {}

    # Step 2: Per-CC bboxes (streaming-safe for zarr).
    cc_bboxes = chunked.find_objects_streaming(neck_cc_labels, logger=None)
    logger.info(
        f"       {n_cc_total} orphan nnU-Net neck CCs found "
        f"({_total_orphan} unassigned neck voxels total) — screening for filopodia.")

    # Step 3: Allocate labels above current max.
    max_spine_label = int(chunked.safe_max(spines_filtered))
    max_neck_label = int(chunked.safe_max(connected_necks))
    if max_neck_label > max_spine_label:
        logger.warning(
            f"       connected_necks max label ({max_neck_label}) > spines_filtered "
            f"max label ({max_spine_label}) — unexpected; allocating from max+1.")
        max_spine_label = max_neck_label
    next_label = max_spine_label + 1

    # Step 4: Per-CC screening + carving.
    n_considered = 0
    n_skip_overlap = 0
    n_skip_too_small = 0
    n_skip_too_far = 0
    n_skip_not_elongated = 0
    n_skip_head_adjacent = 0
    n_skip_carve_failed = 0
    n_recovered = 0
    filopodia_metadata = {}
    sf_shape = spines_filtered.shape

    for cc_label, bbox in cc_bboxes.items():
        n_considered += 1
        z0, z1, y0, y1, x0, x1 = bbox
        dsl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

        cc_crop = np.asarray(neck_cc_labels[dsl])
        cc_mask = (cc_crop == cc_label)
        if not cc_mask.any():
            continue

        # Orphan check: ANY voxel of CC already in connected_necks > 0 → not orphan.
        cn_crop = np.asarray(connected_necks[dsl])
        if (cc_mask & (cn_crop > 0)).any():
            n_skip_overlap += 1
            continue

        cc_voxels = int(cc_mask.sum())
        if cc_voxels < MIN_CC_VOXELS:
            n_skip_too_small += 1
            continue

        # Head-adjacency check: dilate CC by 1 voxel 26-conn in a 1-voxel halo
        # around bbox; if it touches any spine head, this CC should have been
        # claimed by upstream assignment. Skipping avoids incorrectly spawning
        # a filopodium where a real neck was missed.
        hz0 = max(0, z0 - 1); hz1 = min(sf_shape[0], z1 + 1)
        hy0 = max(0, y0 - 1); hy1 = min(sf_shape[1], y1 + 1)
        hx0 = max(0, x0 - 1); hx1 = min(sf_shape[2], x1 + 1)
        halo_sl = (slice(hz0, hz1), slice(hy0, hy1), slice(hx0, hx1))
        sf_halo = np.asarray(spines_filtered[halo_sl])
        # Project cc_mask into halo coords by padding.
        pad_before = (z0 - hz0, y0 - hy0, x0 - hx0)
        pad_after = (hz1 - z1, hy1 - y1, hx1 - x1)
        cc_mask_halo = np.pad(cc_mask,
                               list(zip(pad_before, pad_after)),
                               mode='constant', constant_values=False)
        dilated = ndimage.binary_dilation(cc_mask_halo, structure=struct26, iterations=1)
        if (dilated & (sf_halo > 0)).any():
            n_skip_head_adjacent += 1
            continue

        # Distance / elongation gates.
        dd_crop = np.asarray(dendrite_distance[dsl])
        dd_in_cc = dd_crop[cc_mask]
        dd_min = float(dd_in_cc.min())
        dd_max = float(dd_in_cc.max())
        if dd_min > spine_dist_voxels:
            n_skip_too_far += 1
            continue
        if (dd_max - dd_min) < MIN_EXTENT_VOX:
            n_skip_not_elongated += 1
            continue

        # Find tip voxel (max dendrite_distance within CC).
        masked_dd = np.where(cc_mask, dd_crop, -1.0)
        tip_local = np.unravel_index(int(np.argmax(masked_dd)), cc_crop.shape)
        head_mask_local = _grow_head_from_tip(
            cc_mask, dd_crop, tip_local, MIN_HEAD_VOXELS)
        if head_mask_local is None:
            n_skip_carve_failed += 1
            continue

        neck_mask_local = cc_mask & ~head_mask_local
        if neck_mask_local.sum() < MIN_NECK_VOXELS:
            n_skip_carve_failed += 1
            continue

        # Validate: head is a single 26-CC, neck touches head.
        _, n_head_cc = ndimage.label(head_mask_local, structure=struct26)
        if n_head_cc != 1:
            n_skip_carve_failed += 1
            continue
        head_dilated = ndimage.binary_dilation(
            head_mask_local, structure=struct26, iterations=1)
        if not (head_dilated & neck_mask_local).any():
            n_skip_carve_failed += 1
            continue

        # Allocate + write. Crop-write pattern (zarr-safe).
        new_label = next_label
        next_label += 1

        sf_crop = np.asarray(spines_filtered[dsl])
        # Only write where currently background — defensive.
        writable_h = head_mask_local & (sf_crop == 0)
        sf_crop[writable_h] = new_label
        spines_filtered[dsl] = sf_crop

        cn_crop2 = np.asarray(connected_necks[dsl])
        writable_n = neck_mask_local & (cn_crop2 == 0)
        cn_crop2[writable_n] = new_label
        connected_necks[dsl] = cn_crop2

        # Only commit if the defensive background-only writes actually
        # produced at least the minimum head/neck voxels. Otherwise the
        # metadata would reflect intent, not reality.
        actual_head = int(writable_h.sum())
        actual_neck = int(writable_n.sum())
        if actual_head < MIN_HEAD_VOXELS or actual_neck < MIN_NECK_VOXELS:
            # Roll back any partial writes: re-zero the positions that were
            # assigned above (we haven't yet incremented next_label for this
            # iteration's new_label, so this just cleans up the IDs we wrote).
            if actual_head > 0:
                sf_crop2 = np.asarray(spines_filtered[dsl])
                sf_crop2[sf_crop2 == new_label] = 0
                spines_filtered[dsl] = sf_crop2
            if actual_neck > 0:
                cn_crop3 = np.asarray(connected_necks[dsl])
                cn_crop3[cn_crop3 == new_label] = 0
                connected_necks[dsl] = cn_crop3
            next_label = new_label  # recycle the label — don't advance
            n_skip_carve_failed += 1
            continue

        # Compute centroid of the HEAD voxels in global coordinates — matches
        # the convention of `initial_spine_measurements` which reports head
        # centroid as x/y/z. For filopodia the head is at the tip, so this
        # correctly marks the tip location.
        head_coords = np.argwhere(writable_h)
        centroid_z = float(head_coords[:, 0].mean()) + z0
        centroid_y = float(head_coords[:, 1].mean()) + y0
        centroid_x = float(head_coords[:, 2].mean()) + x0

        # Track the BASE voxel (min-dd within CC, near the dendrite). Used for
        # dendrite-assignment override: calculate_dend_ID_and_geo_distance
        # matches by spine centroid → dendrite via KDTree with a distance cap;
        # for long filopodia the head centroid sits at the tip, far from any
        # dendrite skeleton point. The base voxel is guaranteed within
        # spine_dist_voxels of the dendrite by the rejection filters.
        base_idx_local = np.unravel_index(
            int(np.argmin(np.where(cc_mask, dd_crop, np.inf))), cc_crop.shape)
        base_z = int(base_idx_local[0]) + z0
        base_y = int(base_idx_local[1]) + y0
        base_x = int(base_idx_local[2]) + x0

        filopodia_metadata[int(new_label)] = {
            'base_dist_um': dd_min * res_xy,
            'tip_dist_um': dd_max * res_xy,
            'cc_voxels': cc_voxels,
            'head_voxels': actual_head,
            'neck_voxels': actual_neck,
            'z': centroid_z,
            'y': centroid_y,
            'x': centroid_x,
            'base_z': base_z,
            'base_y': base_y,
            'base_x': base_x,
        }
        n_recovered += 1

    # --------------------------------------------------------------
    # Pass 2: post-bridge distant neck fragments. `bridge_spine_neck_via_pathfinding`
    # can assign a nnU-Net-backed neck region to a spine label by drawing a
    # synthetic corridor through low-signal voxels. The resulting neck looks
    # connected in `connected_necks` but the nnU-Net-backed portion may be
    # spatially disconnected from the spine head — i.e. the neck belongs to
    # a filopodium, not to the distant spine.
    #
    # Detection: for each spine label, CC-label its nnU-Net-backed neck voxels.
    # If multiple CCs exist, the one(s) NOT touching the spine head are
    # candidates to split off and promote to filopodia.
    # --------------------------------------------------------------
    n_pass2_considered = 0
    n_pass2_recovered = 0
    try:
        cn_bboxes = chunked.find_objects_streaming(connected_necks, logger=None)
        sp_bboxes2 = chunked.find_objects_streaming(spines_filtered, logger=None)
    except Exception as e:
        logger.warning(f"       Pass 2 bbox collection failed: {e}; skipping.")
        cn_bboxes = {}
        sp_bboxes2 = {}

    for label, nk_bb in cn_bboxes.items():
        if label == 0:
            continue
        sp_bb = sp_bboxes2.get(int(label))
        if sp_bb is None:
            continue  # neck with no head already handled earlier or is a pass-1 filopodium
        n_pass2_considered += 1
        # Union bbox for per-label crop.
        z0 = min(sp_bb[0], nk_bb[0]); z1 = max(sp_bb[1], nk_bb[1])
        y0 = min(sp_bb[2], nk_bb[2]); y1 = max(sp_bb[3], nk_bb[3])
        x0 = min(sp_bb[4], nk_bb[4]); x1 = max(sp_bb[5], nk_bb[5])
        dsl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

        cn_crop = np.asarray(connected_necks[dsl])
        sp_crop = np.asarray(spines_filtered[dsl])
        nm_crop = np.asarray(neck_mask[dsl]).astype(bool, copy=False)

        label_int = int(label)
        neck_voxels = (cn_crop == label_int)
        neck_nnunet_backed = neck_voxels & nm_crop
        if not neck_nnunet_backed.any():
            continue  # no real neck voxels for this label — bridge-only, skip

        # CC-label the nnU-Net-backed neck voxels for this spine label.
        neck_ccs, n_neck_cc = ndimage.label(neck_nnunet_backed, structure=struct26)
        if n_neck_cc <= 1:
            continue  # single CC — neck is coherent, nothing to promote

        # Find which CC(s) contain the spine head (26-adjacent to spine voxels).
        spine_mask_local = (sp_crop == label_int)
        spine_dilated = ndimage.binary_dilation(
            spine_mask_local, structure=struct26, iterations=1)
        head_cc_ids = set(int(x) for x in np.unique(neck_ccs[spine_dilated]) if x != 0)

        # Screen each non-head CC.
        for cc_id in range(1, n_neck_cc + 1):
            if cc_id in head_cc_ids:
                continue
            cc_mask = (neck_ccs == cc_id)
            cc_voxels = int(cc_mask.sum())
            if cc_voxels < MIN_CC_VOXELS:
                continue
            dd_crop = np.asarray(dendrite_distance[dsl])
            dd_in_cc = dd_crop[cc_mask]
            dd_min = float(dd_in_cc.min())
            dd_max = float(dd_in_cc.max())
            if dd_min > spine_dist_voxels:
                continue
            if (dd_max - dd_min) < MIN_EXTENT_VOX:
                continue

            # Carve head from tip + validate.
            masked_dd = np.where(cc_mask, dd_crop, -1.0)
            tip_local = np.unravel_index(int(np.argmax(masked_dd)), cc_mask.shape)
            head_mask_local = _grow_head_from_tip(
                cc_mask, dd_crop, tip_local, MIN_HEAD_VOXELS)
            if head_mask_local is None:
                continue
            neck_mask_local = cc_mask & ~head_mask_local
            if neck_mask_local.sum() < MIN_NECK_VOXELS:
                continue
            _, n_head_cc = ndimage.label(head_mask_local, structure=struct26)
            if n_head_cc != 1:
                continue
            head_dilated = ndimage.binary_dilation(
                head_mask_local, structure=struct26, iterations=1)
            if not (head_dilated & neck_mask_local).any():
                continue

            # Allocate + reassign. The CC voxels are currently labeled `label_int`
            # in connected_necks; we reassign them to `new_label` (head) and
            # `new_label` (neck), effectively splitting the filopodium off.
            new_label = next_label
            next_label += 1

            # Also zero-out the bridge voxels that previously connected this
            # CC to the spine head — those are connected_necks voxels labeled
            # label_int that are NOT nnU-Net-backed AND lie between head and
            # the CC we're promoting. Conservative: zero any (connected_necks
            # == label_int) & ~nm_crop voxels in the halo around our CC.
            cc_dilated = ndimage.binary_dilation(
                cc_mask, structure=struct26, iterations=3)
            bridge_in_halo = cc_dilated & neck_voxels & ~nm_crop
            if bridge_in_halo.any():
                cn_crop[bridge_in_halo] = 0

            # Reassign head voxels: spines_filtered += new_label (on head).
            # NOTE: writing to sp_crop within the loop on the same dsl mutates
            # the crop; we write back at end of loop iteration.
            writable_h = head_mask_local & (sp_crop == 0)
            actual_head = int(writable_h.sum())
            writable_n = neck_mask_local
            actual_neck = int(writable_n.sum())
            if actual_head < MIN_HEAD_VOXELS or actual_neck < MIN_NECK_VOXELS:
                next_label = new_label  # recycle
                continue

            sp_crop[writable_h] = new_label
            cn_crop[writable_h] = 0     # remove from connected_necks where head now lives
            cn_crop[writable_n] = new_label  # rename neck voxels to filopodia label

            # Capture centroid + base.
            head_coords = np.argwhere(writable_h)
            centroid_z = float(head_coords[:, 0].mean()) + z0
            centroid_y = float(head_coords[:, 1].mean()) + y0
            centroid_x = float(head_coords[:, 2].mean()) + x0
            base_idx_local = np.unravel_index(
                int(np.argmin(np.where(cc_mask, dd_crop, np.inf))), cc_mask.shape)
            base_z = int(base_idx_local[0]) + z0
            base_y = int(base_idx_local[1]) + y0
            base_x = int(base_idx_local[2]) + x0

            filopodia_metadata[int(new_label)] = {
                'base_dist_um': dd_min * res_xy,
                'tip_dist_um': dd_max * res_xy,
                'cc_voxels': cc_voxels,
                'head_voxels': actual_head,
                'neck_voxels': actual_neck,
                'z': centroid_z,
                'y': centroid_y,
                'x': centroid_x,
                'base_z': base_z,
                'base_y': base_y,
                'base_x': base_x,
                'source': f'split_from_label_{label_int}',
            }
            n_pass2_recovered += 1

        # Write back the mutated crops (may include multiple pass-2 promotions
        # per label, plus bridge zeroing).
        spines_filtered[dsl] = sp_crop
        connected_necks[dsl] = cn_crop

    elapsed = time.time() - t0
    logger.info(
        f"       Filopodia recovery: pass1 {n_recovered}/{n_considered} "
        f"(skip: overlap={n_skip_overlap}, too_small={n_skip_too_small}, "
        f"head_adj={n_skip_head_adjacent}, too_far={n_skip_too_far}, "
        f"not_elongated={n_skip_not_elongated}, carve_failed={n_skip_carve_failed}); "
        f"pass2 {n_pass2_recovered}/{n_pass2_considered} split from distant-neck labels. "
        f"Total recovered: {n_recovered + n_pass2_recovered}. ({elapsed:.1f}s)")
    return spines_filtered, connected_necks, filopodia_metadata


def extend_single_object_GPU_v2(label_value, object_subvolume, target_subvolume, intensity_subvolume,
                                 settings, logger, obstacle_subvolume=None):
    # Convert numpy arrays to CuPy arrays
    # logger.info(f"Label {label_value} - Subvolume shapes: subvolume {object_subvolume.shape}, target {target_subvolume.shape}, traversable {traversable_subvolume.shape}")

    pad_width = 1

    object_subvolume_gpu = pad_subvolume_gpu(cp.asarray(object_subvolume), pad_width)
    target_subvolume_gpu = pad_subvolume_gpu(cp.asarray(target_subvolume), pad_width)
    # traversable_subvolume_gpu = cp.asarray(traversable_subvolume)
    intensity_subvolume_gpu = pad_subvolume_gpu(cp.asarray(intensity_subvolume), pad_width)
    # Optional: obstacle mask (e.g., other spine heads) — voxels the path
    # must not enter. Padded to match the other inputs.
    if obstacle_subvolume is not None:
        obstacle_subvolume_gpu = pad_subvolume_gpu(
            cp.asarray(obstacle_subvolume.astype(bool)), pad_width)
    else:
        obstacle_subvolume_gpu = None
    # Use the enhanced simple pathfinding method - below method works well but seeing if we can use intensity as well
    # path, distance_map = strict_z_first_pathfinding(object_subvolume_gpu, target_subvolume_gpu)

    spacing = (settings.input_resZ, settings.input_resXY, settings.input_resXY)
    path, distance_map = pathfinding_v3b(object_subvolume_gpu, target_subvolume_gpu, intensity_subvolume_gpu,
                                               logger, spacing=spacing,
                                               obstacle_subvolume_gpu=obstacle_subvolume_gpu)

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
        return label_value, np.zeros_like(object_subvolume), np.zeros_like(object_subvolume)

    # Exclude both the object (spine) AND target (dendrite / remaining-neck)
    # voxels from the extension. Object exclusion preserves the spine's original
    # labels. Target exclusion prevents the pathfinder's terminal voxel — which
    # now lands on dendrite directly — from writing the spine label INTO the
    # dendrite. The neck extension still reaches the dendrite (last retained
    # voxel is 26-adjacent to the terminal-dendrite voxel), but never enters it.
    final_path_volume_gpu = (path_volume_gpu
                              * (1 - object_subvolume_gpu)
                              * (1 - target_subvolume_gpu))

    # Multiply the final path volume by the label value
    final_path_volume_gpu = final_path_volume_gpu * label_value

    # Convert to NumPy after the operation
    path_volume = unpad_subvolume_gpu(cp.asnumpy(final_path_volume_gpu), pad_width)
    distance_map = unpad_subvolume_gpu(cp.asnumpy(distance_map), pad_width)


    return label_value, path_volume, distance_map

@mp.profile_mem()
def analyze_spines_batch(spine_labels_vol, head_labels_vol, dendrite, neuron, locations, settings, logger, scaling):
    #dendrites and labels and must be binarized for further analysis

    # ZARR GUARD — must run before any cp.asarray / check_gpu_memory_requirements.
    # cp.asarray(zarr_array) silently materializes+uploads the full volume to GPU,
    # which OOMs at T7 (22 GB dendrite alone) and is impossible at 100 GB scale.
    # Also the mem_required gate at L~2204 only sees spine_labels_vol.nbytes, so
    # it cannot catch the case where dendrite or neuron is the zarr hazard.
    if any(hasattr(a, 'chunks') for a in (spine_labels_vol, head_labels_vol, dendrite, neuron)):
        logger.info("       Zarr input detected — routing to CPU-resident per-spine extraction.")
        return analyze_spines_batch_cpu_resident(spine_labels_vol, head_labels_vol, dendrite, neuron,
                                                  locations, settings, logger, scaling)

    start_time = time.time()
    #free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    #logger.info(f"       Available GPU memory: {free_mem / 1e9:.2f} GB")
    mem_required, free_mem, total_mem = check_gpu_memory_requirements(spine_labels_vol, logger)
    # estimate requirements

    if mem_required > 0.5 * free_mem:
        logger.info("       Volume too large for GPU-resident approach. Using CPU-resident per-spine extraction.")
        return analyze_spines_batch_cpu_resident(spine_labels_vol, head_labels_vol, dendrite, neuron,
                                                  locations, settings, logger, scaling)

    # Move data to GPU and create multi-channel array
    cp_multi_channel = cp.stack([cp.asarray(spine_labels_vol),
                                cp.asarray(head_labels_vol),
                                cp.asarray(dendrite),
                                cp.asarray(neuron)], axis=-1)
    logger.info(f"       Multi-channel GPU array created.")
    # Get unique labels from array A (excluding background)
    labels = cp.unique(cp_multi_channel[:, :, :, 0])
    labels = labels[labels != 0]

    avg_sub_volume_shape = 6 / scaling[0], 6 / scaling[1], 6 / scaling[2]  # spine or neck
    dtype_size = 2  # 4 bytes per float32
    batch_size = calculate_batch_size(mem_required, spine_labels_vol, avg_sub_volume_shape, dtype_size)
    batch_size = max(1, batch_size)
    # print(f"Using batch size: {batch_size}")
    logger.info(f"       Using batch size of {batch_size} spines...")

    results = []

    #batch
    total_batches = (len(labels) + batch_size - 1) // batch_size  # Calculate total number of batches (ceiling division)
    milestones = [int(total_batches * p / 100) for p in range(10, 101, 10)]  # Create milestones at 10% intervals

    i = 0
    while i < len(labels):
      try:
        batch_labels = labels[i:i + batch_size]
        current_batch = i // batch_size + 1

        # Print at 10% intervals (10%, 20%, 30%, etc.)
        if current_batch in milestones:
            percentage = (current_batch / total_batches) * 100
            logger.info(f"        Processing batch {current_batch} of {total_batches}... ({int(percentage)}% complete)")


        # Extract subvolumes
        sub_volumes, start_coords = extract_subvolumes_mulitchannel_GPU_batch(cp_multi_channel, batch_labels)

        #logger shape
        #if settings.additional_logging == True:
            #logger.info(f"Subvolumes shape: {len(sub_volumes)}")
            #log dimensions of subvolumes0
            #logger.info(f"Subvolume 0 shape: {sub_volumes[0].shape}")

        #mask subvolumes
        #logger.info(f"      Masking subvolumes...")
        sub_volumes = batch_mask_subvolumes_cp(sub_volumes, batch_labels)

        #if settings.additional_logging == True:
        #    logger.info(f"Subvolume 0 shape after masking: {sub_volumes[0].shape}")


        # Pad subvolumes to allow batching
        #padded_subvolumes, pad_amounts = pad_subvolumes(sub_volumes, logger)

        padded_subvolumes, pad_amounts = pad_and_center_subvolumes(sub_volumes, settings, logger)

        #Save spine arrays - only when intermediate data is requested (disk I/O heavy)
        if settings.save_intermediate_data:
            if settings.additional_logging:
                logger.info(f"      Saving spine arrays...")
            #MZYXC to MZCYX for imagej
            export_subvols = cp.transpose(cp.stack(padded_subvolumes, axis=0), (0, 1, 4, 2,3 ))
            export_subvols[:,:, :3] = export_subvols[:, :, :3] * 65535 / 2
            export_subvols[:, :, 0] = export_subvols[:, :, 0] - export_subvols[:, :, 1]
            export_subvols = cp.asnumpy(export_subvols)

            imwrite(f'{locations.arrays}/Spine_vols_{settings.filename}_b{i // batch_size + 1}.tif', export_subvols.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True,
                    photometric='minisblack',
                    metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'TZCYX', 'mode': 'composite'},
                    resolution=(1 / settings.input_resXY, 1 / settings.input_resXY))

            imwrite(f'{locations.arrays}/Spine_MIPs_{settings.filename}_b{i // batch_size + 1}.tif', np.max(export_subvols, axis=1).astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True,
                    photometric='minisblack',
                    metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX', 'mode': 'composite'},
                    resolution=(1 / settings.input_resXY, 1 / settings.input_resXY))

            del export_subvols

        if settings.additional_logging == True:
            logger.info(f"Subvolume 0 shape after padding: {padded_subvolumes[0].shape}")


        #loop over sub_volumes but also provide a counter
        #if settings.save_intermediate_data == True:
        #    for subvol, start_coord, label in zip(sub_volumes, start_coords, batch_labels):

                #save as tif and resahpe for imageJ
        #        imwrite_filename = os.path.join(locations.Meshes, f"subvol_{label}.tif")
                #imageJ ZCYX - currently ZYXC so fix

                #subvol_out = subvol.get()
        #        imwrite(imwrite_filename, subvol.get().transpose(0, 3, 1, 2).astype(np.uint16), imagej=True,
        #                photometric='minisblack', metadata={'unit': 'um', 'axes': 'ZCYX'})


        #generate data for spine
        batch_data = cp.stack([subvol[:, :, :, 0].astype(cp.float32) for subvol in padded_subvolumes])  # spine volumes
        batch_dendrite = cp.stack([subvol[:, :, :, 2] for subvol in padded_subvolumes]) #dendrite
        #binary of batch_dendrite
        batch_binary = cp.stack([subvol[:, :, :, 2] > 0 for subvol in padded_subvolumes])  # dendrite binary

        # calculate closest points
        closest_points = find_closest_points_batch(batch_data, batch_binary)

        #updated to remove dendrite id calc as more efficient during geodesic dist analysis
        #dendrite_ids = find_closest_dendrite_id(closest_points, batch_dendrite)
        #print(dendrite_ids)
        #if settings.additional_logging == True:
        #    logger.info(f"     batch_data dtype: {batch_data.dtype}, shape: {batch_data.shape}")
        #    logger.info(f"     batch_dendrite dtype: {batch_dendrite.dtype}, shape: {batch_dendrite.shape}")
        #    logger.info(f"     closest_points dtype: {closest_points.dtype}, shape: {closest_points.shape}")
           # logger.info(f"dendrite_ids dtype: {dendrite_ids.dtype}, shape: {dendrite_ids.shape}")

        # Process spines and pass midlines on for neck and head analysis
        t0 = time.time()

        spine_results, midlines = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts, batch_labels, scaling, 0, True, None, True, True,"spine", locations, settings, logger)
        if settings.additional_logging == True:
            logger.info(f"      Time taken for mesh measurements of spines: {time.time() - t0:.2f} seconds")


        # Subtract heads from spines and measure necks and ensure float for marching_cubes
        batch_data = cp.stack([
            cp.maximum(subvol[:, :, :, 0] - subvol[:, :, :, 1], 0).astype(cp.float32)
            for subvol in padded_subvolumes
        ])

        #for subvol in sub_volumes:
        #    subvol[:, :, :, 0] = cp.maximum(subvol[:, :, :, 0] - subvol[:, :, :, 1], 0)
        t0 = time.time()

        neck_results, _ = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts, batch_labels, scaling, 0, False, midlines, True, True, "neck", locations, settings,  logger) #True for min max mean width
        if settings.additional_logging == True:
            logger.info(f"      Time taken for mesh measurements of necks: {time.time() - t0:.2f} seconds")

        #now process heads
        t0 = time.time()
        batch_data = cp.stack([subvol[:, :, :, 1] for subvol in padded_subvolumes])  # spine vols
        # Process just heads - but don't need vol for length just surface area
        head_results, _ = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts, batch_labels, scaling, 1, False, midlines, True, True,"head", locations, settings, logger) #False for width using bounding box
        if settings.additional_logging == True:
                logger.info(f"      Time taken for mesh measurements of spine heads: {time.time() - t0:.2f} seconds")
        # Combine results for this batch

        g = lambda d, k: d.get(k, np.nan)

        for r_spine, r_neck, r_head in zip(spine_results, neck_results, head_results):

            #convex hull ration
            head_conv = g(r_head, 'head_convex_vol')
            head_vol = g(r_head, 'head_volume')
            hull_ratio = (
                np.nan
                if (np.isnan(head_conv) or np.isnan(head_vol) or head_vol == 0)
                else (head_conv - head_vol) / head_vol
            )


            results.append({
                'label': r_spine['ID'],
                # 'dendrite_id': int(dend_id),
                # -------------  spine -----------------
                'start_coords': g(r_spine, 'start_coords'),
                'spine_area': g(r_spine, 'spine_area'),
                'spine_vol': g(r_spine, 'spine_volume'),
                'spine_surf_area': g(r_spine, 'spine_surface_area'),
                'spine_length': g(r_spine, 'spine_length'),
                'spine_bbox_vol': g(r_spine, 'spine_bbox_vol'),
                'spine_extent': g(r_spine, 'spine_extent'),
                'spine_solidity': g(r_spine, 'spine_solidity'),
                'spine_convex_vol': g(r_spine, 'spine_convex_vol'),
                #'spine_axis_major': g(r_spine, 'spine_axis_major'),
                #'spine_axis_minor': g(r_spine, 'spine_axis_minor'),

                # -------------  head ------------------
                'head_width_mean': g(r_head, 'head_mean_width'),
                'head_area': g(r_head, 'head_area'),
                'head_vol': head_vol,
                'head_surf_area': g(r_head, 'head_surface_area'),
                'head_length': g(r_head, 'head_length'),
                'head_bbox_vol': g(r_head, 'head_bbox_vol'),
                'head_extent': g(r_head, 'head_extent'),
                'head_solidity': g(r_head, 'head_solidity'),
                'head_convex_vol': head_conv,
                #'head_axis_major': g(r_head, 'head_axis_major'),
                #'head_axis_minor': g(r_head, 'head_axis_minor'),

                'head_convex_hull_ratio': hull_ratio,

                # -------------  neck ------------------
                'neck_area': g(r_neck, 'neck_area'),
                'neck_vol': g(r_neck, 'neck_volume'),
                'neck_surf_area': g(r_neck, 'neck_surface_area'),
                'neck_length': g(r_neck, 'neck_length'),
                'neck_width_min': g(r_neck, 'neck_min_width'),
                'neck_width_max': g(r_neck, 'neck_max_width'),
                'neck_width_mean': g(r_neck, 'neck_mean_width'),
                'neck_bbox_vol': g(r_neck, 'neck_bbox_vol'),
                'neck_extent': g(r_neck, 'neck_extent'),
                'neck_solidity': g(r_neck, 'neck_solidity'),
                'neck_convex_vol': g(r_neck, 'neck_convex_vol'),
                #'neck_axis_major': g(r_neck, 'neck_axis_major'),
                #'neck_axis_minor': g(r_neck, 'neck_axis_minor'),

                #'head_length_calc': r_spine['spine_length'] - r_neck['neck_length'],
                #'head_vol_calc': r_spine['spine_volume'] - r_neck['neck_volume']
            })

        # Clean up batch GPU arrays between iterations
        del sub_volumes, start_coords, padded_subvolumes, pad_amounts
        del batch_data, batch_dendrite, batch_binary, closest_points
        del spine_results, neck_results, head_results, midlines
        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()

        i += batch_size

      except cp.cuda.memory.OutOfMemoryError:
        # OOM recovery: free GPU memory and halve batch size
        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        old_batch = batch_size
        batch_size = max(1, batch_size // 2)
        logger.warning(f"        GPU OOM at batch_size={old_batch}, retrying with batch_size={batch_size}")
        if batch_size == old_batch:
            logger.error(f"        Cannot reduce batch_size below 1, skipping batch at index {i}")
            i += batch_size
        # Recalculate milestones with new batch size
        total_batches = (len(labels) + batch_size - 1) // batch_size

    spine_results_df = pd.DataFrame(results)

    logger.info(f"      Total time taken for mesh analysis: {time.time() - start_time:.2f} seconds")

    return spine_results_df


@mp.profile_mem()
def analyze_spines_batch_cpu_resident(spine_labels_vol, head_labels_vol, dendrite, neuron,
                                      locations, settings, logger, scaling):
    """CPU-resident mesh analysis for large volumes that don't fit in GPU memory.

    Keeps the 4 input arrays in CPU memory. For each batch of spines, extracts
    small bounding-box crops from CPU, uploads only those to GPU for processing.
    Each spine subvolume is typically <100x100x100 voxels = negligible GPU memory.
    """
    start_time = time.time()
    results = []

    # Get unique spine labels. safe_unique streams on zarr, direct on numpy.
    labels_np = chunked.safe_unique(spine_labels_vol)
    labels_np = labels_np[labels_np != 0]
    logger.info(f"       CPU-resident mode: {len(labels_np)} spines to process.")

    if len(labels_np) == 0:
        return pd.DataFrame(results)

    # Use find_objects for fast bounding box lookup. On zarr inputs we route to
    # the streaming bbox helper — ndimage.find_objects would silently materialize
    # the full volume via np.asarray (22 GB T7, 220 GB at 100 GB scale).
    max_label = int(labels_np.max())
    if hasattr(spine_labels_vol, 'chunks'):
        # Streaming find_objects avoids the per-label scan in _get_object_bboxes
        # (O(N_labels × N_voxels) — pathological at T7+).
        _bbox_map = chunked.find_objects_streaming(spine_labels_vol, logger=logger)
        slices_lookup = [None] * max_label
        for _lbl, (_z0, _z1, _y0, _y1, _x0, _x1) in _bbox_map.items():
            if 1 <= _lbl <= max_label:
                slices_lookup[_lbl - 1] = (slice(_z0, _z1), slice(_y0, _y1), slice(_x0, _x1))
    else:
        slices_lookup = ndimage.find_objects(spine_labels_vol, max_label)

    # Calculate batch size based on available GPU memory
    avg_sub_volume_shape = 6 / scaling[0], 6 / scaling[1], 6 / scaling[2]
    dtype_size = 2
    free_mem, _ = cp.cuda.runtime.memGetInfo()
    # Use a conservative batch size — we're only uploading small crops
    sub_volume_memory = np.prod(avg_sub_volume_shape) * dtype_size * 4
    batch_size = max(1, int(free_mem * 0.3 / (sub_volume_memory * 3)))
    batch_size = min(batch_size, len(labels_np))
    logger.info(f"       Using batch size of {batch_size} spines...")

    padding = 5
    total_batches = (len(labels_np) + batch_size - 1) // batch_size
    milestones = set(int(total_batches * p / 100) for p in range(10, 101, 10))

    # TODO: PERF — double-buffer: use ThreadPoolExecutor to pre-extract NEXT batch's
    # CPU crops while GPU processes current batch. Expected 1.3-1.8x speedup.
    i = 0
    while i < len(labels_np):
      try:
        batch_labels_np = labels_np[i:i + batch_size]
        current_batch = i // batch_size + 1

        if current_batch in milestones:
            pct = (current_batch / total_batches) * 100
            logger.info(f"        Processing batch {current_batch} of {total_batches}... ({int(pct)}% complete)")

        # Extract subvolumes from CPU arrays and upload to GPU
        subvolumes_gpu = []
        start_coords_list = []
        valid_labels = []

        for label in batch_labels_np:
            sl = slices_lookup[int(label) - 1]  # find_objects is 1-indexed via label value
            if sl is None:
                continue

            # Expand bounding box with padding
            starts = [max(0, s.start - padding) for s in sl]
            stops = [min(spine_labels_vol.shape[dim], s.stop + padding) for dim, s in enumerate(sl)]
            crop_sl = tuple(slice(st, sp) for st, sp in zip(starts, stops))

            # Extract small crops from all 4 CPU arrays. np.asarray is a no-op for
            # numpy and a zarr-to-numpy crop read for zarr — keeps per-spine peak RAM
            # to ~1 MB regardless of whether backing is numpy or zarr.
            crop_spine = np.asarray(spine_labels_vol[crop_sl])
            crop_head = np.asarray(head_labels_vol[crop_sl])
            crop_dend = np.asarray(dendrite[crop_sl])
            crop_neuron = np.asarray(neuron[crop_sl])

            # Stack into multi-channel and upload to GPU
            crop_gpu = cp.stack([cp.asarray(crop_spine),
                                 cp.asarray(crop_head),
                                 cp.asarray(crop_dend),
                                 cp.asarray(crop_neuron)], axis=-1)
            subvolumes_gpu.append(crop_gpu)
            start_coords_list.append(cp.array(starts))
            valid_labels.append(label)

        if len(subvolumes_gpu) == 0:
            i += batch_size
            continue

        batch_labels_cp = cp.array(valid_labels)
        start_coords = cp.stack(start_coords_list)

        # Mask subvolumes (same logic as GPU-resident path)
        sub_volumes = batch_mask_subvolumes_cp(subvolumes_gpu, batch_labels_cp)

        # Pad subvolumes for batching
        padded_subvolumes, pad_amounts = pad_and_center_subvolumes(sub_volumes, settings, logger)

        # Save spine arrays — gated to avoid IO errors on network drives
        if settings.save_intermediate_data:
            export_subvols = cp.transpose(cp.stack(padded_subvolumes, axis=0), (0, 1, 4, 2, 3))
            export_subvols[:, :, :3] = export_subvols[:, :, :3] * 65535 / 2
            export_subvols[:, :, 0] = export_subvols[:, :, 0] - export_subvols[:, :, 1]
            export_subvols = cp.asnumpy(export_subvols)

            imwrite(f'{locations.arrays}/Spine_vols_{settings.filename}_b{current_batch}.tif',
                    export_subvols.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True,
                    photometric='minisblack',
                    metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'TZCYX', 'mode': 'composite'},
                    resolution=(1 / settings.input_resXY, 1 / settings.input_resXY))

            imwrite(f'{locations.arrays}/Spine_MIPs_{settings.filename}_b{current_batch}.tif',
                    np.max(export_subvols, axis=1).astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True,
                    photometric='minisblack',
                    metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX', 'mode': 'composite'},
                    resolution=(1 / settings.input_resXY, 1 / settings.input_resXY))

            del export_subvols

        # Generate data for mesh analysis
        batch_data = cp.stack([subvol[:, :, :, 0].astype(cp.float32) for subvol in padded_subvolumes])
        batch_dendrite = cp.stack([subvol[:, :, :, 2] for subvol in padded_subvolumes])
        batch_binary = cp.stack([subvol[:, :, :, 2] > 0 for subvol in padded_subvolumes])

        closest_points = find_closest_points_batch(batch_data, batch_binary)

        # Process spines
        spine_results, midlines = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts,
                                                      batch_labels_cp, scaling, 0, True, None, True, True, "spine",
                                                      locations, settings, logger)

        # Necks
        batch_data = cp.stack([
            cp.maximum(subvol[:, :, :, 0] - subvol[:, :, :, 1], 0).astype(cp.float32)
            for subvol in padded_subvolumes
        ])
        neck_results, _ = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts,
                                              batch_labels_cp, scaling, 0, False, midlines, True, True, "neck",
                                              locations, settings, logger)

        # Heads
        batch_data = cp.stack([subvol[:, :, :, 1] for subvol in padded_subvolumes])
        head_results, _ = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts,
                                              batch_labels_cp, scaling, 1, False, midlines, True, True, "head",
                                              locations, settings, logger)

        # Combine results
        g = lambda d, k: d.get(k, np.nan)

        for r_spine, r_neck, r_head in zip(spine_results, neck_results, head_results):
            head_conv = g(r_head, 'head_convex_vol')
            head_vol = g(r_head, 'head_volume')
            hull_ratio = (
                np.nan
                if (np.isnan(head_conv) or np.isnan(head_vol) or head_vol == 0)
                else (head_conv - head_vol) / head_vol
            )

            results.append({
                'label': r_spine['ID'],
                'start_coords': g(r_spine, 'start_coords'),
                'spine_area': g(r_spine, 'spine_area'),
                'spine_vol': g(r_spine, 'spine_volume'),
                'spine_surf_area': g(r_spine, 'spine_surface_area'),
                'spine_length': g(r_spine, 'spine_length'),
                'spine_bbox_vol': g(r_spine, 'spine_bbox_vol'),
                'spine_extent': g(r_spine, 'spine_extent'),
                'spine_solidity': g(r_spine, 'spine_solidity'),
                'spine_convex_vol': g(r_spine, 'spine_convex_vol'),
                'head_width_mean': g(r_head, 'head_mean_width'),
                'head_area': g(r_head, 'head_area'),
                'head_vol': head_vol,
                'head_surf_area': g(r_head, 'head_surface_area'),
                'head_length': g(r_head, 'head_length'),
                'head_bbox_vol': g(r_head, 'head_bbox_vol'),
                'head_extent': g(r_head, 'head_extent'),
                'head_solidity': g(r_head, 'head_solidity'),
                'head_convex_vol': head_conv,
                'head_convex_hull_ratio': hull_ratio,
                'neck_area': g(r_neck, 'neck_area'),
                'neck_vol': g(r_neck, 'neck_volume'),
                'neck_surf_area': g(r_neck, 'neck_surface_area'),
                'neck_length': g(r_neck, 'neck_length'),
                'neck_width_min': g(r_neck, 'neck_min_width'),
                'neck_width_max': g(r_neck, 'neck_max_width'),
                'neck_width_mean': g(r_neck, 'neck_mean_width'),
                'neck_bbox_vol': g(r_neck, 'neck_bbox_vol'),
                'neck_extent': g(r_neck, 'neck_extent'),
                'neck_solidity': g(r_neck, 'neck_solidity'),
                'neck_convex_vol': g(r_neck, 'neck_convex_vol'),
            })

        # Clean up GPU memory
        del sub_volumes, subvolumes_gpu, start_coords, padded_subvolumes, pad_amounts
        del batch_data, batch_dendrite, batch_binary, closest_points
        del spine_results, neck_results, head_results, midlines
        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()

        i += batch_size

      except cp.cuda.memory.OutOfMemoryError:
        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        old_batch = batch_size
        batch_size = max(1, batch_size // 2)
        logger.warning(f"        GPU OOM at batch_size={old_batch}, retrying with batch_size={batch_size}")
        if batch_size == old_batch:
            logger.error(f"        Cannot reduce batch_size below 1, skipping batch at index {i}")
            i += batch_size
        total_batches = (len(labels_np) + batch_size - 1) // batch_size

    spine_results_df = pd.DataFrame(results)
    logger.info(f"      Total time taken for CPU-resident mesh analysis: {time.time() - start_time:.2f} seconds")
    return spine_results_df


@mp.profile_mem()
def calculate_dend_ID_and_geo_distance(labeled_dendrites, labeled_spines, skeleton_coords, skeleton_labels, filename,
                                       locations,
                                       soma_vol=None, settings=None, skeleton_volume=None, logger=None):
    try:

        def _log(msg):
            if logger is not None:
                logger.info(msg)

        # track time of function
        t0 = time.time()
        _log("       Creating KDTree from skeletonized dendrites...")
        sampling_method = 'topology'
        sampling_param = 4
        dendrite_tree, skeleton_labels = create_kdtree_from_skeleton(
            skeleton_coords, skeleton_labels, sampling_method=sampling_method, sampling_param=sampling_param,
            skeleton_volume=skeleton_volume)
        t1 = time.time()
        _log(f"       Time taken: {t1 - t0:.2f} s")

        _log("       Matching and relabeling objects...")
        t0 = time.time()

        max_match_distance = settings.neuron_spine_dist * 3 if settings is not None else None
        relabeled_spines, spine_labels_1D, spine_labels_dict, indices, distances, dend_IDs = match_and_relabel_objects_geo(
            dendrite_tree, skeleton_labels, labeled_spines, max_distance=max_match_distance)
        t1 = time.time()
        _log(f"       Time taken: {t1 - t0:.2f} s")

        mapping_dend_to_spine_data = defaultdict(list)
        for i in range(len(spine_labels_1D)):
            spine = spine_labels_1D[i]
            dist = spine_labels_dict.get(spine, 0)
            if dist != 0:
                index_in_A = indices[i]
                coord_in_A = dendrite_tree.data[index_in_A]
                mapping_dend_to_spine_data[dist].append({
                    'label_B': spine,
                    'index_in_A': index_in_A,
                    'coord_in_A': coord_in_A,
                    'distance': distances[i],
                })

        # ------------------------------------------------------------------
        # Per-dendrite geodesic distance — bbox-based for scalability
        # ------------------------------------------------------------------
        # For large volumes, avoid allocating full-volume arrays:
        # - geodesic_distance_image (22 GB float32 for T7)
        # - per-dendrite dendrite_mask (5.5 GB bool)
        # - per-dendrite distance_map (22 GB float64)
        # Instead: extract each dendrite's bbox, compute geodesic locally,
        # write results to zarr. MIP code handles zarr natively.
        vol_shape = labeled_dendrites.shape
        n_voxels = int(vol_shape[0]) * int(vol_shape[1]) * int(vol_shape[2])
        # Force the per-dendrite bbox path whenever labeled_dendrites is zarr —
        # the fallback full-volume branch (L2732+) uses ndimage.find_objects and
        # boolean masks that either crash or silently materialize on zarr.
        use_bbox_geodesic = (n_voxels > 500_000_000) or hasattr(labeled_dendrites, 'chunks')

        # Pre-compute soma KDTree once (not per-dendrite). Slab-iterate np.argwhere
        # on zarr soma_vol so we don't materialize a 50 GB uint8 volume at 100 GB scale.
        # Soma is typically <1% of volume so the resulting coords array stays small.
        soma_tree = None
        if soma_vol is not None:
            if isinstance(soma_vol, np.ndarray):
                soma_coords = np.argwhere(soma_vol)
            elif hasattr(soma_vol, 'chunks'):
                _sv_shape = soma_vol.shape
                _sv_bpz = int(_sv_shape[1]) * int(_sv_shape[2]) * np.dtype(soma_vol.dtype).itemsize
                _sv_slz = max(1, min(64, int(2 * 1024**3 / max(_sv_bpz, 1))))
                _parts = []
                for _sz in range(0, _sv_shape[0], _sv_slz):
                    _sze = min(_sz + _sv_slz, _sv_shape[0])
                    _slab = np.asarray(soma_vol[_sz:_sze])
                    _local = np.argwhere(_slab)
                    if _local.size:
                        _local[:, 0] += _sz
                        _parts.append(_local)
                    del _slab
                soma_coords = (np.concatenate(_parts, axis=0) if _parts
                               else np.empty((0, 3), dtype=np.int64))
            else:
                soma_coords = None
            if soma_coords is not None and soma_coords.size > 0:
                from scipy.spatial import cKDTree
                soma_tree = cKDTree(soma_coords)
                del soma_coords

        if settings is not None:
            spacing = (settings.input_resZ, settings.input_resXY, settings.input_resXY)
        else:
            spacing = None

        # XY-downsample geodesic for large images. Dijkstra cost grows ~Z*Y*X
        # voxels; Z stays exact (path-length precision), XY drops to ~0.3 µm.
        # Trigger at >500M voxels (≈1 GB at uint16) — same threshold that already
        # forces the bbox path. Block-OR (np.any) preserves dendrite connectivity.
        # T_LARGE giant dendrite: 240 GB f64 distance_map → ~5 GB at factor=7.
        _GEODESIC_TARGET_RESXY_UM = 0.3
        _GEODESIC_DOWNSAMPLE_MIN_VOXELS = 500_000_000
        _geo_factor = 1
        if (settings is not None and getattr(settings, 'input_resXY', None)
                and n_voxels > _GEODESIC_DOWNSAMPLE_MIN_VOXELS):
            _f = max(1, int(round(_GEODESIC_TARGET_RESXY_UM / settings.input_resXY)))
            if _f > 1:
                _geo_factor = _f
                _log(f"       Geodesic XY downsample ENABLED: factor={_geo_factor} "
                     f"(input_resXY={settings.input_resXY:.4f} µm, "
                     f"target={_GEODESIC_TARGET_RESXY_UM} µm, "
                     f"effective_resXY={settings.input_resXY * _geo_factor:.4f} µm). "
                     f"Z resolution unchanged.")

        _log("       Computing geodesic distances...")
        t0 = time.time()
        geodesic_distances_B = {}

        if use_bbox_geodesic:
            # Per-dendrite bbox approach: O(largest_dendrite_bbox) memory
            # Get bounding boxes for dendrites that have spines
            dendrite_ids_with_spines = sorted(mapping_dend_to_spine_data.keys())
            if hasattr(labeled_dendrites, 'chunks'):
                dend_bboxes = chunked._get_object_bboxes(
                    labeled_dendrites, set(dendrite_ids_with_spines))
            else:
                all_slices = ndimage.find_objects(labeled_dendrites)
                dend_bboxes = {}
                for d in dendrite_ids_with_spines:
                    sl = all_slices[d - 1] if d - 1 < len(all_slices) else None
                    if sl is not None:
                        dend_bboxes[d] = (sl[0].start, sl[0].stop,
                                          sl[1].start, sl[1].stop,
                                          sl[2].start, sl[2].stop)

            # Zarr output for geodesic image (MIP reads via slab accumulation)
            # Write to active
            # workspace when available so geodesic zarr lands on the same
            # filesystem as the rest of the cache (E:/respan_workspace),
            # cleans up via workspace.cleanup() at end of run, and avoids
            # the system-temp fallback which has unknown free-space at
            # T_LARGE scale (geodesic_distance can be 100+ GB).
            _gd_workspace = getattr(settings, '_active_workspace', None) if settings is not None else None
            if _gd_workspace is not None:
                geo_zarr = _gd_workspace.create_array(
                    'geodesic_distance', shape=vol_shape, dtype=np.float32,
                    chunks=chunked.STREAMING_CHUNKS, fill_value=0)
                _geo_cache = None
            else:
                import tempfile
                _geo_cache = tempfile.mkdtemp(prefix='respan_geo_')
                geo_zarr = zarr.open(os.path.join(_geo_cache, 'geodesic'), mode='w',
                                     shape=vol_shape, dtype=np.float32,
                                     chunks=chunked.STREAMING_CHUNKS, fill_value=0)

            # TODO: PERF — parallelize with ThreadPoolExecutor (4-8 workers).
            # Each dendrite is independent. Expected 4-8x speedup on T7.
            _n_dend_total = len(dendrite_ids_with_spines)
            _dend_milestone = max(50, _n_dend_total // 20) if _n_dend_total else 0
            for _d_idx, dendrite in enumerate(dendrite_ids_with_spines):
                if _dend_milestone and (_d_idx + 1) % _dend_milestone == 0:
                    _log(f"       Geodesic: {_d_idx + 1}/{_n_dend_total} dendrites processed...")
                data_list = mapping_dend_to_spine_data[dendrite]
                if dendrite not in dend_bboxes:
                    continue
                z0, z1, y0, y1, x0, x1 = dend_bboxes[dendrite]

                # For giant-bbox dendrites
                # (single dendrite spanning most of the volume at T_LARGE),
                # `np.array(labeled_dendrites[bbox])` materializes the full
                # bbox numpy — could be 60+ GB int32. Slab-fill the bool mask
                # directly via per-Z reads, skipping the intermediate.
                _bbox_shape = (z1 - z0, y1 - y0, x1 - x0)
                _bbox_voxels = int(_bbox_shape[0]) * int(_bbox_shape[1]) * int(_bbox_shape[2])
                _bbox_bytes_int32 = _bbox_voxels * 4
                _bbox_bytes_f64 = _bbox_voxels * 8
                _LARGE_BBOX_THRESHOLD = 1 * (1024 ** 3)  # 1 GB int32 / 2 GB f64
                # Guard: the geodesic
                # itself allocates a float64 distance_map of bbox shape inside
                # compute_geodesic_distance_map (np.full(shape, inf)). For a
                # 50% T_LARGE bbox this would be 240 GB float64 — fatal even
                # if we slab-fill the mask. Guard at 50 GB float64 (≈ 6.7 GB
                # bbox) to leave headroom on a 137 GB host. For genuinely
                # over-large dendrites: skip geodesic, set NaN per-spine
                # values, log warning. Per-spine euclidean distance is still
                # available downstream via dendrite_distance min-over-spine.
                _GEODESIC_F64_HARD_LIMIT = 50 * (1024 ** 3)  # 50 GB float64

                # XY-downsampled geodesic path (large-image mode, _geo_factor > 1).
                # Slab-OR builds the downsampled bool mask directly from zarr,
                # avoiding any full-res bbox materialization. Dijkstra runs at
                # ~0.3 µm XY, then per-spine values are looked up in downsampled
                # coords and the distance map is upsampled (np.repeat) back to
                # the full bbox for the geo_zarr MIP write.
                if _geo_factor > 1:
                    bz, by, bx = _bbox_shape
                    ds_y = (by + _geo_factor - 1) // _geo_factor
                    ds_x = (bx + _geo_factor - 1) // _geo_factor
                    _ds_shape = (bz, ds_y, ds_x)
                    _ds_voxels = int(bz) * int(ds_y) * int(ds_x)
                    _ds_bytes_f64 = _ds_voxels * 8
                    if _ds_bytes_f64 > _GEODESIC_F64_HARD_LIMIT:
                        _log(f"       Geodesic SKIPPED for dendrite {dendrite}: even after "
                             f"XY downsample factor={_geo_factor}, bbox {_ds_shape} "
                             f"would require {_ds_bytes_f64/1e9:.1f} GB float64 distance_map "
                             f"(>50 GB hard limit). Per-spine NaN; euclidean still available.")
                        for data in data_list:
                            geodesic_distances_B[data['label_B']] = float('nan')
                        continue

                    # Slab-fill downsampled mask via XY block-OR
                    local_mask = np.empty(_ds_shape, dtype=bool)
                    _slab_z = max(1, min(64, bz))
                    _py = (-by) % _geo_factor
                    _px = (-bx) % _geo_factor
                    for _z in range(0, bz, _slab_z):
                        _ze = min(_z + _slab_z, bz)
                        _slab = labeled_dendrites[z0 + _z:z0 + _ze, y0:y1, x0:x1]
                        _slab_bool = (np.asarray(_slab) == dendrite)
                        del _slab
                        if _py or _px:
                            _slab_bool = np.pad(
                                _slab_bool, ((0, 0), (0, _py), (0, _px)),
                                mode='constant', constant_values=False)
                        _sh = _slab_bool.shape
                        local_mask[_z:_ze] = _slab_bool.reshape(
                            _sh[0], _sh[1] // _geo_factor, _geo_factor,
                            _sh[2] // _geo_factor, _geo_factor).any(axis=(2, 4))
                        del _slab_bool

                    if not np.any(local_mask):
                        for data in data_list:
                            geodesic_distances_B[data['label_B']] = np.nan
                        continue

                    # Adjusted spacing: Z exact, XY scaled by factor (still in µm)
                    if spacing is not None:
                        _ds_spacing = (spacing[0], spacing[1] * _geo_factor,
                                       spacing[2] * _geo_factor)
                    else:
                        _ds_spacing = None

                    # Starting point in downsampled local coords. Map to global
                    # for soma-distance query (Y/X scaled, Z stays the same).
                    local_coords = np.argwhere(local_mask)
                    if soma_tree is not None:
                        global_coords = local_coords.astype(np.float64).copy()
                        global_coords[:, 0] += z0
                        global_coords[:, 1] = global_coords[:, 1] * _geo_factor + y0
                        global_coords[:, 2] = global_coords[:, 2] * _geo_factor + x0
                        dists_to_soma, _ = soma_tree.query(global_coords)
                        starting_point = tuple(local_coords[np.argmin(dists_to_soma)])
                        del global_coords, dists_to_soma
                    else:
                        starting_point = tuple(local_coords[
                            np.argmin(np.sum(local_coords, axis=1))])
                    del local_coords

                    distance_map = compute_geodesic_distance_map(
                        local_mask, [starting_point], spacing=_ds_spacing)

                    # Per-spine lookup in downsampled coords. Block-OR guarantees
                    # the downsampled voxel is in the mask whenever the original
                    # attachment voxel is in the dendrite.
                    for data in data_list:
                        coord_global = np.round(data['coord_in_A']).astype(int)
                        cz = coord_global[0] - z0
                        cy_ds = (coord_global[1] - y0) // _geo_factor
                        cx_ds = (coord_global[2] - x0) // _geo_factor
                        if (0 <= cz < _ds_shape[0] and
                                0 <= cy_ds < _ds_shape[1] and
                                0 <= cx_ds < _ds_shape[2]):
                            _val = float(distance_map[cz, cy_ds, cx_ds])
                            geodesic_distances_B[data['label_B']] = (
                                _val if np.isfinite(_val) else np.nan)
                        else:
                            geodesic_distances_B[data['label_B']] = np.nan

                    # Upsample distance_map (XY-repeat) for geo_zarr MIP write.
                    # Slab in Z so peak memory stays ~one bbox-Z slice of f32.
                    _g_f32 = distance_map.astype(np.float32, copy=False)
                    for _z in range(0, bz, _slab_z):
                        _ze = min(_z + _slab_z, bz)
                        _ds_slab = _g_f32[_z:_ze]
                        # Repeat in XY, crop to original bbox extent (handles padding)
                        _up = np.repeat(np.repeat(
                            _ds_slab, _geo_factor, axis=1), _geo_factor, axis=2)
                        _up = _up[:, :by, :bx]
                        # Mask to downsampled-mask footprint (also XY-repeated).
                        # Slight "stair-step" at 0.3 µm — acceptable for MIP.
                        _ds_mask_slab = local_mask[_z:_ze]
                        _up_mask = np.repeat(np.repeat(
                            _ds_mask_slab, _geo_factor, axis=1), _geo_factor, axis=2)
                        _up_mask = _up_mask[:, :by, :bx]
                        _local_geo_slab = np.where(
                            _up_mask, _up, 0).astype(np.float32, copy=False)
                        del _up, _up_mask, _ds_slab, _ds_mask_slab
                        _existing = np.asarray(
                            geo_zarr[z0 + _z:z0 + _ze, y0:y1, x0:x1])
                        geo_zarr[z0 + _z:z0 + _ze, y0:y1, x0:x1] = np.maximum(
                            _existing, _local_geo_slab)
                        del _existing, _local_geo_slab
                    del local_mask, distance_map, _g_f32
                    continue

                if _bbox_bytes_f64 > _GEODESIC_F64_HARD_LIMIT:
                    _log(f"       Geodesic SKIPPED for dendrite {dendrite}: bbox {_bbox_shape} "
                         f"would require {_bbox_bytes_f64/1e9:.1f} GB float64 distance_map (>50 GB hard limit). "
                         f"Per-spine geodesic_dist_to_soma set to NaN; euclidean still available.")
                    for data in data_list:
                        geodesic_distances_B[data['label_B']] = float('nan')
                    continue
                if _bbox_bytes_int32 > _LARGE_BBOX_THRESHOLD:
                    # Slab-fill: build local_mask directly from per-Z equality
                    # reads, avoids both the int32 bbox materialization AND
                    # the bool intermediate.
                    local_mask = np.empty(_bbox_shape, dtype=bool)
                    _slab_z = max(1, min(64, _bbox_shape[0]))
                    for _z in range(0, _bbox_shape[0], _slab_z):
                        _ze = min(_z + _slab_z, _bbox_shape[0])
                        _slab = labeled_dendrites[z0 + _z:z0 + _ze, y0:y1, x0:x1]
                        local_mask[_z:_ze] = (np.asarray(_slab) == dendrite)
                        del _slab
                else:
                    # Small bbox — legacy single read
                    local_labels = np.array(labeled_dendrites[z0:z1, y0:y1, x0:x1])
                    local_mask = (local_labels == dendrite)
                    del local_labels

                if not np.any(local_mask):
                    continue

                # Find starting point in local coordinates
                local_coords = np.argwhere(local_mask)
                if soma_tree is not None:
                    global_coords = local_coords + np.array([z0, y0, x0])
                    dists_to_soma, _ = soma_tree.query(global_coords)
                    starting_point = tuple(local_coords[np.argmin(dists_to_soma)])
                    del global_coords, dists_to_soma
                else:
                    starting_point = tuple(local_coords[
                        np.argmin(np.sum(local_coords, axis=1))])
                del local_coords

                # Compute geodesic on local crop only
                distance_map = compute_geodesic_distance_map(
                    local_mask, [starting_point], spacing=spacing)

                # Write to zarr for MIP generation. Phase L7: for giant-bbox
                # dendrites, slab the read+max+write so the float32 `existing`
                # buffer doesn't hit ~60 GB at T_LARGE giant-dendrite case.
                local_geo = np.where(local_mask, distance_map.astype(np.float32), 0)
                _bbox_bytes_f32 = _bbox_voxels * 4
                if _bbox_bytes_f32 > _LARGE_BBOX_THRESHOLD:
                    _slab_z = max(1, min(64, _bbox_shape[0]))
                    for _z in range(0, _bbox_shape[0], _slab_z):
                        _ze = min(_z + _slab_z, _bbox_shape[0])
                        _existing_slab = np.asarray(
                            geo_zarr[z0 + _z:z0 + _ze, y0:y1, x0:x1])
                        geo_zarr[z0 + _z:z0 + _ze, y0:y1, x0:x1] = np.maximum(
                            _existing_slab, local_geo[_z:_ze])
                        del _existing_slab
                else:
                    existing = np.array(geo_zarr[z0:z1, y0:y1, x0:x1])
                    geo_zarr[z0:z1, y0:y1, x0:x1] = np.maximum(existing, local_geo)
                    del existing
                del local_geo

                # Read spine geodesic values at skeleton attachment coords
                for data in data_list:
                    coord_global = np.round(data['coord_in_A']).astype(int)
                    cz, cy, cx = coord_global[0] - z0, coord_global[1] - y0, coord_global[2] - x0
                    if (0 <= cz < local_mask.shape[0] and
                            0 <= cy < local_mask.shape[1] and
                            0 <= cx < local_mask.shape[2]):
                        geodesic_distances_B[data['label_B']] = float(distance_map[cz, cy, cx])
                    else:
                        geodesic_distances_B[data['label_B']] = np.nan

                del local_mask, distance_map

            geodesic_distance_image = geo_zarr  # zarr — MIP reads via slab accumulation
            # Note: _geo_cache cleaned up by caller's workspace.cleanup() or at process exit

        else:
            # Original full-volume path for small images
            # Trip-wire: this branch uses `labeled_dendrites == dendrite` and
            # `np.full(vol_shape, ...)` which silently break on zarr. The
            # use_bbox_geodesic guard above forces the zarr-safe path; this
            # assert fails loudly if a future change slips zarr through.
            assert isinstance(labeled_dendrites, np.ndarray), (
                "full-volume geodesic path requires numpy labeled_dendrites — "
                "use_bbox_geodesic should have routed zarr input to the bbox branch")
            geodesic_distance_image = np.full(vol_shape, np.nan, dtype=np.float32)
            for dendrite, data_list in mapping_dend_to_spine_data.items():
                dendrite_mask = (labeled_dendrites == dendrite)

                # Determine starting point
                object_coords = np.argwhere(dendrite_mask)
                if soma_tree is not None:
                    dists_to_soma, _ = soma_tree.query(object_coords)
                    starting_point = tuple(object_coords[np.argmin(dists_to_soma)])
                else:
                    starting_point = tuple(object_coords[
                        np.argmin(np.sum(object_coords, axis=1))])

                distance_map = compute_geodesic_distance_map(
                    dendrite_mask, [starting_point], spacing=spacing)

                geodesic_distance_image[dendrite_mask] = distance_map[dendrite_mask]

                for data in data_list:
                    coord = tuple(np.round(data['coord_in_A']).astype(int))
                    geodesic_distances_B[data['label_B']] = float(distance_map[coord])

            geodesic_distance_image = np.nan_to_num(
                geodesic_distance_image, nan=0).astype(np.float32)

        t1 = time.time()
        _log(f"       Geodesic distance complete: {t1 - t0:.2f} s")

        for label_B, distance in geodesic_distances_B.items():
            dend_IDs.loc[dend_IDs['label'] == label_B, 'geodesic_dist_to_soma'] = distance

        # Skip full-volume unique counts for large volumes (np.unique OOMs on T7).
        # fast_unique_count is @jit(nopython=True) — numba can't type zarr.Array,
        # so route zarr inputs to the slab-streaming safe_unique helper instead.
        _ld_zarr = hasattr(labeled_dendrites, 'chunks')
        _ls_zarr = hasattr(labeled_spines, 'chunks')
        _rs_zarr = hasattr(relabeled_spines, 'chunks')
        _ld_bytes = labeled_dendrites.nbytes if not _ld_zarr else int(np.prod(labeled_dendrites.shape)) * labeled_dendrites.dtype.itemsize
        if _ld_bytes < 2e9:
            if _ld_zarr:
                _log(f"       Original dendrite labels: {len(chunked.safe_unique(labeled_dendrites))}")
            else:
                _log(f"       Original dendrite labels: {fast_unique_count(labeled_dendrites)}")
            if _ls_zarr:
                _log(f"       Original spine labels: {len(chunked.safe_unique(labeled_spines))}")
            else:
                _log(f"       Original spine labels: {fast_unique_count(labeled_spines)}")
            if _rs_zarr:
                _log(f"       Labels after relabeling: {len(chunked.safe_unique(relabeled_spines))}")
            else:
                _log(f"       Labels after relabeling: {fast_unique_count(relabeled_spines)}")

        return dend_IDs, geodesic_distance_image

    except Exception as e:
        if logger is not None:
            logger.error(f"An error occurred in calculate_dend_ID_and_geo_distance: {repr(e)}")
        import traceback
        traceback.print_exc()
        raise

@mp.profile_mem()
def analyze_spines_batch_tiled(spine_labels_vol, head_labels_vol, dendrite, neuron, locations, settings, logger,
                               scaling):
    start_time = time.time()
    results = []
    shape = spine_labels_vol.shape
    tile_size = calculate_tile_size(shape, settings, logger)
    overlap = (10, 50, 50)  # Adjust based on your needs

    # Calculate total number of tiles
    total_tiles = math.ceil(shape[0] / (tile_size[0] - overlap[0])) * \
                  math.ceil(shape[1] / (tile_size[1] - overlap[1])) * \
                  math.ceil(shape[2] / (tile_size[2] - overlap[2]))

    current_tile = 0

    for z in range(0, shape[0], tile_size[0] - overlap[0]):
        for y in range(0, shape[1], tile_size[1] - overlap[1]):
            for x in range(0, shape[2], tile_size[2] - overlap[2]):
                current_tile += 1
                z_end = min(z + tile_size[0], shape[0])
                y_end = min(y + tile_size[1], shape[1])
                x_end = min(x + tile_size[2], shape[2])

                tile_spine = spine_labels_vol[z:z_end, y:y_end, x:x_end]
                tile_head = head_labels_vol[z:z_end, y:y_end, x:x_end]
                tile_dendrite = dendrite[z:z_end, y:y_end, x:x_end]
                tile_neuron = neuron[z:z_end, y:y_end, x:x_end]


                logger.info(f"      Processing tile {current_tile} of {total_tiles}: z={z}:{z_end}, y={y}:{y_end}, x={x}:{x_end}")


                # Process the tile
                tile_results = process_spine_batch_tile(f'tile{z}_{y}_{x}', tile_spine, tile_head, tile_dendrite, tile_neuron, locations, settings,
                                            logger, scaling)

                # Adjust coordinates to global space
                for result in tile_results:
                    result['start_coords'] = [coord + offset for coord, offset in
                                              zip(result['start_coords'], [z, y, x])]

                results.extend(tile_results)
                cp.cuda.Device().synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

    spine_results_df = pd.DataFrame(results)
    spine_results_df = filter_duplicated_spine_results(spine_results_df)
    logger.info(f"      Total time taken for tiled spine mesh analysis: {time.time() - start_time:.2f} seconds")
    return spine_results_df

@mp.profile_mem()
def filter_duplicated_spine_results(df):
    # Check if 'label' is already the index
    if df.index.name == 'label':
        df = df.reset_index()

    # Ensure 'label' is a column
    if 'label' not in df.columns:
        raise ValueError("'label' is neither a column nor the index of the DataFrame")

        # Convert the 'label' column to strings
    df['label'] = df['label'].astype(str)

    # Create a temporary column for sorting
    df['temp_sort'] = df['label'].astype(int)

    # Group by 'label' and keep the row with the largest spine_vol for each group
    df_final = df.loc[df.groupby('label')['spine_vol'].idxmax()]

    # Sort the final dataframe by the temporary sort column
    df_final = df_final.sort_values('temp_sort')

    # Remove the temporary sort column
    df_final = df_final.drop('temp_sort', axis=1)

    return df_final



def spine_vox_measurements(image, labels, dendrite, max_label, neuron_ch, prefix, dendrite_distance, soma_distance, sizes, dist,
                         settings, locations, filename, logger, soma_present=None):
    """ measures intensity of each channel, as well as distance to dendrite
    Args:
        labels (detected cells)
        settings (dictionary of settings)

    Returns:
        pandas table and labeled spine image
    """


    # User option (settings.measure_intensity, default True): per-channel
    # mean/max intensity adds 3 per_spine_regionprops calls per image (one
    # each for spine head / whole spine / neck) at minutes-per-call scale on
    # T7+. Users who only need morphology can skip this entirely. The geometry
    # measurements at L5326+ (using regionprops on dendrite_distance for
    # head_euclidean_dist_to_dend / spine_length_euclidean) are NOT intensity
    # and stay regardless of this flag.
    _measure_intensity = bool(getattr(settings, 'measure_intensity', True))

    # Normalize image shape + build per-channel intensity_dict.
    # Handles:
    #   - numpy 3D     → expand to 4D (Z,1,Y,X), index channel slice view
    #   - numpy 4D     → index channel slice view
    #   - zarr 3D      → single channel, pass zarr handle directly
    #   - zarr 4D      → multi-channel, materialize per channel (RAM heavy)
    _image_is_zarr = hasattr(image, 'chunks')
    if _image_is_zarr:
        if image.ndim == 3:
            n_channels = 1
            _intensity_channels = {f'{prefix}_C1': image} if _measure_intensity else {}
        elif image.ndim == 4:
            n_channels = image.shape[1]
            # Skip the np.asarray(image[:, ch]) materialization when intensity
            # is disabled — at T_LARGE multi-channel this would be the BLOCKER #4
            # site ([_intensity_channels = {... np.asarray(image[:, ch]) ...}])
            # exactly. measure_intensity=False side-steps it entirely.
            if _measure_intensity:
                _intensity_channels = {f'{prefix}_C{ch+1}': np.asarray(image[:, ch, :, :])
                                       for ch in range(n_channels)}
            else:
                _intensity_channels = {}
        else:
            raise ValueError(f"Unexpected zarr image.ndim={image.ndim}")
    else:
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=1)
        n_channels = image.shape[1]
        _intensity_channels = ({f'{prefix}_C{ch+1}': image[:, ch, :, :]
                                 for ch in range(n_channels)}
                                if _measure_intensity else {})

    # Compute bboxes once and reuse across all per_spine_regionprops calls
    # below — recomputing per call would re-scan the labels volume each time.
    # Under Phase H the labels are zarr; this avoids 4 redundant decompressions.
    # Reuse safety: zarr-route branches do NOT mutate labels. The non-zarr
    # `prefix=='head'` retry path (further down) does call
    # filter_invalid_objects, which mutates labels — that branch invalidates
    # _shared_bboxes locally so a subsequent zarr soma-distance consumer
    # auto-recomputes. Skip the precompute entirely when no zarr branch will
    # fire (numpy-only path uses measure.regionprops_table directly).
    _zarr_path_active = (hasattr(labels, 'chunks')
                         or _image_is_zarr
                         or hasattr(dendrite_distance, 'chunks')
                         or (soma_distance is not None and hasattr(soma_distance, 'chunks')))
    if _zarr_path_active:
        if hasattr(labels, 'chunks'):
            _shared_bboxes = chunked.find_objects_streaming(labels, logger=logger)
        elif isinstance(labels, np.ndarray) and labels.max() > 0:
            _shared_bboxes = ndimage.find_objects(labels, int(labels.max()))
        else:
            _shared_bboxes = None
    else:
        _shared_bboxes = None

    if not _measure_intensity:
        # Skip per-channel intensity entirely — produce a label-only main_table
        # so downstream joins still work. Saves N_channels per_spine_regionprops
        # calls (3 prefixes × N_channels per image; ~minutes per call at T7+).
        if hasattr(labels, 'chunks'):
            _label_bboxes_for_skip = (_shared_bboxes
                                       if _shared_bboxes is not None
                                       else chunked.find_objects_streaming(labels, logger=logger))
            main_table = pd.DataFrame({'label': sorted(_label_bboxes_for_skip.keys())})
        else:
            _max_lbl = int(labels.max()) if labels.size else 0
            main_table = pd.DataFrame({'label': list(range(1, _max_lbl + 1))})
        if logger:
            logger.info(f"     Channel intensity skipped (settings.measure_intensity=False) for prefix='{prefix}'")
    elif hasattr(labels, 'chunks') or _image_is_zarr:
        # Zarr-safe: single slab-coalesced pass over all channels
        intensity_dict = _intensity_channels
        main_table = chunked.per_spine_regionprops(
            labels, intensity_dict,
            ['label', 'mean_intensity', 'max_intensity'],
            spine_bboxes=_shared_bboxes, logger=logger)
        # Rename columns: per_spine_regionprops outputs {src_name}_{prop}
        rename_map = {}
        for ch in range(n_channels):
            src = f'{prefix}_C{ch+1}'
            rename_map[f'{src}_mean_intensity'] = f'{prefix}_C{ch+1}_mean_int'
            rename_map[f'{src}_max_intensity'] = f'{prefix}_C{ch+1}_max_int'
        main_table.rename(columns=rename_map, inplace=True)
    else:
        main_table = pd.DataFrame(
            measure.regionprops_table(
                labels,
                intensity_image=image[:, 0, :, :],
                properties=['label', 'mean_intensity', 'max_intensity'],
            )
        )
        main_table.rename(columns={'mean_intensity': f'{prefix}_C1_mean_int',
                                    'max_intensity': f'{prefix}_C1_max_int'}, inplace=True)
        for ch in range(1, n_channels):
            table = pd.DataFrame(
                measure.regionprops_table(
                    labels,
                    intensity_image=image[:, ch, :, :],
                    properties=['label', 'mean_intensity', 'max_intensity'],
                )
            )
            table.rename(columns={'mean_intensity': f'{prefix}_C{ch+1}_mean_int',
                                   'max_intensity': f'{prefix}_C{ch+1}_max_int'}, inplace=True)
            main_table = main_table.join(table[[f'{prefix}_C{ch+1}_mean_int',
                                                 f'{prefix}_C{ch+1}_max_int']])


    if prefix == 'head':
        _is_zarr_dd = hasattr(dendrite_distance, 'chunks')
        if _is_zarr_dd:
            # Zarr-safe path: per-spine bbox reads for intensity + morphology
            _head_props = ['label', 'min_intensity', 'max_intensity']
            if settings.use_vox_measurements:
                _head_props += ['area_bbox', 'extent', 'solidity', 'area_convex',
                                'axis_major_length', 'axis_minor_length', 'feret_diameter_max']
            distance_table = chunked.per_spine_regionprops(
                labels, {'dd': dendrite_distance}, _head_props,
                spine_bboxes=_shared_bboxes, logger=logger)
            distance_table.rename(columns={'dd_min_intensity': 'min_intensity',
                                            'dd_max_intensity': 'max_intensity'}, inplace=True)
        elif settings.use_vox_measurements == False:
            distance_table = pd.DataFrame(
                measure.regionprops_table(
                    labels,
                    intensity_image=dendrite_distance,
                    properties=['label', 'min_intensity', 'max_intensity'
                               ],  # area is volume for 3D images
                )
            )
        else:
            try:

                distance_table = pd.DataFrame(
                    measure.regionprops_table(
                        labels,
                        intensity_image=dendrite_distance,
                        properties=['label', 'min_intensity', 'max_intensity',
                                    'area_bbox', 'extent', 'solidity', 'area_convex',
                                    'axis_major_length', 'axis_minor_length', 'feret_diameter_max'
                                    ],  # area is volume for 3D images
                    )
                )
            except ValueError as e:
                logger.info(f"                Convex hull calculation failed: {str(e)}. Filtering problematic spines and retrying.\n    "
                            f"                Typically occurs when spines are detected at the edge of the image volume and can be addressed by padding \n                or adding a few slices above and below.")
                try:
                    labels, num_valid = filter_invalid_objects(labels, min_volume=settings.neuron_spine_size[0], min_dims=3)
                    # labels was just mutated; invalidate _shared_bboxes so
                    # any downstream zarr-route consumer (e.g. the soma-distance
                    # call below when soma_distance is zarr) auto-recomputes
                    # from current labels rather than reusing stale bboxes.
                    _shared_bboxes = None
                    distance_table = pd.DataFrame(
                        measure.regionprops_table(
                            labels,
                            intensity_image=dendrite_distance,
                            properties=['label', 'min_intensity', 'max_intensity',
                                        'area_bbox', 'extent', 'solidity', 'area_convex',
                                        'axis_major_length', 'axis_minor_length', 'feret_diameter_max'
                                        ],  # area is volume for 3D images
                        )
                    )
                except ValueError as e2:
                    logger.info(f"                Convex hull calculation failed again: {str(e2)}. Using fallback properties.")
                    distance_table = pd.DataFrame(
                        measure.regionprops_table(
                            labels,
                            intensity_image=dendrite_distance,
                            properties=['label', 'min_intensity','max_intensity']))

                    # Add placeholder columns for missing properties
                    distance_table['area_bbox'] = np.nan
                    distance_table['extent'] = np.nan
                    distance_table['solidity'] = np.nan
                    distance_table['area_convex'] = np.nan
                    distance_table['axis_major_length'] = np.nan
                    distance_table['axis_minor_length'] = np.nan
                    distance_table['feret_diameter_max'] = np.nan

        # rename distance column
        distance_table.rename(columns={'min_intensity': 'head_euclidean_dist_to_dend'}, inplace=True)
        distance_table.rename(columns={'max_intensity': 'spine_length_euclidean'}, inplace=True)

        if settings.use_vox_measurements == True:
            distance_table.rename(columns={'area_bbox': 'head_bbox_vox'}, inplace=True)
            distance_table.rename(columns={'extent': 'head_extent_vox'}, inplace=True)
            distance_table.rename(columns={'solidity': 'head_solidity_vox'}, inplace=True)
            distance_table.rename(columns={'area_convex': 'head_vol_convex_vox'}, inplace=True)
            distance_table.rename(columns={'axis_major_length': 'head_major_length_vox'}, inplace=True)
            distance_table.rename(columns={'axis_minor_length': 'head_minor_length_vox'}, inplace=True)

            distance_col = distance_table[
                               "head_bbox_vox"] * settings.input_resXY * settings.input_resXY * settings.input_resZ
            main_table = main_table.join(distance_col)

            distance_col = distance_table["head_extent_vox"]
            main_table = main_table.join(distance_col)

            distance_col = distance_table["head_solidity_vox"]
            main_table = main_table.join(distance_col)

            distance_col = distance_table[
                               "head_vol_convex_vox"] * settings.input_resXY * settings.input_resXY * settings.input_resZ
            main_table = main_table.join(distance_col)

            distance_col = distance_table["head_major_length_vox"] * settings.input_resXY
            main_table = main_table.join(distance_col)

            distance_col = distance_table["head_minor_length_vox"] * settings.input_resXY
            main_table = main_table.join(distance_col)

        #distance_col = distance_table["spine_length_dm"] * settings.input_resXY
        #main_table = main_table.join(distance_col)
        #main_table = tables.move_column(main_table, 'spine_length_dm', 2)

        distance_col = distance_table["head_euclidean_dist_to_dend"]* settings.input_resXY
        main_table = main_table.join(distance_col)
        main_table = tables.move_column(main_table, 'head_euclidean_dist_to_dend', 2)
        distance_col = distance_table["spine_length_euclidean"]  * settings.input_resXY
        main_table = main_table.join(distance_col)
        main_table = tables.move_column(main_table, 'spine_length_euclidean', 3)



        # Use precomputed flag to avoid np.max() on potentially zarr-backed arrays.
        # Fallback uses chunked.safe_max so it remains zarr-safe if a future caller
        # omits soma_present (avoids a silent full-volume materialization).
        _soma_present = soma_present if soma_present is not None else bool(chunked.safe_max(soma_distance) > 0)
        if _soma_present:
            # measure distance to soma
            logger.info("      Measuring distances to soma...")
            if hasattr(soma_distance, 'chunks'):
                # Zarr-safe path: per-spine bbox reads
                soma_distance_table = chunked.per_spine_regionprops(
                    labels, {'sd': soma_distance},
                    ['label', 'min_intensity'],
                    spine_bboxes=_shared_bboxes, logger=logger)
                soma_distance_table.rename(columns={'sd_min_intensity': 'euclidean_dist_to_soma'}, inplace=True)
            else:
                soma_distance_table = pd.DataFrame(
                    measure.regionprops_table(
                        labels,
                        intensity_image=soma_distance,
                        properties=['label', 'min_intensity', 'max_intensity'],
                    )
                )
                soma_distance_table.rename(columns={'min_intensity': 'euclidean_dist_to_soma'}, inplace=True)
            distance_col = soma_distance_table["euclidean_dist_to_soma"] * settings.input_resXY
            main_table = main_table.join(distance_col)
        else:
            main_table['euclidean_dist_to_soma'] = pd.NA


        main_table = tables.move_column(main_table, 'euclidean_dist_to_soma', 3)

    if prefix == 'spine' and settings.use_vox_measurements == True:
        _is_zarr_dd = hasattr(dendrite_distance, 'chunks')
        if _is_zarr_dd:
            # Zarr-safe path: morphology only (no intensity stats needed for spine prefix)
            _spine_morph_props = ['label', 'area_bbox', 'extent', 'solidity', 'area_convex',
                                  'axis_major_length', 'axis_minor_length', 'feret_diameter_max']
            distance_table = chunked.per_spine_regionprops(
                labels, {}, _spine_morph_props,
                spine_bboxes=_shared_bboxes, logger=logger)
        else:
            try:
                distance_table = pd.DataFrame(
                    measure.regionprops_table(
                        labels,
                        intensity_image=dendrite_distance,
                        properties=['label',
                                    'area_bbox', 'extent', 'solidity', 'area_convex',
                                    'axis_major_length', 'axis_minor_length', 'feret_diameter_max'
                                    ],
                    )
                )
            except ValueError as e:
                logger.info(f"                Convex hull calculation failed for some spines: {str(e)} Filtering problematic spines and retrying.")
                try:
                    labels, num_valid = filter_invalid_objects(labels, min_volume=settings.neuron_spine_size[0], min_dims=3)
                    distance_table = pd.DataFrame(
                        measure.regionprops_table(
                            labels,
                            intensity_image=dendrite_distance,
                            properties=['label',
                                        'area_bbox', 'extent', 'solidity', 'area_convex',
                                        'axis_major_length', 'axis_minor_length', 'feret_diameter_max'
                                        ],
                        )
                    )
                except ValueError as e2:

                    # Fall back to properties that don't require convex hull
                    logger.info(f"                Convex hull calculation failed for some spines: {str(e2)} Using fallback properties.")
                    distance_table = pd.DataFrame()

                    # Add placeholder columns for missing properties
                    distance_table['area_bbox'] = np.nan
                    distance_table['extent'] = np.nan
                    distance_table['solidity'] = np.nan
                    distance_table['area_convex'] = np.nan
                    distance_table['axis_major_length'] = np.nan
                    distance_table['axis_minor_length'] = np.nan
                    distance_table['feret_diameter_max'] = np.nan

        # rename distance column
        distance_table.rename(columns={'area_bbox': 'spine_bbox_vox'}, inplace=True)
        distance_table.rename(columns={'extent': 'spine_extent_vox'}, inplace=True)
        distance_table.rename(columns={'solidity': 'spine_solidity_vox'}, inplace=True)
        #distance_table.rename(columns={'area_convex': 'spine_vol_convex'}, inplace=True)
        distance_table.rename(columns={'axis_major_length': 'spine_major_length_vox'}, inplace=True)
        distance_table.rename(columns={'axis_minor_length': 'spine_minor_length_vox'}, inplace=True)

        distance_col = distance_table["spine_bbox_vox"] * settings.input_resXY * settings.input_resXY * settings.input_resZ
        main_table = main_table.join(distance_col)

        distance_col = distance_table["spine_extent_vox"] * settings.input_resXY
        main_table = main_table.join(distance_col)

        distance_col = distance_table["spine_solidity_vox"] * settings.input_resXY
        main_table = main_table.join(distance_col)

       # distance_col = distance_table["spine_vol_convex"] * settings.input_resXY * settings.input_resXY * settings.input_resZ
       # main_table = main_table.join(distance_col)

        distance_col = distance_table["spine_major_length_vox"] * settings.input_resXY
        main_table = main_table.join(distance_col)

        #distance_col = distance_table["spine_minor_length"] * settings.input_resXY
        #main_table = main_table.join(distance_col)


    if settings.Track != True:
        # update label numbers based on offset
        main_table['label'] += max_label
        # Zarr-safe slab-iterated version of `labels[labels > 0] += max_label`.
        # See spine_measurements max_label guard for rationale.
        if max_label:
            _ly, _lx = int(labels.shape[1]), int(labels.shape[2])
            _dtbytes = np.dtype(labels.dtype).itemsize
            _slab_z = max(1, min(64, int(2 * 1024**3 / max(_ly * _lx * _dtbytes, 1))))
            for _z in range(0, labels.shape[0], _slab_z):
                _ze = min(_z + _slab_z, labels.shape[0])
                _slab = np.asarray(labels[_z:_ze])
                _slab[_slab > 0] += max_label
                labels[_z:_ze] = _slab
                del _slab
        labels = create_filtered_labels_image(labels, main_table, logger)

    #else:

        # Clean up label image to remove objects from image.
        #ids_to_keep = set(filtered_table['label'])  # Extract IDs to keep from your filtered DataFrame
        # Create a mask
        #mask_to_keep = np.isin(labels, list(ids_to_keep))
        # Apply the mask: set pixels not in `ids_to_keep` to 0
        #labels = np.where(mask_to_keep, labels, 0)

    # update to included dendrite_id
    #filtered_table.insert(4, 'dendrite_id', dendrite)


    # logger.info(f"  filtered table before image filter = {len(filtered_table)}. ")
    # logger.info(f"  image labels before filter = {np.max(labels)}.")
    # integrated_density

    #main_table = main_table.drop([f'{prefix}_vol_vox'], axis=1)
    # Drop unwanted columns
    # filtered_table = filtered_table.drop(['spine_vol','spine_length', 'dist_to_dendrite', 'dist_to_soma'], axis=1)
    #logger.info(
    #    f"     After filtering {len(filtered_table)} spines were analyzed from a total of {len(main_table)} putative spines")

    return main_table, labels


def filter_invalid_objects(labels, min_volume=10, min_dims=2):
    """
    Enhanced filtering of objects that would cause problems with convex hull or Feret diameter.

    Parameters:
    -----------
    labels : ndarray
        Label image containing the segmented objects
    min_volume : int
        Minimum volume (in voxels) for an object to be considered valid
    min_dims : int
        Minimum number of dimensions that should have > 1 pixel extent

    Returns:
    --------
    valid_labels : ndarray
        Label image with only valid objects
    num_valid : int
        Number of valid objects after filtering
    """
    # Create a copy of labels
    valid_labels = np.zeros_like(labels)

    # Get all region properties at once (much faster than one by one)
    props = measure.regionprops(labels)

    for prop in props:
        # Skip small objects
        if prop.area < min_volume:
            continue

        # Check dimensionality via bounding box
        bbox = prop.bbox
        if len(bbox) == 6:  # 3D case
            dims = np.array([(bbox[i + 3] - bbox[i]) for i in range(3)])

            # Calculate dimension ratios to detect flat objects
            sorted_dims = np.sort(dims)
            smallest_to_largest_ratio = sorted_dims[0] / sorted_dims[2] if sorted_dims[2] > 0 else 0

            # Skip objects that are too flat (causing QHull errors)
            # The threshold can be adjusted based on your specific data
            if smallest_to_largest_ratio < 0.4:
                continue

            # Check for coplanarity by analyzing coordinates
            coords = prop.coords
            if coords.shape[0] > 0:
                # Calculate principal components to detect if points lie in a plane
                centered_coords = coords - np.mean(coords, axis=0)
                # Use SVD to check for planarity - if smallest singular value is very small, object is planar
                _, s, _ = np.linalg.svd(centered_coords, full_matrices=False)
                if len(s) >= 3 and s[0] > 1e-10 and s[2] / s[0] < 0.05:  # Ratio of smallest to largest singular value
                    continue
        else:  # 2D case
            dims = np.array([(bbox[i + 2] - bbox[i]) for i in range(2)])

        # Only keep objects with sufficient dimensionality
        if np.sum(dims > 1) >= min_dims:
            valid_labels[labels == prop.label] = prop.label

        # Get the number of valid objects
    num_valid = len(np.unique(valid_labels)) - (1 if 0 in valid_labels else 0)

    return valid_labels, num_valid


##############################################################################
# Helper Functions
##############################################################################


def check_image_shape(image,logger):
    if len(image.shape) > 3:
        #Multichannel input format variability
        # Enable/modify if issues with different datasets to ensure consistency
        smallest_axis = np.argmin(image.shape)
        if smallest_axis != 1:
            # Move the smallest axis to position 2
            image = np.moveaxis(image, smallest_axis, 1)
            logger.info(f"   Channels moved ZCYX - raw data now has shape {image.shape}") #ImageJ supports TZCYX order
    else:
        image = np.expand_dims(image, axis=1)

    return image


def contrast_stretch(image, pmin=2, pmax=98):
    p2, p98 = np.percentile(image, (pmin, pmax))
    return exposure.rescale_intensity(image, in_range=(p2, p98))


##############################################################################
# Table Functions
##############################################################################



##############################################################################
# Sub-Functions
##############################################################################


@jit(nopython=True, cache=True)
def _greedy_path_numba(augmented_map, target_volume,
                       start_z, start_y, start_x, max_iterations):
    """Numba-compiled greedy path search — replaces Python loop for ~3-5x speedup.

    At each step, evaluates all 26 neighbors and splits them into two classes:
      - terminal candidates: neighbors inside `target_volume` (dendrite)
      - step candidates:     neighbors outside `target_volume`, not yet visited

    If any terminal candidate exists, the path takes the cheapest one as its
    final voxel and stops (target reached). Otherwise it steps to the cheapest
    non-target neighbor and continues. This preserves the no-tunneling
    invariant (target_volume is never entered mid-path) while eliminating the
    1-voxel gap that resulted from stopping on the dendrite's outer shell.
    """
    sz, sy, sx = augmented_map.shape
    visited = np.zeros((sz, sy, sx), dtype=np.bool_)

    path_z = np.empty(max_iterations + 1, dtype=np.int32)
    path_y = np.empty(max_iterations + 1, dtype=np.int32)
    path_x = np.empty(max_iterations + 1, dtype=np.int32)
    path_len = 0

    z, y, x = start_z, start_y, start_x

    # Defensive guard: if the start voxel is already inside target_volume
    # (e.g., the object's dilated shell overlaps the dendrite), treat as
    # immediate success — no neighbor search needed, and avoid marking a
    # target voxel as visited.
    reached_target = False
    if target_volume[z, y, x]:
        path_z[0] = z; path_y[0] = y; path_x[0] = x
        path_len = 1
        return (path_z[:path_len], path_y[:path_len], path_x[:path_len], True)

    path_z[0] = z; path_y[0] = y; path_x[0] = x
    path_len = 1
    visited[z, y, x] = True
    for _ in range(max_iterations):
        best_step_cost = np.inf
        best_step_nz, best_step_ny, best_step_nx = z, y, x
        best_term_cost = np.inf
        best_term_nz, best_term_ny, best_term_nx = z, y, x
        found_step = False
        found_term = False

        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz = z + dz
                    ny = y + dy
                    nx = x + dx
                    if not (0 <= nz < sz and 0 <= ny < sy and 0 <= nx < sx):
                        continue
                    if target_volume[nz, ny, nx]:
                        # Terminal candidate — can land here as the final step.
                        # No visited check needed (target voxels are never stepped into).
                        cost = augmented_map[nz, ny, nx]
                        if cost < best_term_cost:
                            best_term_cost = cost
                            best_term_nz = nz
                            best_term_ny = ny
                            best_term_nx = nx
                            found_term = True
                    else:
                        if visited[nz, ny, nx]:
                            continue
                        cost = augmented_map[nz, ny, nx]
                        if cost < best_step_cost:
                            best_step_cost = cost
                            best_step_nz = nz
                            best_step_ny = ny
                            best_step_nx = nx
                            found_step = True

        # If a target voxel is adjacent, land on it (terminal step).
        if found_term:
            path_z[path_len] = best_term_nz
            path_y[path_len] = best_term_ny
            path_x[path_len] = best_term_nx
            path_len += 1
            reached_target = True
            break

        if not found_step:
            break  # No valid unvisited neighbor found — dead end.

        # Z-preference: only if Z-only move is strictly better (same invariant
        # as before: prefer pure-Z moves for smoother paths in anisotropic data).
        # Z-only moves are only considered as non-terminal step candidates here.
        if best_step_nz != z:
            dz_sign = 1 if best_step_nz > z else -1
            z_only_nz = z + dz_sign
            if (0 <= z_only_nz < sz and
                    not target_volume[z_only_nz, y, x] and
                    not visited[z_only_nz, y, x]):
                z_cost = augmented_map[z_only_nz, y, x]
                if z_cost < best_step_cost:
                    best_step_nz = z_only_nz
                    best_step_ny = y
                    best_step_nx = x

        z, y, x = best_step_nz, best_step_ny, best_step_nx
        path_z[path_len] = z
        path_y[path_len] = y
        path_x[path_len] = x
        path_len += 1
        visited[z, y, x] = True

    return path_z[:path_len], path_y[:path_len], path_x[:path_len], reached_target


def pathfinding_v3b(object_subvolume_gpu, target_subvolume_gpu, intensity_image_gpu, logger,
                    spacing=None, obstacle_subvolume_gpu=None):
    """
    Pathfinding that combines distance map and intensity image to prioritize brighter voxels,
    with optimized start and end point selection.

    Parameters
    ----------
    spacing : tuple of float, optional
        Voxel spacing (Z, Y, X) for anisotropy-aware EDT. If None, uses isotropic.
    obstacle_subvolume_gpu : cupy.ndarray, optional
        Bool/uint8 mask of voxels the path may NOT enter. Typical use:
        OTHER spine heads when computing this spine's neck path. Without this,
        the cost grid (which favours bright voxels) routes paths through
        neighbouring spine heads — biologically wrong and visible at MIP
        scale as necks skirting/clipping adjacent heads. Enforced AFTER
        Gaussian smoothing of the cost grid so the smoothing kernel does not
        smear inf-values into legitimate corridors.
    """
    # Early validation: check for start/end candidates BEFORE expensive GPU ops.
    # Transfer small intermediate results to CPU for candidate checking.
    object_volume = object_subvolume_gpu.get()
    target_volume = target_subvolume_gpu.get()

    dilated_object = ndimage.binary_dilation(object_volume, iterations=1)
    start_candidates = np.argwhere(dilated_object & ~object_volume)
    if len(start_candidates) == 0:
        print("        No valid start candidates found in pathfinding_v3b")
        return None, np.zeros_like(object_volume, dtype=np.float32)

    # Target must exist somewhere in the subvolume. The pathfinder now lands
    # directly on target_volume voxels (no shell offset), so any target voxel
    # is a valid landing — unlike the previous `modified_target = shell` check,
    # which additionally required a non-target neighbor to exist around it.
    if not target_volume.any():
        print("        No target voxels found in pathfinding_v3b")
        return None, np.zeros_like(object_volume, dtype=np.float32)

    # Candidates exist — proceed with expensive GPU computation
    # Compute the distance map from the target subvolume.
    # Use physical spacing for anisotropy-aware distance (prevents Z-bias).
    edt_kwargs = {'sampling': spacing} if spacing is not None else {}
    distance_map_gpu = cp_ndimage.distance_transform_edt(1 - target_subvolume_gpu, **edt_kwargs)

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
        logger.info("        Intensity image is all zeros, skipping normalization")

    # Apply gamma correction to increase the influence of brighter voxels
    gamma = 0.4  # Adjust this value to fine-tune the brightness influence
    gamma_corrected_intensity = cp.power(normalized_intensity_gpu, gamma)

    # Create blurred intensity image — moderate smoothing to bridge gaps in signal
    # while preserving fine neck structure (sigma=[0.5, 3, 3] vs old [0.25, 10, 10])
    blurred_intensity_gpu = cp_ndimage.gaussian_filter(gamma_corrected_intensity, sigma=[0.5, 3, 3])

    # Normalize the distance map to [0, 1]
    max_distance = cp.max(distance_map_gpu)
    if max_distance > 0:
        normalized_distance_map_gpu = distance_map_gpu / (max_distance + 1e-6)
    else:
        normalized_distance_map_gpu = distance_map_gpu
        logger.info("        Distance map is all zeros, skipping normalization")

    # Invert the intensity so that brighter voxels have lower values (to minimize).
    # Higher intensity_weight makes paths follow bright signal more closely.
    intensity_weight = 0.4
    blurred_intensity_weight = 0.2
    epsilon = 1e-6
    augmented_map_gpu = normalized_distance_map_gpu / (
            intensity_weight * gamma_corrected_intensity +
            blurred_intensity_weight * blurred_intensity_gpu +
            epsilon
    )

    # Apply Gaussian smoothing to reduce noise and create a smoother path.
    # Use anisotropic sigma to account for different Z vs XY resolution —
    # less smoothing in Z (coarser) to preserve Z structure.
    sigma = [0.3, 0.5, 0.5]
    augmented_map_gpu = cp_ndimage.gaussian_filter(augmented_map_gpu, sigma)

    # Obstacle enforcement (applied post-smoothing) — voxels in
    # `obstacle_subvolume_gpu` get +inf cost so the greedy walk never enters
    # them. Applied AFTER gaussian_filter so the smoothing kernel does not
    # smear large values into legitimate neck corridors.
    if obstacle_subvolume_gpu is not None:
        augmented_map_gpu = cp.where(
            obstacle_subvolume_gpu.astype(bool),
            cp.float32(cp.inf),
            augmented_map_gpu)

    # Convert augmented map to NumPy for greedy path search.
    # object_volume, target_volume, and start_candidates were already computed
    # in the early validation above.
    augmented_map = augmented_map_gpu.get()

    # Find the best start point
    best_start = min(start_candidates, key=lambda p: augmented_map[tuple(p)])

    # Numba-compiled greedy path search (3-5x faster than Python loop).
    # Path can now land on target_volume voxels directly as its terminal step;
    # non-terminal steps are still excluded from target (no tunneling).
    path_z, path_y, path_x, reached_target = _greedy_path_numba(
        augmented_map, target_volume,
        int(best_start[0]), int(best_start[1]), int(best_start[2]),
        500)

    if reached_target:
        path = list(zip(path_z.tolist(), path_y.tolist(), path_x.tolist()))
        return path, augmented_map
    else:
        print("      Path finding failed: did not reach the target.")
        return None, augmented_map

def unpad_subvolume_gpu(subvolume_gpu, pad_width):
    return subvolume_gpu[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]

def mesh_volume(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    volume = cp.sum(cp.cross(v0, v1) * v2) / 6.0
    return abs(float(volume))


def smooth_skeleton_3d(skeleton_points, num_samples=50):
    """
    Smooth the 3D skeleton using spline interpolation.
    """
    if len(skeleton_points) < 4:
        return cp.asarray(skeleton_points)  # Not enough points for smoothing

    try:
        # Use NumPy for spline fitting as SciPy doesn't support CuPy arrays
        #tck, u = splprep([skeleton_points[:, 0].get(), skeleton_points[:, 1].get(), skeleton_points[:, 2].get()], s=0, k=3)
        tck, u = splprep([skeleton_points[:, i].get() for i in range(3)], s=0, k=3)
        u_fine = np.linspace(0, 1, num_samples)
        smoothed = np.array(splev(u_fine, tck)).T
        return cp.asarray(smoothed)
    except Exception as e:
        print(f"Spline fitting failed: {e}. Using original points.")
        return skeleton_points


def calculate_neck_length_and_width(skeleton_points, verts):
    """
    Calculate neck length and width relative to the skeleton.

    """
    # Smooth the 3D skeleton
    smooth_skeleton_points = smooth_skeleton_3d(skeleton_points)

    # Calculate the total length of the neck
    diff = cp.diff(smooth_skeleton_points, axis=0)
    total_length = float(cp.sum(cp.sqrt(cp.sum(diff ** 2, axis=1))))

    # Calculate tangent vectors along the skeleton
    tangents = cp.diff(smooth_skeleton_points, axis=0)
    tangents = tangents / cp.linalg.norm(tangents, axis=1)[:, None]
    tangents = cp.vstack((tangents, tangents[-1]))  # Add last tangent

    # Function to compute perpendicular distance in XY plane
    @cp.fuse()
    def perpendicular_distance_xy(point_y, point_x, skeleton_y, skeleton_x, tangent_y, tangent_x):
        vec_y = point_y - skeleton_y
        vec_x = point_x - skeleton_x
        proj = vec_y * tangent_y + vec_x * tangent_x
        perp_vec_y = vec_y - proj * tangent_y
        perp_vec_x = vec_x - proj * tangent_x
        return cp.sqrt(perp_vec_y ** 2 + perp_vec_x ** 2)

    # Calculate widths in XY plane

    widths = cp.zeros(len(smooth_skeleton_points))
    for i, (skeleton_point, tangent) in enumerate(zip(smooth_skeleton_points, tangents)):
        distances = perpendicular_distance_xy(
            verts[:, 1], verts[:, 2],
            skeleton_point[1], skeleton_point[2],
            tangent[1], tangent[2]
        )
        widths[i] = cp.max(distances) * 2  # Diameter

    # Compute statistics
    min_width = float(cp.min(widths))
    max_width = float(cp.max(widths))
    mean_width = float(cp.mean(widths))

    return total_length, min_width, max_width, mean_width, smooth_skeleton_points


def mesh_neck_width_and_length(volume, verts, logger, scaling):
    """
    Calculate the neck length and width using the skeleton and vertices.

    Args:
        volume (ndarray): 3D binary volume representing the object.
        verts (ndarray): Vertices representing the surface or points of the object.
        scaling (tuple): Voxel scaling in (Z, Y, X) order.

    Returns:
        total_length (float): Total length of the neck.
        min_width (float): Minimum width of the neck.
        max_width (float): Maximum width of the neck.
        mean_width (float): Mean width of the neck.
    """
    #print logger and scaling
    #logger.info(f"Calculating neck length and width with scaling: {scaling}")
    skeleton = morphology.skeletonize(cp.asnumpy(volume))
    skeleton_points = cp.asarray(np.argwhere(skeleton)) * cp.asarray(scaling)
    #skeleton_points = np.argwhere(skeleton)

    if len(skeleton_points) == 0:
        # No skeleton points, return 0 for all metrics
        return 0, 0, 0, 0, 0
        # Apply voxel scaling to the vertices
    #skeleton_points_scaled = skeleton_points * cp.asarray(scaling)

    if len(skeleton_points) < 2:

        # Use two skeleton points to form a basic axis for width measurement
        # Treat as a straight line between the two bounding box corners
        bbox_min = np.min(verts, axis=0)
        bbox_max = np.max(verts, axis=0)

        # Skeleton axis: straight line between bounding box corners
        skeleton_axis = bbox_max - bbox_min
        skeleton_axis /= np.linalg.norm(skeleton_axis)  # Normalize axis

        # Project the vertices onto the skeleton axis to measure distances perpendicular to the axis
        diffs = verts - bbox_min  # Differences relative to one endpoint of the axis
        projections = cp.dot(diffs, skeleton_axis)[:, None] * skeleton_axis  # Project onto axis
        perpendicular_diffs = verts - (bbox_min + projections)  # Perpendicular vectors

        # Compute the width in the XY plane (ignore Z, only use Y and X for width)
        perpendicular_diffs_xy = perpendicular_diffs[:, 1:]  # Only Y and X
        dists = cp.linalg.norm(perpendicular_diffs_xy, axis=1)

        # Compute width statistics (min, max, mean) in the XY plane
        min_width = float(cp.min(dists))
        max_width = float(cp.max(dists))
        mean_width = float(cp.mean(dists))

        # Compute the total length of the neck as the bounding box diagonal
        total_length = np.linalg.norm(bbox_max - bbox_min)
        #total_length *= np.linalg.norm(scaling)  # Adjust for voxel scaling
        #create a line based on bbox min and max that can be plotted
        #bbox_min are cp we need np

        skeleton_output = np.array([bbox_min.get(), bbox_max.get()])



    else:
        total_length, min_width, max_width, mean_width, skeleton_output = calculate_neck_length_and_width(
            skeleton_points, verts)

    # Calculate neck length and width using the smooth skeleton
    return total_length, min_width, max_width, mean_width, skeleton_output



def mesh_neck_width_and_length_midline_calculated(midline, verts, logger, scaling):

    if len(midline) == 1:

        #print("Midline is too short to calculate tangents")
        return scaling[1], scaling[1], scaling[1], scaling[1], cp.array(midline)

    scaling = cp.array(scaling)
    # Calculate total length
    total_length = cp.sum(cp.sqrt(cp.sum(cp.diff(midline * scaling, axis=0) ** 2, axis=1)))
    #print(f"Total length: {total_length}")

    # Calculate tangent vectors along the midline
    tangents = cp.diff(midline, axis=0)
    tangent_norms = cp.linalg.norm(tangents, axis=1)
    tangent_norms = cp.where(tangent_norms == 0, 1e-8, tangent_norms)
    tangents = tangents / tangent_norms[:, None]
    tangents = cp.vstack((tangents, tangents[-1]))
    #print(f"Tangents shape: {tangents.shape}")

    # Function to compute perpendicular distance in XY plane
    @cp.fuse()
    def perpendicular_distance_xy(point_y, point_x, skeleton_y, skeleton_x, tangent_y, tangent_x):
        vec_y = point_y - skeleton_y
        vec_x = point_x - skeleton_x
        proj = vec_y * tangent_y + vec_x * tangent_x
        perp_vec_y = vec_y - proj * tangent_y
        perp_vec_x = vec_x - proj * tangent_x
        return cp.sqrt(perp_vec_y ** 2 + perp_vec_x ** 2)

    # Calculate widths in XY plane
    widths = cp.zeros(len(midline))
    for i, (skeleton_point, tangent) in enumerate(zip(midline, tangents)):
        distances = perpendicular_distance_xy(
            verts[:, 1], verts[:, 2],
            skeleton_point[1], skeleton_point[2],
            tangent[1], tangent[2]
        )
        widths[i] = cp.max(distances) * 2  # Diameter

    #print(f"Widths shape: {widths.shape}")
    if len(widths) == 0:
       #print("No widths calculated")
        return total_length, 0, 0, 0, None

    # Compute statistics
    min_width = float(cp.min(widths)) * scaling[1]
    max_width = float(cp.max(widths)) * scaling[1]
    mean_width = float(cp.mean(widths)) * scaling[1]


    # Calculate neck length and width using the smooth skeleton
    #print(midline)
    return total_length, min_width, max_width, mean_width, midline

def mesh_neck_width_and_length_closest(volume, verts, closest_point, logger, scaling):
    """
    Calculate the neck length and width using the skeleton and vertices.

    """
    #print(f"Input volume shape: {volume.shape}")
    #print(f"Number of vertices: {len(verts)}")
    #print(f"Closest point: {closest_point}")
    #print(f"Scaling: {scaling}")

    if cp.sum(volume) == 0:
        return 0, 0, 0, 0, None
    elif cp.sum(volume) == 1:
        #print("Volume contains only one voxel")
        return 1 * scaling[1], 1 * scaling[1], 1 * scaling[1], 1 * scaling[1], cp.argwhere(volume)

    if cp.isnan(volume).any() or cp.isinf(volume).any():
        logger.error("Volume contains NaN or Inf values")
        return 0, 0, 0, 0, None

    # Create distance map from the object surface
    mask = cp.asnumpy(volume>0)
    dist_map = distance_transform_edt(mask, sampling=scaling)
    dist_map = cp.asarray(dist_map)
    #print(f"Distance map shape: {dist_map.shape}")
    #print max value in dist_map
    #print(f"Max value in dist_map: {cp.max(dist_map)}")
    if cp.isnan(dist_map).any() or cp.isinf(dist_map).any():
        logger.error("Distance map contains NaN or Inf values")
        return 0, 0, 0, 0, None

    # Find the furthest point from the closest_point
    closest_point = tuple(map(int, closest_point))
    #print(f"Closest point: {closest_point}")

    temp_map = dist_map.copy()
    temp_map[~mask] = 0
    furthest_point = cp.unravel_index(cp.argmax(temp_map), volume.shape)
    #print(f"Furthest point: {furthest_point}")

    # Create a cost map (inverse of distance map)
    cost_map = cp.max(dist_map) - dist_map
    cost_map[volume == 0] = cp.inf  # Set cost to infinity outside the object

    # More aggressive boosting around the furthest point
    gaussian_boost_radius = 3  # Smaller sigma for sharper boost
    boost_map = cp.zeros_like(cost_map)

    # Set the furthest point in the boost_map as 1 (center of the Gaussian)
    boost_map[furthest_point] = 1.0

    # Apply Gaussian filter to smooth the boost map, creating a sharper boost
    boost_map = cp.asarray(gaussian_filter(cp.asnumpy(boost_map), sigma=gaussian_boost_radius))

    # Now modify the cost map with a stronger Gaussian boost
    cost_map = cost_map * (1.0 - boost_map * 2) + boost_map * 0.1  # More aggressive scaling near furthest point

    # Convert to NumPy for route_through_array
    cost_map_np = cp.asnumpy(cost_map)
    furthest_point_np = tuple(int(i) for i in furthest_point)
    #print(f"Furthest point (NP): {furthest_point_np}")

    # Find the optimal path through the center of the object
    try:
        indices, _ = route_through_array(cost_map_np, closest_point, furthest_point_np, fully_connected=True)
        if not indices:
            raise ValueError("No path found between closest and furthest points")
    except Exception as e:
        #print(f"Error in route_through_array: {e}")
        return 0, 0, 0, 0, None

    midline = cp.array(indices)
    #midline = extend_midline_to_boundary(midline, volume)
    #midline = extend_midline_to_furthest_point(midline, furthest_point, volume, logger)


    # Geodesic distance map from the furthest point
    #binary_volume = volume > 0
    #geodesic_map = distance_transform_edt(cp.asnumpy(binary_volume), sampling=scaling)  # Use NumPy for geodesic map


    # Follow the geodesic path from furthest_point to the midline
    #midline = extend_to_furthest_point_geodesic(midline, geodesic_map, furthest_point)


    if len(midline) < 2:
        #print("Midline is too short to calculate tangents")
        return 1 * scaling[1], 1 * scaling[1], 1 * scaling[1], 1 * scaling[1], cp.argwhere(volume)

    scaling = cp.array(scaling)
    # Calculate total length
   # total_length = cp.sum(cp.sqrt(cp.sum(cp.diff(midline * scaling, axis=0) ** 2, axis=1)))
    # Calculate the differences between consecutive midline points
    diff = cp.diff(midline, axis=0)

    # Apply scaling to each axis independently to account for anisotropy
    scaled_diff = diff * scaling  # Element-wise multiplication of voxel differences by the corresponding scaling factors

    # Calculate the anisotropic length by summing the Euclidean distance with correct scaling for each axis
    distances = cp.sqrt(cp.sum(scaled_diff ** 2, axis=1))

    # Sum the distances to get the total length
    total_length = cp.sum(distances)
    #print(f"Total length: {total_length}")

    # Calculate tangent vectors along the midline
    tangents = cp.diff(midline, axis=0)
    tangent_norms = cp.linalg.norm(tangents, axis=1)
    tangent_norms = cp.where(tangent_norms == 0, 1e-8, tangent_norms)
    tangents = tangents / tangent_norms[:, None]
    tangents = cp.vstack((tangents, tangents[-1]))
    #print(f"Tangents shape: {tangents.shape}")

    # Function to compute perpendicular distance in XY plane
    @cp.fuse()
    def perpendicular_distance_xy(point_y, point_x, skeleton_y, skeleton_x, tangent_y, tangent_x):
        vec_y = point_y - skeleton_y
        vec_x = point_x - skeleton_x
        proj = vec_y * tangent_y + vec_x * tangent_x
        perp_vec_y = vec_y - proj * tangent_y
        perp_vec_x = vec_x - proj * tangent_x
        return cp.sqrt(perp_vec_y ** 2 + perp_vec_x ** 2)

    # Calculate widths in XY plane
    widths = cp.zeros(len(midline))
    for i, (skeleton_point, tangent) in enumerate(zip(midline, tangents)):
        distances = perpendicular_distance_xy(
            verts[:, 1], verts[:, 2],
            skeleton_point[1], skeleton_point[2],
            tangent[1], tangent[2]
        )
        widths[i] = cp.max(distances) * 2  # Diameter

    #print(f"Widths shape: {widths.shape}")
    if len(widths) == 0:
       #print("No widths calculated")
        return total_length, 0, 0, 0, None

    # Compute statistics
    min_width = float(cp.min(widths)) * scaling[1]
    max_width = float(cp.max(widths)) * scaling[1]
    mean_width = float(cp.mean(widths)) * scaling[1]


    # Calculate neck length and width using the smooth skeleton
    #print(midline)
    return total_length, min_width, max_width, mean_width, midline



def extract_subvolumes_mulitchannel_GPU_batch(multi_channel_array, labels, padding=5,
                                               _cached_slices=None):
    """Extract per-label subvolumes using precomputed bounding boxes.

    Parameters
    ----------
    _cached_slices : list, optional
        Pre-computed find_objects result to avoid repeated GPU→CPU transfer
        when called in a batch loop. Caller should compute once and pass in.
    """
    subvolumes = []
    start_coords = []

    label_channel = multi_channel_array[:, :, :, 0]
    vol_shape = multi_channel_array.shape[:3]  # plain tuple, no CuPy array

    # Use cached bounding boxes if provided, otherwise compute
    if _cached_slices is not None:
        slices_lookup = _cached_slices
    else:
        label_np = cp.asnumpy(label_channel).astype(np.int32)
        max_label = int(labels.max()) if len(labels) > 0 else 0
        slices_lookup = ndimage.find_objects(label_np, max_label)

    for label in labels:
        label_idx = int(label) - 1
        if label_idx < 0 or label_idx >= len(slices_lookup) or slices_lookup[label_idx] is None:
            continue

        sl = slices_lookup[label_idx]
        # Use plain ints for slicing — avoids 6 implicit GPU syncs per label
        z0 = max(sl[0].start - padding, 0)
        y0 = max(sl[1].start - padding, 0)
        x0 = max(sl[2].start - padding, 0)
        z1 = min(sl[0].stop + padding, vol_shape[0])
        y1 = min(sl[1].stop + padding, vol_shape[1])
        x1 = min(sl[2].stop + padding, vol_shape[2])

        subvolume = multi_channel_array[z0:z1, y0:y1, x0:x1]
        subvolumes.append(subvolume)
        start_coords.append(cp.array([z0, y0, x0]))

    if len(start_coords) == 0:
        return subvolumes, cp.empty((0, 3), dtype=cp.int64)
    return subvolumes, cp.stack(start_coords)


def extract_subvolumes_mulitchannel_GPU_batch_2ndpass(multi_channel_array, labels, padding=5,
                                                      _cached_slices=None):
    """Extract per-label subvolumes for 2nd pass (same optimization as primary)."""
    subvolumes = []
    start_coords = []

    label_channel = multi_channel_array[:, :, :, 0]
    vol_shape = multi_channel_array.shape[:3]

    if _cached_slices is not None:
        slices_lookup = _cached_slices
    else:
        label_np = cp.asnumpy(label_channel).astype(np.int32)
        max_label = int(labels.max()) if len(labels) > 0 else 0
        slices_lookup = ndimage.find_objects(label_np, max_label)

    for label in labels:
        label_idx = int(label) - 1
        if label_idx < 0 or label_idx >= len(slices_lookup) or slices_lookup[label_idx] is None:
            continue

        sl = slices_lookup[label_idx]
        z0 = max(sl[0].start - padding, 0)
        y0 = max(sl[1].start - padding, 0)
        x0 = max(sl[2].start - padding, 0)
        z1 = min(sl[0].stop + padding, vol_shape[0])
        y1 = min(sl[1].stop + padding, vol_shape[1])
        x1 = min(sl[2].stop + padding, vol_shape[2])

        subvolume = multi_channel_array[z0:z1, y0:y1, x0:x1]
        subvolumes.append(subvolume)
        start_coords.append(cp.array([z0, y0, x0]))

    if len(start_coords) == 0:
        return subvolumes, cp.empty((0, 3), dtype=cp.int64)
    return subvolumes, cp.stack(start_coords)



def import_tiff_files_to_cupy_list(folder_path):
    """
    Import all TIFF files from a folder into a list of CuPy arrays.

    Returns:
    list: A list of CuPy arrays, each representing a TIFF file.
    """
    tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tif') or f.endswith('.tiff')]
    tiff_files.sort()  # Ensure consistent ordering

    imported_list = []

    for filename in tiff_files:
        file_path = os.path.join(folder_path, filename)
        # Read the TIFF file and convert to CuPy array
        image = cp.asarray(imread(file_path))


        imported_list.append(image)

    return imported_list


def log_memory_usage(logger):
    # Get current process
    process = psutil.Process(os.getpid())

    # Log overall memory usage
    mem_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MiB
    logger.info(f"Current memory usage: {mem_usage:.6f} MiB")

    # Log detailed memory usage
    logger.info("Detailed memory usage:")
    mem_usage = memory_profiler.memory_usage()
    logger.info(f"Total memory usage: {mem_usage[0]:.6f} MiB")

def check_gpu_memory_requirements(spine_labels_vol, logger):
    # Calculate memory for one volume
    single_vol_memory = spine_labels_vol.nbytes

    # Calculate memory for all four volumes
    total_memory_required = single_vol_memory * 4

    # Get total GPU memory
    # Free stale GPU memory before checking availability
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    free_mem, total_gpu_memory = cp.cuda.runtime.memGetInfo()

    #logger.info(f"       Memory required for one volume: {single_vol_memory / 1e9:.2f} GB")
    logger.info(f"       Total memory required for full multi-channel array: {total_memory_required / 1e9:.2f} GB")
    pct = (total_memory_required / free_mem * 100) if free_mem > 0 else float('inf')
    logger.info(f"       Total available GPU memory: {free_mem / 1e9:.2f} GB. Percentage of GPU memory required for array: {pct:.2f}%")

    return total_memory_required, free_mem, total_gpu_memory

def calculate_batch_size(mem_required, labels, sub_volume_shape, dtype_size):
    # Get the default memory pool
    memory_pool = cp.get_default_memory_pool()
    # Get total and used bytes from the memory pool
    used_bytes = memory_pool.used_bytes()
    _, total_bytes = cp.cuda.runtime.memGetInfo()

    # Calculate free memory in bytes
    free_mem = total_bytes - used_bytes

    # Calculate memory required for one sub-volume (4 channels)
    sub_volume_memory = np.prod(sub_volume_shape) * dtype_size * 4  # bytes

    # Estimate additional overhead (temporary arrays, etc.)
    # Adjust the overhead factor based on empirical observations
    overhead_factor = 3
    memory_per_item = sub_volume_memory * overhead_factor

    # Calculate batch size, leaving 20% of free memory as buffer
    available_memory = (free_mem - mem_required) * 0.8
    batch_size = int(available_memory / memory_per_item)
    batch_size = max(1, min(batch_size, len(labels)))

    return batch_size

def calculate_batch_size_v0(mem_required, labels, sub_volume_shape, dtype_size):
    free_mem, _ = cp.cuda.runtime.memGetInfo()

    # Calculate memory required for one sub-volume
    sub_volume_memory = np.prod(sub_volume_shape) * dtype_size *4  # 4 channels

    # Calculate memory required for other operations
    other_operations_memory = sub_volume_memory * 2

    # Total memory per item in batch
    memory_per_item = sub_volume_memory + other_operations_memory

    # Calculate batch size, leaving 20% of free memory as buffer
    batch_size = int(((free_mem-mem_required) * 0.8) / memory_per_item)

    return int(max(1, min(batch_size, len(labels)))/2.5)

def calculate_tile_size(shape, settings, logger):
    _, total_mem = cp.cuda.runtime.memGetInfo()
    target_mem = 0.10 * total_mem  # 10% of total GPU memory (conservative for 4x intermediate overhead)
    voxel_size = 4 * 4  # 4 volumes, 4 bytes per voxel (assuming float32)
    total_voxels = target_mem / voxel_size

    # Use actual data aspect ratio instead of fixed ratio
    aspect = shape[1] / max(1, shape[0])  # YX vs Z ratio
    z_size = int((total_voxels / (aspect * aspect)) ** (1/3))
    y_size = x_size = int(z_size * aspect)

    # Adjust sizes to fit within the original shape
    z_size = min(z_size, shape[0])
    y_size = min(y_size, shape[1])
    x_size = min(x_size, shape[2])

    # Ensure sizes are at least 1
    z_size = max(1, z_size)
    y_size = max(1, y_size)
    x_size = max(1, x_size)

    logger.info(f"     Calculated tile size: z={z_size}, y={y_size}, x={x_size}")
    return (z_size, y_size, x_size)


def process_spine_batch_tile(tile, tile_spine, tile_head, tile_dendrite, tile_neuron, locations, settings, logger, scaling):
    # Create multi-channel array for the tile

    mem_required, free_mem, total_mem = check_gpu_memory_requirements(tile_spine, logger)

    cp_multi_channel = cp.stack([cp.asarray(tile_spine),
                                 cp.asarray(tile_head),
                                 cp.asarray(tile_dendrite),
                                 cp.asarray(tile_neuron)], axis=-1)

    # Get unique labels from the tile
    labels = cp.unique(cp_multi_channel[:, :, :, 0])
    labels = labels[labels != 0]

    logger.info(f"       Processing tile {tile} with {len(labels)} spines...")

    # Calculate batch size for this tile
    avg_sub_volume_shape = (6 / scaling[0], 6 / scaling[1], 6 / scaling[2])
    dtype_size = 2  # Assuming float32
    free_mem, _ = cp.cuda.runtime.memGetInfo()
    batch_size = calculate_batch_size(mem_required, tile_spine, avg_sub_volume_shape, dtype_size)
    #logger.info(f"       Using batch size of {batch_size} spines... free_mem: {mem_required} avg_sub_volume_shape, dtype_size {avg_sub_volume_shape} {dtype_size}")

    tile_results = []

    if batch_size == 0:
        return  tile_results

    logger.info(f"       Using batch size of {batch_size} spines...")



    # batch
    i = 0
    while i < len(labels):
      try:
        batch_labels = labels[i:i + batch_size]
        logger.info(
            f"        Processing batch {i // batch_size + 1} of {len(labels) // batch_size + 1}...")

        # Extract subvolumes
        sub_volumes, start_coords = extract_subvolumes_mulitchannel_GPU_batch(cp_multi_channel, batch_labels)

        # logger shape
        if settings.additional_logging == True:
            logger.info(f"Subvolumes shape: {len(sub_volumes)}")
            # log dimensions of subvolumes0
            logger.info(f"Subvolume 0 shape: {sub_volumes[0].shape}")

        # mask subvolumes
        # logger.info(f"      Masking subvolumes...")
        sub_volumes = batch_mask_subvolumes_cp(sub_volumes, batch_labels)

        if settings.additional_logging == True:
            logger.info(f"Subvolume 0 shape after masking: {sub_volumes[0].shape}")

        # Pad subvolumes to allow batching
        # padded_subvolumes, pad_amounts = pad_subvolumes(sub_volumes, logger)

        padded_subvolumes, pad_amounts = pad_and_center_subvolumes(sub_volumes, settings, logger)

        # Save spine arrays — gated to avoid IO errors on network drives
        if settings.save_intermediate_data:
            if settings.additional_logging == True:
                logger.info(f"      Saving spine arrays...")
            # MZYXC to MZCYX for imagej
            export_subvols = cp.transpose(cp.stack(padded_subvolumes, axis=0), (0, 1, 4, 2, 3))
            export_subvols[:, :, :3] = export_subvols[:, :, :3] * 65535 / 2
            export_subvols[:, :, 0] = export_subvols[:, :, 0] - export_subvols[:, :, 1]
            export_subvols = cp.asnumpy(export_subvols)

            imwrite(f'{locations.arrays}/Spine_vols_{settings.filename}_tile{tile}_b{i // batch_size + 1}.tif',
                    export_subvols.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True,
                    photometric='minisblack',
                    metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'TZCYX', 'mode': 'composite'},
                    resolution=(1 / settings.input_resXY, 1 / settings.input_resXY))

            imwrite(f'{locations.arrays}/Spine_MIPs_{settings.filename}_tile{tile}_b{i // batch_size + 1}.tif',
                    np.max(export_subvols, axis=1).astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True,
                    photometric='minisblack',
                    metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX', 'mode': 'composite'},
                    resolution=(1 / settings.input_resXY, 1 / settings.input_resXY))

            del export_subvols

        if settings.additional_logging == True:
            logger.info(f"Subvolume 0 shape after padding: {padded_subvolumes[0].shape}")

        # loop over sub_volumes but also provide a counter
        # if settings.save_intermediate_data == True:
        #    for subvol, start_coord, label in zip(sub_volumes, start_coords, batch_labels):

        # save as tif and resahpe for imageJ
        #        imwrite_filename = os.path.join(locations.Meshes, f"subvol_{label}.tif")
        # imageJ ZCYX - currently ZYXC so fix

        # subvol_out = subvol.get()
        #        imwrite(imwrite_filename, subvol.get().transpose(0, 3, 1, 2).astype(np.uint16), imagej=True,
        #                photometric='minisblack', metadata={'unit': 'um', 'axes': 'ZCYX'})

        # generate data for spine
        batch_data = cp.stack([subvol[:, :, :, 0].astype(cp.float32) for subvol in padded_subvolumes])  # spine volumes
        batch_dendrite = cp.stack([subvol[:, :, :, 2] for subvol in padded_subvolumes])  # dendrite
        # binary of batch_dendrite
        batch_binary = cp.stack([subvol[:, :, :, 2] > 0 for subvol in padded_subvolumes])  # dendrite binary

        # calculate closest points
        closest_points = find_closest_points_batch(batch_data, batch_binary)

        #dendrite_ids = find_closest_dendrite_id(closest_points, batch_dendrite)
        # print(dendrite_ids)
        #if settings.additional_logging == True:
        #    logger.info(f"batch_data dtype: {batch_data.dtype}, shape: {batch_data.shape}")
        #    logger.info(f"batch_dendrite dtype: {batch_dendrite.dtype}, shape: {batch_dendrite.shape}")
        #    logger.info(f"closest_points dtype: {closest_points.dtype}, shape: {closest_points.shape}")
            #logger.info(f"dendrite_ids dtype: {dendrite_ids.dtype}, shape: {dendrite_ids.shape}")

        # Process spines and pass midlines on for neck and head analysis
        t0 = time.time()

        spine_results, midlines = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts,
                                                      batch_labels, scaling, 0, True, None, True, True, "spine",
                                                      locations, settings, logger)
        if settings.additional_logging == True:
            logger.info(f"      Time taken for mesh measurements of spines: {time.time() - t0:.2f} seconds")

        # Subtract heads from spines and measure necks and ensure float for marching_cubes
        batch_data = cp.stack([
            cp.maximum(subvol[:, :, :, 0] - subvol[:, :, :, 1], 0).astype(cp.float32)
            for subvol in padded_subvolumes
        ])

        # for subvol in sub_volumes:
        #    subvol[:, :, :, 0] = cp.maximum(subvol[:, :, :, 0] - subvol[:, :, :, 1], 0)
        t0 = time.time()

        neck_results, _ = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts, batch_labels,
                                              scaling, 0, False, midlines, True, True, "neck", locations, settings,
                                              logger)  # True for min max mean width
        if settings.additional_logging == True:
            logger.info(f"      Time taken for mesh measurements of necks: {time.time() - t0:.2f} seconds")

        # now process heads
        t0 = time.time()
        batch_data = cp.stack([subvol[:, :, :, 1] for subvol in padded_subvolumes])  # spine vols
        # Process just heads - but don't need vol for length just surface area
        head_results, _ = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts, batch_labels,
                                              scaling, 1, False, midlines, True, True, "head", locations, settings,
                                              logger)  # False for width using bounding box
        if settings.additional_logging == True:
            logger.info(f"      Time taken for mesh measurements of spine heads: {time.time() - t0:.2f} seconds")
        # Combine results for this batch

        g = lambda d, k: d.get(k, np.nan)

        for r_spine, r_neck, r_head in zip(spine_results, neck_results, head_results):
            # convex hull ration
            head_conv = g(r_head, 'head_convex_vol')
            head_vol = g(r_head, 'head_volume')
            hull_ratio = (
                np.nan
                if (np.isnan(head_conv) or np.isnan(head_vol) or head_vol == 0)
                else (head_conv - head_vol) / head_vol
            )

            tile_results.append({
                'label': r_spine['ID'],
                # 'dendrite_id': int(dend_id),
                # -------------  spine -----------------
                'start_coords': g(r_spine, 'start_coords'),
                'spine_area': g(r_spine, 'spine_area'),
                'spine_vol': g(r_spine, 'spine_volume'),
                'spine_surf_area': g(r_spine, 'spine_surface_area'),
                'spine_length': g(r_spine, 'spine_length'),
                'spine_bbox_vol': g(r_spine, 'spine_bbox_vol'),
                'spine_extent': g(r_spine, 'spine_extent'),
                'spine_solidity': g(r_spine, 'spine_solidity'),
                'spine_convex_vol': g(r_spine, 'spine_convex_vol'),
                'spine_axis_major': g(r_spine, 'spine_axis_major'),
                'spine_axis_minor': g(r_spine, 'spine_axis_minor'),

                # -------------  head ------------------
                'head_width_mean': g(r_head, 'head_mean_width'),
                'head_area': g(r_head, 'head_area'),
                'head_vol': head_vol,
                'head_surf_area': g(r_head, 'head_surface_area'),
                'head_length': g(r_head, 'head_length'),
                'head_bbox_vol': g(r_head, 'head_bbox_vol'),
                'head_extent': g(r_head, 'head_extent'),
                'head_solidity': g(r_head, 'head_solidity'),
                'head_convex_vol': head_conv,
                'head_axis_major': g(r_head, 'head_axis_major'),
                'head_axis_minor': g(r_head, 'head_axis_minor'),

                'head_convex_hull_ratio': hull_ratio,

                # -------------  neck ------------------
                'neck_area': g(r_neck, 'neck_area'),
                'neck_vol': g(r_neck, 'neck_volume'),
                'neck_surf_area': g(r_neck, 'neck_surface_area'),
                'neck_length': g(r_neck, 'neck_length'),
                'neck_width_min': g(r_neck, 'neck_min_width'),
                'neck_width_max': g(r_neck, 'neck_max_width'),
                'neck_width_mean': g(r_neck, 'neck_mean_width'),
                'neck_bbox_vol': g(r_neck, 'neck_bbox_vol'),
                'neck_extent': g(r_neck, 'neck_extent'),
                'neck_solidity': g(r_neck, 'neck_solidity'),
                'neck_convex_vol': g(r_neck, 'neck_convex_vol'),
                'neck_axis_major': g(r_neck, 'neck_axis_major'),
                'neck_axis_minor': g(r_neck, 'neck_axis_minor'),

                # 'head_length_calc': r_spine['spine_length'] - r_neck['neck_length'],
                # 'head_vol_calc': r_spine['spine_volume'] - r_neck['neck_volume']
            })

        # Clean up batch GPU arrays between iterations
        del sub_volumes, start_coords, padded_subvolumes, pad_amounts
        del batch_data, batch_dendrite, batch_binary, closest_points
        del spine_results, neck_results, head_results, midlines
        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()

        i += batch_size

      except cp.cuda.memory.OutOfMemoryError:
        # OOM recovery: free GPU memory and halve batch size
        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        old_batch = batch_size
        batch_size = max(1, batch_size // 2)
        logger.warning(f"        GPU OOM at batch_size={old_batch}, retrying with batch_size={batch_size}")
        if batch_size == old_batch:
            logger.error(f"        Cannot reduce batch_size below 1, skipping batch at index {i}")
            i += batch_size

    return tile_results

def find_closest_dendrite_id(closest_points, batch_dendrite):
    dendrite_ids = []

    for closest_point, dendrite in zip(closest_points, batch_dendrite):
        # Create a distance transform centered on the closest point
        point_mask = cp.zeros_like(dendrite)
        point_mask[closest_point] = 1
        dist_transform = cp.asarray(distance_transform_edt(cp.asnumpy(point_mask)))

        # Mask the distance transform with non-zero dendrite voxels
        masked_dist = cp.where(dendrite > 0, dist_transform, cp.inf)

        # Find the minimum distance point in the dendrite
        closest_dendrite_point = cp.unravel_index(cp.argmin(masked_dist), dendrite.shape)
        dendrite_id = dendrite[closest_dendrite_point]
        dendrite_ids.append(dendrite_id)

    return cp.array(dendrite_ids)

def batch_mask_subvolumes_cp(sub_volumes_list, batch_labels):
    # Convert batch_labels to a CuPy array if it's not already
    batch_labels = cp.asarray(batch_labels)

    masked_subvolumes = []

    for subvol, label in zip(sub_volumes_list, batch_labels):
        # Ensure subvol is a CuPy array (it should already be, but just in case)
        subvol = cp.asarray(subvol)

        # Reshape label to broadcast correctly
        label = label.reshape(1, 1, 1, 1)

        # Create binary masks for both channels
        mask_channel0 = (subvol[:, :, :, 0] == label)
        mask_channel1 = (subvol[:, :, :, 1] == label)

        # Create a copy of subvol to avoid modifying the original
        masked_subvol = subvol.copy()

        # Apply the masks
        masked_subvol[:, :, :, 0] = mask_channel0.astype(subvol.dtype)
        masked_subvol[:, :, :, 1] = mask_channel1.astype(subvol.dtype)

        masked_subvolumes.append(masked_subvol)

    return masked_subvolumes

def create_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def pad_subvolumes(subvolumes, logger):
    # Find the maximum dimensions
    max_shape = np.max([subvol.shape for subvol in subvolumes], axis=0)
    logger.info(f"       Max spine volume shape: {max_shape}")

    # Pad each subvolume to the maximum dimensions
    padded_subvolumes = []
    pad_amounts = []
    for subvol in subvolumes:
        pad_width = [(0, max_dim - dim) for max_dim, dim in zip(max_shape, subvol.shape)]
        padded_subvol = np.pad(subvol, pad_width, mode='constant', constant_values=0)
        padded_subvolumes.append(padded_subvol)
        pad_amounts.append(pad_width)

    return padded_subvolumes, pad_amounts


def pad_and_center_subvolumes(subvolumes, settings, logger):

    # Find the maximum dimensions
    max_shape = np.max([subvol.shape for subvol in subvolumes], axis=0)
    if settings.additional_logging == True:
        logger.info(f"       Max spine volume shape: {max_shape}")

    # Pad each subvolume to the maximum dimensions
    padded_subvolumes = []
    pad_amounts = []
    for subvol in subvolumes:
        # Calculate padding for each dimension
        pad_width = [(max_dim - dim) // 2 + ((max_dim - dim) % 2 > 0) for max_dim, dim in zip(max_shape, subvol.shape)]

        # Create the full pad_width tuple for np.pad
        full_pad_width = [(pad, max_dim - dim - pad) for pad, max_dim, dim in zip(pad_width, max_shape, subvol.shape)]

        padded_subvol = np.pad(subvol, full_pad_width, mode='constant', constant_values=0)
        padded_subvolumes.append(padded_subvol)
        pad_amounts.append(full_pad_width)

    return padded_subvolumes, pad_amounts

def unpad_result(result, pad_amount):
    # Adjust start_coords based on padding
    result['start_coords'] = [coord - pad for coord, (pad, _) in zip(result['start_coords'], pad_amount)]
    return result


def batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts, object_ids, scaling, channel, calc_midline, midlines,
                        multiple_widths_option, save_meshes, prefix, locations, settings, logger):
    # Pad subvolumes to allow batching
    #subvolumes will be prepadded so and provided as the batch data cp format previously in fucntion
    #padded_subvolumes, pad_amounts = pad_subvolumes(subvolumes)

    #batch_data = cp.stack([subvol[:, :, :, channel] for subvol in padded_subvolumes])
    #batch_binary = cp.stack([subvol[:, :, :, 2] for subvol in padded_subvolumes])
    results = []

    if calc_midline:

        midlines = [None] * len(batch_data)
    else:
        closest_points = [None] * len(batch_data)
    #cp.cuda.Stream.null.synchronize()
    #cp.get_default_memory_pool().free_all_blocks()
    #cp.get_default_pinned_memory_pool().free_all_blocks()

    #cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    #cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)



    if settings.additional_logging == True:
        if cp.issubdtype(batch_data.dtype, cp.floating):
            if cp.isnan(batch_data).any() or cp.isinf(batch_data).any():
                logger.error("batch_data contains NaN or Inf values")
                # Handle the error as appropriate
        #else:
            #logger.info("batch_data is of integer type; skipping NaN and Inf checks")
        #logger.info(f"batch_data dtype: {batch_data.dtype}, shape: {batch_data.shape}")


    # create folder f'{locations.Meshes}/{settings.filename}
    if settings.save_intermediate_data:
        if not os.path.exists(f'{locations.Meshes}/{settings.filename}'):
            os.makedirs(f'{locations.Meshes}/{settings.filename}')
            os.makedirs(f'{locations.Meshes}/{settings.filename}/head')
            os.makedirs(f'{locations.Meshes}/{settings.filename}/spine')
            os.makedirs(f'{locations.Meshes}/{settings.filename}/neck')
    for i, (subvol, start_coord, closest_point, pad_amount, obj_id, midline) in enumerate(
            zip(batch_data, start_coords, closest_points, pad_amounts, object_ids, midlines)):

        #if settings.additional_logging == True:
            #logger.info(f"Processing object ID {obj_id}")
            #logger.info(f"Type of subvol: {type(subvol)}, shape: {subvol.shape}")
            #logger.info(f"Type of start_coord: {type(start_coord)}, value: {start_coord}")
            #logger.info(f"Type of closest_point: {type(closest_point)}, value: {closest_point}")
            #logger.info(f"Type of pad_amount: {type(pad_amount)}, value: {pad_amount}")
            #logger.info(f"Type of midline: {type(midline)}")
            #logger.error(f"subvol contains NaN or Inf values for object ID {obj_id}")
            #logger.info(f"subvol dtype: {subvol.dtype}, shape: {subvol.shape}")

            #mempool = cp.get_default_memory_pool()
            #logger.info(f"GPU memory used: {mempool.used_bytes() / 1024 ** 2:.2f} MB")

        if cp.issubdtype(subvol.dtype, cp.floating):
            if cp.isnan(subvol).any() or cp.isinf(subvol).any():
                logger.error(f"subvol contains NaN or Inf values for object ID {obj_id}")
                # Handle the error as appropriate
        #elif settings.additional_logging == True:
        #    logger.info(f"subvol is of integer type; skipping NaN and Inf checks for object ID {obj_id}")

        # Calculate voxel count

        try:
            voxel_count = int(cp.sum(subvol > 0).get())

        except Exception as e:
            logger.error(f"Error calculating voxel count for {prefix} {i} spine {obj_id}: {str(e)}")
            voxel_count = 0

        #create a MIP of the subvol and then sum the pixels
        mip = cp.max(subvol, axis=0)
        area = cp.sum(mip > 0) * scaling[1] * scaling[2]

        # add in zero values for non existant necks
        length = 0.0
        min_width = 0.0
        max_width = 0.0
        mean_width = 0.0
        #print(area)

        try:
            # Attempt to create mesh
            verts, faces, _, _ = measure.marching_cubes(subvol.get())

            if len(verts) == 0 or len(faces) == 0:
                result = create_empty_result(obj_id, start_coord, closest_point, prefix, multiple_widths_option)
                results.append(result)
                continue

            verts = cp.asarray(verts) * cp.asarray(scaling)
            faces = cp.asarray(faces)

            volume = mesh_volume(verts, faces)
            surface_area = measure_surface_area_gpu(verts, faces)


            bbox_dims = (verts.max(axis=0) - verts.min(axis=0))
            bbox_vol = float(cp.prod(bbox_dims))
            hull_vol = float(ConvexHull(cp.asnumpy(verts)).volume)
            # 2026-09-20: extent and solidity used a voxel-counted numerator
            # (voxel_count x voxel size) against denominators derived from the
            # marching-cubes vertices. The isosurface sits half a voxel inside the
            # voxel block, so the mesh bbox and hull bound a smaller region than the
            # voxel count implies and both ratios could exceed 1 (measured on T2:
            # head solidity to 2.12, neck to 6.00, spine to 1.09). Take the numerator
            # from the same geometry as the denominator.
            # A single voxel has no measurable shape - marching cubes turns it into
            # an octahedron of one sixth of a voxel - so report NaN rather than a
            # ratio that cannot be compared with anything.
            if voxel_count <= 1:
                extent = float('nan')
                solidity = float('nan')
            else:
                extent = volume / bbox_vol if bbox_vol > 0 else float('nan')
                solidity = volume / hull_vol if hull_vol > 0 else float('nan')

            #cov = cp.cov(verts, rowvar=False)
            #eigvals = cp.linalg.eigvalsh(cov)
            #axis_major_len = 2.0 * cp.sqrt(3.0 * eigvals[-1]).item()
            #axis_minor_len = 2.0 * cp.sqrt(3.0 * eigvals[0]).item()

            ## NEEDS ATTENTION
            if multiple_widths_option:
                if voxel_count == 1:
                    length, min_width, max_width, mean_width =  1*scaling[1], 1*scaling[1],1*scaling[1], 1*scaling[1]
                    #create volume by multplying voxel by scaling XYZ dims
                    volume = scaling[0] * scaling[1] * scaling[2]
                if calc_midline and voxel_count > 1:
                    length, min_width, max_width, mean_width, skeleton_result = mesh_neck_width_and_length_closest(
                        subvol, verts, closest_point, logger, scaling)
                    midlines[i] = skeleton_result
                else:
                    if midline is not None:
                        midline_voxels = cp.round(midline).astype(cp.int32)
                        mask = cp.zeros(subvol.shape, dtype=cp.bool_)
                        for voxel in midline_voxels:
                            mask[tuple(voxel)] = True
                        midline_mask = mask * (subvol > 0)
                        midline = cp.argwhere(midline_mask)
                    else:
                        #print(f"Midline is None for {prefix} {i}. Using default values.")
                        length, min_width, max_width, mean_width, skeleton_result = 0.0, 0.0, 0.0, 0.0, None

                    if midline is not None and len(midline) > 0 and voxel_count > 1:
                        try:
                            length, min_width, max_width, mean_width, skeleton_result = mesh_neck_width_and_length_midline_calculated(
                            midline, verts, logger, scaling)
                        except Exception as e:
                            logger.error(f"Unexpected error calculating with predifined midline {prefix} {i} spine {obj_id}: {str(e)}")
                            logger.info(f'volume is {volume} for {prefix} {i}')
                            #logger other values such as length width and skeleton result
                            logger.info(f"Length is {length} for {prefix} {i}")
                            logger.info(f"Min width is {min_width} for {prefix} {i}")
                            logger.info(f"Max width is {max_width} for {prefix} {i}")
                            logger.info(f"Mean width is {mean_width} for {prefix} {i}")
                            logger.info(f"Skeleton result is {skeleton_result} for {prefix} {i}")
                            #length, min_width, max_width, mean_width = 0.0, 0.0, 0.0, 0.0
                            skeleton_result = None
                    else:
                        skeleton_result = midline
                #if prefix == "neck" and i == 0:
                #    print(f"Volume is {volume} for {prefix} {i}")
                result = {
                    'ID': obj_id,
                    'start_coords': start_coord.get().tolist() if closest_point is None else (
                                start_coord + closest_point).get().tolist(),
                    f'{prefix}_area': float(area),
                    f'{prefix}_volume': float(volume),
                    f'{prefix}_surface_area': float(surface_area),
                    f'{prefix}_length': float(length),
                    f'{prefix}_min_width': float(min_width),
                    f'{prefix}_max_width': float(max_width),
                    f'{prefix}_mean_width': float(mean_width),
                    f'{prefix}_bbox_vol': bbox_vol,
                    f'{prefix}_extent': extent,
                    f'{prefix}_solidity': solidity,
                    f'{prefix}_convex_vol': hull_vol,
                    #f'{prefix}_axis_major': axis_major_len,
                    #f'{prefix}_axis_minor': axis_minor_len,
                    #f'{prefix}_feret_max': feret
                }

            else:
                length, width = calculate_simple_length_and_width(verts)
                skeleton_result = None

                result = {
                    'ID': obj_id,
                    'start_coords': start_coord.get().tolist() if closest_point is None else (
                                start_coord + closest_point).get().tolist(),
                    f'{prefix}_area': float(area),
                    f'{prefix}_volume': float(volume),
                    f'{prefix}_surface_area': float(surface_area),
                    f'{prefix}_length': float(length),
                    f'{prefix}_width': float(width),
                    f'{prefix}_bbox_vol': bbox_vol,
                    f'{prefix}_extent': extent,
                    f'{prefix}_solidity': solidity,
                    f'{prefix}_convex_vol': hull_vol,
                   # f'{prefix}_axis_major': axis_major_len,
                   # f'{prefix}_axis_minor': axis_minor_len,
                }

            if settings.save_intermediate_data:
                mesh_filename = f'{locations.Meshes}/{settings.filename}/{prefix}/{obj_id}.obj'
                save_mesh(verts.get(), faces.get(), mesh_filename)

                subvol_data = subvol.get().astype(np.uint16) * 65535

                if skeleton_result is not None:
                    skeleton_channel = np.zeros_like(subvol_data)
                    closet_point_channel = np.zeros_like(subvol_data)
                    skeleton_voxels = np.atleast_2d(skeleton_result.get())

                    for point in skeleton_voxels:
                        point = tuple(point)
                        if np.all(np.array(point) >= 0) and np.all(np.array(point) < np.array(skeleton_channel.shape)):
                            skeleton_channel[point] = 65535

                    if closest_point is not None:
                        closet_point_channel[tuple(cp.asnumpy(closest_point))] = 65535

                    multi_channel_image = np.stack([subvol_data, skeleton_channel, closet_point_channel], axis=1)
                    multi_channel_image = multi_channel_image * 65535

                    imsave_filename = f'{locations.Meshes}/{settings.filename}/{prefix}/{obj_id}.tif'
                    imwrite(imsave_filename, multi_channel_image.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True,
                            metadata={'spacing': scaling[0], 'unit': 'um'})
                else:
                    imsave_filename = f'{locations.Meshes}/{settings.filename}/{prefix}/{obj_id}.tif'
                    imwrite(imsave_filename, subvol_data, compression='zlib', compressionargs={'level': 1}, imagej=True, metadata={'spacing': scaling[0], 'unit': 'um'})

            # Unpad the coordinate
            result = unpad_result(result, pad_amount)
            results.append(result)

        except RuntimeError as e:
            if str(e) == 'No surface found at the given iso value.':
                #logger.info(f"No surface found for {prefix} {i}. Creating empty result.")
                result = create_empty_result(obj_id, start_coord, closest_point, prefix, multiple_widths_option)
                result = unpad_result(result, pad_amount)
                results.append(result)
            else:
                logger.error(f"Error processing {prefix} {i}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error processing {prefix} {i} spine {obj_id}: {str(e)}")

            result = create_empty_result(obj_id, start_coord, closest_point, prefix, multiple_widths_option)
            result = unpad_result(result, pad_amount)
            results.append(result)

    return results, midlines


def create_empty_result(obj_id, start_coord, closest_point, prefix, multiple_widths_option):
    base_result = {
        'ID': obj_id,
        'start_coords': start_coord.get().tolist() if closest_point is None else (
                    start_coord + closest_point).get().tolist(),
        f'{prefix}_volume': 0.0,
        f'{prefix}_surface_area': 0.0,
    }

    if multiple_widths_option:
        base_result.update({
            f'{prefix}_area':0.0,
            f'{prefix}_length': 0.0,
            f'{prefix}_min_width': 0.0,
            f'{prefix}_max_width': 0.0,
            f'{prefix}_mean_width': 0.0
        })
    else:
        base_result.update({
            f'{prefix}_length': 0.0,
            f'{prefix}_width': 0.0
        })

    return base_result

def preprocess_faces_for_viz(faces):
    # Add the count of vertices (3) at the start of each face
    return np.column_stack((np.full(faces.shape[0], 3), faces))


def save_mesh(verts, faces, filename):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)

def calculate_simple_length_and_width(verts):
    # Calculate the overall length (max dimension in any direction)
    bbox_min = cp.min(verts, axis=0)
    bbox_max = cp.max(verts, axis=0)
    length = float(cp.max(bbox_max - bbox_min))

    # Project vertices onto XY plane
    verts_xy = verts[:, :2]

    # Calculate the centroid of the XY projection
    centroid = cp.mean(verts_xy, axis=0)

    # Calculate distances from each point to the centroid
    distances = cp.linalg.norm(verts_xy - centroid, axis=1)

    # Find the two points furthest from each other
    max_distance_idx = cp.argmax(distances)
    furthest_point = verts_xy[max_distance_idx]

    # Calculate vectors from centroid to each point
    vectors = verts_xy - centroid

    # Calculate the perpendicular direction to the longest axis
    longest_vector = furthest_point - centroid
    perpendicular = cp.array([-longest_vector[1], longest_vector[0]])
    perpendicular /= cp.linalg.norm(perpendicular)

    # Project all points onto this perpendicular direction
    projections = cp.dot(vectors, perpendicular)

    # Calculate width as the difference between max and min projections
    width = float(cp.max(projections) - cp.min(projections))


    return length, width

def find_closest_points_batch(batch_data, batch_binary):
    closest_points = []
    for data, binary in zip(batch_data, batch_binary):
        # Calculate distance transform on the binary image
        dist_transform = cp.asarray(distance_transform_edt(cp.asnumpy(binary == 0)))

        # Mask the distance transform with the object
        masked_dist = cp.where(data > 0, dist_transform, cp.inf)

        # Find the minimum distance point
        closest_point = cp.unravel_index(cp.argmin(masked_dist), data.shape)
        closest_points.append(closest_point)

    return cp.array(closest_points)



def measure_surface_area_gpu(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    area = 0.5 * cp.linalg.norm(cp.cross(v1 - v0, v2 - v0), axis=1)
    return float(cp.sum(area))


def measure_skeleton_length(points):
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points.get())
    dist, _ = tree.query(points.get(), k=len(points))
    return cp.max(dist).item()


def filter_dendrites(dendrites, settings,logger):
    # Label initial dendrites
    dendrite_labels, num_detected = ndimage.label(dendrites)

    # Calculate volumes and filter
    dend_vols = ndimage.sum_labels(dendrites, dendrite_labels, index=range(1, num_detected + 1))
    large_dendrites = dend_vols >= settings.min_dendrite_vol

    # Create new dendrite binary using LUT (avoids full-volume bool from np.isin)
    keep_ids = np.nonzero(large_dendrites)[0] + 1
    max_dl = int(dendrite_labels.max())
    dl_lut = np.zeros(max_dl + 1, dtype=np.uint8)
    dl_lut[keep_ids] = 1
    filt_dendrites, num_filtered = ndimage.label(dl_lut[dendrite_labels])

    # Free memory of temporary variables
    del dendrite_labels, dend_vols, large_dendrites, dl_lut

    logger.info(
        f"    Processing {num_filtered} of {num_detected} detected dendrites larger than minimum volume threshold of {settings.min_dendrite_vol} voxels")

    return filt_dendrites


def dendrite_repair(labeled_dendrites, max_dist_um, res_xy, logger):
    """Bridge disconnected dendrite fragments within max_dist_um (XY only).

    For each Z slice, finds pairs of dendrite CCs whose minimum XY gap is
    <= max_dist_um and draws a 1-voxel-wide line between the closest voxels.
    Existing dendrite voxels are NEVER overwritten — only background voxels
    along bridges are painted. After bridging, fragments that share a bridge
    are remapped to the **parent** label (largest CC by voxel count of the
    merged group), so a single dendrite that was broken into 3 fragments
    ends up with the parent's ID throughout.

    Algorithm
    ---------
    1. Per-slice EDT of background with `return_indices=True`. Each background
       voxel knows its nearest dendrite voxel position; that lookup gives the
       Voronoi label of every voxel in the slice.
    2. Boundaries between Voronoi regions of different labels are candidate
       bridge sites. For each (a, b) pair, keep the boundary with the shortest
       resulting bridge length (Euclidean between the two nearest dendrite
       voxels on either side).
    3. Reject bridges whose endpoint-to-endpoint Euclidean length > max_dist.
    4. Size-weighted union-find merges remaining (a, b) pairs — the larger CC
       (by global voxel count) becomes the parent.
    5. Apply the LUT slice-by-slice and rasterise bridges with skimage.draw.line.
       Bridge voxels take the parent's label.

    Works for both numpy and zarr input (slab iteration is identical).

    Parameters
    ----------
    labeled_dendrites : numpy.ndarray or zarr.Array
        3D int label volume from `filter_dendrites` / `filter_dendrites_streaming`.
    max_dist_um : float
        Maximum XY gap to bridge (microns).
    res_xy : float
        XY voxel resolution (microns).
    logger : logging.Logger

    Returns
    -------
    labeled_dendrites : same type as input (modified in place + returned)
    n_bridges_drawn : int
        Number of bridge voxels painted.
    """
    from skimage.draw import line as _line

    is_zarr = hasattr(labeled_dendrites, 'chunks')
    max_dist_vox = float(max_dist_um) / float(res_xy)
    max_dist_vox_sq = max_dist_vox * max_dist_vox
    shape = labeled_dendrites.shape
    if len(shape) != 3:
        logger.info("    Dendrite repair: expected 3D labels, skipping.")
        return labeled_dendrites, 0

    # ---- Pass 1: compute global per-label voxel counts (for size-weighted union)
    sizes = {}  # label -> total voxels
    max_label = 0
    for z in range(shape[0]):
        slab = np.asarray(labeled_dendrites[z])
        if slab.size == 0:
            continue
        m = int(slab.max())
        if m == 0:
            continue
        if m > max_label:
            max_label = m
        bc = np.bincount(slab.ravel(), minlength=m + 1)
        for i in range(1, m + 1):
            if bc[i] > 0:
                sizes[i] = sizes.get(i, 0) + int(bc[i])

    if max_label < 2 or len(sizes) < 2:
        logger.info(
            f"    Dendrite repair: only {len(sizes)} dendrite CC(s) — nothing to bridge.")
        return labeled_dendrites, 0

    # ---- Union-find with size weighting: parent = LARGER CC of merged group
    parent = np.arange(max_label + 1, dtype=np.int32)
    size_arr = np.zeros(max_label + 1, dtype=np.int64)
    for k, v in sizes.items():
        size_arr[k] = v

    def _find(x):
        # Path-halving
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return False
        # parent = larger CC; tie-break by smaller label
        if (size_arr[ra] < size_arr[rb]) or (size_arr[ra] == size_arr[rb] and ra > rb):
            ra, rb = rb, ra
        parent[rb] = ra
        size_arr[ra] += size_arr[rb]
        return True

    # ---- Pass 2: per-slice bridge detection + union
    bridges_per_slice = {}  # z -> list of (ya, xa, yb, xb)
    n_pairs_total = 0
    for z in range(shape[0]):
        slice_lab = np.asarray(labeled_dendrites[z])
        present = np.unique(slice_lab)
        present = present[present > 0]
        if len(present) < 2:
            continue
        bg = (slice_lab == 0)
        if not bg.any():
            continue

        # 2D EDT in voxel units (XY isotropic at typical SDC; skip sampling)
        dist, indices = ndimage.distance_transform_edt(
            bg, return_distances=True, return_indices=True)
        nearest = np.where(bg, slice_lab[indices[0], indices[1]], slice_lab)

        bridge_per_pair = {}  # (min_lbl, max_lbl) -> (sq_len, ya, xa, yb, xb)
        H, W = slice_lab.shape
        for dy, dx in ((0, 1), (1, 0)):
            curr = nearest[0:H - dy, 0:W - dx] if dy or dx else nearest
            shifted = nearest[dy:H, dx:W]
            different = (curr != shifted) & (curr > 0) & (shifted > 0)
            if not different.any():
                continue
            ys, xs = np.nonzero(different)
            for y_, x_ in zip(ys.tolist(), xs.tolist()):
                la = int(curr[y_, x_]); lb = int(shifted[y_, x_])
                if la == lb:
                    continue
                key = (la, lb) if la < lb else (lb, la)
                ya, xa = int(indices[0, y_, x_]),       int(indices[1, y_, x_])
                yb, xb = int(indices[0, y_ + dy, x_ + dx]), int(indices[1, y_ + dy, x_ + dx])
                line_sq = (ya - yb) * (ya - yb) + (xa - xb) * (xa - xb)
                if line_sq > max_dist_vox_sq:
                    continue
                prev = bridge_per_pair.get(key)
                if prev is None or prev[0] > line_sq:
                    bridge_per_pair[key] = (line_sq, ya, xa, yb, xb)

        if not bridge_per_pair:
            continue
        n_pairs_total += len(bridge_per_pair)
        bridges_per_slice[z] = []
        for (la, lb), (line_sq, ya, xa, yb, xb) in bridge_per_pair.items():
            _union(la, lb)
            bridges_per_slice[z].append((ya, xa, yb, xb))

    if not bridges_per_slice:
        logger.info(
            f"    Dendrite repair: no fragments within {max_dist_um:.2f} µm — no bridges added.")
        return labeled_dendrites, 0

    # ---- Build LUT mapping each label to its parent root
    lut = np.zeros(max_label + 1, dtype=labeled_dendrites.dtype)
    for i in range(1, max_label + 1):
        lut[i] = _find(i)
    n_remapped = int((lut[1:max_label + 1] != np.arange(1, max_label + 1, dtype=lut.dtype)).sum())

    # ---- Pass 3: apply LUT + draw bridges in each slice
    n_bridges_drawn = 0
    n_slices_changed = 0
    for z in range(shape[0]):
        slice_lab = np.asarray(labeled_dendrites[z])
        relabeled = lut[slice_lab]
        slice_changed = bool(np.any(slice_lab != relabeled))
        out = relabeled  # numpy ndarray already (lut[slice_lab] returns ndarray)

        if z in bridges_per_slice:
            H, W = out.shape
            for (ya, xa, yb, xb) in bridges_per_slice[z]:
                # Both endpoints reside on dendrite voxels; their LUT-mapped
                # labels are now equal (the merged parent). Use that as the
                # bridge label.
                pa = int(out[ya, xa]) if 0 <= ya < H and 0 <= xa < W else 0
                if pa == 0:
                    continue
                rr, cc = _line(int(ya), int(xa), int(yb), int(xb))
                # Paint only background voxels along the rasterised path.
                mask = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                rr = rr[mask]; cc = cc[mask]
                if rr.size == 0:
                    continue
                bg_mask = (out[rr, cc] == 0)
                if bg_mask.any():
                    out[rr[bg_mask], cc[bg_mask]] = pa
                    n_bridges_drawn += int(bg_mask.sum())
                    slice_changed = True

        if slice_changed:
            n_slices_changed += 1
            labeled_dendrites[z] = out

    logger.info(
        f"    Dendrite repair: {n_pairs_total} candidate CC pairs across "
        f"{len(bridges_per_slice)} slices; painted {n_bridges_drawn} bridge voxels in "
        f"{n_slices_changed} slices; remapped {n_remapped} fragment labels to parent.")
    return labeled_dendrites, n_bridges_drawn


def filter_dendrites_cupy(dendrites_np, settings, logger,
                     connectivity=3):            # 3 => 26‑conn in 3‑D
    """
    Parameters
    ----------
    dendrites_np : ndarray  (bool / uint8)
        Binary dendrite mask in CPU memory.
    settings     : object   (expects .min_dendrite_vol attr)
    logger       : logging.Logger
    connectivity : {1,2,3}
        1 = 6‑conn, 2 = 18‑conn, 3 = 26‑conn (same as SciPy)
    """

    # ---- move to GPU ----------------------------------------------------
    dend_cp = cp.asarray(dendrites_np, dtype=cp.bool_)

    struct = cp.ones((3, 3, 3), dtype=cp.int8) if connectivity == 3 else None

    # 1. label connected components
    labels_cp, num_detected = cp_ndimage.label(dend_cp, structure=struct)

    # 2. compute volumes (voxel counts) of each component
    idx = cp.arange(1, num_detected + 1, dtype=cp.int32)
    dend_vols_cp = cp_ndimage.sum(dend_cp, labels_cp, index=idx)

    # 3. filter by minimum volume threshold
    large_mask_cp = dend_vols_cp >= settings.min_dendrite_vol
    keep_labels_cp = idx[large_mask_cp]              # label IDs to keep

    # 4. create filtered binary & relabel
    filt_binary_cp = cp.isin(labels_cp, keep_labels_cp)
    filt_labels_cp, num_filtered = cp_ndimage.label(filt_binary_cp,
                                                     structure=struct)

    # ---- back to CPU ----------------------------------------------------
    filt_labels_np = cp.asnumpy(filt_labels_cp).astype(cp.uint16)

    # optional: free GPU memory early
    del dend_cp, labels_cp, dend_vols_cp, large_mask_cp, keep_labels_cp
    del filt_binary_cp, filt_labels_cp
    cp.cuda.Device().synchronize()

    logger.info(
        f"    Processing {num_filtered} of {num_detected} detected dendrites "
        f"larger than minimum volume threshold of {settings.min_dendrite_vol} voxels"
    )

    return filt_labels_np

def adaptive_distance_transform(image, logger, threshold_size=20 * 2000 * 2000,
                                  workspace=None, dest_name=None):
    """Adaptive (downsampling-based) Euclidean distance transform.

    The image-domain `astype(bool)`
    and skimage.resize() calls are fatal at T_LARGE — `.astype(bool)` on a
    27.7 GB zarr materializes 27.7 GB bool numpy, and skimage.resize allocates
    a 222 GB float64 internal buffer for 29.8B-voxel inputs. The chunked path
    below handles both:
      - For zarr or numpy.size > threshold: slab-iterate bool conversion into
        a smaller intermediate, call `chunked_resize_3d` (grid_mode=True →
        byte-identical to skimage at small scale), run EDT
        on the downsampled bool numpy (fits in RAM), upsample via
        `chunked_resize_3d`.
      - For small numpy: legacy skimage.resize path preserved (byte-identical
        T2/T4 baseline).

    Note: soma_distance, with typical magnitudes of 1000+ voxels, accepts
    ±20 voxel error from downsample rounding. dendrite_distance uses
    chunked GPU EDT (no downsample) for
    stricter tolerance. This function should ONLY be called for soma /
    other-where-coarse-OK measurements.
    """
    # Ensure the input is binary
    time_initial = time.time()

    # Detect zarr / large inputs that need the chunked path
    _is_zarr = hasattr(image, 'chunks')
    _shape = tuple(int(s) for s in image.shape)
    current_size = int(np.prod(np.array(_shape, dtype=np.int64)))
    _use_chunked_path = _is_zarr or current_size > 1_000_000_000  # 1B voxels

    if not _use_chunked_path:
        # Legacy small-volume path — byte-identical to T2/T4 baseline
        binary_image = image.astype(bool)
        size_ratio = current_size / threshold_size
        scale_factor = max(1, np.ceil(np.sqrt(size_ratio) * 2) / 2)
        if scale_factor == 1:
            return ndimage.distance_transform_edt(np.invert(binary_image))
        logger.info(f"    Using scale factor {round(scale_factor,1)} to reduce computation time for adaptive distance calculation.")
        original_shape = binary_image.shape
        new_shape = tuple(int(s / scale_factor) for s in original_shape)
        small_image = resize(binary_image, new_shape, order=0, preserve_range=True, anti_aliasing=False)
        del binary_image
        gc.collect()
        small_distance = ndimage.distance_transform_edt(np.invert(small_image)).astype(np.float32)
        del small_image
        gc.collect()
        large_distance = resize(small_distance, original_shape, order=1, preserve_range=True).astype(np.float32)
        del small_distance
        gc.collect()
        large_distance *= scale_factor
        logger.info(f"    Time taken for adaptive distance calculation: {time.time() - time_initial:.2f} seconds")
        return large_distance

    # Chunked path for zarr inputs / huge volumes
    size_ratio = current_size / threshold_size
    scale_factor = max(1, np.ceil(np.sqrt(size_ratio) * 2) / 2)
    new_shape = tuple(max(1, int(s / scale_factor)) for s in _shape)
    logger.info(f"    Adaptive EDT (chunked path): {_shape} -> {new_shape} (scale={scale_factor:.1f})")

    # Step 1: build downsampled bool image without materializing the full input.
    # Slab-iterate the input, convert to bool per slab, accumulate into a small
    # bool array sized for the downsampled shape via per-slab block sums.
    # At T_LARGE: input 27.7 GB uint8 zarr -> ~85 MB downsampled bool (scale~7).
    import RESPAN.ImageAnalysis.ChunkedProcessing as _chunked
    # Workspace was previously read via
    # `getattr(image, '_respan_workspace', None)` but no upstream code sets
    # that attribute, so the chunked path always fell back to system temp
    # (~30 GB at T_LARGE — wrong filesystem, wrong cleanup semantics). Now
    # accepted as an explicit kwarg from the caller.
    if workspace is None:
        # No workspace plumbed → fall back to system temp dir (small-volume
        # callers; T_LARGE callers pass workspace explicitly).
        import tempfile
        _temp_dir = tempfile.mkdtemp(prefix='respan_soma_edt_')
        _bool_zarr_path = os.path.join(_temp_dir, 'soma_bool')
        _bool_zarr = zarr.open(
            _bool_zarr_path, mode='w', shape=_shape, dtype=np.uint8,
            chunks=_chunked.STREAMING_CHUNKS)
    else:
        _temp_dir = None
        _bool_zarr = workspace.create_array(
            'soma_bool_for_edt', _shape, dtype=np.uint8,
            chunks=_chunked.STREAMING_CHUNKS)

    _slab_z = max(1, _chunked.STREAMING_CHUNKS[0])
    for _z in range(0, _shape[0], _slab_z):
        _ze = min(_z + _slab_z, _shape[0])
        _bool_zarr[_z:_ze] = (np.asarray(image[_z:_ze]) > 0).astype(np.uint8)

    # Step 2: chunked-resize bool zarr -> downsampled numpy bool
    # chunked_resize_3d expects an array-like; pass the zarr handle.
    small_uint8 = _chunked.chunked_resize_3d(
        _bool_zarr, new_shape, order=0, dtype=np.uint8, logger=logger)
    small_image = small_uint8.astype(bool)
    del small_uint8

    # Cleanup intermediate bool zarr
    if _temp_dir is not None:
        import shutil
        shutil.rmtree(_temp_dir, ignore_errors=True)
    elif workspace is not None:
        try:
            # Remove the temporary array directory but keep the workspace
            _bool_path = os.path.join(workspace.base_dir, 'soma_bool_for_edt.zarr')
            import shutil as _shutil
            _shutil.rmtree(_bool_path, ignore_errors=True)
        except Exception:
            pass

    # Step 3: EDT on downsampled bool (small enough for numpy)
    small_distance = ndimage.distance_transform_edt(np.invert(small_image)).astype(np.float32)
    del small_image
    gc.collect()

    # Step 4: upsample back to original shape.
    # At T_LARGE the full-volume float32 alloc is
    # 30B × 4 = 120 GB → tight margin on 137 GB host. When workspace +
    # dest_name are provided, stream the upsample directly into a zarr to
    # avoid the peak. Caller (e.g. analyze_spines soma_distance call) then
    # uses the zarr handle without ever materializing a 120 GB numpy.
    if workspace is not None and dest_name is not None:
        # Allocate dest zarr at full shape — Blosc-compressed on disk, peak
        # in-memory transient is per-slab during chunked_resize_3d's tile
        # loop (a few hundred MB for typical chunk sizes).
        dest_zarr = workspace.create_array(
            dest_name, _shape, dtype=np.float32,
            chunks=_chunked.STREAMING_CHUNKS)
        # chunked_resize_3d writes to a numpy by default; for zarr destination
        # we slab the upsample manually using its xy_pass + z_pass primitives.
        # Pass 1 (XY) produces a float32 intermediate of (small_Z, target_Y, target_X).
        # That intermediate IS still numpy but smaller (e.g. T_LARGE: 9 GB
        # vs 120 GB final — fits in RAM). Pass 2 streams Z-tiles into the
        # zarr destination.
        _intermediate = _chunked.chunked_resize_xy_pass(
            small_distance, _shape, order=1, logger=logger)
        del small_distance
        gc.collect()
        # Stream Z-pass into zarr: per-Y-X tile, write to zarr[z_range, y_tile, x_tile]
        _tile_yx = 1024
        _zf_z = _shape[0] / _intermediate.shape[0]
        for y_start in range(0, _intermediate.shape[1], _tile_yx):
            y_end = min(y_start + _tile_yx, _intermediate.shape[1])
            for x_start in range(0, _intermediate.shape[2], _tile_yx):
                x_end = min(x_start + _tile_yx, _intermediate.shape[2])
                _tile = _intermediate[:, y_start:y_end, x_start:x_end]
                _resized = ndimage.zoom(_tile, (_zf_z, 1.0, 1.0), order=1,
                                          prefilter=False, grid_mode=True,
                                          mode='grid-constant')
                # Apply scale_factor and write to zarr — slab-iterate the Z dim
                # of the resized tile to avoid keeping the full target-Z tile
                # in addition to dest_zarr's compressed write buffer.
                _slab_z_w = max(1, min(64, _resized.shape[0]))
                for _zw in range(0, _resized.shape[0], _slab_z_w):
                    _zwe = min(_zw + _slab_z_w, _resized.shape[0])
                    dest_zarr[_zw:_zwe, y_start:y_end, x_start:x_end] = (
                        _resized[_zw:_zwe] * scale_factor).astype(np.float32)
                del _tile, _resized
        del _intermediate
        gc.collect()
        logger.info(f"    Time taken for adaptive distance calculation (chunked, zarr dest): {time.time() - time_initial:.2f} seconds")
        return dest_zarr

    # Legacy: numpy output (small/medium volumes that fit)
    large_distance = _chunked.chunked_resize_3d(
        small_distance, _shape, order=1, dtype=np.float32, logger=logger)
    del small_distance
    gc.collect()

    large_distance *= scale_factor
    logger.info(f"    Time taken for adaptive distance calculation (chunked): {time.time() - time_initial:.2f} seconds")
    return large_distance




def categorize_spine(L, dH, dN, 
                     big_ratio=2.0,   # threshold for ">>"
                     gt_ratio=1.2,    # threshold for ">"
                     eq_tolerance=0.2 # threshold for "~=" around 1:1
                    ):
    """
    Categorize a spine based on L, dH, dN, using approximate numeric thresholds
    for '>>', '>', and '~='.

    Returns a string representing the spine category.
    """
    def roughly_equal(a, b, tol=eq_tolerance):
        # Check if a and b are 'close' in ratio, i.e. a/b in [1 - tol, 1 + tol]
        ratio = a / b if b != 0 else np.inf
        return (1 - tol <= ratio <= 1 + tol)

    # Avoid division by zero
    L  = max(L, 1e-9)
    dH = max(dH, 1e-9)
    dN = max(dN, 1e-9)

    L_dH  = L / dH
    dH_dN = dH / dN
    
    # 1) filopodia:   L >> dH ~= dN
    if (L_dH > big_ratio) and roughly_equal(dH, dN):
        return "filopodia"
    # 2) long thin:   L >> dH > dN
    elif (L_dH > big_ratio) and (dH_dN > gt_ratio):
        return "spine"
    # 3) thin:        L > dH > dN
    elif (L_dH > gt_ratio) and (dH_dN > gt_ratio):
        return "spine"
    # 4) stubby:      L ~= dH ~= dN
    elif roughly_equal(L, dH) and roughly_equal(dH, dN):
        return "spine"
    # 5) mushroom:    dH >> dN
    elif dH_dN > big_ratio:
        return "spine"
    # Otherwise
    return "spine"

def create_kdtree_from_skeleton(skeleton_coords, skeleton_labels, sampling_method='systematic', sampling_param=4,
                                skeleton_volume=None):

    # Optional: Downsample the skeleton points if needed
    if sampling_method == 'topology':
        skeleton_coords, skeleton_labels = downsample_skeleton_preserving_topology(
            skeleton_coords, skeleton_labels, skeleton_volume, step=sampling_param)
    elif sampling_method == 'random':
        sampling_param = 0.25
        skeleton_coords, skeleton_labels = downsample_skeleton_points_random(
            skeleton_coords, skeleton_labels, sampling_rate=sampling_param)
    elif sampling_method == 'systematic':
        skeleton_coords, skeleton_labels = downsample_skeleton_points_systematic(
            skeleton_coords, skeleton_labels, step=sampling_param)
    elif sampling_method == 'voxel_grid':
        sampling_param = 2
        skeleton_coords, skeleton_labels = voxel_grid_filter(
            skeleton_coords, skeleton_labels, voxel_size=sampling_param)
    elif sampling_method is not None:
        raise ValueError('Invalid sampling method')
    # If sampling_method is None, skip downsampling

    # Build the KDTree
    tree = cKDTree(skeleton_coords)
    return tree, skeleton_labels


def downsample_skeleton_points_random(skeleton_points, skeleton_labels, sampling_rate=0.25):
    num_points = skeleton_points.shape[0]
    num_samples = int(num_points * sampling_rate)
    if num_samples == 0:
        num_samples = 1  # Ensure at least one point is sampled

    indices = np.random.choice(num_points, size=num_samples, replace=False)
    sampled_points = skeleton_points[indices]
    sampled_labels = skeleton_labels[indices]
    return sampled_points, sampled_labels

def downsample_skeleton_points_systematic(skeleton_points, skeleton_labels, step=4):
    # Ensure step is at least 1
    step = max(1, step)
    sampled_points = skeleton_points[::step]
    sampled_labels = skeleton_labels[::step]
    return sampled_points, sampled_labels

def downsample_skeleton_preserving_topology(skeleton_points, skeleton_labels, skeleton_volume, step=4):
    """Downsample skeleton points but always keep branch points and endpoints.

    Branch points (3+ neighbors) and endpoints (1 neighbor) are identified
    via 3D convolution on the skeleton volume, then always retained.
    Remaining points are systematically sampled with the given step.
    """
    if skeleton_volume is None:
        # Fallback to systematic if no volume provided
        return downsample_skeleton_points_systematic(skeleton_points, skeleton_labels, step=step)

    step = max(1, step)

    # Build a set of (z,y,x) tuples for fast lookup
    point_set = set(map(tuple, skeleton_points))

    # 3x3x3 kernel for counting neighbors (excluding center)
    kernel = np.ones((3, 3, 3), dtype=np.int32)
    kernel[1, 1, 1] = 0

    # Count neighbors for each skeleton voxel
    neighbor_count = ndimage.convolve(skeleton_volume.astype(np.int32), kernel, mode='constant', cval=0)

    # Identify critical points: branch points (3+ neighbors) or endpoints (1 neighbor)
    critical_mask = skeleton_volume.astype(bool) & ((neighbor_count >= 3) | (neighbor_count == 1))

    # Get critical point coordinates
    critical_coords = set(map(tuple, np.argwhere(critical_mask)))

    # Separate skeleton points into critical (always keep) and regular (subsample)
    is_critical = np.array([tuple(pt) in critical_coords for pt in skeleton_points])

    critical_points = skeleton_points[is_critical]
    critical_labels = skeleton_labels[is_critical]

    regular_points = skeleton_points[~is_critical]
    regular_labels = skeleton_labels[~is_critical]

    # Systematically sample regular points
    if len(regular_points) > 0:
        sampled_regular = regular_points[::step]
        sampled_regular_labels = regular_labels[::step]
        # Combine critical + sampled regular
        out_points = np.concatenate([critical_points, sampled_regular], axis=0)
        out_labels = np.concatenate([critical_labels, sampled_regular_labels], axis=0)
    else:
        out_points = critical_points
        out_labels = critical_labels

    return out_points, out_labels

def voxel_grid_filter(skeleton_points, skeleton_labels, voxel_size=2):
    # Quantize coordinates to the voxel grid
    quantized_coords = (skeleton_points // voxel_size).astype(np.int32)
    # Create a unique key for each voxel
    keys = quantized_coords.view([('', quantized_coords.dtype)] * quantized_coords.shape[1])
    # Find unique voxels and their indices
    _, unique_indices = np.unique(keys, return_index=True)
    sampled_points = skeleton_points[unique_indices]
    sampled_labels = skeleton_labels[unique_indices]
    return sampled_points, sampled_labels

def match_and_relabel_objects_geo(tree_A, labels_A, image_B, max_distance=None):
    #from scipy.ndimage import label, generate_binary_structure, measurements

    # Label the connected components in image_B
    #s = generate_binary_structure(3, 1)
    #labeled_B, num_features_B = label(image_B, structure=s)

    # Extract centroids and labels for image_B. The zarr-safe path uses
    # per_spine_regionprops (per-bbox crops) instead of measure.regionprops,
    # which calls np.asarray() on its input — at T7 with Phase H zarr labels
    # that's a 22 GB materialization. Same reason np.zeros_like(image_B) is
    # avoided in the empty branch (uses .shape/.dtype only).
    _is_zarr_b = hasattr(image_B, 'chunks')
    if _is_zarr_b:
        _props_df = chunked.per_spine_regionprops(
            image_B, {}, ['label', 'centroid'])
        n_labels = len(_props_df)
    else:
        props_B = measure.regionprops(image_B)
        n_labels = len(props_B)

    if n_labels == 0:
        print("Warning: No labeled regions found in image_B")
        # Use shape/dtype only — np.zeros_like(zarr) materializes via np.asarray().
        empty_relabeled = np.zeros(image_B.shape, dtype=image_B.dtype)
        empty_labels = np.array([], dtype=np.int32)
        empty_dict = {}
        empty_indices = np.array([], dtype=np.int32)
        empty_distances = np.array([], dtype=np.float64)
        empty_df = pd.DataFrame(columns=['label', 'dendrite_id'])

        return (empty_relabeled, empty_labels, empty_dict,
                empty_indices, empty_distances, empty_df)

    # Extract centroids and labels (zarr-safe DataFrame path) or from regionprops (numpy)
    if _is_zarr_b:
        centroids_B = _props_df[['centroid-0', 'centroid-1', 'centroid-2']].to_numpy()
        labels_B_1D = _props_df['label'].astype(np.int32).to_numpy()
    else:
        centroids_B = np.array([prop.centroid for prop in props_B])
        labels_B_1D = np.array([prop.label for prop in props_B], dtype=np.int32)

    # Perform batch KDTree query with all centroids
    distances, indices = tree_A.query(centroids_B)

    # Optionally filter by max_distance
    if max_distance is not None:
        valid = distances <= max_distance
        n_invalid = np.sum(~valid)
        if n_invalid > 0:
            print(f"Warning: {n_invalid} spine(s) exceeded max matching distance ({max_distance:.1f}) and were unmatched")
    else:
        valid = np.ones_like(distances, dtype=bool)

    # Get the closest labels from labels_A using indices from KDTree query
    closest_labels = labels_A[indices]

    # Create a mapping array from labels in image_B to labels in image_A
   # num_features_B = image_B.max()
   # label_map_array = np.zeros(num_features_B + 1, dtype=np.int32)  # +1 because labels start from 1

    # Assign closest labels to the mapping array for valid matches
  #  label_map_array[labels_B_1D[valid]] = closest_labels[valid]

    # For invalid matches (if max_distance is specified), labels remain 0 (background)

    # Apply the mapping to relabel the entire image_B
   # relabeled_B = label_map_array[image_B]
    # Create a mapping dictionary from labels in image_B to labels in image_A
    label_map_dict = {}
    for spine_label, dendrite_label, is_valid in zip(labels_B_1D, closest_labels, valid):
        if is_valid:
            label_map_dict[spine_label] = dendrite_label
        else:
            label_map_dict[spine_label] = 0  # Background or invalid match

    # Vectorized relabeling using numpy vectorization
    #relabeled_B = np.vectorize(label_map_dict.get)(image_B)
    #relabeled_B = np.vectorize(label_map_dict.get, otypes=[np.int32])(image_B)

    # Create arrays from the mapping dictionary
    original_labels = np.array(list(label_map_dict.keys()), dtype=np.int32)
    mapped_labels = np.array(list(label_map_dict.values()), dtype=np.int32)

    # Create mapping array. safe_max handles zarr image_B under Phase H —
    # direct .max() AttributeError'd on zarr.Array (no .max method).
    max_label_in_B = int(chunked.safe_max(image_B))
    max_label_in_mapping = original_labels.max()
    mapping_array_size = max(max_label_in_B, max_label_in_mapping) + 1
    mapping_array = np.zeros(mapping_array_size, dtype=np.int32)
    mapping_array[original_labels] = mapped_labels

    # For large volumes, skip creating the relabeled volume (20 GB).
    # The caller only uses the DataFrame (dend_IDs) and geodesic distances,
    # not the relabeled_B volume. Creating it in-place would corrupt the
    # spine labels that downstream measurements depend on.
    if image_B.nbytes > 2e9:
        relabeled_B = image_B  # return original — not relabeled, but not used
    else:
        relabeled_B = mapping_array[image_B]


    # Create a DataFrame with original and updated labels (respect valid mask)
    masked_labels = np.where(valid, closest_labels, 0)
    dend_IDs = pd.DataFrame({
        'label': labels_B_1D,
        'dendrite_id': masked_labels
    })

    # Return additional data for geodesic distance calculations
    return relabeled_B, labels_B_1D, label_map_dict, indices, distances, dend_IDs


@jit(nopython=True)
def fast_unique_count(arr):
    return len(np.unique(arr))


def compute_geodesic_distance_map(object_mask, starting_points, spacing=None):
    """Compute geodesic distances using Dijkstra with anisotropic spacing.

    Parameters
    ----------
    object_mask : ndarray (Z, Y, X)
        Boolean mask of the object to compute distances within.
    starting_points : list of tuples
        Seed voxels for distance computation.
    spacing : tuple of float, optional
        Physical voxel spacing as (resZ, resY, resX) in microns.
        If None, unit spacing is used (backward compatible).

    Returns
    -------
    distance_map : ndarray
        Geodesic distances in physical units (microns) if spacing provided,
        otherwise in voxel units.
    """
    import heapq

    if spacing is None:
        spacing = (1.0, 1.0, 1.0)

    sz, sy, sx = spacing

    # Initialize distance map with infinity
    distance_map = np.full(object_mask.shape, np.inf, dtype=np.float64)

    # Build 6-connected neighbor offsets with physical step costs
    # Each entry: (dz, dy, dx, cost)
    neighbors_6 = [
        (-1, 0, 0, sz), (1, 0, 0, sz),
        (0, -1, 0, sy), (0, 1, 0, sy),
        (0, 0, -1, sx), (0, 0, 1, sx),
    ]

    # Priority queue: (distance, (z, y, x))
    heap = []
    for pt in starting_points:
        distance_map[pt] = 0.0
        heapq.heappush(heap, (0.0, pt))

    shape = object_mask.shape

    while heap:
        current_dist, current = heapq.heappop(heap)

        # Skip if we already found a shorter path
        if current_dist > distance_map[current]:
            continue

        for dz, dy, dx, cost in neighbors_6:
            nz = current[0] + dz
            ny = current[1] + dy
            nx = current[2] + dx

            if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                if object_mask[nz, ny, nx]:
                    new_dist = current_dist + cost
                    if new_dist < distance_map[nz, ny, nx]:
                        distance_map[nz, ny, nx] = new_dist
                        heapq.heappush(heap, (new_dist, (nz, ny, nx)))

    return distance_map



def analyze_spines_4D(settings, locations, log, logger):
    logger.info("\nAnalyzing spines across time...")
    #spines = 1
    #dendrites = 2
    #soma = 3

    datasetname = os.path.basename(os.path.normpath(locations.input_dir))

    image = imread(locations.input_dir+"/Registered/Registered_images_4D.tif")

    labels = imread(locations.input_dir+"/Registered/Registered_labels_4D.tif")

    if image.shape != labels.shape:
        logger.info(log)
        raise RuntimeError("Image and labels are not the same shape.")

    #Currently registering one channel - but need to add capacity to deal with multi channels
    #in that situation the data will be CMZYX
    #if only one channel then MZYX, so if if shape only 4 add a empty axis at the beginning and then proceed
    #Add channel axis if not present
    if len(image.shape) == 4:
        image = np.expand_dims(image, axis=2)

    #select neuron channel
    neuron = image[:,:,settings.neuron_channel-1,:,:]


    spines = (labels == 1)
    dendrites = (labels == 2)
    soma = (labels==3)

    #
    spine_labels = spine_detection_4d(spines, settings.erode_shape, settings.remove_touching_boarders, logger) #value used to remove small holes


    #logger.info(f" {np.max(spine_labels[0,:,:,:])}.")

    #create lists for each 3D volume
    dendrite_distance_list = []
    skeleton_list = []
    soma_distance_list = []

    spine_summary = pd.DataFrame()



    for t in range(labels.shape[0]):

        logger.info(f" Processing timepoint {t+1} of {labels.shape[0]}.")

        dendrites_3d = dendrites[t, :, :, :]

        #fitler out small dendites
        dendrite_labels, num_detected = ndimage.label(dendrites_3d)


        # Calculate volumes and filter
        dend_vols = ndimage.sum_labels(dendrites_3d, dendrite_labels, index=range(1, num_detected + 1))

        large_dendrites = dend_vols >= settings.min_dendrite_vol

        # Create new dendrite binary using LUT (avoids full-volume bool from np.isin)
        _keep = np.nonzero(large_dendrites)[0] + 1
        _max_dl = int(dendrite_labels.max())
        _dl_lut = np.zeros(_max_dl + 1, dtype=np.uint8)
        _dl_lut[_keep] = 1
        dendrites_3d = _dl_lut[dendrite_labels].astype(bool)

        filt_dendrites = np.max(measure.label(dendrites_3d))

        logger.info(f"  Processing{filt_dendrites} of {num_detected} detected dendrites larger than minimum volume threshold of {settings.min_dendrite_vol * settings.input_resXY *settings.input_resXY*settings.input_resZ} µm<sup>3</sup> or {settings.min_dendrite_vol} voxels...")

        if filt_dendrites > 0:

            #logger.info(f"   Processing {filt_dendrites} dendrites...")


            #Create Distance Map



            dendrite_distance = ndimage.distance_transform_edt(np.invert(dendrites_3d)) #invert neuron mask to get outside distance
            dendrite_distance_list.append(dendrite_distance)

            dendrites_3d = dendrites_3d.astype(np.uint8)

            skeleton = morphology.skeletonize(dendrites_3d)
            skeleton_list.append(skeleton)

            soma_3d = soma[t, :, :, :]

            if np.max(soma_3d) == 0:
                soma_distance = soma[t, :, :, :]

            else:
                soma_distance = ndimage.distance_transform_edt(np.invert(soma))


            #if settings.save_val_data == True:
            #    save_3D_tif(neuron_distance.astype(np.uint16), locations.validation_dir+"/Neuron_Mask_Distance_3D"+file, settings)

            #Create Neuron MIP for validation - include distance map too
            #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines, dendrites, skeleton, dendrite_distance], locations.MIPs+"MIP_"+files[file], 'float', settings)
            #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, neuron_mask, soma_mask, soma_distance, skeleton, neuron_distance, density_image], locations.analyzed_images+"/Neuron/Neuron_MIP_"+file, 'float', settings)

            #Detection
            #logger.info(" Detecting spines...")
        else:
            logger.info("  *No dendrites were analyzed for this image.")

        dendrite_distance = np.stack(dendrite_distance_list, axis=0)
        skeleton = np.stack(skeleton_list, axis=0)
        soma_distance = np.stack(dendrite_distance_list, axis=0)

        spines_filtered_list = []
        all_spines_table = pd.DataFrame()
        all_summary_table = pd.DataFrame()

            #Measurements


    for t in range(labels.shape[0]):
        logger.info(f" Measuring timepoint {t+1} of {labels.shape[0]}.")
        #max label for 4d?
        spine_table, spines_filtered = spine_measurementsV2(image[t, :, :, :], spine_labels[t, :, :, :], 0, 0, settings.neuron_channel, dendrite_distance[t, :, :, :], soma_distance[t, :, :, :], settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, datasetname, logger)
                                                            #soma_mask, soma_distance, )
        if t == 0:
            previous_spines = set(np.unique(spines_filtered))
            new = 0
            pruned = 0

        else:
            current_spines = set(np.unique(spines_filtered))
            new = len(current_spines - previous_spines)
            pruned = len(previous_spines - current_spines)

        dendrite_length = np.sum(skeleton[t,:,:,:] == 1)

        dendrite_volume = np.sum(dendrites[t,:,:,:] ==1)


        if len(spine_table) == 0 and np.max(dendrites[t,:,:,:]) == 0:
            logger.info(f"  *No spines or dendrites were analyzed for this image.")

        else:
            #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines, spines_filtered, dendrites, skeleton, dendrite_distance], locations.MIPs+"MIP_"+filename, 'float', settings)

            spine_MIPs, spine_slices, spine_vols = create_spine_arrays_in_blocks(image[t, :, :, :], labels[t,:,:,:], spines_filtered, spine_table, settings.roi_volume_size, settings, locations, str(t+1)+'.tif',  logger, settings.GPU_block_size)

            #spine_table.to_csv(locations.tables + str(t)+ 'Detected_spines_1.csv',index=False)

            label_areas = spine_MIPs[:, 1, :, :]
            spine_areas = np.sum(label_areas > 0, axis=(1, 2))

            spine_masks = label_areas > 0
            spine_ids = np.nan_to_num(np.sum(label_areas * spine_masks, axis=(1, 2)) / np.sum(spine_masks, axis=(1, 2), where=spine_masks))

            df_spine_areas = pd.DataFrame({'spine_area': spine_areas})
            df_spine_areas['label'] = spine_ids

            #df_spine_areas['label'] = spine_table['label'].values

            #df_spine_areas.to_csv(locations.tables + str(t)+ 'Detected_spines_areas.csv',index=False)
            # Reindex df_spine_areas to match the index of spine_table
            #df_spine_areas_reindex = df_spine_areas.reindex(spine_table.index)
            #df_spine_areas_reindex.to_csv(locations.tables + 'Detected_spines_'+filename+'reindex.csv',index=False)
            spine_table = spine_table.merge(df_spine_areas, on='label', how='left')
            spine_table.insert(5, 'spine_area', spine_table.pop('spine_area')) #pops and inserts
            #spine_table.insert(5, 'spine_area', df_spine_areas['spine_area'])
            spine_table.insert(6, 'spine_area_um2', spine_table['spine_area'] * (settings.input_resXY **2))

            spine_table.insert(0, 'timepoint', t+1)

            #spine_table.to_csv(locations.tables + str(t)+ 'Detected_spines_2.csv',index=False)

            #append tables
            all_spines_table = pd.concat([all_spines_table, spine_table], ignore_index=True)

            #all_spines_table.to_csv(locations.tables + str(t)+ 'Detected_spines_appened.csv',index=False)

            summary = tables.create_spine_summary_neuron(spine_table, str(t+1), dendrite_length, dendrite_volume, settings)

            summary.insert(7, 'new_spines', new)
            summary.insert(8, 'pruned_spines', pruned)

            # Append to the overall summary DataFrame
            all_summary_table = pd.concat([all_summary_table, summary], ignore_index=True)
            #append images
            spines_filtered_list.append(spines_filtered)
            spines_filtered_all = np.stack(spines_filtered_list, axis=0)


    #Create Pivot Tables for volume and coordinates
    all_spines_table.drop(['dendrite_id'], axis=1, inplace=True)
    all_spines_table.rename(columns={'label': 'spine_id'}, inplace=True)
    all_summary_table.drop(['avg_dendrite_id'], axis=1, inplace=True)
    all_summary_table.rename(columns={'Filename': 'timepoint'}, inplace=True)


    vol_over_t = all_spines_table.pivot(index='spine_id', columns='timepoint', values='spine_vol')
    z_over_t = all_spines_table.pivot(index='spine_id', columns='timepoint', values='z')
    y_over_t = all_spines_table.pivot(index='spine_id', columns='timepoint', values='y')
    x_over_t = all_spines_table.pivot(index='spine_id', columns='timepoint', values='x')

    #save required tables
    vol_over_t.to_csv(locations.tables + 'Volume_4D.csv',index=False)
    all_summary_table.to_csv(locations.tables + 'Detected_spines_4D_summary.csv',index=False)
    all_spines_table.to_csv(locations.tables + 'Detected_spines_4D.csv',index=False)


    # Create MIP
    neuron_MIP = io.create_mip_and_save_multichannel_tiff_4d([neuron, spines, spines_filtered_all, dendrites, skeleton, dendrite_distance], locations.input_dir+"/Registered/Registered_MIPs_4D.tif", 'float', settings)

            #Create 4D Labels
    imwrite(locations.input_dir+'/Registered/Detected_spines.tif', spines_filtered_all.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'TZYX'})

    #Extract MIPs for each spine
    #if len(y_over_t) >=1:
    #    spine_MIPs, filtered_spine_MIP = create_MIP_spine_arrays_in_blocks_4d(neuron_MIP, y_over_t, x_over_t, settings.roi_volume_size, settings, locations, datasetname, logger, settings.GPU_block_size)

    #Cleanup
    #if os.path.exists(locations.nnUnet_input): shutil.rmtree(locations.nnUnet_input)
    if os.path.exists(locations.MIPs): shutil.rmtree(locations.MIPs)
    if os.path.exists(locations.validation_dir+"/Registered_segmentation_labels/"): shutil.rmtree(locations.validation_dir+"/Registered_segmentation_labels/")


    #logger.info("Spine analysis complete.\n")


def create_ellipsoidal_element(radius_z, radius_y, radius_x):
    # Create a grid of points
    z = np.arange(-radius_x, radius_x + 1)
    y = np.arange(-radius_y, radius_y + 1)
    x = np.arange(-radius_z, radius_z + 1)
    z, y, x = np.meshgrid(z, y, x, indexing='ij')

    # Create an ellipsoid
    ellipsoid = (z**2 / radius_z**2) + (y**2 / radius_y**2) + (x**2 / radius_x**2) <= 1

    return ellipsoid


def create_filtered_labels_image(labels, filtered_table, logger):
    """
    Filter a labels image in-place using a LUT, slab-by-slab.

    Remaps labels not in filtered_table to 0.  Works slab-by-slab to
    avoid allocating a full-volume copy (20+ GB for T7-class images).
    The LUT itself is only max_label × 4 bytes (a few KB–MB).

    Args:
        labels (numpy.ndarray): Input labels image (modified in-place).
        filtered_table (pd.DataFrame): Filtered regionprops table.

    Returns:
        numpy.ndarray: The same labels array, filtered in-place.
    """
    filtered_labels_list = filtered_table['label'].astype(labels.dtype).values
    # Zarr-safe max — `labels.max()` on zarr would materialize the full volume.
    # chunked.safe_max slab-iterates on zarr and falls through to .max() on numpy.
    max_label = int(chunked.safe_max(labels))
    lut = np.zeros(max_label + 1, dtype=labels.dtype)
    lut[filtered_labels_list] = filtered_labels_list
    # Slab-by-slab to avoid allocating a full-volume copy.
    # Cap each slab at ~2 GB to stay safe on high-res XY images.
    bytes_per_z = int(labels.shape[1]) * int(labels.shape[2]) * labels.dtype.itemsize
    slab_z = max(1, min(64, int(2 * 1024**3 / bytes_per_z)))
    for z in range(0, labels.shape[0], slab_z):
        ze = min(z + slab_z, labels.shape[0])
        labels[z:ze] = lut[labels[z:ze]]
    return labels

def create_filtered_and_unfiltered_spine_arrays_cupy(image, spines_filtered, labels, table, volume_size, settings, locations, file, logger):
    merge_mip_list = [] #image MIP
    merge_slice_list = [] # image slize
    merge_vol_list = []

    merge_masked_mip_list = [] #Mip filtered by label - we don't need this
    merge_masked_slice_list = [] #slice filtered by label - we don't need this either

    smallest_axis = np.argmin(image.shape)
    image = np.moveaxis(image, smallest_axis, -1)

    image_cp = cp.array(image)
    spines_filtered_cp = cp.array(spines_filtered)
    labels_cp = cp.array(labels)

    volume_size_z = int(volume_size / settings.input_resZ)
    volume_size_y = int(volume_size / settings.input_resXY)
    volume_size_x = int(volume_size / settings.input_resXY)


    # Pad the image and spines_filtered and keep the channel axis intact for the image
    image_cp = cp.pad(image_cp, ((volume_size_z // 2, volume_size_z // 2), (volume_size_y // 2, volume_size_y // 2), (volume_size_x // 2, volume_size_x // 2), (0, 0)), mode='constant', constant_values=0)
    spines_filtered_cp = cp.pad(spines_filtered_cp, ((volume_size_z // 2, volume_size_z // 2), (volume_size_y // 2, volume_size_y // 2), (volume_size_x // 2, volume_size_x // 2)), mode='constant', constant_values=0)
    labels_cp = cp.pad(labels_cp, ((volume_size_z // 2, volume_size_z // 2), (volume_size_y // 2, volume_size_y // 2), (volume_size_x // 2, volume_size_x // 2)), mode='constant', constant_values=0)


    for index, row in table.iterrows():
        #logger.info(f' Extracting and saving 3D ROIs for cell {index + 1}/{len(table)}', end='\r')
        z, y, x, label = int(row['z']), int(row['y']), int(row['x']), int(row['label'])

        z_min, z_max = z - volume_size_z // 2, z + volume_size_z // 2
        y_min, y_max = y - volume_size_y // 2, y + volume_size_y // 2
        x_min, x_max = x - volume_size_x // 2, x + volume_size_x // 2

        # Extract the volume and keep the channel axis intact
        image_vol = image_cp[z_min + volume_size_z // 2:z_max + volume_size_z // 2, y_min + volume_size_y // 2:y_max + volume_size_y // 2, x_min + volume_size_x // 2:x_max + volume_size_x // 2, :]
        spine_vol = spines_filtered_cp[z_min + volume_size_z // 2:z_max + volume_size_z // 2, y_min + volume_size_y // 2:y_max + volume_size_y // 2, x_min + volume_size_x // 2:x_max + volume_size_x // 2]
        label_vol = labels_cp[z_min + volume_size_z // 2:z_max + volume_size_z // 2, y_min + volume_size_y // 2:y_max + volume_size_y // 2, x_min + volume_size_x // 2:x_max + volume_size_x // 2]

        # Filter the volume to only show image data inside the label
        spine_mask = (spine_vol == label)
        spine_vol = spine_vol * spine_mask

        spine_vol = cp.expand_dims(spine_vol, axis=-1)
        label_vol = cp.expand_dims(label_vol, axis=-1)
        spine_mask = cp.expand_dims(spine_mask, axis=-1)

        #logger.info(f' image{image_vol.shape} ,spine_vol.shape {spine_vol.shape}labelvol shape {label_vol.shape}, spine maskshape {spine_mask.shape} ')

        #for the labels we actually want the dendrite label, but only the spine label for the center spine - requires masking spine label, but not dendrite label
        #cleaned_label = cp.copy(extracted_label)
        #clear outside spine for spine label only - leave dendrite, soma
        label_vol[(label_vol == 1) & (spine_mask == 0)] = 0

        #mask the spine image
        masked_image_vol = image_vol * spine_mask

        #

        # Extract 2D slice at position z
        image_slice = image_vol[volume_size_z // 2, :, :, :]
        spine_slice = spine_vol[volume_size_z // 2, :, :, :]
        label_slice = label_vol[volume_size_z // 2, :, :, :]

        masked_image_slice = masked_image_vol[volume_size_z // 2, :, :, :]
        #taking out the masked arrays - I don't think we need these        

        #why expanding -taking htis out temporarily
        ##spine_mask_expanded = cp.expand_dims(spine_mask, axis=-1)

        #extracted_volume_label_filtered = extracted_volume * spine_mask_expanded

        # Extract 2D slice before the MIP in step 3 is created
        #slice_before_mip = extracted_volume_label_filtered[volume_size_z // 2, :, :, :]
        #spine_slice = spine_mask_expanded[volume_size_z // 2, :, :, :]

        # Compute MIPs - this could be done by merging then mip but leave as this for now
        image_mip = cp.max(image_vol, axis=0)
        ##image_mip = image_mip[cp.newaxis, :, :] #expand to add label
        spine_mip=cp.max(spine_vol, axis = 0)
        ##spine_mip = spine_mip[cp.newaxis, :, :] # *65535#expand to add label
        label_mip=cp.max(label_vol, axis = 0)
        ##label_mip = label_mip[cp.newaxis, :, :]

        masked_image_mip = cp.max(masked_image_vol, axis=0)

        #logger.info(f' image mip {image_mip.shape} ,spine mip.shape {spine_mip.shape} label mip shape {label_mip.shape} ')

        merge_mip = cp.concatenate((image_mip, spine_mip, label_mip), axis=-1)

        merge_masked_mip = cp.concatenate((masked_image_mip, spine_mip, label_mip), axis=-1)


        merge_mip_list.append(merge_mip.get())

        merge_masked_mip_list.append(merge_masked_mip.get())


        ##image_slice = image_slice[cp.newaxis, :, :]#expand to add label
        ##spine_slice = spine_slice[cp.newaxis, :, :]
        ##label_slice = label_slice[cp.newaxis, :, :]


        #logger.info(f' image slice {image_slice.shape} ,spine slice.shape {spine_slice.shape} label slice shape {label_slice.shape} ')

        merge_slice = cp.concatenate((image_slice, spine_slice, label_slice), axis=-1)

        merge_masked_slice = cp.concatenate((masked_image_slice, spine_slice, label_slice), axis=-1)

        merge_slice_list.append(merge_slice.get())

        merge_masked_slice_list.append(merge_masked_slice.get())
        # now add Label slice to image slice

        #mip_label_filtered = cp.max(extracted_volume_label_filtered, axis=0)
        #mip_label_filtered = mip_label_filtered[cp.newaxis, :, :] #expand to add label
        #spine_slice = spine_slice[cp.newaxis, :, :] *65535 #expand to add label
        #mip_label_filtered= cp.concatenate((mip_label_filtered, spine_slice), axis=0)

        #mip_label_filtered_list.append(mip_label_filtered.get())

        #slice_before_mip = slice_before_mip[cp.newaxis, :, :] #expand to add label
        #slice_before_mip= cp.concatenate((slice_before_mip, spine_slice), axis=0)

        #slice_before_mip_list.append(slice_before_mip.get())

        #Could potentially add axis to volumes at beginning, then do all the mips w/o expansion - would be cleaner merge on axis 1 instead of 0
        # ie.. merge vol then create MIP and slice- come back and clean up if time...
        #CZYX



        ##image_vol = cp.expand_dims(image_vol, axis=0)
        #image_vol = image_vol[cp.newaxis, :, :, :]
        ##spine_vol = cp.expand_dims(spine_vol, axis=0)
        #spine_vol = image_vol[cp.newaxis, :, :, :, :]
        ##label_vol = cp.expand_dims(label_vol, axis=0)

        #logger.info(f' image{image_vol.shape} ,spine_vol.shape {spine_vol.shape}labelvol shape {label_vol.shape} ')



        merge_vol = cp.concatenate((image_vol, spine_vol, label_vol), axis=-1)
        merge_vol_list.append(merge_vol.get())




        del image_vol, label_vol, spine_vol, image_slice, spine_slice, label_slice, image_mip, spine_mip, label_mip, merge_vol, masked_image_mip, masked_image_slice, masked_image_vol
        cp.cuda.Stream.null.synchronize()
        gc.collect()

    mip_array = np.stack(merge_mip_list)

    masked_mip_array = np.stack(merge_masked_mip_list)

    #mip_array = np.moveaxis(mip_array, 3, 1)
    ##mip_array = mip_array.squeeze(axis=0)

    slice_array = np.stack(merge_slice_list)

    masked_slice_array = np.stack(merge_masked_slice_list)

    #slice_z_array = np.moveaxis(slice_z_array, 3, 1)
    ##slice_array = slice_array.squeeze(axis=0)

    vol_array = np.stack(merge_vol_list)
    ##vol_array = vol_array.squeeze(axis=0)


    #mip_label_filtered_array = np.stack(mip_label_filtered_list)
    #mip_label_filtered_array = np.moveaxis(mip_label_filtered_array, 3, 1)
    #mip_label_filtered_array = mip_label_filtered_array.squeeze(axis=-1)

    #slice_before_mip_array = np.stack(slice_before_mip_list)
    #slice_before_mip_array = np.moveaxis(slice_before_mip_array, 3, 1)
    #slice_before_mip_array = slice_before_mip_array.squeeze(axis=-1)

    return mip_array, slice_array, masked_mip_array, masked_slice_array, vol_array

def create_spine_arrays_in_blocks(image, labels, spines_filtered, table, volume_size, settings, locations, file, logger, block_size=(50, 300, 300)):
    #suppress warning about subtracting from table without copying
    original_chained_assignment = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None

    smallest_axis = np.argmin(image.shape)
    image = np.moveaxis(image, smallest_axis, -1)



    mip_list = []
    slice_list = []
    masked_mip_list = []
    masked_slice_list = []
    vol_list = []

    block_size_z, block_size_y, block_size_x = block_size

    z_blocks = math.ceil(image.shape[0] / block_size_z)
    y_blocks = math.ceil(image.shape[1] / block_size_y)
    x_blocks = math.ceil(image.shape[2] / block_size_x)
    total_blocks = z_blocks * y_blocks * x_blocks

    #logger.info(f'    Total blocks used for GPU spine array calculations: {total_blocks} ')

    for i in range(z_blocks):
        for j in range(y_blocks):
            for k in range(x_blocks):
                z_start = i * block_size_z
                z_end = min((i + 1) * block_size_z, image.shape[0])
                y_start = j * block_size_y
                y_end = min((j + 1) * block_size_y, image.shape[1])
                x_start = k * block_size_x
                x_end = min((k + 1) * block_size_x, image.shape[2])

                padding_z = int(max(0, (volume_size // settings.input_resZ) // 2))
                padding_y = int(max(0, (volume_size // settings.input_resXY) // 2))
                padding_x = int(max(0, (volume_size // settings.input_resXY) // 2))

                padded_z_start = int(max(0, z_start - padding_z))
                padded_z_end = int(min(image.shape[0], z_end + padding_z))
                padded_y_start = int(max(0, y_start - padding_y))
                padded_y_end = int(min(image.shape[1], y_end + padding_y))
                padded_x_start = int(max(0, x_start - padding_x))
                padded_x_end = int(min(image.shape[2], x_end + padding_x))

                #print(padded_z_start,padded_z_end, padded_y_start, padded_y_end, padded_x_start,padded_x_end)
                block_image = image[padded_z_start:padded_z_end, padded_y_start:padded_y_end, padded_x_start:padded_x_end]
                block_spines_filtered = spines_filtered[padded_z_start:padded_z_end, padded_y_start:padded_y_end, padded_x_start:padded_x_end]
                block_labels = labels[padded_z_start:padded_z_end, padded_y_start:padded_y_end, padded_x_start:padded_x_end]
                #print(table['z'])
                block_table = table[(table['z'] >= z_start) & (table['z'] < z_end) & (table['y'] >= y_start) & (table['y'] < y_end) & (table['x'] >= x_start) & (table['x'] < x_end)]
                #print(block_table)
                if len(block_table) > 0:
                    #block_table['z'] = block_table['z'] - padded_z_start
                    #block_table['y'] = block_table['y'] - padded_y_start
                    #block_table['x'] = block_table['x'] - padded_x_start
                    block_table.loc[:, 'z'] = block_table['z'] - padded_z_start
                    block_table.loc[:, 'y'] = block_table['y'] - padded_y_start
                    block_table.loc[:, 'x'] = block_table['x'] - padded_x_start

                    block_mip, block_slice, block_masked_mip, block_masked_slice, block_vol = create_filtered_and_unfiltered_spine_arrays_cupy(
                        block_image, block_spines_filtered, block_labels, block_table, volume_size, settings, locations, file, logger
                    )
                    mip_list.extend(block_mip)
                    slice_list.extend(block_slice)
                    masked_mip_list.extend(block_masked_mip)
                    masked_slice_list.extend(block_masked_slice)
                    vol_list.extend(block_vol)
                gc.collect()

        progress_percentage = ((i + 1) * y_blocks * x_blocks) / total_blocks * 100
        #print(f'Progress: {progress_percentage:.2f}%   ', end='', flush=True)

    #print(mip_list)
    # Convert lists to arrays and concatenate
    mip_array = np.stack(mip_list, axis = 0).transpose(0, 3, 1, 2)
    slice_array = np.stack(slice_list, axis=0).transpose(0, 3, 1, 2)
    masked_mip_array = np.stack(masked_mip_list, axis = 0).transpose(0, 3, 1, 2)
    masked_slice_array = np.stack(masked_slice_list, axis=0).transpose(0, 3, 1, 2)


    vol_array = np.stack(vol_list, axis=0).transpose(0, 1, 4, 2, 3)

    #vol_array = np.transpose(vol_array, (0, 2, 1, 3, 4))
    #mip_label_filtered_array = np.stack(mip_label_filtered_list, axis=0)
    #slice_before_mip_array = np.stack(slice_before_mip_list, axis=0)
    #print(mip_array.shape)

    #logger.info(f' {mip_array.shape}')
    imwrite(locations.arrays+"/Spine_MIPs_"+file, mip_array.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    imwrite(locations.arrays + "/Spine_slices_" + file, slice_array.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))

    imwrite(locations.arrays+"/Spine_masked_MIPs_"+file, masked_mip_array.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    imwrite(locations.arrays + "/Spine_masked_slices_" + file, masked_slice_array.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    imwrite(locations.arrays + "/Spine_vols_" + file, vol_array.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'TZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    #imwrite(locations.arrays+"/Masked_Spines_MIPs_"+file, mip_label_filtered_array.astype(np.uint16), imagej=True, photometric='minisblack',
    #        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
    #        resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    #imwrite(locations.arrays + "/Masked_Spines_Slices_" + file, slice_before_mip_array.astype(np.uint16), imagej=True, photometric='minisblack',
    #                 metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
    #                 resolution=(1/settings.input_resXY, 1/settings.input_resXY))

    #reenable pandas warning:
    pd.options.mode.chained_assignment = original_chained_assignment
    logger.info(f'     Complete.\n ')
    return mip_array, slice_array, vol_array

def create_spine_arrays_in_blocks_4d(image, labels_filtered, table, volume_size, settings, locations, file, logger, block_size=(50, 300, 300)):
    #suppress warning about subtracting from table without copying
    original_chained_assignment = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None

    smallest_axis = np.argmin(image.shape)
    image = np.moveaxis(image, smallest_axis, -1)

    mip_list = []
    slice_z_list = []
    mip_label_filtered_list = []
    slice_before_mip_list = []

    block_size_z, block_size_y, block_size_x = block_size

    z_blocks = math.ceil(image.shape[0] / block_size_z)
    y_blocks = math.ceil(image.shape[1] / block_size_y)
    x_blocks = math.ceil(image.shape[2] / block_size_x)
    total_blocks = z_blocks * y_blocks * x_blocks

    logger.info(f' Total blocks used for GPU spine array calculations: {total_blocks} ')

    for i in range(z_blocks):
        for j in range(y_blocks):
            for k in range(x_blocks):
                z_start = i * block_size_z
                z_end = min((i + 1) * block_size_z, image.shape[0])
                y_start = j * block_size_y
                y_end = min((j + 1) * block_size_y, image.shape[1])
                x_start = k * block_size_x
                x_end = min((k + 1) * block_size_x, image.shape[2])

                padding_z = int(max(0, (volume_size // settings.input_resZ) // 2))
                padding_y = int(max(0, (volume_size // settings.input_resXY) // 2))
                padding_x = int(max(0, (volume_size // settings.input_resXY) // 2))

                padded_z_start = int(max(0, z_start - padding_z))
                padded_z_end = int(min(image.shape[0], z_end + padding_z))
                padded_y_start = int(max(0, y_start - padding_y))
                padded_y_end = int(min(image.shape[1], y_end + padding_y))
                padded_x_start = int(max(0, x_start - padding_x))
                padded_x_end = int(min(image.shape[2], x_end + padding_x))

                #print(padded_z_start,padded_z_end, padded_y_start, padded_y_end, padded_x_start,padded_x_end)

                block_image = image[padded_z_start:padded_z_end, padded_y_start:padded_y_end, padded_x_start:padded_x_end]
                block_labels_filtered = labels_filtered[padded_z_start:padded_z_end, padded_y_start:padded_y_end, padded_x_start:padded_x_end]
                #print(table['z'])
                block_table = table[(table['z'] >= z_start) & (table['z'] < z_end) & (table['y'] >= y_start) & (table['y'] < y_end) & (table['x'] >= x_start) & (table['x'] < x_end)]
                #print(block_table)
                if len(block_table) > 0:
                    #block_table['z'] = block_table['z'] - padded_z_start
                    #block_table['y'] = block_table['y'] - padded_y_start
                    #block_table['x'] = block_table['x'] - padded_x_start
                    block_table.loc[:, 'z'] = block_table['z'] - padded_z_start
                    block_table.loc[:, 'y'] = block_table['y'] - padded_y_start
                    block_table.loc[:, 'x'] = block_table['x'] - padded_x_start

                    block_mip, block_slice_z, block_mip_label_filtered, block_slice_before_mip = create_filtered_and_unfiltered_spine_arrays_cupy(
                        block_image, block_labels_filtered, block_table, volume_size, settings, locations, file, logger
                    )
                    mip_list.extend(block_mip)
                    slice_z_list.extend(block_slice_z)
                    mip_label_filtered_list.extend(block_mip_label_filtered)
                    slice_before_mip_list.extend(block_slice_before_mip)
                    gc.collect()


        progress_percentage = ((i + 1) * y_blocks * x_blocks) / total_blocks * 100
        #print(f'Progress: {progress_percentage:.2f}%   ', end='', flush=True)

    #print(mip_list)
    # Convert lists to arrays and concatenate
    mip_array = np.stack(mip_list, axis = 0)
    slice_z_array = np.stack(slice_z_list, axis=0)
    mip_label_filtered_array = np.stack(mip_label_filtered_list, axis=0)
    slice_before_mip_array = np.stack(slice_before_mip_list, axis=0)

    imwrite(locations.arrays+"/Spines_MIPs_"+file, mip_array.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    imwrite(locations.arrays + "/Spines_Slices_" + file, slice_z_array.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    imwrite(locations.arrays+"/Masked_Spines_MIPs_"+file, mip_label_filtered_array.astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    imwrite(locations.arrays + "/Masked_Spines_Slices_" + file, slice_before_mip_array.astype(np.uint16),  compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))

    #reenable pandas warning:
    pd.options.mode.chained_assignment = original_chained_assignment
    return mip_array, slice_z_array, mip_label_filtered_array, slice_before_mip_array

def create_MIP_spine_arrays_in_blocks_4d(image, y_locs, x_locs, volume_size, settings, locations, file, logger, block_size=(50, 300, 300)):
    #spine_MIPs, filtered_spine_MIP = create_MIP_spine_arrays_in_blocks_4d(neuron_MIP, y_over_t, x_over_t, settings.roi_volume_size, settings, locations, datasetname, logger, settings.GPU_block_size)

    #image will be TCYX

    #we only want raw and object ID (0 and 2)

    image = image[:, [0, 2], :, :]

    volume_size = int(volume_size // settings.input_resXY)
    # Pad the image
    pad_width = int(volume_size // 2)
    padded_image = np.pad(image, ((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)), mode='constant')


    stacked_4D = []
    for (index1, yrow), (index2, xrow) in zip(y_locs.iterrows(), x_locs.iterrows()):
        regions = []
        for t in range(padded_image.shape[0]):
            if np.isnan(yrow[t]):
                region = np.zeros((padded_image.shape[1], volume_size, volume_size))
            else:
                y_coord = int(yrow[t])
                x_coord = int(xrow[t])
                #mask to object specifically            
                region = padded_image[t, :, y_coord:y_coord+volume_size, x_coord:x_coord+volume_size]

            regions.append(region)


        # Stack regions horizontally

        stacked_regions = np.concatenate(regions, axis=2)
        binary_mask = (stacked_regions[ 1, :, :] == index1)
        binary_mask = binary_mask.astype(int)*100
        stacked_regions[1, :, :] = binary_mask
        stacked_regions = np.expand_dims(stacked_regions, axis=0)

        stacked_4D.append(stacked_regions)

    # Stack each ID along the Z-axis
    final_4d = np.concatenate(stacked_4D, axis=0)

    imwrite(locations.input_dir+"/Registered/Isolated_spines_4D.tif", final_4d.astype(np.uint16),  compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
            metadata={'unit': 'um','axes': 'TCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))

    return final_4d, final_4d



def spine_measurementsV2(image, labels, dendrite, max_label, neuron_ch, dendrite_distance, soma_distance, sizes, dist,
                         settings, locations, filename, logger):
    """ measures intensity of each channel, as well as distance to dendrite
    Args:
        labels (detected cells)
        settings (dictionary of settings)

    Returns:
        pandas table and labeled spine image
    """

    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=1)

    # Measure channel 1:
    logger.info("    Making initial morphology and intensity measurements for channel 1...")
    # logger.info(f" {labels.shape}, {image.shape}")
    main_table = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=image[:, 0, :, :],
            properties=['label', 'centroid', 'area', 'mean_intensity', 'max_intensity'],  # area is volume for 3D images
        )
    )

    # rename mean intensity
    main_table.rename(columns={'mean_intensity': 'C1_mean_int'}, inplace=True)
    main_table.rename(columns={'max_intensity': 'C1_max_int'}, inplace=True)
    main_table.rename(columns={'centroid-0': 'z'}, inplace=True)
    main_table.rename(columns={'centroid-1': 'y'}, inplace=True)
    main_table.rename(columns={'centroid-2': 'x'}, inplace=True)

    # measure remaining channels
    for ch in range(image.shape[1] - 1):
        logger.info(f"    Measuring channel {ch + 2}...")
        # Measure
        table = pd.DataFrame(
            measure.regionprops_table(
                labels,
                intensity_image=image[:, ch + 1, :, :],
                properties=['label', 'mean_intensity', 'max_intensity'],  # area is volume for 3D images
            )
        )

        # rename mean intensity
        table.rename(columns={'mean_intensity': 'C' + str(ch + 2) + '_mean_int'}, inplace=True)
        table.rename(columns={'max_intensity': 'C' + str(ch + 2) + '_max_int'}, inplace=True)

        Mean = table['C' + str(ch + 2) + '_mean_int']
        Max = table['C' + str(ch + 2) + '_max_int']

        # combine columns with main table
        main_table = main_table.join(Mean)
        main_table = main_table.join(Max)

    # measure distance to dendrite
    logger.info("    Measuring distances to dendrite/s...")
    # logger.info(f" {labels.shape}, {dendrite_distance.shape}")
    distance_table = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=dendrite_distance,
            properties=['label', 'min_intensity', 'max_intensity'],  # area is volume for 3D images
        )
    )

    # rename distance column
    distance_table.rename(columns={'min_intensity': 'dist_to_dendrite'}, inplace=True)
    distance_table.rename(columns={'max_intensity': 'spine_length'}, inplace=True)

    distance_col = distance_table["dist_to_dendrite"]
    main_table = main_table.join(distance_col)
    distance_col = distance_table["spine_length"]
    main_table = main_table.join(distance_col)

    if np.max(soma_distance) > 0:
        # measure distance to dendrite
        logger.info("    Measuring distances to soma...")
        distance_table = pd.DataFrame(
            measure.regionprops_table(
                labels,
                intensity_image=soma_distance,
                properties=['label', 'min_intensity', 'max_intensity'],  # area is volume for 3D images
            )
        )
        distance_table.rename(columns={'min_intensity': 'euclidean_dist_to_soma'}, inplace=True)
        distance_col = distance_table["euclidean_dist_to_soma"]
        main_table = main_table.join(distance_col)
    else:
        main_table['euclidean_dist_to_soma'] = pd.NA

    # filter out small objects
    volume_min = sizes[0]  # 3
    volume_max = sizes[1]  # 1500?

    # logger.info(f" Filtering spines between size {volume_min} and {volume_max} voxels...")

    # filter based on volume
    # logger.info(f"  filtered table before area = {len(main_table)}")
    spinebefore = len(main_table)

    # Inclusive bounds (>=, <=). Previous strict inequalities silently dropped
    # spines whose volume exactly matched the configured min/max — at low
    # thresholds (e.g., spine_vol=(0.0005, 15) → min=1 voxel) this erased
    # every 1-voxel spine.
    filtered_table = main_table[(main_table['area'] >= volume_min) & (main_table['area'] <= volume_max)]

    logger.info(f"     Total putative spines: {spinebefore}")
    logger.info(f"     Spines after volume filtering = {len(filtered_table)} ")
    # logger.info(f"  filtered table after area = {len(filtered_table)}")

    # filter based on distance to dendrite
    spinebefore = len(filtered_table)
    # logger.info(f" Filtering spines less than {dist} voxels from dendrite...")
    # logger.info(f"  filtered table before dist = {len(filtered_table)}. and distance = {dist}")
    filtered_table = filtered_table[(filtered_table['spine_length'] < dist)]
    logger.info(f"     Spines after distance filtering = {len(filtered_table)} ")

    if settings.Track != True:
        # update label numbers based on offset
        filtered_table['label'] += max_label
        # Zarr-safe slab-iterated version of `labels[labels > 0] += max_label`.
        # In practice max_label=0 in the test-runner/GUI path so this is a
        # no-op; guard kept for non-zero legacy callers.
        if max_label:
            _ly, _lx = int(labels.shape[1]), int(labels.shape[2])
            _dtbytes = np.dtype(labels.dtype).itemsize
            _slab_z = max(1, min(64, int(2 * 1024**3 / max(_ly * _lx * _dtbytes, 1))))
            for _z in range(0, labels.shape[0], _slab_z):
                _ze = min(_z + _slab_z, labels.shape[0])
                _slab = np.asarray(labels[_z:_ze])
                _slab[_slab > 0] += max_label
                labels[_z:_ze] = _slab
                del _slab
        labels = create_filtered_labels_image(labels, filtered_table, logger)
    else:

        # Clean up label image — LUT remap, slab-by-slab to avoid
        # allocating a full-volume copy (20+ GB for T7-class images)
        ids_to_keep = filtered_table['label'].astype(labels.dtype).values
        # safe_max handles zarr labels under Phase H; falls through to
        # arr.max() for numpy. Direct labels.max() would materialize
        # the entire zarr volume.
        max_lbl = int(chunked.safe_max(labels))
        lut = np.zeros(max_lbl + 1, dtype=labels.dtype)
        lut[ids_to_keep] = ids_to_keep
        _bpz = int(labels.shape[1]) * int(labels.shape[2]) * labels.dtype.itemsize
        _slz = max(1, min(64, int(2 * 1024**3 / _bpz)))
        for _z in range(0, labels.shape[0], _slz):
            _ze = min(_z + _slz, labels.shape[0])
            labels[_z:_ze] = lut[labels[_z:_ze]]

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
    filtered_table = tables.move_column(filtered_table, 'euclidean_dist_to_soma', 11)
    #filtered_table['euclidean_dist_to_soma'] *= settings.input_resXY
    # filtered_table.insert(12, 'dist_to_soma_um', filtered_table['dist_to_soma'] * (settings.input_resXY))

    # logger.info(f"  filtered table before image filter = {len(filtered_table)}. ")
    # logger.info(f"  image labels before filter = {np.max(labels)}.")
    # integrated_density
    filtered_table['C1_int_density'] = filtered_table['spine_vol'] * filtered_table['C1_mean_int']

    # measure remaining channels
    for ch in range(image.shape[1] - 1):
        filtered_table['C' + str(ch + 2) + '_int_density'] = filtered_table['spine_vol'] * filtered_table[
            'C' + str(ch + 2) + '_mean_int']

    # Drop unwanted columns
    # filtered_table = filtered_table.drop(['spine_vol','spine_length', 'dist_to_dendrite', 'dist_to_soma'], axis=1)
    #logger.info(
     #   f"     After filtering {len(filtered_table)} spines were analyzed from a total of {len(main_table)} putative spines")

    return filtered_table, labels


#############################################################
#CUPY functions
#############################################################

def pad_subvolume_gpu(subvolume_gpu, pad_width):
    subvolume_gpu = cp.asarray(subvolume_gpu)
    return cp.pad(subvolume_gpu, ((pad_width, pad_width), (pad_width, pad_width), (pad_width, pad_width)),
                  mode='constant', constant_values=0)



def get_bounding_box_cupy(mask, margin, shape, settings):
    """Get the bounding box of a binary mask with added margin."""
    margin_y = margin_x = int(margin / settings.input_resXY)
    margin_z = int(margin / settings.input_resZ)
    z, y, x = cp.where(mask)

    z_min, z_max = int(max(z.min().item() - margin_z, 0)), int(min(z.max().item() + margin_z + 1, shape[0]))
    y_min, y_max = int(max(y.min().item() - margin_y, 0)), int(min(y.max().item() + margin_y + 1, shape[1]))
    x_min, x_max = int(max(x.min().item() - margin_x, 0)), int(min(x.max().item() + margin_x + 1, shape[2]))

    return z_min, z_max, y_min, y_max, x_min, x_max





#############################################################
# Excluded functionality and other functions
#############################################################

class FakeStream(object):
    def isatty(self):
        return False


def extract_subvolumes_GPU_batch(labeled_array, labels, padding=5):
    """
    Extract subvolumes for multiple labels in parallel on GPU using CuPy.

    Args:
        labeled_array (cp.ndarray): Labeled array where each unique label represents a distinct region.
        labels (cp.ndarray): Array of unique labels to extract.
        padding (int): Padding to apply around the bounding box of each labeled region.

    Returns:
        subvolumes (list of cp.ndarray): List of subvolumes for each label.
        coords (list of cp.ndarray): List of start coordinates for each subvolume.
    """
    subvolumes = []
    start_coords = []

    # Convert to CuPy array if it's not already
    labeled_array = cp.asarray(labeled_array)
    labels = cp.asarray(labels)

    # Get binary masks for all labels in one batch operation
    binary_masks = labeled_array[:, :, :, None] == labels[None, None, None, :]  # Shape: (z, y, x, n_labels)

    # Iterate through the labels and process each label's subvolume in parallel
    for idx, label in enumerate(labels):
        # Find the coordinates of the current label
        #print(f"Processing label {label}...")
        binary_mask = binary_masks[..., idx]
        coords = cp.argwhere(binary_mask)

        # Check if there are any coordinates for the current label
        if coords.size == 0:
            continue

        # Compute the start and end coordinates with padding
        start = cp.maximum(cp.min(coords, axis=0) - padding, 0)
        end = cp.minimum(cp.max(coords, axis=0) + padding + 1, cp.array(labeled_array.shape))

        # Extract the subvolume for the current label
        subvolume = labeled_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]] == label

        # Store the subvolume and the start coordinates
        subvolumes.append(subvolume)
        start_coords.append(start)

    return subvolumes, cp.stack(start_coords)


def spine_neck_analysis_gpu_batch(subvolumes, logger, scaling):

    cp_subvolumes = [cp.asarray(sv) for sv in subvolumes]

    results = []

    for cp_subvolume in cp_subvolumes:

        # Extract surface using marching cubes
        verts, faces, _, _ = marching_cubes(cp.asnumpy(cp_subvolume))

        # Convert vertices and faces to CuPy arrays
        verts = cp.asarray(verts) * cp.asarray(scaling)
        faces = cp.asarray(faces)

        # Compute volume using triangles
        volume = mesh_volume(verts, faces)

        # compute length and then widths
        length, min_width, max_width, mean_width, skeleton_result = mesh_neck_width_and_length(cp_subvolume, verts, logger, scaling)
        # Compute length using the longest path
        #length = mesh_length(verts, faces)

        # Compute width statistics
        #min_width, max_width, mean_width = mesh_neck_width(verts, faces)
        results.append((volume, length, min_width, max_width, mean_width))

    return results #volume, length, min_width, max_width, mean_width



def second_pass_annotation(head_labels_vol, neck_labels_vol, dendrite_vol, neuron_vol, locations, settings, logger):
    #dendrites and labels and must be binarized for further analysis
    #ensure data is rescaled to match second pass model

    start_time = time.time()
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    logger.info(f"       Available GPU memory: {free_mem / 1e9:.2f} GB")
    # estimate requirements
    avg_size = 8 / settings.refinement_Z * 8 /  settings.refinement_XY * 8 /  settings.refinement_XY  # spine or neck
    batch_size = max(1, int(free_mem * 0.7 / (avg_size * 4)))  # 4 bytes per float32
    #print(f"Using batch size: {batch_size}")

    # Move data to GPU and create multi-channel array
    cp_multi_channel = cp.stack([cp.asarray(head_labels_vol+neck_labels_vol),
                                            cp.asarray(neck_labels_vol > 0),
                                            cp.asarray(dendrite_vol >0),
                                            cp.asarray(neuron_vol)], axis=-1)

    # Get unique labels from array A (excluding background)
    labels = cp.unique(cp_multi_channel[:, :, :, 0])
    labels = labels[labels != 0]



      #batch
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i:i + batch_size]
        logger.info(f"      Processing batch {i // batch_size + 1} of {len(labels) // batch_size + 1}...")

        # Extract subvolumes
        sub_volumes, start_coords = extract_subvolumes_mulitchannel_GPU_batch_2ndpass(cp_multi_channel, batch_labels)

        #save each subvolume as a tiff in a folder
        for subvol, label in zip(sub_volumes, batch_labels):


            # save as tif and resahpe for imageJ
            imwrite_filename = os.path.join(locations.nnUnet_2nd_pass, f"subvol_{10000+label}_0000.tif")
            # imageJ ZCYX - currently ZYXC so fix

            # subvol_out = subvol.get()
            imwrite(imwrite_filename, subvol[:,:,:,3].get().transpose(0, 1, 2).astype(np.uint16), compression='zlib', compressionargs={'level': 1}, imagej=True,
                    photometric='minisblack', metadata={'unit': 'um', 'axes': 'ZYX'})

            #process with nnunet

        # split the path into subdirectories
        subdirectories = os.path.normpath(settings.refinement_model_path).split(os.sep)
        last_subdirectory = subdirectories[-1]
        # find all three digit sequences in the last subdirectory
        matches = re.findall(r'\d{3}', last_subdirectory)
        # If there's a match, assign it to a variable
        dataset_id = matches[0] if matches else None

        logger.info("\nPerforming spine refinement on GPU...\n")
        #create dir locations.nnUnet_2nd_pass+'\labels'
        if not os.path.exists(locations.nnUnet_2nd_pass+'\labels'):
            os.makedirs(locations.nnUnet_2nd_pass+'\labels')

        ##uncomment if issues with nnUnet
        # logger.info(f"{settings.nnUnet_conda_path} , {settings.nnUnet_env} , {locations.nnUnet_input}, {locations.labels} , {dataset_id} , {settings.nnUnet_type} , {settings}")

        return_code = sr. run_nnunet_predict(settings.nnUnet_conda_path, settings.nnUnet_env,
                                         locations.nnUnet_2nd_pass, locations.nnUnet_2nd_pass+'\labels', dataset_id, settings.nnUnet_type,
                                         settings, logger)

        logger.info(f"Updating subvolumes with refined labels...")

        #import all tifs in output folder as updated_sub_volumes .tif
        updated_sub_volumes = import_tiff_files_to_cupy_list(locations.nnUnet_2nd_pass+'\labels')

        multi_channel_subvolumes = []

        for subvolume, label in zip(updated_sub_volumes, labels):
            # Create boolean masks for each channel
            c0 = (subvolume == 1).astype(cp.float32) * label  # Multiply by label
            c1 = (subvolume == 2).astype(cp.float32)
            c2 = (subvolume == 4).astype(cp.float32)  * label  # Multiply by label

            # Stack the channels to create a multi-channel array
            multi_channel = cp.stack([c0, c1, c2], axis=-1)

            multi_channel_subvolumes.append(multi_channel)
        # Clean up label folder
        # delete nnunet input folder and files
        if settings.save_intermediate_data == False:

            shutil.rmtree(locations.nnUnet_2nd_pass)
            shutil.rmtree(locations.nnUnet_2nd_pass+'\labels')

        # Insert processed subvolumes back into the original image
        logger.info(f"Inserting refined subvolumes back into the original image...")
        cp_multi_channel = insert_subvolumes(cp_multi_channel, multi_channel_subvolumes, start_coords, labels)

    # Extract refined spine labels (channel 0) and neck labels (channel 1)
    spines_out = cp.asnumpy(cp_multi_channel[:, :, :, 0]).astype(np.int32)
    necks_out = cp.asnumpy(cp_multi_channel[:, :, :, 1]).astype(np.int32)

    del cp_multi_channel
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    return spines_out, necks_out


def insert_subvolumes(original_image, processed_subvolumes, start_coords, labels):
    # Create a mask for the areas we've modified
    modified_mask = cp.zeros(original_image.shape[:3], dtype=bool)

    for subvolume, start, label in zip(processed_subvolumes, start_coords, labels):
        end = start + cp.array(subvolume.shape[:3])

        # Create a mask for the current label in the original image
        label_mask = original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 0] == label

        # Update the modified mask
        modified_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]] |= label_mask

        # Clear the areas corresponding to the current label in c0 and c1
        original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 0] = cp.where(label_mask, 0,
                                                                                        original_image[start[0]:end[0],
                                                                                        start[1]:end[1],
                                                                                        start[2]:end[2], 0])
        original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 1] = cp.where(label_mask, 0,
                                                                                        original_image[start[0]:end[0],
                                                                                        start[1]:end[1],
                                                                                        start[2]:end[2], 1])

        # Add the subvolume data where it's non-zero
        subvolume_mask = subvolume > 0
        original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 0] += cp.where(subvolume_mask[..., 0],
                                                                                         subvolume[..., 0], 0)
        original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 1] += cp.where(subvolume_mask[..., 2],
                                                                                         subvolume[..., 2], 0)

        # For c2, combine using logical OR
        original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 2] = cp.logical_or(
            original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 2],
            subvolume[..., 1] > 0
        ).astype(original_image.dtype)

    # Clear any remaining voxels in c0 and c1 that weren't replaced
    original_image[:, :, :, 0] = cp.where(modified_mask, original_image[:, :, :, 0], 0)
    original_image[:, :, :, 1] = cp.where(modified_mask, original_image[:, :, :, 1], 0)

    return original_image

def flush_ram_and_gpu_memory(settings, logger):
    """
    Force garbage collection on CPU memory and free blocks in CuPy's memory pools.
    """
    # Log memory usage before flush
    size_threshold = 2e6 #(2MB)
    if settings.save_intermediate_data == True:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # in MB
        logger.info(f"CPU memory usage before flush: {mem_before:.2f} MB")

    namespace = globals()
    if namespace is not None:
        keys_to_delete = []
        for var_name, obj in list(namespace.items()):
            # Check if it's a NumPy or CuPy array with large nbytes
            if isinstance(obj, (np.ndarray, cp.ndarray)):
                # obj.nbytes is the data size in bytes
                if obj.nbytes >= size_threshold:
                    keys_to_delete.append(var_name)

        # Actually remove them
        for var_name in keys_to_delete:
            if logger is not None:
                size_mb = namespace[var_name].nbytes / (1024 * 1024)
                logger.info(f"[flush] Deleting large array '{var_name}' (~{size_mb:.2f} MB).")
            del namespace[var_name]


    # Force Python garbage collection
    gc.collect()

    # Synchronize and free GPU memory blocks
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # Log memory usage after flush
    if settings.save_intermediate_data == True:
        process = psutil.Process(os.getpid())
        mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
        logger.info(f"CPU memory usage after flush: {mem_after:.2f} MB")
        logger.info("Flushed CPU & GPU memory.")
