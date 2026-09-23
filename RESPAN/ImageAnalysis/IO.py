


import os

import numpy as np
from tifffile import imwrite


def long_path(path):
    """Return `path` in a form Windows will accept at any length.

    2026-09-20: Windows refuses to open a path longer than 260 characters
    unless it carries the `\\?\` extended-length prefix. RESPAN output paths
    concatenate the dataset folder, an output subfolder and the source image
    filename, so a cloud-synced data root plus a descriptive filename crosses
    that limit easily - a T2-series validation MIP measured 263 characters and
    the whole image was lost, its stale CSVs from a previous run left in place
    looking current.

    The prefix requires a fully qualified path with no forward slashes and no
    relative components, which `abspath` guarantees. Applied unconditionally on
    Windows rather than above a length threshold, so there is one code path
    instead of a branch that only runs on the rare inputs that break.
    Non-Windows platforms have no such limit and get the path unchanged.
    """
    if os.name != "nt":
        return path
    resolved = os.path.abspath(path)
    if resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):          # UNC share: \\server\share -> \\?\UNC\server\share
        return "\\\\?\\UNC" + resolved[1:]
    return "\\\\?\\" + resolved


# ImageJ LUTs (256x3 uint8) used when saving validation MIPs as composite TIFFs.
# One LUT per channel, cycled if more channels than LUTs.
def _lut_gray():
    """Standard grayscale."""
    g = np.arange(256, dtype=np.uint8)
    return np.stack([g, g, g])


def _lut_single_color(rgb):
    """Black-to-color ramp for an (R, G, B) target color."""
    r, g, b = rgb
    ramp = np.arange(256, dtype=np.float32) / 255.0
    return np.stack([
        (ramp * r).astype(np.uint8),
        (ramp * g).astype(np.uint8),
        (ramp * b).astype(np.uint8),
    ])


def _lut_fire():
    """Approximate ImageJ 'Fire' LUT — black → red → orange → yellow → white."""
    stops = [(0, 0, 0), (160, 0, 0), (255, 80, 0), (255, 200, 0), (255, 255, 255)]
    rgb = np.zeros((256, 3), dtype=np.uint8)
    n_stops = len(stops)
    for i in range(255 + 1):
        t = i * (n_stops - 1) / 255.0
        lo = int(t)
        hi = min(lo + 1, n_stops - 1)
        f = t - lo
        for c in range(3):
            rgb[i, c] = int(stops[lo][c] * (1 - f) + stops[hi][c] * f)
    return rgb.T  # return shape (3, 256)


# ImageJ "Glasbey on dark" — distinct colors optimized for label visualisation,
# index 0 is black for background. Used for C2 (spines, raw nnU-Net) and C3
# (spines_filtered, post-watershed labels) so each spine gets its own colour.
_GLASBEY_ON_DARK = np.array([
    [  0,   0,   0], [255, 255,   0], [255,  25, 255], [  0, 147, 147],
    [156,  64,   0], [ 88,   0, 199], [241, 235, 255], [ 20,  75,   0],
    [  0, 188,   1], [255, 159,  98], [145, 144, 255], [ 93,   0,  63],
    [  0, 255, 214], [255,   0,  95], [120, 100, 119], [  0,  73,  96],
    [140, 136,  74], [ 82, 207, 255], [207, 152, 186], [157,   0, 177],
    [191, 211, 152], [  0, 107, 210], [163,  51,  91], [ 88,  70,  43],
    [107, 255, 102], [156, 171, 174], [  0, 132,  65], [ 92,  16,   0],
    [  0,   0, 143], [240,  81,   0], [205, 170,   0], [182, 114, 100],
    [ 76, 190, 141], [148,  60, 255], [ 82,  54, 100], [ 73, 101,  93],
    [110, 132, 165], [175, 105, 192], [208, 184, 255], [255, 211, 190],
    [212, 255, 237], [255, 123, 137], [ 96,  98,   0], [222,   0, 157],
    [  0, 159, 249], [197, 120,   1], [  0,   1, 255], [197,   1,  29],
    [190, 163, 136], [ 98,  90, 157], [255, 144, 255], [160, 205,   0],
    [255, 215,  97], [107,  59,  73], [101, 144,   0], [124, 131, 125],
    [255, 255, 195], [149, 215, 214], [ 18, 112, 141], [255, 195, 239],
    [195, 102, 146], [140,   0,  30], [138, 177,  93], [135,  98,  59],
    [183, 209, 245], [163, 153, 193], [ 16, 189, 193], [255, 102, 194],
    [ 48,  57, 118], [ 77,  82,  99], [205, 192, 201], [ 94,  63, 255],
    [197, 135, 255], [195,   0, 255], [  0,  80,  57], [139,   3, 110],
    [208, 252, 136], [127, 229, 159], [151,  78, 136], [110,   0, 159],
    [130, 170, 209], [100, 150, 109], [158, 132, 145], [204,  76,  87],
    [ 66,   0, 124], [255, 172, 180], [136, 119, 190], [144,  86,  89],
    [109,  52,   0], [ 93, 116,  74], [  0, 246, 255], [ 96, 116, 255],
    [ 84,   0,  98], [  0, 169,  83], [137,  79, 175], [219, 182, 107],
    [197, 213, 203], [ 16, 144, 184], [230, 121,  85], [ 65,  85,  45],
    [ 42, 106,   0], [104,  96,  87], [255, 165,   4], [  2, 220,  95],
    [151, 183, 151], [147, 109,   0], [247,   0,  29], [195,  50, 190],
    [  2,  85, 148], [192,  93,  39], [  0, 125, 102], [156, 149,   0],
    [248, 125,   0], [255, 252, 243], [105, 165, 154], [184, 245,   0],
    [132,  56,  45], [214, 144, 141], [209,   0,  98], [197, 241, 255],
    [222, 214,   3], [163, 180, 255], [ 90, 124, 130], [105,  26,  42],
    [186, 150,  73], [114,  79,   0], [  0, 216, 189], [120,  53, 126],
    [157, 133, 109], [215, 124, 206], [254,  85,  81], [  0,  96, 100],
    [238,  85, 137], [ 23, 176, 218], [187, 255, 192], [126,   0, 220],
    [255, 150, 208], [ 73,  65,   0], [216,  90, 255], [176,  33, 135],
    [163, 110, 255], [ 64,  71, 177], [ 49,   0, 186], [186, 196,  86],
    [ 14, 103,  55], [ 85, 105, 136], [137,   0,  68], [  0, 158, 125],
    [125, 174,   0], [209, 196, 168], [140, 143, 156], [158, 224, 110],
    [ 86,  73,  78], [154, 255, 245], [176, 162, 168], [171,  62,  51],
    [103, 153, 169], [146, 116, 156], [106,  80, 124], [ 77, 131, 204],
    [179, 182, 207], [160,  25,   0], [143, 154, 123], [170, 117,  64],
    [ 91,  59, 145], [ 91, 227,   0], [205, 156, 223], [235, 177, 149],
    [  0, 140,   1], [204,  57,   2], [239, 218, 157], [175, 176, 117],
    [138, 111, 109], [156, 104, 129], [ 97,  42,  85], [167, 229, 200],
    [129, 186, 199], [254, 219, 232], [120, 123,  19], [ 99, 104, 111],
    [101,  94,  51], [217,  85, 183], [200, 139,  98], [115,  97, 214],
    [ 73,  80,  70], [227, 126, 166], [222, 203, 238], [132,  57, 100],
    [213, 110, 121], [158,   3, 224], [175,   0,  58], [117,  76,  61],
    [ 89, 127, 109], [139, 229, 255], [125,  33,   0], [123, 117,  89],
    [133, 147, 204], [179, 134, 184], [116, 164, 254], [126, 193, 175],
    [162,  91,   3], [ 26,  70, 255], [255,   0, 202], [215, 236, 199],
    [255, 248, 115], [115, 108, 145], [  0, 255, 150], [114, 201, 120],
    [ 80, 124,  40], [160,  88,  60], [ 68,  81, 128], [206, 213, 225],
    [194,  65, 126], [ 57,  80,  83], [225, 149,  58], [240, 178, 255],
    [176, 158, 238], [255,   0, 148], [116, 151,  74], [214, 171, 175],
    [101,  43,  29], [253, 150, 133], [  0, 215, 232], [243, 190,   0],
    [164,  87, 221], [  9, 120, 123], [ 54,  91, 112], [176,  88, 108],
    [102, 114, 176], [216, 222, 120], [191, 129, 149], [176, 178, 166],
    [127, 190, 255], [115,   0, 117], [  0,  60, 158], [229, 220, 214],
    [100,  67, 195], [113, 200,  65], [ 74,   0, 250], [  0, 220, 157],
    [235, 249, 254], [170, 177,   2], [199, 174, 206], [131,  74,  29],
    [255, 104, 237], [  0, 181, 158], [116,  77, 103], [171, 131,   6],
    [218,  90,  65], [173, 197, 203], [ 64,  85,   0], [255, 194, 132],
    [220,  27,  68], [168, 131, 209], [125, 121, 131], [ 17, 101,  82],
], dtype=np.uint8).T  # shape (3, 256)


# Default channel LUT sequence matches the MIP channel order used by
# create_mip_and_save_multichannel_tiff callers in ImageAnalysis.py:
#   0 neuron, 1 spines, 2 spines_filtered, 3 connected_necks,
#   4 labeled_dendrites, 5 skeleton, 6 dendrite_distance, 7 geodesic_distance
# Label channels (spines, spines_filtered, connected_necks, labeled_dendrites)
# use Glasbey-on-dark so each label ID renders as a distinct colour.
_DEFAULT_MIP_LUTS = [
    _lut_gray(),                           # 0 neuron (intensity)
    _GLASBEY_ON_DARK,                      # 1 spines (raw nnU-Net labels)
    _GLASBEY_ON_DARK,                      # 2 spines_filtered (watershed labels)
    _GLASBEY_ON_DARK,                      # 3 connected_necks (labeled per-spine)
    _GLASBEY_ON_DARK,                      # 4 labeled_dendrites (per-dendrite labels)
    _lut_single_color((255,   0, 255)),    # 5 skeleton (magenta binary)
    _lut_fire(),                           # 6 dendrite_distance (fire)
    _lut_single_color((  0, 128, 255)),    # 7 geodesic_distance (blue ramp)
]

# Indices of LUT channels that expect a 0-to-max-label display range so every
# label ID is visible under Glasbey. Other channels get their natural range.
_LABEL_CHANNEL_INDICES = {1, 2, 3, 4}


def _imagej_resolution_kwargs(settings):
    """Return tifffile imwrite kwargs for ImageJ physical scale in µm."""
    try:
        res_xy = float(settings.input_resXY)
    except (AttributeError, TypeError, ValueError):
        return {}
    if res_xy <= 0:
        return {}
    # tifffile wants pixels-per-unit and a TIFF-spec ResolutionUnit. For
    # ImageJ-readable metadata, pass the µm pixel size directly through the
    # metadata dict so ImageJ shows "X Resolution / Y Resolution" correctly.
    # pixels-per-cm = 1 / (pixel_size_um * 1e-4)
    px_per_cm = 1.0 / (res_xy * 1e-4)
    return {
        'resolution': (px_per_cm, px_per_cm),
        'resolutionunit': 'CENTIMETER',
    }



def _materialize_for_volume_save(arr):
    """Return a numpy ndarray. If arr is zarr-backed, slab-stream materialize.

    Used by create_and_save_multichannel_tiff so the 3D val volume can be saved
    even when chunked-mode kept arrays as zarr handles. Full materialization is
    O(volume) RAM — caller is responsible for ensuring host has headroom.
    """
    if isinstance(arr, np.ndarray):
        return arr
    out = np.empty(arr.shape, dtype=arr.dtype)
    chunk_z = arr.chunks[0] if hasattr(arr, 'chunks') else 64
    _bpz = int(arr.shape[1]) * int(arr.shape[2]) * arr.dtype.itemsize
    chunk_z = max(1, min(chunk_z, int(2 * 1024**3 / _bpz)))
    for z in range(0, arr.shape[0], chunk_z):
        z1 = min(z + chunk_z, arr.shape[0])
        out[z:z1] = np.asarray(arr[z:z1])
    return out


def create_and_save_multichannel_tiff(images_3d, filename, bitdepth, settings,
                                      channel_zero_above=None):
    """
    Create TIF from a list of 3D images, merge them into a image,
    save as a 16-bit TIFF file with the same per-channel LUTs, display ranges,
    and physical-scale metadata as the MIP saved by
    `create_mip_and_save_multichannel_tiff`. Channel order must match the MIP
    callers (neuron, spines, spines_filtered, connected_necks, labeled_dendrites,
    skeleton, dendrite_distance, geodesic_distance_image).

    Accepts numpy arrays or zarr handles; zarr inputs are slab-streamed into
    numpy via _materialize_for_volume_save before stacking.

    Args:
        images_3d (list of numpy.ndarray or zarr.Array): List of 3D arrays.
        filename (str): Filename for the output TIFF file.
        bitdepth: kept for backward-compat; output is always uint16.
        settings: used for resXY/resZ metadata.
        channel_zero_above (dict, optional): Per-channel zero-out thresholds
            ({channel_idx: threshold}). Visualization-only.
    """
    images_3d = [_materialize_for_volume_save(a) for a in images_3d]
    # Stack channel-axis — result shape is (Z, C, Y, X).
    multichannel_image = np.stack(images_3d, axis=1)
    multichannel_image = multichannel_image.astype(np.uint16)

    # Per-channel far-field zero-out (visualization only, raw arrays unchanged)
    if channel_zero_above:
        for ch_idx, thresh in channel_zero_above.items():
            if 0 <= ch_idx < multichannel_image.shape[1] and thresh is not None:
                ch = multichannel_image[:, ch_idx]
                ch[ch > thresh] = 0

    n_channels = multichannel_image.shape[1]
    luts = [_DEFAULT_MIP_LUTS[i % len(_DEFAULT_MIP_LUTS)] for i in range(n_channels)]
    # Per-channel display range — mirror MIP semantics: label channels use
    # 0..max-label for Glasbey visibility, others use true min..max per channel.
    ranges_flat = []
    for ch_idx in range(n_channels):
        ch = multichannel_image[:, ch_idx, :, :]
        ch_max = int(ch.max()) if ch.size else 0
        if ch_idx in _LABEL_CHANNEL_INDICES:
            ranges_flat.extend([0.0, float(max(ch_max, 1))])
        else:
            lo = float(ch.min()) if ch.size else 0.0
            hi = float(ch_max) if ch_max > 0 else 1.0
            ranges_flat.extend([lo, hi])
    meta = {
        'spacing': settings.input_resZ,
        'unit': 'um',
        'axes': 'ZCYX',
        'mode': 'composite',
        'LUTs': luts,
        'Ranges': tuple(ranges_flat),
    }
    try:
        res_xy = float(settings.input_resXY)
        meta['PhysicalSizeX'] = res_xy
        meta['PhysicalSizeXUnit'] = 'um'
        meta['PhysicalSizeY'] = res_xy
        meta['PhysicalSizeYUnit'] = 'um'
        meta['PhysicalSizeZ'] = float(settings.input_resZ)
        meta['PhysicalSizeZUnit'] = 'um'
    except (AttributeError, TypeError, ValueError):
        pass

    imwrite(long_path(filename), multichannel_image,
            compression='zlib', compressionargs={'level': 1},
            imagej=True, photometric='minisblack',
            metadata=meta, **_imagej_resolution_kwargs(settings))


def mip_from_source(arr, max_val=None):
    """Compute Z projection, zarr-safe.

    Default: max-intensity projection (np.amax along Z).

    If max_val is set, the channel is treated as a distance map:
      - Voxels with value > max_val are excluded (per-slab clip to 0 first,
        which would otherwise be the saturated junk dominating each column).
      - Reduction switches to MIN over non-zero values: each (Y, X) shows the
        CLOSEST distance to dendrite within reach. Columns with no in-range
        voxels (or that pass through a dendrite) collapse to 0.

    For zarr inputs, both the clip and the min-of-nonzero accumulate slab-wise.
    """
    if max_val is None:
        # Plain max projection
        if isinstance(arr, np.ndarray):
            return np.amax(arr, axis=0)
        chunk_z = arr.chunks[0] if hasattr(arr, 'chunks') else 64
        _bpz = int(arr.shape[1]) * int(arr.shape[2]) * arr.dtype.itemsize
        chunk_z = max(1, min(chunk_z, int(2 * 1024**3 / _bpz)))
        mip = None
        for z in range(0, arr.shape[0], chunk_z):
            slab = np.array(arr[z:min(z + chunk_z, arr.shape[0])])
            slab_mip = np.amax(slab, axis=0)
            mip = slab_mip.copy() if mip is None else np.maximum(mip, slab_mip, out=mip)
        return mip if mip is not None else np.zeros(arr.shape[1:], dtype=arr.dtype)

    # Distance-map projection: per-slab clip + min-over-nonzero
    SENTINEL = np.iinfo(np.uint16).max if np.issubdtype(arr.dtype, np.integer) else np.float32(1e30)
    out = None
    if isinstance(arr, np.ndarray):
        slabs = [arr]
    else:
        chunk_z = arr.chunks[0] if hasattr(arr, 'chunks') else 64
        _bpz = int(arr.shape[1]) * int(arr.shape[2]) * arr.dtype.itemsize
        chunk_z = max(1, min(chunk_z, int(2 * 1024**3 / _bpz)))
        slabs = (np.array(arr[z:min(z + chunk_z, arr.shape[0])])
                 for z in range(0, arr.shape[0], chunk_z))
    for slab in slabs:
        # Replace out-of-range AND zeros with SENTINEL so min ignores them
        masked = np.where((slab > 0) & (slab <= max_val), slab, SENTINEL)
        slab_min = np.amin(masked, axis=0)
        out = slab_min if out is None else np.minimum(out, slab_min)
    if out is None:
        return np.zeros(arr.shape[1:], dtype=arr.dtype)
    # Columns with no in-range voxel (still SENTINEL) → 0
    out = np.where(out >= SENTINEL, 0, out)
    return out.astype(arr.dtype if isinstance(arr, np.ndarray) else arr.dtype)


def create_mip_and_save_multichannel_tiff(images_3d, filename, bitdepth, settings,
                                          channel_zero_above=None):
    """
    Create MIPs from a list of 3D images, merge them into a multi-channel 2D image,
    save as a 16-bit TIFF file, and return the merged 2D multi-channel image.

    Args:
        images_3d (list of numpy.ndarray or zarr.Array): List of 3D arrays.
        filename (str): Filename for the output TIFF file.
        channel_zero_above (dict, optional): Per-channel zero-out thresholds
            ({channel_idx: threshold}). Used to hide chunked-EDT far-field
            noise in distance channels — pixels above threshold become 0.
            Visualization-only; does not affect underlying arrays.

    Returns:
        numpy.ndarray: Merged 2D multi-channel image as a numpy array.
    """
    # Create MIPs from the 3D images (zarr-safe via slab accumulation).
    # For distance channels with channel_zero_above set, zero per-slab BEFORE
    # the max-projection — otherwise far-field junk dominates every column's
    # max and post-MIP zeroing kills the entire channel.
    mips = []
    for i, img in enumerate(images_3d):
        thresh = channel_zero_above.get(i) if channel_zero_above else None
        mips.append(mip_from_source(img, max_val=thresh))

    # Convert the MIPs to a single multichannel image
    multichannel_image = np.stack(mips, axis=0)

    # Convert the multichannel image to 16-bit
    multichannel_image = multichannel_image.astype(np.uint16)
    #multichannel_image = rescale_all_channels_to_full_range(multichannel_image)

    # --- ImageJ composite TIFF metadata ---
    # Composite mode = channels displayed on top of each other in ImageJ/Fiji
    # with their per-channel LUTs. Pixel size (µm) is written as both ImageJ
    # metadata (for the Info panel) and TIFF resolution tags (for File > Image
    # Properties). Combined, opening the TIF in ImageJ/Fiji gives a composite
    # with scale already set — no manual Set Scale step needed.
    n_channels = multichannel_image.shape[0]
    luts = [_DEFAULT_MIP_LUTS[i % len(_DEFAULT_MIP_LUTS)] for i in range(n_channels)]
    # Per-channel display range. Label channels scale 0..max-label so every
    # ID maps to a distinct Glasbey entry; intensity/distance channels get a
    # robust percentile range so bright pixels don't wash out signal.
    ranges_flat = []
    for ch_idx in range(n_channels):
        ch = multichannel_image[ch_idx]
        ch_max = int(ch.max()) if ch.size else 0
        if ch_idx in _LABEL_CHANNEL_INDICES:
            # 0 → max label (ensure at least 1 to avoid min==max in ImageJ)
            ranges_flat.extend([0.0, float(max(ch_max, 1))])
        else:
            lo = float(ch.min()) if ch.size else 0.0
            hi = float(ch_max) if ch_max > 0 else 1.0
            ranges_flat.extend([lo, hi])
    meta = {
        'spacing': settings.input_resZ,
        'unit': 'um',
        'axes': 'CYX',
        'mode': 'composite',
        'LUTs': luts,
        'Ranges': tuple(ranges_flat),
    }
    # Also include pixel-size in the ImageJ metadata so Image > Properties
    # shows microns even when the TIFF resolution tags are ignored.
    try:
        res_xy = float(settings.input_resXY)
        meta['PhysicalSizeX'] = res_xy
        meta['PhysicalSizeXUnit'] = 'um'
        meta['PhysicalSizeY'] = res_xy
        meta['PhysicalSizeYUnit'] = 'um'
    except (AttributeError, TypeError, ValueError):
        pass

    imwrite(long_path(filename), multichannel_image,
            compression='zlib', compressionargs={'level': 1},
            imagej=True, photometric='minisblack',
            metadata=meta, **_imagej_resolution_kwargs(settings))

    # Return the merged 2D multi-channel image as a numpy array
    return multichannel_image


def create_mip_and_save_multichannel_tiff_4d(images_3d, filename, bitdepth, settings):
    """
    Create MIPs from a list of 3D images, merge them into a multi-channel 2D image,
    save as a 16-bit TIFF file, and return the merged 2D multi-channel image.

    Args:
        images_3d (list of numpy.ndarray): List of 3D numpy arrays representing input images.
        filename (str): Filename for the output TIFF file.

    Returns:
        numpy.ndarray: Merged 2D multi-channel image as a numpy array.
    """
    # Create MIPs from the 3D images
    mips = [np.amax(img, axis=1) for img in images_3d]

    # Convert the MIPs to a single multichannel image
    multichannel_image = np.stack(mips, axis=0)
    multichannel_image = np.swapaxes(multichannel_image, 0, 1)

    # Convert the multichannel image to 16-bit
    multichannel_image = multichannel_image.astype(np.uint16)
    #multichannel_image = rescale_all_channels_to_full_range(multichannel_image)

    # Save the multichannel image as a 16-bit TIFF file
    #imwrite(filename, multichannel_image, photometric='minisblack')

    imwrite(long_path(filename), multichannel_image, compression='zlib', compressionargs={'level': 1}, imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'TCYX'})

    # Return the merged 2D multi-channel image as a numpy array
    return multichannel_image


def rescale_all_channels_to_full_range(array):
    """
    Rescale all channels in a multi-channel numpy array to use the full range of 16-bit values.

    Args:
        array (numpy.ndarray): Input multi-channel numpy array.

    Returns:
        numpy.ndarray: Rescaled multi-channel numpy array.
    """
    num_channels = array.shape[0]

    for channel in range(num_channels):
        # Calculate the minimum and maximum values of the current channel
        min_val = np.min(array[channel])
        max_val = np.max(array[channel])

        # Rescale the current channel to the 16-bit range [0, 65535] using a linear transformation
        if max_val > min_val:
            array[channel] = (array[channel] - min_val) / (max_val - min_val) * 65535
        else:
            array[channel] = 0

        # Convert the rescaled channel to uint16 data type
        array[channel] = array[channel].astype(np.uint16)

    return array
